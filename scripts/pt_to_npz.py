"""Convert carry-box PT motion files to windowed NPZ files.

The robot part of the output matches ``scripts/csv_to_npz.py``. By default this
script also appends carry-box object features:

  windows      (N, window_size, 75)
  fps          (1,)
  window_size  (1,)
  stride       (1,)
  ee_body_names
  feature_dims

Expected PT payload is a tensor dict with at least:
  base_position, base_quat, joint_position
and, when ``include_box_features`` is enabled:
  box_pos_local

Velocity keys are optional when finite-difference velocity sources are used.
``joint_velocity`` is not part of the SMP pretraining feature layout and is
ignored. ``box_height_global`` is used when present; otherwise box height falls
back to the computed global box z.

Feature layout:
  robot features:
    root_pos(3), root_rot(6), joint_pos(29), ee_pos(15),
    root_lin_vel(3), root_ang_vel(3)
  box features:
    box_pos(3), box_height(1), box_rot(6), box_lin_vel(3), box_ang_vel(3)

Carrybox dataset notes:
  - ``base_quat`` is stored as xyzw, matching replay_carrybox_motion.py.
  - No explicit box rotation is stored in the current PT files, so this script
    uses the root heading quaternion for the box, matching replay_pt_motion.py.
  - The files are 60 fps; this script does not resample.
  - ``link_position`` contains carrybox-specific root-relative points, not the
    five world-frame end-effectors used by csv_to_npz.py. By default this
    script recomputes the five tracked G1 body positions with the same mjlab FK
    path used by csv_to_npz.py.

Usage:
  uv run scripts/pt_to_npz.py \
    --input-path datasets/motions/carry_box \
    --output-dir datasets/npz/carry_box
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro

from csv_to_npz import (
  EE_BODY_NAMES,
  JOINT_NAMES,
  NUM_EE,
  NUM_JOINTS,
  _compute_windows,
  _setup_sim,
  _tan_norm_from_quat,
)
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)
from smp.utils import detect_device

BASE_REQUIRED_KEYS: tuple[str, ...] = (
  "base_position",
  "base_quat",
  "joint_position",
)
ROBOT_FEATURE_NAMES: tuple[str, ...] = (
  "root_pos",
  "root_rot",
  "joint_pos",
  "ee_pos",
  "root_lin_vel",
  "root_ang_vel",
)
ROBOT_FEATURE_DIMS: tuple[int, ...] = (3, 6, NUM_JOINTS, NUM_EE * 3, 3, 3)
BOX_FEATURE_NAMES: tuple[str, ...] = (
  "box_pos",
  "box_height",
  "box_rot",
  "box_lin_vel",
  "box_ang_vel",
)
BOX_FEATURE_DIMS: tuple[int, ...] = (3, 1, 6, 3, 3)

RootZSource = Literal["base_position", "base_height", "motionlib"]
VelocitySource = Literal["file", "finite_difference"]
QuatOrder = Literal["wxyz", "xyzw"]
EeSource = Literal["fk", "link_position"]
LinkPositionFrame = Literal["world", "root_relative"]


@dataclass
class Cfg:
  input_path: str = "datasets/motions/carry_box"
  """Input .pt file, or directory containing .pt files."""
  output_dir: str = "datasets/npz/carry_box"
  """Directory to write output NPZ files."""
  output_name: str = ""
  """Optional output filename for a single input file. Defaults to <pt stem>.npz."""
  recursive: bool = True
  """Recursively find .pt files when input_path is a directory."""
  window_size: int = 10
  """Number of frames per window."""
  stride: int = 1
  """Stride between consecutive windows."""
  fps: int = 60
  """PT frame rate. The PT data is not resampled; finite differences use this."""
  device: str = ""
  """Compute device. Empty = auto (cuda if available else cpu)."""
  quat_order: QuatOrder = "xyzw"
  """Quaternion order in base_quat. Project math expects wxyz internally."""
  root_z_source: RootZSource = "motionlib"
  """Root z source: base_position.z, base_height, or base_height + z_offset."""
  z_offset: float = 0.05
  """Extra z offset for root_z_source=motionlib, matching replay_carrybox_motion.py."""
  linear_velocity_source: VelocitySource = "finite_difference"
  """Use file velocities, or recompute world linear velocity from base_position."""
  angular_velocity_source: VelocitySource = "finite_difference"
  """Use file angular velocities, or recompute world angular velocity from base_quat."""
  ee_source: EeSource = "fk"
  """End-effector source: recompute with G1 FK, or read link_position."""
  link_position_frame: LinkPositionFrame = "root_relative"
  """Frame of link_position when ee_source=link_position."""
  joint_indexes: str = ""
  """Comma-separated 29 joint indexes if joint_position has extra joints."""
  ee_indexes: str = ""
  """Comma-separated 5 link indexes if ee_source=link_position and link_position has extra links."""
  include_box_features: bool = True
  """Append carry-box object features after the 59 robot features."""
  box_quat_order: QuatOrder = "xyzw"
  """Quaternion order if a future PT payload provides box_quat."""
  box_linear_velocity_source: VelocitySource = "finite_difference"
  """Use box_linear_velocity from file, or finite-difference box position."""
  box_angular_velocity_source: VelocitySource = "finite_difference"
  """Use box_angular_velocity from file, or finite-difference box rotation."""
  shard_index: int = 0
  """Index of this shard for directory inputs."""
  num_shards: int = 1
  """Total number of shards for directory inputs."""


@dataclass
class FkContext:
  sim: Any
  scene: Any
  joint_indexes: torch.Tensor
  ee_indexes: torch.Tensor


@dataclass
class BoxTensors:
  pos: torch.Tensor
  quat: torch.Tensor
  lin_vel: torch.Tensor
  ang_vel: torch.Tensor
  height: torch.Tensor


def _parse_indexes(raw: str, expected: int, name: str) -> list[int]:
  if not raw:
    return []
  indexes = [int(part.strip()) for part in raw.split(",") if part.strip()]
  if len(indexes) != expected:
    msg = f"{name}: expected {expected} comma-separated indexes, got {len(indexes)}"
    raise ValueError(msg)
  if min(indexes) < 0:
    msg = f"{name}: indexes must be non-negative"
    raise ValueError(msg)
  return indexes


def _load_pt(path: Path) -> Mapping[str, Any]:
  try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
  except TypeError:
    payload = torch.load(path, map_location="cpu")
  if not isinstance(payload, Mapping):
    msg = f"{path}: expected a dict-like PT payload, got {type(payload).__name__}"
    raise TypeError(msg)
  missing = [key for key in BASE_REQUIRED_KEYS if key not in payload]
  if missing:
    msg = f"{path}: missing required keys: {missing}"
    raise KeyError(msg)
  return payload


def _as_float_tensor(
  payload: Mapping[str, Any], key: str, device: torch.device
) -> torch.Tensor:
  if key not in payload:
    msg = f"missing required key for selected options: {key}"
    raise KeyError(msg)
  value = payload[key]
  tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
  return tensor.to(device=device, dtype=torch.float32)


def _prod(shape: tuple[int, ...]) -> int:
  if not shape:
    return 1
  return reduce(mul, shape, 1)


def _reshape_clips(
  name: str,
  tensor: torch.Tensor,
  prefix_shape: tuple[int, ...],
  sample_shape: tuple[int, ...],
) -> torch.Tensor:
  if tuple(tensor.shape[-len(sample_shape) :]) != sample_shape:
    msg = f"{name}: expected trailing shape {sample_shape}, got {tuple(tensor.shape)}"
    raise ValueError(msg)
  if tuple(tensor.shape[: -len(sample_shape)]) != prefix_shape:
    msg = (
      f"{name}: expected frame prefix {prefix_shape}, got "
      f"{tuple(tensor.shape[:-len(sample_shape)])}"
    )
    raise ValueError(msg)
  if not prefix_shape:
    msg = f"{name}: missing time dimension"
    raise ValueError(msg)
  num_clips = _prod(prefix_shape[:-1])
  time_steps = prefix_shape[-1]
  return tensor.reshape(num_clips, time_steps, *sample_shape)


def _height_tensor(
  payload: Mapping[str, Any],
  device: torch.device,
  prefix_shape: tuple[int, ...],
) -> torch.Tensor:
  base_height = _as_float_tensor(payload, "base_height", device)
  if tuple(base_height.shape) == prefix_shape + (1,):
    base_height = base_height.squeeze(-1)
  if tuple(base_height.shape) != prefix_shape:
    msg = f"base_height: expected shape {prefix_shape}, got {tuple(base_height.shape)}"
    raise ValueError(msg)
  return base_height


def _root_position(
  payload: Mapping[str, Any],
  device: torch.device,
  root_z_source: RootZSource,
  z_offset: float,
) -> torch.Tensor:
  base_position = _as_float_tensor(payload, "base_position", device)
  if base_position.ndim < 2:
    msg = f"base_position: expected (..., 2|3), got {tuple(base_position.shape)}"
    raise ValueError(msg)

  last_dim = base_position.shape[-1]
  prefix_shape = tuple(base_position.shape[:-1])
  if last_dim not in (2, 3):
    msg = f"base_position: expected last dim 2 or 3, got {last_dim}"
    raise ValueError(msg)

  if root_z_source == "base_position":
    if last_dim != 3:
      msg = "root_z_source=base_position requires base_position shape (..., 3)"
      raise ValueError(msg)
    return base_position

  z = _height_tensor(payload, device, prefix_shape)
  if root_z_source == "motionlib":
    z = z + float(z_offset)

  if last_dim == 2:
    return torch.cat([base_position, z.unsqueeze(-1)], dim=-1)

  base_position = base_position.clone()
  base_position[..., 2] = z
  return base_position


def _root_quat(
  payload: Mapping[str, Any],
  device: torch.device,
  quat_order: QuatOrder,
  prefix_shape: tuple[int, ...],
) -> torch.Tensor:
  quat = _as_float_tensor(payload, "base_quat", device)
  if tuple(quat.shape[:-1]) != prefix_shape or quat.shape[-1] != 4:
    msg = f"base_quat: expected shape {prefix_shape + (4,)}, got {tuple(quat.shape)}"
    raise ValueError(msg)
  if quat_order == "xyzw":
    quat = quat[..., [3, 0, 1, 2]]
  norm = quat.norm(dim=-1, keepdim=True)
  if torch.any(norm < 1e-8):
    msg = "base_quat: found near-zero quaternion"
    raise ValueError(msg)
  return quat / norm


def _quat_tensor(
  payload: Mapping[str, Any],
  key: str,
  device: torch.device,
  quat_order: QuatOrder,
  prefix_shape: tuple[int, ...],
) -> torch.Tensor:
  quat = _as_float_tensor(payload, key, device)
  if tuple(quat.shape[:-1]) != prefix_shape or quat.shape[-1] != 4:
    msg = f"{key}: expected shape {prefix_shape + (4,)}, got {tuple(quat.shape)}"
    raise ValueError(msg)
  if quat_order == "xyzw":
    quat = quat[..., [3, 0, 1, 2]]
  norm = quat.norm(dim=-1, keepdim=True)
  if torch.any(norm < 1e-8):
    msg = f"{key}: found near-zero quaternion"
    raise ValueError(msg)
  return quat / norm


def _joint_position(
  payload: Mapping[str, Any],
  device: torch.device,
  prefix_shape: tuple[int, ...],
  joint_indexes: list[int],
) -> torch.Tensor:
  joint_pos = _as_float_tensor(payload, "joint_position", device)
  if tuple(joint_pos.shape[:-1]) != prefix_shape:
    msg = (
      f"joint_position: expected frame prefix {prefix_shape}, got "
      f"{tuple(joint_pos.shape[:-1])}"
    )
    raise ValueError(msg)
  total_joints = joint_pos.shape[-1]
  if joint_indexes:
    if max(joint_indexes) >= total_joints:
      msg = f"joint_indexes: max index {max(joint_indexes)} >= {total_joints}"
      raise ValueError(msg)
    idx = torch.tensor(joint_indexes, dtype=torch.long, device=device)
    return joint_pos.index_select(-1, idx)
  if total_joints != NUM_JOINTS:
    msg = (
      f"joint_position has {total_joints} joints, expected {NUM_JOINTS}. "
      "Pass --joint-indexes with the 29 G1 joint indexes in csv_to_npz.py order."
    )
    raise ValueError(msg)
  return joint_pos


def _ee_position_from_link_position(
  payload: Mapping[str, Any],
  device: torch.device,
  prefix_shape: tuple[int, ...],
  root_pos: torch.Tensor,
  ee_indexes: list[int],
  link_position_frame: LinkPositionFrame,
) -> torch.Tensor:
  link_pos = _as_float_tensor(payload, "link_position", device)

  if link_pos.shape[-1] == NUM_EE * 3 and tuple(link_pos.shape[:-1]) == prefix_shape:
    ee_pos = link_pos.reshape(*prefix_shape, NUM_EE, 3)
  else:
    if link_pos.ndim < 3 or link_pos.shape[-1] != 3:
      msg = (
        "link_position: expected shape (..., num_links, 3), or flattened "
        f"(..., {NUM_EE * 3}) for the tracked end-effectors"
      )
      raise ValueError(msg)
    if tuple(link_pos.shape[:-2]) != prefix_shape:
      msg = (
        f"link_position: expected frame prefix {prefix_shape}, got "
        f"{tuple(link_pos.shape[:-2])}"
      )
      raise ValueError(msg)

    num_links = link_pos.shape[-2]
    if ee_indexes:
      if max(ee_indexes) >= num_links:
        msg = f"ee_indexes: max index {max(ee_indexes)} >= {num_links}"
        raise ValueError(msg)
      idx = torch.tensor(ee_indexes, dtype=torch.long, device=device)
      ee_pos = link_pos.index_select(-2, idx)
    elif num_links == NUM_EE:
      ee_pos = link_pos
    else:
      msg = (
        f"link_position has {num_links} links. Either use --ee-source fk "
        "or pass --ee-indexes with the 5 indexes for "
        f"{EE_BODY_NAMES}, in that order."
      )
      raise ValueError(msg)

  if link_position_frame == "root_relative":
    ee_pos = ee_pos + root_pos[..., None, :]
  return ee_pos


def _box_pose_tensors(
  payload: Mapping[str, Any],
  cfg: Cfg,
  device: torch.device,
  prefix_shape: tuple[int, ...],
  base_pos: torch.Tensor,
  base_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  box_pos_local = _as_float_tensor(payload, "box_pos_local", device)
  if tuple(box_pos_local.shape[:-1]) != prefix_shape or box_pos_local.shape[-1] != 3:
    msg = (
      f"box_pos_local: expected shape {prefix_shape + (3,)}, "
      f"got {tuple(box_pos_local.shape)}"
    )
    raise ValueError(msg)

  box_pos = base_pos + box_pos_local

  if "box_quat" in payload:
    box_quat = _quat_tensor(
      payload,
      "box_quat",
      device,
      cfg.box_quat_order,
      prefix_shape,
    )
  else:
    box_quat = yaw_quat(base_quat.reshape(-1, 4)).reshape(*prefix_shape, 4)

  if "box_height_global" in payload:
    box_height = _as_float_tensor(payload, "box_height_global", device)
    if tuple(box_height.shape) == prefix_shape + (1,):
      box_height = box_height.squeeze(-1)
    if tuple(box_height.shape) != prefix_shape:
      msg = (
        f"box_height_global: expected shape {prefix_shape}, "
        f"got {tuple(box_height.shape)}"
      )
      raise ValueError(msg)
    if cfg.root_z_source == "motionlib":
      box_height = box_height + float(cfg.z_offset)
    box_height = box_height.unsqueeze(-1)
  else:
    box_height = box_pos[..., 2:3]

  return box_pos, box_quat, box_height


def _finite_difference_linear_velocity(
  base_pos: torch.Tensor,
  fps: int | float,
) -> torch.Tensor:
  vel = torch.zeros_like(base_pos)
  if base_pos.shape[1] < 2:
    return vel
  vel[:, 0] = (base_pos[:, 1] - base_pos[:, 0]) * float(fps)
  vel[:, -1] = (base_pos[:, -1] - base_pos[:, -2]) * float(fps)
  if base_pos.shape[1] > 2:
    vel[:, 1:-1] = (base_pos[:, 2:] - base_pos[:, :-2]) * (0.5 * float(fps))
  return vel


def _quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
  out = quat.clone()
  out[..., 1:] = -out[..., 1:]
  return out


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(dim=-1)
  bw, bx, by, bz = b.unbind(dim=-1)
  return torch.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    dim=-1,
  )


def _finite_difference_angular_velocity(
  base_quat: torch.Tensor,
  fps: int | float,
) -> torch.Tensor:
  ang_vel = torch.zeros(*base_quat.shape[:-1], 3, device=base_quat.device)
  if base_quat.shape[1] < 3:
    return ang_vel

  delta = _quat_mul_wxyz(
    base_quat[:, 2:],
    _quat_conjugate_wxyz(base_quat[:, :-2]),
  )
  delta = torch.where(delta[..., :1] < 0.0, -delta, delta)
  delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)

  vec = delta[..., 1:]
  sin_half = vec.norm(dim=-1)
  angle = 2.0 * torch.atan2(sin_half, delta[..., 0].clamp(-1.0, 1.0))
  axis = vec / sin_half[..., None].clamp_min(1e-8)
  rotvec = axis * angle[..., None]
  rotvec = torch.where(sin_half[..., None] > 1e-8, rotvec, 2.0 * vec)

  omega = rotvec * (0.5 * float(fps))
  ang_vel[:, 1:-1] = omega
  ang_vel[:, 0] = omega[:, 0]
  ang_vel[:, -1] = omega[:, -1]
  return ang_vel


def _box_tensors(
  payload: Mapping[str, Any],
  cfg: Cfg,
  device: torch.device,
  prefix_shape: tuple[int, ...],
  base_pos: torch.Tensor,
  base_quat: torch.Tensor,
) -> BoxTensors | None:
  if not cfg.include_box_features:
    return None

  box_pos, box_quat, box_height = _box_pose_tensors(
    payload,
    cfg,
    device,
    prefix_shape,
    base_pos,
    base_quat,
  )
  box_pos = _reshape_clips("box_position", box_pos, prefix_shape, (3,))
  box_quat = _reshape_clips("box_quat", box_quat, prefix_shape, (4,))
  box_height = _reshape_clips("box_height", box_height, prefix_shape, (1,))

  if cfg.box_linear_velocity_source == "file":
    box_lin_vel = _as_float_tensor(payload, "box_linear_velocity", device)
    box_lin_vel = _reshape_clips("box_linear_velocity", box_lin_vel, prefix_shape, (3,))
  else:
    box_lin_vel = _finite_difference_linear_velocity(box_pos, cfg.fps)

  if cfg.box_angular_velocity_source == "file":
    box_ang_vel = _as_float_tensor(payload, "box_angular_velocity", device)
    box_ang_vel = _reshape_clips("box_angular_velocity", box_ang_vel, prefix_shape, (3,))
  else:
    box_ang_vel = _finite_difference_angular_velocity(box_quat, cfg.fps)

  return BoxTensors(
    pos=box_pos,
    quat=box_quat,
    lin_vel=box_lin_vel,
    ang_vel=box_ang_vel,
    height=box_height,
  )


def _compute_box_windows(
  base_pos: torch.Tensor,
  base_quat: torch.Tensor,
  box_pos: torch.Tensor,
  box_quat: torch.Tensor,
  box_lin_vel: torch.Tensor,
  box_ang_vel: torch.Tensor,
  box_height: torch.Tensor,
  window_size: int,
  stride: int,
) -> torch.Tensor | None:
  T = base_pos.shape[0]
  if T < window_size:
    return None

  starts = torch.arange(
    0, T - window_size + 1, stride, device=base_pos.device, dtype=torch.long
  )
  offsets = torch.arange(window_size, device=base_pos.device, dtype=torch.long)
  win_idx = starts[:, None] + offsets[None, :]
  N, W = win_idx.shape[0], window_size

  flat_idx = win_idx.reshape(-1)
  win_base_pos = base_pos.index_select(0, flat_idx).reshape(N, W, 3)
  win_base_quat = base_quat.index_select(0, flat_idx).reshape(N, W, 4)
  win_box_pos = box_pos.index_select(0, flat_idx).reshape(N, W, 3)
  win_box_quat = box_quat.index_select(0, flat_idx).reshape(N, W, 4)
  win_box_lin_vel = box_lin_vel.index_select(0, flat_idx).reshape(N, W, 3)
  win_box_ang_vel = box_ang_vel.index_select(0, flat_idx).reshape(N, W, 3)
  win_box_height = box_height.index_select(0, flat_idx).reshape(N, W, 1)

  anchor_quat_T = win_base_quat[:, -1, :]
  yaw_T = yaw_quat(anchor_quat_T)
  heading_inv_T_WF = quat_conjugate(yaw_T)[:, None, :].expand(N, W, 4).reshape(-1, 4)
  yaw_T_W = yaw_T[:, None, :].expand(N, W, 4).reshape(-1, 4)

  box_offset_w = win_box_pos - win_base_pos
  box_pos_local = quat_apply_inverse(yaw_T_W, box_offset_w.reshape(-1, 3)).reshape(
    N, W, 3
  )
  box_rot_local_quat = quat_mul(
    heading_inv_T_WF, win_box_quat.reshape(-1, 4)
  ).reshape(N, W, 4)
  box_rot_6d = _tan_norm_from_quat(box_rot_local_quat)
  box_lin_vel_local = quat_apply_inverse(
    yaw_T_W, win_box_lin_vel.reshape(-1, 3)
  ).reshape(N, W, 3)
  box_ang_vel_local = quat_apply_inverse(
    yaw_T_W, win_box_ang_vel.reshape(-1, 3)
  ).reshape(N, W, 3)

  return torch.cat(
    [
      box_pos_local,
      win_box_height,
      box_rot_6d,
      box_lin_vel_local,
      box_ang_vel_local,
    ],
    dim=-1,
  )


def _motion_tensors(
  payload: Mapping[str, Any],
  cfg: Cfg,
  device: torch.device,
) -> tuple[
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor | None,
  torch.Tensor,
  BoxTensors | None,
]:
  joint_indexes = _parse_indexes(cfg.joint_indexes, NUM_JOINTS, "joint_indexes")
  ee_indexes = _parse_indexes(cfg.ee_indexes, NUM_EE, "ee_indexes")

  base_pos_raw = _root_position(payload, device, cfg.root_z_source, cfg.z_offset)
  prefix_shape = tuple(base_pos_raw.shape[:-1])
  base_quat_raw = _root_quat(payload, device, cfg.quat_order, prefix_shape)
  joint_pos_raw = _joint_position(payload, device, prefix_shape, joint_indexes)
  box = _box_tensors(
    payload,
    cfg,
    device,
    prefix_shape,
    base_pos_raw,
    base_quat_raw,
  )

  base_pos = _reshape_clips("base_position", base_pos_raw, prefix_shape, (3,))
  base_quat = _reshape_clips("base_quat", base_quat_raw, prefix_shape, (4,))
  joint_pos = _reshape_clips(
    "joint_position", joint_pos_raw, prefix_shape, (NUM_JOINTS,)
  )

  if cfg.linear_velocity_source == "file":
    base_lin_vel = _as_float_tensor(payload, "base_linear_velocity", device)
    base_lin_vel = _reshape_clips(
      "base_linear_velocity", base_lin_vel, prefix_shape, (3,)
    )
  else:
    base_lin_vel = _finite_difference_linear_velocity(base_pos, cfg.fps)

  if cfg.angular_velocity_source == "file":
    base_ang_vel = _as_float_tensor(payload, "base_angular_velocity", device)
    base_ang_vel = _reshape_clips(
      "base_angular_velocity", base_ang_vel, prefix_shape, (3,)
    )
  else:
    base_ang_vel = _finite_difference_angular_velocity(base_quat, cfg.fps)

  ee_pos: torch.Tensor | None
  if cfg.ee_source == "link_position":
    ee_pos_raw = _ee_position_from_link_position(
      payload,
      device,
      prefix_shape,
      _root_position(payload, device, cfg.root_z_source, cfg.z_offset),
      ee_indexes,
      cfg.link_position_frame,
    )
    ee_pos = _reshape_clips("link_position", ee_pos_raw, prefix_shape, (NUM_EE, 3))
  else:
    ee_pos = None

  finite_checks = [
    ("base_position", base_pos),
    ("base_quat", base_quat),
    ("base_linear_velocity", base_lin_vel),
    ("base_angular_velocity", base_ang_vel),
    ("joint_position", joint_pos),
  ]
  if ee_pos is not None:
    finite_checks.append(("link_position", ee_pos))
  if box is not None:
    finite_checks.extend(
      [
        ("box_position", box.pos),
        ("box_quat", box.quat),
        ("box_linear_velocity", box.lin_vel),
        ("box_angular_velocity", box.ang_vel),
        ("box_height", box.height),
      ]
    )
  for name, tensor in finite_checks:
    if not torch.isfinite(tensor).all():
      msg = f"{name}: found non-finite values"
      raise ValueError(msg)

  return base_pos, base_quat, base_lin_vel, base_ang_vel, ee_pos, joint_pos, box


def _setup_fk_context(device: str) -> FkContext:
  sim, scene = _setup_sim(device)
  robot = scene["robot"]
  joint_indexes = torch.tensor(
    robot.find_joints(list(JOINT_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=sim.device,
  )
  ee_indexes = torch.tensor(
    robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=sim.device,
  )
  return FkContext(
    sim=sim,
    scene=scene,
    joint_indexes=joint_indexes,
    ee_indexes=ee_indexes,
  )


@torch.no_grad()
def _fk_ee_positions(
  base_pos: torch.Tensor,
  base_quat: torch.Tensor,
  joint_pos: torch.Tensor,
  fk: FkContext,
) -> torch.Tensor:
  robot = fk.scene["robot"]
  ee_pos_list: list[torch.Tensor] = []

  fk.scene.reset()
  for frame_idx in range(base_pos.shape[0]):
    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos[frame_idx]
    root_states[:, :2] += fk.scene.env_origins[:, :2]
    root_states[:, 3:7] = base_quat[frame_idx]
    robot.write_root_state_to_sim(root_states)

    joint_pos_full = robot.data.default_joint_pos.clone()
    joint_vel_full = robot.data.default_joint_vel.clone()
    joint_pos_full[:, fk.joint_indexes] = joint_pos[frame_idx]
    robot.write_joint_state_to_sim(joint_pos_full, joint_vel_full)

    fk.sim.forward()
    fk.scene.update(fk.sim.mj_model.opt.timestep)
    ee_pos_list.append(robot.data.body_link_pos_w[0, fk.ee_indexes].clone())

  return torch.stack(ee_pos_list, dim=0)


def _input_files(
  input_path: Path,
  recursive: bool,
  shard_index: int,
  num_shards: int,
) -> list[Path]:
  if input_path.is_file():
    files = [input_path]
  elif input_path.is_dir():
    pattern = "**/*.pt" if recursive else "*.pt"
    files = sorted(input_path.glob(pattern))
    if not files:
      msg = f"No .pt files found in {input_path}"
      raise FileNotFoundError(msg)
  else:
    msg = f"Input path does not exist: {input_path}"
    raise FileNotFoundError(msg)

  if num_shards < 1:
    msg = "num_shards must be >= 1"
    raise ValueError(msg)
  if not 0 <= shard_index < num_shards:
    msg = f"shard_index must be in [0, {num_shards}), got {shard_index}"
    raise ValueError(msg)
  return files[shard_index::num_shards]


def _output_path(pt_path: Path, out_dir: Path, output_name: str, num_files: int) -> Path:
  if output_name:
    if num_files != 1:
      msg = "--output-name can only be used with a single input file"
      raise ValueError(msg)
    name = output_name if output_name.endswith(".npz") else f"{output_name}.npz"
    return out_dir / name
  return out_dir / f"{pt_path.stem}.npz"


def _check_output_collisions(out_paths: list[Path]) -> None:
  seen: dict[Path, Path] = {}
  for out_path in out_paths:
    old = seen.get(out_path)
    if old is not None:
      msg = f"multiple PT files would write the same output file: {out_path}"
      raise ValueError(msg)
    seen[out_path] = out_path


def _convert_file(
  pt_path: Path,
  out_path: Path,
  cfg: Cfg,
  device: torch.device,
  fk: FkContext | None,
) -> int:
  payload = _load_pt(pt_path)
  (
    base_pos,
    base_quat,
    base_lin_vel,
    base_ang_vel,
    ee_pos,
    joint_pos,
    box,
  ) = _motion_tensors(payload, cfg, device)

  windows_per_clip: list[torch.Tensor] = []
  for clip_idx in range(base_pos.shape[0]):
    if ee_pos is None:
      if fk is None:
        msg = "ee_source=fk requires an FK context"
        raise RuntimeError(msg)
      clip_ee_pos = _fk_ee_positions(
        base_pos[clip_idx],
        base_quat[clip_idx],
        joint_pos[clip_idx],
        fk,
      )
    else:
      clip_ee_pos = ee_pos[clip_idx]

    windows = _compute_windows(
      base_pos[clip_idx],
      base_quat[clip_idx],
      base_lin_vel[clip_idx],
      base_ang_vel[clip_idx],
      clip_ee_pos,
      joint_pos[clip_idx],
      cfg.window_size,
      cfg.stride,
    )
    if windows is not None:
      if box is not None:
        box_windows = _compute_box_windows(
          base_pos[clip_idx],
          base_quat[clip_idx],
          box.pos[clip_idx],
          box.quat[clip_idx],
          box.lin_vel[clip_idx],
          box.ang_vel[clip_idx],
          box.height[clip_idx],
          cfg.window_size,
          cfg.stride,
        )
        if box_windows is None:
          msg = "robot windows were produced but box windows were None"
          raise RuntimeError(msg)
        windows = torch.cat([windows, box_windows], dim=-1)
      windows_per_clip.append(windows)

  if not windows_per_clip:
    print(f"  [SKIP] too short for window_size={cfg.window_size}")
    return 0

  windows = torch.cat(windows_per_clip, dim=0)
  feature_dims = list(ROBOT_FEATURE_DIMS)
  feature_names = list(ROBOT_FEATURE_NAMES)
  if box is not None:
    feature_dims.extend(BOX_FEATURE_DIMS)
    feature_names.extend(BOX_FEATURE_NAMES)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    out_path,
    windows=windows.cpu().numpy().astype(np.float32),
    fps=np.array([cfg.fps], dtype=np.float32),
    window_size=np.array([cfg.window_size], dtype=np.int32),
    stride=np.array([cfg.stride], dtype=np.int32),
    ee_body_names=np.array(EE_BODY_NAMES),
    feature_dims=np.array(feature_dims, dtype=np.int32),
    feature_names=np.array(feature_names),
    includes_box=np.array([box is not None], dtype=np.bool_),
  )
  print(
    f"  saved {out_path.name}: windows={tuple(windows.shape)} "
    f"clips={base_pos.shape[0]} frames/clip={base_pos.shape[1]}"
  )
  return int(windows.shape[0])


def main(cfg: Cfg) -> None:
  if not cfg.device:
    cfg.device = detect_device()
  device = torch.device(cfg.device)

  input_path = Path(cfg.input_path)
  out_dir = Path(cfg.output_dir)
  pt_files = _input_files(
    input_path,
    recursive=cfg.recursive,
    shard_index=cfg.shard_index,
    num_shards=cfg.num_shards,
  )
  if not pt_files:
    msg = f"Shard {cfg.shard_index}/{cfg.num_shards} has no files"
    raise FileNotFoundError(msg)
  out_paths = [
    _output_path(pt_path, out_dir, cfg.output_name, len(pt_files))
    for pt_path in pt_files
  ]
  _check_output_collisions(out_paths)

  print(f"Device: {device}")
  print(f"Files: {len(pt_files)} from {input_path}")
  print(f"Output: {out_dir}")
  print(f"Window: size={cfg.window_size} stride={cfg.stride} fps={cfg.fps}")
  print(f"Root z: {cfg.root_z_source} (z_offset={cfg.z_offset:g})")
  print(
    "Velocity source: "
    f"linear={cfg.linear_velocity_source} angular={cfg.angular_velocity_source}"
  )
  print(f"EE source: {cfg.ee_source}")
  print(
    "Box features: "
    f"{'on' if cfg.include_box_features else 'off'} "
    f"(linear_vel={cfg.box_linear_velocity_source}, "
    f"angular_vel={cfg.box_angular_velocity_source})"
  )
  print(f"End-effectors: {NUM_EE} {EE_BODY_NAMES} | Joints: {NUM_JOINTS}")
  feature_dim = sum(ROBOT_FEATURE_DIMS)
  if cfg.include_box_features:
    feature_dim += sum(BOX_FEATURE_DIMS)
  print(f"Feature dim: {feature_dim}")
  if cfg.num_shards > 1:
    print(f"Shard {cfg.shard_index}/{cfg.num_shards}: {len(pt_files)} files")

  fk = _setup_fk_context(cfg.device) if cfg.ee_source == "fk" else None

  total_windows = 0
  for i, (pt_path, out_path) in enumerate(zip(pt_files, out_paths, strict=True)):
    print(f"\n[{i + 1}/{len(pt_files)}] {pt_path.name}")
    total_windows += _convert_file(pt_path, out_path, cfg, device, fk)

  if total_windows == 0:
    msg = "No windows were written. Check clip length, window_size and stride."
    raise RuntimeError(msg)
  print(f"\nDone: wrote {total_windows} windows")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
