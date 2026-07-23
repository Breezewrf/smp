"""Convert CSV motion files to windowed NPZ files.

Each output NPZ contains a ``windows`` array of shape ``(N, window_size, F)``
with the following 59-dim per-frame layout for G1 and X2:

  root_pos        (3)              xy in last-frame heading-inv frame
                                    relative to last root; z in world
  root_rot        (6)              6D tan-norm of heading_inv(T) ⊗ root_quat[t]
  joint_pos       (num_joints)     raw joint angles
  ee_pos          (num_ee*3=15)    end-effectors, per-frame root offset,
                                    last-frame heading-inv rotation
  root_lin_vel    (3)              last-frame heading-inv
  root_ang_vel    (3)              last-frame heading-inv

The anchor frame for every spatial quantity is the LAST window frame's
yaw-only local frame (origin at pelvis_T, x-axis = heading_T direction).

Usage:
  uv run scripts/csv_to_npz.py --input-dir datasets/csv --output-dir datasets/npz
  uv run scripts/csv_to_npz.py --robot x2 --input-dir datasets/x2/csv \
    --output-dir datasets/x2/npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro
from mjlab.entity import Entity, EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.scripts.csv_to_npz import MotionLoader as CsvMotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)

from smp.robots import (
  G1_EE_BODY_NAMES,
  G1_JOINT_NAMES,
  X2_CSV_JOINT_NAMES,
  X2_EE_BODY_NAMES,
  X2_JOINT_NAMES,
)
from smp.robots.x2 import X2_XML_PATH
from smp.utils import detect_device


@dataclass(frozen=True)
class RobotMotionCfg:
  csv_joint_names: tuple[str, ...]
  joint_names: tuple[str, ...]
  ee_body_names: tuple[str, ...]


ROBOT_CONFIGS: dict[str, RobotMotionCfg] = {
  "g1": RobotMotionCfg(G1_JOINT_NAMES, G1_JOINT_NAMES, G1_EE_BODY_NAMES),
  "x2": RobotMotionCfg(X2_CSV_JOINT_NAMES, X2_JOINT_NAMES, X2_EE_BODY_NAMES),
}

@dataclass
class Cfg:
  robot: Literal["g1", "x2"] = "g1"
  """Robot model and corresponding CSV joint layout."""
  input_dir: str = "datasets/csv"
  """Directory of input CSV motion files."""
  output_dir: str = "datasets/npz"
  """Directory to write output NPZ window files."""
  window_size: int = 10
  """Number of frames per window."""
  stride: int = 1
  """Stride between consecutive windows."""
  input_fps: int = 30
  """CSV frame rate."""
  output_fps: int = 50
  """Output (and sim) frame rate after interpolation."""
  device: str = ""
  """Compute device. Empty = auto (cuda if available else cpu)."""
  shard_index: int = 0
  """Index of this shard (for parallel runs). Files are sliced as [shard_index::num_shards]."""
  num_shards: int = 1
  """Total number of shards (for parallel runs)."""


def _setup_sim(device: str, robot_name: str) -> tuple[Simulation, Scene]:
  """Build the selected robot's FK simulation once."""
  sim_cfg = SimulationCfg()
  if robot_name == "g1":
    scene_cfg = unitree_g1_flat_tracking_env_cfg().scene
  elif robot_name == "x2":
    if not X2_XML_PATH.is_file():
      raise FileNotFoundError(f"X2 MJCF not found: {X2_XML_PATH}")

    def load_x2_spec() -> mujoco.MjSpec:
      return mujoco.MjSpec.from_file(str(X2_XML_PATH))

    # FK does not require terrain, sensors, or a tracking environment.
    scene_cfg = SceneCfg(
      entities={"robot": EntityCfg(spec_fn=load_x2_spec)},
      num_envs=1,
    )
  else:
    raise ValueError(f"Unsupported robot: {robot_name}")

  scene = Scene(scene_cfg, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene


@torch.no_grad()
def _fk_motion(
  csv_path: Path,
  sim: Simulation,
  scene: Scene,
  joint_indexes: torch.Tensor,
  ee_indexes: torch.Tensor,
  motion_joint_indexes: torch.Tensor,
  expected_csv_joints: int,
  input_fps: int,
  output_fps: int,
) -> tuple[
  torch.Tensor,  # base_pos
  torch.Tensor,  # base_quat
  torch.Tensor,  # base_lin_vel
  torch.Tensor,  # base_ang_vel
  torch.Tensor,  # ee_pos  (T, num_ee, 3)
  torch.Tensor,  # joint_pos  (T, num_joints)
  torch.Tensor,  # joint_vel  (T, num_joints)
]:
  """Replay a CSV through the sim, returning interpolated base state and
  FK'd world-frame positions for the end-effector bodies."""
  motion = CsvMotionLoader(
    motion_file=str(csv_path),
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
  )
  if motion.motion_dof_poss.shape[1] != expected_csv_joints:
    raise ValueError(
      f"{csv_path.name}: robot expects {expected_csv_joints} joint columns after the "
      f"7 root-state columns, got {motion.motion_dof_poss.shape[1]}"
    )
  robot: Entity = scene["robot"]

  ee_pos_list: list[torch.Tensor] = []

  scene.reset()
  for _ in range(motion.output_frames):
    state, _ = motion.get_next_state()
    base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel = state
    dof_pos = dof_pos.index_select(1, motion_joint_indexes)
    dof_vel = dof_vel.index_select(1, motion_joint_indexes)

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos_full = robot.data.default_joint_pos.clone()
    joint_vel_full = robot.data.default_joint_vel.clone()
    joint_pos_full[:, joint_indexes] = dof_pos
    joint_vel_full[:, joint_indexes] = dof_vel
    robot.write_joint_state_to_sim(joint_pos_full, joint_vel_full)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    ee_pos_list.append(robot.data.body_link_pos_w[0, ee_indexes].clone())

  return (
    motion.motion_base_poss,
    motion.motion_base_rots,
    motion.motion_base_lin_vels,
    motion.motion_base_ang_vels,
    torch.stack(ee_pos_list),
    motion.motion_dof_poss.index_select(1, motion_joint_indexes),
    motion.motion_dof_vels.index_select(1, motion_joint_indexes),
  )


def _tan_norm_from_quat(quat: torch.Tensor) -> torch.Tensor:
  """Convert quaternion (wxyz) to 6D tan-norm.

  Stacks the rotation matrix's first column (rotated x-axis) and third
  column (rotated z-axis).  This is the "tangent + normal" 6D
  representation, NOT Zhou-2019's "first two columns" form.  Input
  ``(..., 4)``, output ``(..., 6)`` as ``[col0_xyz, col2_xyz]``.
  """
  mat = matrix_from_quat(quat)
  col0 = mat[..., :, 0]
  col2 = mat[..., :, 2]
  return torch.cat([col0, col2], dim=-1)


def _compute_windows(
  base_pos: torch.Tensor,
  base_quat: torch.Tensor,
  base_lin_vel: torch.Tensor,
  base_ang_vel: torch.Tensor,
  ee_pos: torch.Tensor,
  joint_pos: torch.Tensor,
  window_size: int,
  stride: int,
) -> torch.Tensor | None:
  """Slice into windows and compute the per-frame motion features.

  All spatial quantities anchored to the LAST window frame's yaw-only local
  frame (origin at pelvis_T, heading = yaw_T).  Joint velocities are NOT
  part of the feature output.

  Returns ``(num_windows, window_size, 3+6+J+E*3+3+3)`` or ``None`` if the
  input is too short.
  """
  T = base_pos.shape[0]
  if T < window_size:
    return None

  E = ee_pos.shape[1]
  J = joint_pos.shape[1]
  starts = torch.arange(
    0, T - window_size + 1, stride, device=base_pos.device, dtype=torch.long
  )
  offsets = torch.arange(window_size, device=base_pos.device, dtype=torch.long)
  win_idx = starts[:, None] + offsets[None, :]
  N, W = win_idx.shape[0], window_size

  flat_idx = win_idx.reshape(-1)
  win_base_pos = base_pos.index_select(0, flat_idx).reshape(N, W, 3)
  win_base_quat = base_quat.index_select(0, flat_idx).reshape(N, W, 4)
  win_base_lin_vel = base_lin_vel.index_select(0, flat_idx).reshape(N, W, 3)
  win_base_ang_vel = base_ang_vel.index_select(0, flat_idx).reshape(N, W, 3)
  win_ee_pos = ee_pos.index_select(0, flat_idx).reshape(N, W, E, 3)
  win_joint = joint_pos.index_select(0, flat_idx).reshape(N, W, J)

  anchor_pos_T = win_base_pos[:, -1, :]
  anchor_quat_T = win_base_quat[:, -1, :]
  yaw_T = yaw_quat(anchor_quat_T)
  heading_inv_T_WF = quat_conjugate(yaw_T)[:, None, :].expand(N, W, 4).reshape(-1, 4)
  yaw_T_W = yaw_T[:, None, :].expand(N, W, 4).reshape(-1, 4)

  # root_pos: xy in heading-inv frame, z in world.
  root_offset = win_base_pos - anchor_pos_T[:, None, :]
  root_pos_local = quat_apply_inverse(yaw_T_W, root_offset.reshape(-1, 3)).reshape(
    N, W, 3
  )
  root_pos_local = root_pos_local.clone()
  root_pos_local[..., 2] = win_base_pos[..., 2]

  # root_rot: tan-norm of heading_inv(T) ⊗ root_quat[t].
  root_rot_local_quat = quat_mul(
    heading_inv_T_WF, win_base_quat.reshape(-1, 4)
  ).reshape(N, W, 4)
  root_rot_6d = _tan_norm_from_quat(root_rot_local_quat)

  # EE: (ee[t] - root[t]) rotated into the last-frame heading-inv frame.
  ee_offset_w = win_ee_pos - win_base_pos[:, :, None, :]
  yaw_T_E = yaw_T[:, None, None, :].expand(N, W, E, 4).reshape(-1, 4)
  ee_pos_local = quat_apply_inverse(yaw_T_E, ee_offset_w.reshape(-1, 3)).reshape(
    N, W, E * 3
  )

  lin_vel_local = quat_apply_inverse(yaw_T_W, win_base_lin_vel.reshape(-1, 3)).reshape(
    N, W, 3
  )
  ang_vel_local = quat_apply_inverse(yaw_T_W, win_base_ang_vel.reshape(-1, 3)).reshape(
    N, W, 3
  )

  return torch.cat(
    [
      root_pos_local,
      root_rot_6d,
      win_joint,
      ee_pos_local,
      lin_vel_local,
      ang_vel_local,
    ],
    dim=-1,
  )


def main(cfg: Cfg) -> None:
  if not cfg.device:
    cfg.device = detect_device()
  print(f"Device: {cfg.device}")

  robot_motion_cfg = ROBOT_CONFIGS[cfg.robot]
  csv_joint_names = robot_motion_cfg.csv_joint_names
  joint_names = robot_motion_cfg.joint_names
  ee_body_names = robot_motion_cfg.ee_body_names
  num_joints = len(joint_names)
  num_ee = len(ee_body_names)

  in_dir = Path(cfg.input_dir)
  out_dir = Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  csv_files = sorted(in_dir.glob("*.csv"))
  if not csv_files:
    msg = f"No CSV files found in {in_dir}"
    raise FileNotFoundError(msg)
  if cfg.num_shards > 1:
    csv_files = csv_files[cfg.shard_index :: cfg.num_shards]
    print(f"Shard {cfg.shard_index}/{cfg.num_shards}: {len(csv_files)} files")

  sim, scene = _setup_sim(cfg.device, cfg.robot)
  robot: Entity = scene["robot"]
  joint_indexes = torch.tensor(
    robot.find_joints(list(joint_names), preserve_order=True)[0],
    dtype=torch.long,
    device=sim.device,
  )
  ee_indexes = torch.tensor(
    robot.find_bodies(list(ee_body_names), preserve_order=True)[0],
    dtype=torch.long,
    device=sim.device,
  )
  motion_joint_indexes = torch.tensor(
    [csv_joint_names.index(name) for name in joint_names],
    dtype=torch.long,
    device=sim.device,
  )
  if joint_indexes.numel() != num_joints:
    raise ValueError(
      f"{cfg.robot}: found {joint_indexes.numel()} of {num_joints} configured joints"
    )
  if ee_indexes.numel() != num_ee:
    raise ValueError(
      f"{cfg.robot}: found {ee_indexes.numel()} of {num_ee} configured end-effectors"
    )

  feature_dims = [3, 6, num_joints, num_ee * 3, 3, 3]
  total_feature_dim = sum(feature_dims)

  print(f"Robot: {cfg.robot}")
  print(f"Files: {len(csv_files)} in {in_dir}")
  print(f"Output: {out_dir}")
  print(f"Window: size={cfg.window_size} stride={cfg.stride} fps={cfg.output_fps}")
  print(
    f"End-effectors: {num_ee} {ee_body_names} | "
    f"CSV joints: {len(csv_joint_names)} | Output joints: {num_joints}"
  )
  print(
    f"Feature dim: {total_feature_dim} "
    f"(= 3 root_pos + 6 root_rot + {num_joints} joint_pos + {num_ee * 3} "
    f"ee_pos + 3 lin_vel + 3 ang_vel)"
  )

  for i, csv_path in enumerate(csv_files):
    print(f"\n[{i + 1}/{len(csv_files)}] {csv_path.name}")
    (
      base_pos,
      base_quat,
      base_lin_vel,
      base_ang_vel,
      ee_pos,
      joint_pos,
      joint_vel,
    ) = _fk_motion(
      csv_path,
      sim,
      scene,
      joint_indexes,
      ee_indexes,
      motion_joint_indexes,
      expected_csv_joints=len(csv_joint_names),
      input_fps=cfg.input_fps,
      output_fps=cfg.output_fps,
    )
    if joint_pos.shape[-1] != num_joints:
      msg = (
        f"{csv_path.name}: expected {num_joints} dof columns, got {joint_pos.shape[-1]}"
      )
      raise ValueError(msg)
    del joint_vel
    windows = _compute_windows(
      base_pos,
      base_quat,
      base_lin_vel,
      base_ang_vel,
      ee_pos,
      joint_pos,
      cfg.window_size,
      cfg.stride,
    )
    if windows is None:
      print(f"  [SKIP] too short for window_size={cfg.window_size}")
      continue

    out_path = out_dir / f"{csv_path.stem}.npz"
    np.savez_compressed(
      out_path,
      windows=windows.cpu().numpy().astype(np.float32),
      fps=np.array([cfg.output_fps], dtype=np.float32),
      window_size=np.array([cfg.window_size], dtype=np.int32),
      stride=np.array([cfg.stride], dtype=np.int32),
      robot=np.array([cfg.robot]),
      source_joint_names=np.array(csv_joint_names),
      joint_names=np.array(joint_names),
      ee_body_names=np.array(ee_body_names),
      feature_dims=np.array(feature_dims, dtype=np.int32),
    )
    print(f"  saved {out_path.name}: windows={tuple(windows.shape)}")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
