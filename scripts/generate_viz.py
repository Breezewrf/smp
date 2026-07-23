"""Unconditionally generate a motion window with a trained SMP diffusion model
and visualize the predicted trajectory on the matching robot in a viser viewer.

Features carry ``root_pos`` (xy heading-inv + world z) and ``root_rot``
(6D tan-norm, heading-inv relative to the last-frame root), so the
world-frame pelvis trajectory is reconstructed directly from those two —
no velocity integration needed.  The last window frame is placed at a
chosen anchor pose (default: the robot's default standing state) and the
rest of the window is reconstructed relative to it.  EE positions come
from the sampled ``ee_pos`` feature lifted into world via the per-frame
pelvis pose.

Usage:
  uv run scripts/generate_viz.py --ckpt-path <pretrained.pt>
  uv run scripts/generate_viz.py --robot x2 --ckpt-path <pretrained_x2.pt>
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
import viser
from mjlab.entity import Entity
from mjlab.viewer.viser.scene import MjlabViserScene

from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler
from smp.robots import (
  G1_EE_BODY_NAMES,
  G1_JOINT_NAMES,
  X2_EE_BODY_NAMES,
  X2_JOINT_NAMES,
  get_x2_robot_cfg,
)
from smp.sampling.feature_to_state import (
  NUM_EE,
  window_to_ee_trajectories,
  window_to_pelvis_trajectory,
)
from smp.utils import detect_device

RobotName = Literal["g1", "x2"]

_ROBOT_JOINT_NAMES: dict[RobotName, tuple[str, ...]] = {
  "g1": G1_JOINT_NAMES,
  "x2": X2_JOINT_NAMES,
}
_ROBOT_EE_BODY_NAMES: dict[RobotName, tuple[str, ...]] = {
  "g1": G1_EE_BODY_NAMES,
  "x2": X2_EE_BODY_NAMES,
}


@dataclass
class Cfg:
  ckpt_path: str = ""
  """Path to a local SMP diffusion checkpoint .pt file. Mutually exclusive with --wandb-run."""
  wandb_run: str = ""
  """W&B run path '<entity>/<project>/<run_id>'. Downloads the latest .pt from the run."""
  robot: Literal["auto", "g1", "x2"] = "auto"
  """Robot to visualize. Auto uses checkpoint metadata and falls back to G1."""
  device: str = ""
  """Compute device. Empty = auto."""
  fps: float = 50.0
  """Playback frame rate."""


def _resolve_ckpt_path(cfg: Cfg) -> str:
  """Return a local ckpt path, downloading from wandb if --wandb-run is set."""
  if bool(cfg.ckpt_path) == bool(cfg.wandb_run):
    msg = "Specify exactly one of --ckpt-path or --wandb-run"
    raise ValueError(msg)
  if cfg.ckpt_path:
    return cfg.ckpt_path

  import wandb

  api = wandb.Api()
  run = api.run(cfg.wandb_run)
  pt_files = [f for f in run.files() if f.name.endswith(".pt")]
  if not pt_files:
    msg = f"No .pt files in wandb run {cfg.wandb_run}"
    raise FileNotFoundError(msg)
  target = next(
    (f for f in pt_files if Path(f.name).name == "pretrained.pt"),
    sorted(pt_files, key=lambda f: f.name)[-1],
  )
  download_dir = Path("logs") / "wandb_ckpt_cache" / cfg.wandb_run.replace("/", "_")
  download_dir.mkdir(parents=True, exist_ok=True)
  target.download(root=str(download_dir), replace=True)
  local = download_dir / target.name
  print(f"Downloaded {target.name} from {cfg.wandb_run} -> {local}")
  return str(local)


def _build_model_and_scheduler(
  ckpt: dict, device: torch.device
) -> tuple[DiffusionDenoiser, DDPMScheduler, np.ndarray, np.ndarray]:
  cfg = ckpt["cfg"]
  model = DiffusionDenoiser(
    feature_dim=cfg["feature_dim"],
    window_size=cfg["window_size"],
    d_model=cfg.get("d_model", 256),
    nhead=cfg.get("nhead", 8),
    num_layers=cfg.get("num_layers", 2),
    dropout=cfg.get("dropout", 0.0),
  ).to(device)
  state = ckpt.get("model_ema") or ckpt["model"]
  model.load_state_dict(state)
  model.eval()
  scheduler = DDPMScheduler(
    num_timesteps=cfg.get("num_timesteps", 50),
  ).to(device)
  return model, scheduler, ckpt["q_low"], ckpt["q_high"]


def _resolve_robot_name(cfg_robot: str, ckpt: dict) -> RobotName:
  """Resolve the robot from the CLI and checkpoint metadata."""
  ckpt_robot_raw = ckpt.get("cfg", {}).get("robot")
  ckpt_robot = str(ckpt_robot_raw).lower() if ckpt_robot_raw is not None else None
  if ckpt_robot not in (None, "g1", "x2"):
    raise ValueError(f"Unsupported robot in checkpoint metadata: {ckpt_robot_raw!r}")

  if cfg_robot == "auto":
    if ckpt_robot is None:
      print("Checkpoint has no robot metadata; defaulting to G1.")
      return "g1"
    return ckpt_robot

  if cfg_robot not in ("g1", "x2"):
    raise ValueError(f"Unsupported robot: {cfg_robot!r}")
  if ckpt_robot is not None and cfg_robot != ckpt_robot:
    raise ValueError(
      f"--robot {cfg_robot} conflicts with checkpoint robot metadata {ckpt_robot!r}"
    )
  return cfg_robot


def _validate_checkpoint_layout(ckpt: dict, robot_name: RobotName) -> None:
  """Reject checkpoints whose feature layout does not match the selected robot."""
  ckpt_cfg = ckpt["cfg"]
  expected_joints = _ROBOT_JOINT_NAMES[robot_name]
  expected_ee = _ROBOT_EE_BODY_NAMES[robot_name]

  joint_names = ckpt_cfg.get("joint_names")
  if joint_names is not None and tuple(joint_names) != expected_joints:
    raise ValueError(
      f"Checkpoint joint layout does not match {robot_name}: "
      f"expected {expected_joints}, got {tuple(joint_names)}"
    )
  ee_body_names = ckpt_cfg.get("ee_body_names")
  if ee_body_names is not None and tuple(ee_body_names) != expected_ee:
    raise ValueError(
      f"Checkpoint end-effector layout does not match {robot_name}: "
      f"expected {expected_ee}, got {tuple(ee_body_names)}"
    )

  expected_feature_dim = 3 + 6 + len(expected_joints) + len(expected_ee) * 3 + 3 + 3
  feature_dim = int(ckpt_cfg["feature_dim"])
  if feature_dim != expected_feature_dim:
    raise ValueError(
      f"Checkpoint feature_dim={feature_dim} does not match {robot_name} "
      f"layout ({expected_feature_dim})"
    )
  if len(expected_ee) != NUM_EE:
    raise ValueError(
      f"Visualization expects {NUM_EE} end effectors, but {robot_name} has "
      f"{len(expected_ee)}"
    )


def _setup_sim(device: str, robot_name: RobotName):
  """Build a single-robot sim matching the motion checkpoint."""
  from mjlab.scene import Scene, SceneCfg
  from mjlab.sim.sim import Simulation, SimulationCfg
  from mjlab.terrains import TerrainEntityCfg

  sim_cfg = SimulationCfg()
  if robot_name == "g1":
    from mjlab.tasks.tracking.config.g1.env_cfgs import (
      unitree_g1_flat_tracking_env_cfg,
    )

    scene_cfg = unitree_g1_flat_tracking_env_cfg().scene
  else:
    scene_cfg = SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_x2_robot_cfg()},
      num_envs=1,
      extent=2.0,
    )

  scene = Scene(scene_cfg, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene


def _quantile_denormalize(
  x: torch.Tensor, q_low: torch.Tensor, q_high: torch.Tensor
) -> torch.Tensor:
  return (x + 1.0) / 2.0 * (q_high - q_low) + q_low


@torch.no_grad()
def _run_generate(
  model: DiffusionDenoiser,
  scheduler: DDPMScheduler,
  q_low: np.ndarray,
  q_high: np.ndarray,
  window_size: int,
  feature_dim: int,
  device: torch.device,
) -> torch.Tensor:
  """Unconditional DDPM ancestral sampling. Returns (W, F) denormalized window on CPU."""
  x_t = torch.randn(1, window_size, feature_dim, device=device)
  for t in reversed(range(scheduler.num_timesteps)):
    t_batch = torch.full((1,), t, dtype=torch.long, device=device)
    eps = model(x_t, t_batch)
    x_t = scheduler.step(eps, x_t, t)
  q_low_t = torch.from_numpy(q_low).float().to(device)
  q_high_t = torch.from_numpy(q_high).float().to(device)
  return _quantile_denormalize(x_t.squeeze(0), q_low_t, q_high_t).cpu()


def _write_pose_to_robot(
  robot: Entity,
  pelvis_pos: np.ndarray,
  pelvis_quat_wxyz: np.ndarray,
  joint_pos: np.ndarray,
  joint_indexes: torch.Tensor,
  device: str,
) -> None:
  """Mirror scripts/csv_to_npz.py:_fk_motion's per-frame state write."""
  root = robot.data.default_root_state.clone()
  root[:, 0:3] = torch.as_tensor(pelvis_pos, device=device, dtype=root.dtype)
  root[:, 3:7] = torch.as_tensor(pelvis_quat_wxyz, device=device, dtype=root.dtype)
  robot.write_root_state_to_sim(root)
  jp = robot.data.default_joint_pos.clone()
  jp[:, joint_indexes] = torch.as_tensor(joint_pos, device=device, dtype=jp.dtype)
  jv = robot.data.default_joint_vel.clone()
  robot.write_joint_state_to_sim(jp, jv)


def main(cfg: Cfg) -> None:
  device_str = cfg.device or detect_device()
  device = torch.device(device_str)
  print(f"Device: {device_str}")

  ckpt_path = _resolve_ckpt_path(cfg)
  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  robot_name = _resolve_robot_name(cfg.robot, ckpt)
  _validate_checkpoint_layout(ckpt, robot_name)
  model, scheduler, q_low, q_high = _build_model_and_scheduler(ckpt, device)
  print(f"Loaded checkpoint epoch={ckpt.get('epoch')} from {ckpt_path}")
  print(f"Robot: {robot_name}")

  feature_dim = int(ckpt["cfg"]["feature_dim"])
  window_size = int(ckpt["cfg"]["window_size"])

  sim_device = device_str
  sim, scene = _setup_sim(sim_device, robot_name)
  robot: Entity = scene["robot"]
  mj_model = sim.mj_model
  joint_name_to_index = {name: index for index, name in enumerate(robot.joint_names)}
  expected_joint_names = _ROBOT_JOINT_NAMES[robot_name]
  missing_joint_names = [
    name for name in expected_joint_names if name not in joint_name_to_index
  ]
  if missing_joint_names:
    raise ValueError(f"{robot_name} model is missing joints: {missing_joint_names}")
  joint_indexes = torch.tensor(
    [joint_name_to_index[name] for name in expected_joint_names],
    device=sim_device,
    dtype=torch.long,
  )

  # Place the last window frame at the robot's default standing pose.
  anchor_pelvis_pos = robot.data.default_root_state[0, 0:3].detach().cpu()
  anchor_pelvis_quat = robot.data.default_root_state[0, 3:7].detach().cpu()

  def run() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_denorm = _run_generate(
      model,
      scheduler,
      q_low,
      q_high,
      window_size,
      feature_dim,
      device,
    )
    p_pos, p_quat, p_joint = window_to_pelvis_trajectory(
      pred_denorm,
      anchor_pelvis_pos,
      anchor_pelvis_quat,
    )
    ee_pos = window_to_ee_trajectories(pred_denorm, p_pos, p_quat)
    return (
      p_pos.cpu().numpy(),
      p_quat.cpu().numpy(),
      p_joint.cpu().numpy(),
      ee_pos.cpu().numpy(),
    )

  state: dict = {"pred": run()}

  server = viser.ViserServer()
  viser_scene = MjlabViserScene(server, mj_model, num_envs=1)
  viser_scene.debug_visualization_enabled = True

  # /fixed_bodies parents under mjviser's camera-tracking scene offset, so
  # the points stay aligned with the re-centered robot.
  ee_points = server.scene.add_point_cloud(
    name="/fixed_bodies/predicted_ee_positions",
    points=np.zeros((NUM_EE, 3), dtype=np.float32),
    colors=np.tile(np.array([255, 80, 0], dtype=np.uint8), (NUM_EE, 1)),
    point_size=0.03,
  )

  with server.gui.add_folder("Generate"):
    frame_slider = server.gui.add_slider(
      "Frame", min=0, max=window_size - 1, step=1, initial_value=0
    )
    play_btn = server.gui.add_button("Play / Pause")
    resample_btn = server.gui.add_button("Resample")

  playing = {"v": True}

  @play_btn.on_click
  def _(_evt) -> None:
    playing["v"] = not playing["v"]

  @resample_btn.on_click
  def _(_evt) -> None:
    state["pred"] = run()

  def render(frame: int) -> None:
    p_pos, p_quat, p_joint, ee_pos = state["pred"]
    _write_pose_to_robot(
      robot,
      p_pos[frame],
      p_quat[frame],
      p_joint[frame],
      joint_indexes,
      sim_device,
    )
    sim.forward()
    wd = sim.wp_data
    viser_scene.update_from_arrays(
      body_xpos=np.asarray(wd.xpos.numpy()),
      body_xmat=np.asarray(wd.xmat.numpy()),
      qpos=np.asarray(wd.qpos.numpy()),
      env_idx=0,
    )
    ee_points.points = ee_pos[frame]
    viser_scene.refresh_visualization()

  print("Viser server running. Open the printed URL.")
  dt_play = 1.0 / cfg.fps
  try:
    while True:
      render(int(frame_slider.value))
      if playing["v"]:
        nxt = (int(frame_slider.value) + 1) % window_size
        frame_slider.value = nxt
      time.sleep(dt_play)
  except KeyboardInterrupt:
    print("Shutting down.")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
