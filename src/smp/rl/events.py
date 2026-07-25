"""Startup + reset events for SMP RL.

Run from mjlab's event manager so the task stays a plain ``ManagerBasedRlEnv``.
Motion features carry no absolute root pose, so GSI writes a default root frame
(each env's origin, identity yaw) to sim and primes the feature buffer in an
env-origin-relative frame, so the SMP reward is invariant to env placement.
"""

from __future__ import annotations

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply, quat_mul, yaw_quat

from smp.rl.utils import DiffNormalizer, MotionFeatureBuffer, load_denoiser
from smp.robots import G1_EE_BODY_NAMES, G1_JOINT_NAMES
from smp.sampling.feature_to_state import rot6d_to_quat, slice_features


def _maybe_compile(model, compile_model: bool, compile_mode: str | None):
  """``torch.compile`` ``model`` (no-op if ``compile_model`` false), working
  around the Inductor ``pad_mm`` TF32 crash by disabling shape padding."""
  if not compile_model:
    return model
  torch.set_float32_matmul_precision("high")
  try:
    import torch._inductor.config as _ic

    _ic.shape_padding = False
  except ImportError:
    pass
  if compile_mode is not None:
    return torch.compile(model, fullgraph=True, mode=compile_mode)
  return torch.compile(model, fullgraph=True)


def init_smp_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  ckpt_path: str = "",
  gsi_buffer_size: int = 4096,
  gsi_batch_size: int = 256,
  compile_model: bool = True,
  compile_mode: str | None = None,
  gsi_max_head_height: float | None = None,
  gsi_max_draw_multiplier: int = 20,
  robot_name: str = "g1",
  joint_names: tuple[str, ...] = G1_JOINT_NAMES,
  ee_body_names: tuple[str, ...] = G1_EE_BODY_NAMES,
) -> None:
  """Startup-mode event: load the frozen denoiser, allocate the feature buffer +
  ``DiffNormalizer`` (stashed on the env), and pre-generate the GSI pool of
  ``gsi_buffer_size`` windows that ``gsi_reset`` samples from (amortizes the DDPM
  cost).  If ``compile_model``, the denoiser is ``torch.compile``-d and pre-warmed
  so Inductor compiles here, not on the first sim step."""
  del env_ids
  if not ckpt_path:
    msg = (
      "init_smp_state called without `ckpt_path`. Set it on the EventTermCfg: "
      "EventTermCfg(func=init_smp_state, mode='startup', "
      "params={'ckpt_path': '/path/to/pretrained.pt'})."
    )
    raise RuntimeError(msg)
  (
    model,
    scheduler,
    q_low,
    q_high,
    feature_dim,
    window_size,
    checkpoint_cfg,
  ) = load_denoiser(ckpt_path, env.device)
  num_joints = len(joint_names)
  num_ee = len(ee_body_names)
  expected_feature_dim = 3 + 6 + num_joints + num_ee * 3 + 3 + 3
  if feature_dim != expected_feature_dim:
    raise ValueError(
      f"SMP checkpoint feature_dim={feature_dim}, but the configured robot "
      f"layout requires {expected_feature_dim} ({num_joints} joints, {num_ee} EEs)"
    )
  metadata_checks = {
    "robot": robot_name,
    "joint_names": joint_names,
    "ee_body_names": ee_body_names,
  }
  for field_name, expected in metadata_checks.items():
    actual = checkpoint_cfg.get(field_name)
    mismatch = (
      actual != expected
      if field_name == "robot"
      else actual is not None and tuple(actual) != tuple(expected)
    )
    if actual is not None and mismatch:
      raise ValueError(
        f"SMP checkpoint {field_name}={actual!r} does not match "
        f"the configured robot's {expected!r}"
      )
  model = _maybe_compile(model, compile_model, compile_mode)
  env._smp_bundle = (  # type: ignore[attr-defined]
    model,
    scheduler,
    q_low,
    q_high,
    feature_dim,
    window_size,
  )
  robot = env.scene["robot"]
  joint_indexes = robot.find_joints(list(joint_names), preserve_order=True)[0]
  ee_indexes = robot.find_bodies(list(ee_body_names), preserve_order=True)[0]
  if len(joint_indexes) != num_joints:
    raise ValueError(
      f"Found {len(joint_indexes)} of {num_joints} configured SMP joints"
    )
  if len(ee_indexes) != num_ee:
    raise ValueError(f"Found {len(ee_indexes)} of {num_ee} configured SMP bodies")
  env._smp_joint_indexes = torch.tensor(  # type: ignore[attr-defined]
    joint_indexes,
    dtype=torch.long,
    device=env.device,
  )
  env._smp_ee_indexes = torch.tensor(  # type: ignore[attr-defined]
    ee_indexes,
    dtype=torch.long,
    device=env.device,
  )
  env._smp_num_joints = num_joints  # type: ignore[attr-defined]
  env._smp_num_ee = num_ee  # type: ignore[attr-defined]
  env._smp_buffer = MotionFeatureBuffer(  # type: ignore[attr-defined]
    num_envs=env.num_envs,
    window_size=window_size,
    num_joints=num_joints,
    num_ee=num_ee,
    device=env.device,
  )
  env._smp_normalizer = DiffNormalizer(scheduler.num_timesteps, env.device)  # type: ignore[attr-defined]

  if gsi_buffer_size <= 0:
    msg = f"gsi_buffer_size must be positive, got {gsi_buffer_size}."
    raise ValueError(msg)
  if gsi_batch_size <= 0:
    msg = f"gsi_batch_size must be positive, got {gsi_batch_size}."
    raise ValueError(msg)
  if gsi_max_draw_multiplier <= 0:
    msg = (
      "gsi_max_draw_multiplier must be positive, got "
      f"{gsi_max_draw_multiplier}."
    )
    raise ValueError(msg)
  env._smp_gsi_batch_size = gsi_batch_size  # type: ignore[attr-defined]
  env._smp_gsi_max_head_height = gsi_max_head_height  # type: ignore[attr-defined]
  env._smp_gsi_max_draw_multiplier = gsi_max_draw_multiplier  # type: ignore[attr-defined]
  pool, head_heights = _sample_gsi_windows(
    env,
    gsi_buffer_size,
    max_head_height=gsi_max_head_height,
    batch_size=gsi_batch_size,
    max_draw_multiplier=gsi_max_draw_multiplier,
  )
  env._smp_gsi_pool = pool  # type: ignore[attr-defined]
  env._smp_gsi_head_heights = head_heights  # type: ignore[attr-defined]

  if compile_model and env.num_envs != gsi_batch_size:
    # Warm the reward-path shape so its Inductor compile happens here.
    with torch.no_grad():
      dummy_x = torch.randn(env.num_envs, window_size, feature_dim, device=env.device)
      dummy_t = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
      _ = model(dummy_x, dummy_t)

  gsi_reset(env)


def _prime_sim_and_buffer(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  window: torch.Tensor,
) -> None:
  """Common GSI tail: write the window's last frame to sim, fill the feature
  buffer.  The buffer is env-origin-RELATIVE (placement-invariant features) while
  the sim write adds each env's origin so robots spread across the grid.
  ``joint_vel`` is finite-differenced from ``joint_pos`` (not in the window)."""
  n, W, _ = window.shape
  J = env._smp_num_joints  # type: ignore[attr-defined]
  E = env._smp_num_ee  # type: ignore[attr-defined]
  parts = slice_features(window, num_joints=J, num_ee=E)
  root_pos_local = parts["root_pos"]
  root_rot_6d = parts["root_rot"]
  joint_pos = parts["joint_pos"]
  ee_pos_local = parts["ee_pos"].reshape(n, W, E, 3)
  root_lin_vel_local = parts["root_lin_vel"]
  root_ang_vel_local = parts["root_ang_vel"]

  control_dt = float(env.cfg.sim.mujoco.timestep) * float(env.cfg.decimation)
  if W > 1:
    joint_vel = torch.zeros_like(joint_pos)
    joint_vel[:, :-1] = (joint_pos[:, 1:] - joint_pos[:, :-1]) / control_dt
    joint_vel[:, -1] = joint_vel[:, -2]
  else:
    joint_vel = torch.zeros_like(joint_pos)

  robot = env.scene["robot"]
  default_root = robot.data.default_root_state[env_ids].clone()
  default_pos = default_root[:, 0:3]
  default_quat = default_root[:, 3:7]
  yaw_T = yaw_quat(default_quat)
  yaw_T_W = yaw_T[:, None, :].expand(n, W, 4).reshape(-1, 4)

  local_xy = root_pos_local.clone()
  local_xy[..., 2] = 0.0
  world_offset_xy = quat_apply(yaw_T_W, local_xy.reshape(-1, 3)).reshape(n, W, 3)
  pelvis_pos_w = world_offset_xy.clone()
  pelvis_pos_w[..., 0] += default_pos[:, None, 0]
  pelvis_pos_w[..., 1] += default_pos[:, None, 1]
  pelvis_pos_w[..., 2] = root_pos_local[..., 2]

  root_rot_local_quat = rot6d_to_quat(root_rot_6d.reshape(-1, 6)).reshape(n, W, 4)
  pelvis_quat_w = quat_mul(yaw_T_W, root_rot_local_quat.reshape(-1, 4)).reshape(n, W, 4)

  lin_vel_w = quat_apply(yaw_T_W, root_lin_vel_local.reshape(-1, 3)).reshape(n, W, 3)
  ang_vel_w = quat_apply(yaw_T_W, root_ang_vel_local.reshape(-1, 3)).reshape(n, W, 3)

  yaw_T_E = yaw_T[:, None, None, :].expand(n, W, E, 4).reshape(-1, 4)
  ee_offset_w = quat_apply(yaw_T_E, ee_pos_local.reshape(-1, 3)).reshape(n, W, E, 3)
  ee_pos_w = ee_offset_w + pelvis_pos_w[:, :, None, :]

  # Buffer stays env-relative; the sim write is offset to each env's origin.
  origins = env.scene.env_origins[env_ids]
  last_root_state = torch.cat(
    [
      pelvis_pos_w[:, -1] + origins,
      pelvis_quat_w[:, -1],
      lin_vel_w[:, -1],
      ang_vel_w[:, -1],
    ],
    dim=-1,
  )
  robot.write_root_state_to_sim(last_root_state, env_ids=env_ids)
  robot.write_joint_state_to_sim(
    joint_pos[:, -1],
    joint_vel[:, -1],
    joint_ids=env._smp_joint_indexes,  # type: ignore[attr-defined]
    env_ids=env_ids,
  )

  buf: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  buf.reset(
    env_ids,
    pelvis_pos_w,
    pelvis_quat_w,
    lin_vel_w,
    ang_vel_w,
    ee_pos_w,
    joint_pos,
    joint_vel,
  )


@torch.no_grad()
def _ddpm_sample(env: ManagerBasedRlEnv, n: int) -> torch.Tensor:
  """Run DDPM ancestral sampling and return ``n`` denormalized windows."""
  model, scheduler, q_low, q_high, feature_dim, window_size = env._smp_bundle  # type: ignore[attr-defined]
  x_t = torch.randn(n, window_size, feature_dim, device=env.device)
  for t_int in reversed(range(scheduler.num_timesteps)):
    t = torch.full((n,), t_int, dtype=torch.long, device=env.device)
    eps = model(x_t, t)
    x_t = scheduler.step(eps, x_t, t_int)
  return (x_t + 1.0) / 2.0 * (q_high - q_low) + q_low


@torch.no_grad()
def _gsi_head_heights(env: ManagerBasedRlEnv, windows: torch.Tensor) -> torch.Tensor:
  """Compute each window's final head-site height with host MuJoCo kinematics."""
  robot = env.scene["robot"]
  head_local_ids = robot.find_sites(["head"], preserve_order=True)[0]
  if len(head_local_ids) != 1:
    raise ValueError(
      "GSI head-height filtering requires exactly one robot site named 'head'."
    )

  n = windows.shape[0]
  parts = slice_features(
    windows,
    num_joints=env._smp_num_joints,  # type: ignore[attr-defined]
    num_ee=env._smp_num_ee,  # type: ignore[attr-defined]
  )
  root_pos = parts["root_pos"][:, -1].detach().cpu()
  root_rot = parts["root_rot"][:, -1]
  root_quat = rot6d_to_quat(root_rot.reshape(n, 6)).detach().cpu()
  joint_pos = parts["joint_pos"][:, -1].detach().cpu()

  model = env.sim.mj_model
  data = mujoco.MjData(model)
  qpos0 = model.qpos0.copy()
  root_q_adrs = robot.indexing.free_joint_q_adr.detach().cpu().numpy()
  smp_joint_ids = env._smp_joint_indexes.detach().cpu()  # type: ignore[attr-defined]
  joint_q_adrs = robot.indexing.joint_q_adr[smp_joint_ids].detach().cpu().numpy()
  head_site_id = int(robot.indexing.site_ids[head_local_ids[0]].item())
  heights = torch.empty(n, dtype=windows.dtype)

  if len(root_q_adrs) != 7:
    raise ValueError(
      "GSI head-height filtering requires a floating-base robot with one free joint."
    )
  if len(joint_q_adrs) != env._smp_num_joints:  # type: ignore[attr-defined]
    raise ValueError("SMP joint layout does not match the robot qpos layout.")

  for i in range(n):
    data.qpos[:] = qpos0
    data.qpos[root_q_adrs[:3]] = root_pos[i].numpy()
    data.qpos[root_q_adrs[3:]] = root_quat[i].numpy()
    data.qpos[joint_q_adrs] = joint_pos[i].numpy()
    mujoco.mj_kinematics(model, data)
    heights[i] = float(data.site_xpos[head_site_id, 2])
  return heights.to(device=windows.device)


@torch.no_grad()
def _sample_gsi_windows(
  env: ManagerBasedRlEnv,
  num_samples: int,
  *,
  max_head_height: float | None,
  batch_size: int,
  max_draw_multiplier: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
  """Generate a GSI pool, rejecting windows whose final head is too high."""
  if max_head_height is None:
    chunks = []
    for start in range(0, num_samples, batch_size):
      chunks.append(_ddpm_sample(env, min(batch_size, num_samples - start)))
    return torch.cat(chunks, dim=0), None

  accepted_windows: list[torch.Tensor] = []
  accepted_heights: list[torch.Tensor] = []
  num_accepted = 0
  num_drawn = 0
  max_draws = max(num_samples, num_samples * max_draw_multiplier)
  while num_accepted < num_samples and num_drawn < max_draws:
    draw_size = min(batch_size, max_draws - num_drawn)
    candidates = _ddpm_sample(env, draw_size)
    heights = _gsi_head_heights(env, candidates)
    keep = heights < max_head_height
    if torch.any(keep):
      accepted_windows.append(candidates[keep])
      accepted_heights.append(heights[keep])
      num_accepted += int(torch.count_nonzero(keep).item())
    num_drawn += draw_size

  if num_accepted < num_samples:
    rate = num_accepted / max(num_drawn, 1)
    raise RuntimeError(
      f"Only {num_accepted}/{num_samples} GSI windows with head_z < "
      f"{max_head_height:.3f} m were accepted after {num_drawn} draws "
      f"(acceptance rate {rate:.1%}). Increase gsi_max_draw_multiplier or "
      "relax gsi_max_head_height."
    )

  pool = torch.cat(accepted_windows, dim=0)[:num_samples]
  head_heights = torch.cat(accepted_heights, dim=0)[:num_samples]
  return pool, head_heights


@torch.no_grad()
def gsi_refresh(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  num_samples: int = 1024,
  step_interval: int = 2400,
) -> None:
  """Step-mode event: every ``step_interval`` steps, FIFO-replace ``num_samples``
  GSI-pool windows with fresh DDPM samples so the init distribution stays fresh."""
  del env_ids
  cur = int(env.common_step_counter)
  if cur == 0 or (cur % step_interval) != 0:
    return

  pool: torch.Tensor = env._smp_gsi_pool  # type: ignore[attr-defined]
  pool_size = pool.shape[0]
  if num_samples > pool_size:
    msg = f"num_samples ({num_samples}) cannot exceed pool size ({pool_size})"
    raise ValueError(msg)

  new_windows, new_head_heights = _sample_gsi_windows(
    env,
    num_samples,
    max_head_height=env._smp_gsi_max_head_height,  # type: ignore[attr-defined]
    batch_size=env._smp_gsi_batch_size,  # type: ignore[attr-defined]
    max_draw_multiplier=env._smp_gsi_max_draw_multiplier,  # type: ignore[attr-defined]
  )
  pool_head_heights: torch.Tensor | None = env._smp_gsi_head_heights  # type: ignore[attr-defined]
  head = int(getattr(env, "_smp_gsi_head", 0))
  end = head + num_samples
  if end <= pool_size:
    pool[head:end] = new_windows
    if pool_head_heights is not None and new_head_heights is not None:
      pool_head_heights[head:end] = new_head_heights
  else:
    first = pool_size - head
    pool[head:] = new_windows[:first]
    pool[: end - pool_size] = new_windows[first:]
    if pool_head_heights is not None and new_head_heights is not None:
      pool_head_heights[head:] = new_head_heights[:first]
      pool_head_heights[: end - pool_size] = new_head_heights[first:]
  env._smp_gsi_head = end % pool_size  # type: ignore[attr-defined]


@torch.no_grad()
def gsi_reset(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
  """Generative State Initialization: sample ``n`` windows from the GSI pool and
  prime sim + feature buffer from them.  Must run AFTER mjlab's ``reset_base``.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = int(env_ids.numel())
  if n == 0:
    return

  pool: torch.Tensor = env._smp_gsi_pool  # type: ignore[attr-defined]
  idx = torch.randint(0, pool.shape[0], (n,), device=env.device)
  window = pool[idx]
  pool_head_heights: torch.Tensor | None = env._smp_gsi_head_heights  # type: ignore[attr-defined]
  if pool_head_heights is not None:
    initial_head_height = getattr(env, "_getup_initial_head_height", None)
    if initial_head_height is None:
      initial_head_height = torch.zeros(env.num_envs, device=env.device)
    initial_head_height[env_ids] = pool_head_heights[idx]
    env._getup_initial_head_height = initial_head_height  # type: ignore[attr-defined]
  _prime_sim_and_buffer(env, env_ids, window)
