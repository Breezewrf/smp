"""Reward functions for SMP RL tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from smp.rl.utils import (
  BOX_FEATURE_END,
  ROBOT_FEATURE_DIM,
  DiffNormalizer,
  MotionBoxFeatureBuffer,
  MotionFeatureBuffer,
)

if TYPE_CHECKING:
  from collections.abc import Callable

  from mjlab.envs import ManagerBasedRlEnv

  TaskTerm = tuple["Callable[..., torch.Tensor]", float, dict]


def _update_buffer_from_sim(env: ManagerBasedRlEnv) -> None:
  """Push current sim kinematics onto the buffer tail, env-origin-relative
  (matching ``_prime_sim_and_buffer``) so features are placement-invariant."""
  robot = env.scene["robot"]
  ee_indexes = env._smp_ee_indexes  # type: ignore[attr-defined]
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  origins = env.scene.env_origins
  buffer.update(
    robot.data.root_link_pos_w - origins,
    robot.data.root_link_quat_w,
    robot.data.root_link_lin_vel_w,
    robot.data.root_link_ang_vel_w,
    robot.data.body_link_pos_w[:, ee_indexes] - origins[:, None, :],
    robot.data.joint_pos,
    robot.data.joint_vel,
  )


def _update_box_buffer_from_sim(
  env: ManagerBasedRlEnv,
  box_name: str = "box",
) -> None:
  """Push current robot + box kinematics onto the 75-D carry-box buffer."""
  robot = env.scene["robot"]
  box = env.scene[box_name]
  ee_indexes = env._smp_ee_indexes  # type: ignore[attr-defined]
  buffer: MotionBoxFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  origins = env.scene.env_origins
  box_pos_origin = box.data.root_link_pos_w - origins
  box_height = box_pos_origin[:, 2]
  buffer.update(
    robot.data.root_link_pos_w - origins,
    robot.data.root_link_quat_w,
    robot.data.root_link_lin_vel_w,
    robot.data.root_link_ang_vel_w,
    robot.data.body_link_pos_w[:, ee_indexes] - origins[:, None, :],
    robot.data.joint_pos,
    robot.data.joint_vel,
    box_pos_origin,
    box.data.root_link_quat_w,
    box.data.root_link_lin_vel_w,
    box.data.root_link_ang_vel_w,
    box_height,
  )


def reset_box_buffer_to_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  box_pos_w: torch.Tensor,
  box_quat_w: torch.Tensor,
) -> None:
  """Fill box history from a known world-frame pose."""
  buffer = getattr(env, "_smp_buffer", None)
  if not isinstance(buffer, MotionBoxFeatureBuffer) or env_ids.numel() == 0:
    return
  origins = env.scene.env_origins[env_ids]
  box_pos_origin = box_pos_w - origins
  W = buffer.window_size
  zeros = torch.zeros(env_ids.numel(), W, 3, device=env.device)
  buffer.reset_box(
    env_ids,
    box_pos_origin[:, None, :].expand(-1, W, 3),
    box_quat_w[:, None, :].expand(-1, W, 4),
    zeros,
    zeros,
    box_pos_origin[:, None, 2:3].expand(-1, W, 1),
  )


def smp_guidance_reward(
  env: ManagerBasedRlEnv,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 4.0,
  normalize: bool = True,
) -> torch.Tensor:
  """SDS-style guidance reward over fixed timesteps ``K``:
  ``exp(-w_s/|K| · Σ_{i∈K} ‖ε̂_i − ε_i‖²)``.  ``normalize`` divides each MSE by a
  ``DiffNormalizer`` running mean (policy-relative) vs. raw (absolute scale);
  always stashes the mean raw MSE on ``env._smp_raw_err``."""
  device = torch.device(env.device)
  model, scheduler, q_low, q_high, feature_dim, _ = env._smp_bundle  # type: ignore[attr-defined]
  if feature_dim != ROBOT_FEATURE_DIM:
    msg = (
      f"smp_guidance_reward expects {ROBOT_FEATURE_DIM}-D robot features, "
      f"got feature_dim={feature_dim}. Use smp_box_guidance_reward for "
      f"{BOX_FEATURE_END}-D carry-box features."
    )
    raise ValueError(msg)
  normalizer: DiffNormalizer = env._smp_normalizer  # type: ignore[attr-defined]
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  _update_buffer_from_sim(env)

  features = buffer.compute_features()
  x_0 = 2.0 * (features - q_low) / (q_high - q_low + 1e-8) - 1.0
  num_envs = x_0.shape[0]

  total_err = torch.zeros(num_envs, device=device)
  total_raw = torch.zeros(num_envs, device=device)
  with torch.no_grad():
    for t_scalar in fixed_timesteps:
      if not 0 <= t_scalar < scheduler.num_timesteps:
        msg = f"fixed_timestep {t_scalar} out of range [0, {scheduler.num_timesteps})"
        raise ValueError(msg)
      t = torch.full((num_envs,), t_scalar, dtype=torch.long, device=device)
      noise = torch.randn_like(x_0)
      x_t = scheduler.add_noise(x_0, noise, t)
      eps_hat = model(x_t, t)
      mse_per_env = ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
      total_raw += mse_per_env
      if normalize:
        total_err += normalizer.update_and_normalize(t_scalar, mse_per_env)
      else:
        total_err += mse_per_env

  env._smp_raw_err = total_raw / len(fixed_timesteps)  # type: ignore[attr-defined]
  err = total_err / len(fixed_timesteps)
  return torch.exp(-err * ws)


def smp_box_guidance_reward(
  env: ManagerBasedRlEnv,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 4.0,
  normalize: bool = True,
  box_name: str = "box",
) -> torch.Tensor:
  """SMP guidance over 75-D robot+box carry-box features."""
  device = torch.device(env.device)
  model, scheduler, q_low, q_high, feature_dim, _ = env._smp_bundle  # type: ignore[attr-defined]
  if feature_dim != BOX_FEATURE_END:
    msg = (
      f"smp_box_guidance_reward expects {BOX_FEATURE_END}-D carry-box features, "
      f"got feature_dim={feature_dim}."
    )
    raise ValueError(msg)
  normalizer: DiffNormalizer = env._smp_normalizer  # type: ignore[attr-defined]
  buffer: MotionBoxFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  _update_box_buffer_from_sim(env, box_name=box_name)

  features = buffer.compute_features()
  x_0 = 2.0 * (features - q_low) / (q_high - q_low + 1e-8) - 1.0
  num_envs = x_0.shape[0]

  total_err = torch.zeros(num_envs, device=device)
  total_raw = torch.zeros(num_envs, device=device)
  with torch.no_grad():
    for t_scalar in fixed_timesteps:
      if not 0 <= t_scalar < scheduler.num_timesteps:
        msg = f"fixed_timestep {t_scalar} out of range [0, {scheduler.num_timesteps})"
        raise ValueError(msg)
      t = torch.full((num_envs,), t_scalar, dtype=torch.long, device=device)
      noise = torch.randn_like(x_0)
      x_t = scheduler.add_noise(x_0, noise, t)
      eps_hat = model(x_t, t)
      mse_per_env = ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
      total_raw += mse_per_env
      if normalize:
        total_err += normalizer.update_and_normalize(t_scalar, mse_per_env)
      else:
        total_err += mse_per_env

  env._smp_raw_err = total_raw / len(fixed_timesteps)  # type: ignore[attr-defined]
  err = total_err / len(fixed_timesteps)
  return torch.exp(-err * ws)


def task_smp_product(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
) -> torch.Tensor:
  """``(Σ wᵢ · taskᵢ(env)) · r_smp`` — multiplicative SMP gating; ``task_terms`` is
  a tuple of ``(func, weight, kwargs)``.  Calls ``smp_guidance_reward`` once (the
  sole SMP-buffer update), so it must be the task's only SMP reward term."""
  task = sum(w * func(env, **kw) for func, w, kw in task_terms)
  return task * smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)


def task_smp_box_product(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  box_name: str = "box",
) -> torch.Tensor:
  """Carry-box ``task_smp_product`` variant using 75-D robot+box guidance."""
  task = sum(w * func(env, **kw) for func, w, kw in task_terms)
  return task * smp_box_guidance_reward(
    env,
    fixed_timesteps=fixed_timesteps,
    ws=ws,
    box_name=box_name,
  )
