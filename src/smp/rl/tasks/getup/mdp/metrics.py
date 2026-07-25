"""Episode success metrics for the getup task."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = ["episode_success"]


def episode_success(
  env: ManagerBasedRlEnv,
  head_height: float = 1.2,
  max_speed: float = 0.5,
  hold_steps: int = 25,
) -> torch.Tensor:
  """Latch stable-standing success independently of success termination."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  z = robot.data.site_pos_w[:, head_idx, 2]
  speed = torch.linalg.norm(robot.data.root_link_lin_vel_w, dim=-1)
  is_standing = (z >= head_height) & (speed < max_speed)

  count = getattr(env, "_getup_metric_stand_count", None)
  if count is None:
    count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  count = torch.where(is_standing, count + 1, torch.zeros_like(count))
  env._getup_metric_stand_count = count  # type: ignore[attr-defined]

  success = getattr(env, "_getup_episode_success", None)
  if success is None:
    success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  success |= count >= hold_steps
  env._getup_episode_success = success  # type: ignore[attr-defined]
  return success.float()
