"""Reset events for the getup task."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = ["record_success_by_initial_head_height", "reset_stand_counter"]


@torch.no_grad()
def record_success_by_initial_head_height(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  bin_edges: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9),
) -> None:
  """Log conditional getup success for episodes grouped by initial head height."""
  initial_height = getattr(env, "_getup_initial_head_height", None)
  success = getattr(env, "_getup_episode_success", None)
  if initial_height is None or success is None:
    return
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  completed = env.episode_length_buf[env_ids] > 0
  if not torch.any(completed):
    return
  height = initial_height[env_ids]
  did_succeed = success[env_ids].float()
  lower = float("-inf")
  for upper in bin_edges:
    in_bin = completed & (height >= lower) & (height < upper)
    if torch.any(in_bin):
      lower_label = "lt" if lower == float("-inf") else f"{lower:.2f}_to"
      key = f"Metrics/GetupSuccess/head_z_{lower_label}_{upper:.2f}"
      env.extras["log"][key] = did_succeed[in_bin].mean()
      env.extras["log"][key + "_episodes"] = torch.count_nonzero(in_bin)
    lower = upper

  in_bin = completed & (height >= lower)
  if torch.any(in_bin):
    key = f"Metrics/GetupSuccess/head_z_ge_{lower:.2f}"
    env.extras["log"][key] = did_succeed[in_bin].mean()
    env.extras["log"][key + "_episodes"] = torch.count_nonzero(in_bin)


@torch.no_grad()
def reset_stand_counter(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None
) -> None:
  """Clear termination and metric standing state for the reset environments."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  for attr in (
    "_getup_stand_count",
    "_getup_metric_stand_count",
    "_getup_episode_success",
  ):
    value = getattr(env, attr, None)
    if value is not None:
      value[env_ids] = 0
