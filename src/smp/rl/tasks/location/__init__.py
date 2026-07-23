"""Register G1 and X2 SMP location tasks."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import agibot_x2_smp_ppo_runner_cfg, unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.location.location_env_cfg import (
  g1_location_smp_env_cfg,
  x2_location_smp_env_cfg,
)

_location_rl = unitree_g1_smp_ppo_runner_cfg()
_location_rl.experiment_name = "smp_location_g1"
_location_rl.run_name = "smp_location_g1"

register_mjlab_task(
  task_id="Smp-Location-G1",
  env_cfg=g1_location_smp_env_cfg(play=False),
  play_env_cfg=g1_location_smp_env_cfg(play=True),
  rl_cfg=_location_rl,
)

_x2_location_rl = agibot_x2_smp_ppo_runner_cfg()
_x2_location_rl.experiment_name = "smp_location_x2"
_x2_location_rl.run_name = "smp_location_x2"

register_mjlab_task(
  task_id="Smp-Location-X2",
  env_cfg=x2_location_smp_env_cfg(play=False),
  play_env_cfg=x2_location_smp_env_cfg(play=True),
  rl_cfg=_x2_location_rl,
)

__all__ = ["g1_location_smp_env_cfg", "x2_location_smp_env_cfg"]
