"""Register G1 and X2 SMP forward/steering tasks."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import agibot_x2_smp_ppo_runner_cfg, unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.steering.forward_env_cfg import (
  g1_forward_smp_env_cfg,
  x2_forward_smp_env_cfg,
)
from smp.rl.tasks.steering.steering_env_cfg import (
  g1_steering_smp_env_cfg,
  x2_steering_smp_env_cfg,
)

_steering_rl = unitree_g1_smp_ppo_runner_cfg()
_steering_rl.experiment_name = "smp_steering_g1"
_steering_rl.run_name = "smp_steering_g1"

register_mjlab_task(
  task_id="Smp-Steering-G1",
  env_cfg=g1_steering_smp_env_cfg(play=False),
  play_env_cfg=g1_steering_smp_env_cfg(play=True),
  rl_cfg=_steering_rl,
)

_forward_rl = unitree_g1_smp_ppo_runner_cfg()
_forward_rl.experiment_name = "smp_forward_g1"
_forward_rl.run_name = "smp_forward_g1"

register_mjlab_task(
  task_id="Smp-Forward-G1",
  env_cfg=g1_forward_smp_env_cfg(play=False),
  play_env_cfg=g1_forward_smp_env_cfg(play=True),
  rl_cfg=_forward_rl,
)

_x2_steering_rl = agibot_x2_smp_ppo_runner_cfg()
_x2_steering_rl.experiment_name = "smp_steering_x2"
_x2_steering_rl.run_name = "smp_steering_x2"

register_mjlab_task(
  task_id="Smp-Steering-X2",
  env_cfg=x2_steering_smp_env_cfg(play=False),
  play_env_cfg=x2_steering_smp_env_cfg(play=True),
  rl_cfg=_x2_steering_rl,
)

_x2_forward_rl = agibot_x2_smp_ppo_runner_cfg()
_x2_forward_rl.experiment_name = "smp_forward_x2"
_x2_forward_rl.run_name = "smp_forward_x2"

register_mjlab_task(
  task_id="Smp-Forward-X2",
  env_cfg=x2_forward_smp_env_cfg(play=False),
  play_env_cfg=x2_forward_smp_env_cfg(play=True),
  rl_cfg=_x2_forward_rl,
)

__all__ = [
  "g1_forward_smp_env_cfg",
  "g1_steering_smp_env_cfg",
  "x2_forward_smp_env_cfg",
  "x2_steering_smp_env_cfg",
]
