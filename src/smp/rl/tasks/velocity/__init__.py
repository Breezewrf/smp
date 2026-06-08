"""SMP velocity task -- registers ``Smp-Velocity-G1`` on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.velocity.velocity_env_cfg import g1_velocity_smp_env_cfg

_velocity_rl = unitree_g1_smp_ppo_runner_cfg()
_velocity_rl.experiment_name = "smp_velocity_g1"
_velocity_rl.run_name = "smp_velocity_g1"

register_mjlab_task(
  task_id="Smp-Velocity-G1",
  env_cfg=g1_velocity_smp_env_cfg(play=False),
  play_env_cfg=g1_velocity_smp_env_cfg(play=True),
  rl_cfg=_velocity_rl,
)

__all__ = ["g1_velocity_smp_env_cfg"]
