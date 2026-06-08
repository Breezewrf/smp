"""SMP carry-box task — registers ``Smp-CarryBox-G1`` on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.carrybox.carrybox_env_cfg import g1_carrybox_smp_env_cfg

_carrybox_rl = unitree_g1_smp_ppo_runner_cfg()
_carrybox_rl.experiment_name = "smp_carrybox_g1"
_carrybox_rl.run_name = "smp_carrybox_g1"

register_mjlab_task(
  task_id="Smp-CarryBox-G1",
  env_cfg=g1_carrybox_smp_env_cfg(play=False),
  play_env_cfg=g1_carrybox_smp_env_cfg(play=True),
  rl_cfg=_carrybox_rl,
)

__all__ = ["g1_carrybox_smp_env_cfg"]
