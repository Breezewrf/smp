"""Location task with SMP guidance.

Each env gets a periodically-resampled world-frame xy goal; reward is
position-tracking gated by the SMP guidance reward.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import g1_smp_env_cfg, x2_loco_ckpt_path, x2_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.location import mdp


def _location_smp_env_cfg(
  cfg: ManagerBasedRlEnvCfg, ckpt_path: str
) -> ManagerBasedRlEnvCfg:

  # --- Commands ------------------------------------------------------------
  cfg.commands["location"] = mdp.LocationCommandCfg(
    entity_name="robot",
    resampling_time_range=(5.0, 10.0),
    tar_dist_min=1.0,
    tar_dist_max=10.0,
    debug_vis=True,
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "location"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs

  # --- Rewards -------------------------------------------------------------
  # task = position tracking, gated by SMP.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.location_position,
          1.0,
          {"command_name": "location", "pos_err_scale": 0.25},
        ),
      ),
      "ws": 4,
    },
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = ckpt_path

  # --- Terminations --------------------------------------------------------
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  return cfg


def g1_location_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 location environment."""
  return _location_smp_env_cfg(
    g1_smp_env_cfg(play=play), "datasets/pretrain_ckpt/pretrained_lafan_run.pt"
  )


def x2_location_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the X2 location environment."""
  return _location_smp_env_cfg(x2_smp_env_cfg(play=play), x2_loco_ckpt_path())
