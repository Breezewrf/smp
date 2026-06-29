"""G1 body-frame velocity tracking task with SMP guidance.

The command is ``[lin_vel_x, lin_vel_y, ang_vel_z]`` in the robot body frame.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.velocity import mdp


def g1_velocity_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 velocity env cfg with SMP guidance."""
  cfg = g1_smp_env_cfg(play=play)

  # --- Commands ------------------------------------------------------------
  cfg.commands["twist"] = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.05,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 2.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
      heading=(-math.pi, math.pi),
    ),
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "twist"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs

  # --- Rewards -------------------------------------------------------------
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.track_linear_velocity,
          1.0,
          {"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        (
          mdp.track_angular_velocity,
          1.0,
          {"command_name": "twist", "std": math.sqrt(0.5)},
        ),
        (
          mdp.stand_still,
          -1.0,
          {
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
          },
        ),
        (
          mdp.joint_acc_l2,
          -2.5e-7,
          {"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        ),
        (
          mdp.body_orientation_l2,
          -1.0,
          {"asset_cfg": SceneEntityCfg("robot", body_names=())},
        ),
      ),
    },
  )
  cfg.rewards["body_orientation_l2"] = RewardTermCfg(
    func=mdp.body_orientation_l2,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_loco.pt"
  )

  # --- Terminations --------------------------------------------------------
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  if play:
    cfg.observations["actor"].enable_corruption = False

    # Disable termination for play mode to allow uninterrupted episodes.
    cfg.terminations = {}
  return cfg
