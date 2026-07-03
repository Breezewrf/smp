"""G1 carry-box task with SMP guidance."""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.events import gsi_box_refresh, gsi_box_reset, init_smp_box_state
from smp.rl.tasks.carrybox import mdp as carrybox_mdp


def carrybox_box_spec(
  half_size: tuple[float, float, float] = (0.10, 0.15, 0.12),
  mass: float = 1.0,
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="box")
  body.add_freejoint(name="box_joint")
  body.add_geom(
    name="box_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=half_size,
    mass=mass,
    rgba=(0.8, 0.3, 0.2, 0.9),
    friction=(1.0, 0.08, 0.002),
    solref=(0.008, 1.0),
    solimp=(0.98, 0.99, 0.001, 0.9, 4.0),
    priority=1,
  )
  return spec


def carrybox_box_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=carrybox_box_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.5, 0.0, 0.18),
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
      joint_pos={},
      joint_vel={},
    ),
  )


def g1_carrybox_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 carry-box env cfg with SMP guidance."""
  cfg = g1_smp_env_cfg(play=play)

  cfg.scene.entities["box"] = carrybox_box_cfg()
  cfg.scene.env_spacing = 4.0
  cfg.scene.extent = 3.0
  cfg.sim.nconmax = 80
  cfg.sim.njmax = 2000

  # --- Commands ------------------------------------------------------------
  cfg.commands["carrybox"] = carrybox_mdp.CarryBoxCommandCfg(
    robot_name="robot",
    box_name="box",
    resampling_time_range=(20.0, 20.0),
    debug_vis=True,
    success_threshold=0.25,
    reset_stage_weights=(0.45, 0.35, 0.20),
    pickup_goal_distance=(0.35, 0.70),
    carry_goal_distance=(0.60, 1.20),
    place_goal_distance=(0.00, 0.20),
  )
  # Start with a shorter target range. The old 1.2-2.4m range let PPO discover
  # foot pushes before it learned the harder grasp/lift/carry sequence.
  cfg.commands["carrybox"].relative_goal_range.distance = (0.35, 1.0)

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=carrybox_mdp.generated_commands,
    params={"command_name": "carrybox"},
  )
  box_vel_obs = ObservationTermCfg(
    func=carrybox_mdp.box_velocity_b,
    params={"box_name": "box", "robot_name": "robot"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs
  cfg.observations["actor"].terms["box_velocity"] = box_vel_obs
  cfg.observations["critic"].terms["box_velocity"] = box_vel_obs

  # --- Rewards -------------------------------------------------------------
  cfg.rewards["task_smp_stage_gated"] = RewardTermCfg(
    func=carrybox_mdp.stage_gated_carrybox_task,
    weight=1.0,
    params={
      "command_name": "carrybox",
      "robot_name": "robot",
      "box_name": "box",
      "smp_ws": 4.0,
    },
  )


  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].func = init_smp_box_state
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_carrybox_wowalk.pt"
  )
  cfg.events["init_smp_state"].params["box_name"] = "box"
  cfg.events["gsi_reset"].func = gsi_box_reset
  cfg.events["gsi_reset"].params = {
    "box_name": "box",
    "stage_weights": (0.45, 0.35, 0.20),
    "pickup_height_max": 0.30,
    "pickup_dist_max": 0.85,
    "carry_height_min": 0.45,
    "carry_dist_max": 0.85,
    "place_height_min": 0.45,
    "place_dist_max": 0.85,
    "place_speed_max": 1.25,
  }
  if "gsi_refresh" in cfg.events:
    cfg.events["gsi_refresh"].func = gsi_box_refresh
  cfg.events.pop("push_robot", None)

  # --- Terminations --------------------------------------------------------
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=carrybox_mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    cfg.commands["carrybox"].resampling_time_range = (20.0, 20.0)
    cfg.commands["carrybox"].fixed_start_pos = (0.60, 0.0, 0.18)
    cfg.commands["carrybox"].fixed_goal_offset = (0.80, 0.0, 0.0)
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.events.pop("init_smp_state", None)
    cfg.events.pop("gsi_reset", None)
    cfg.events.pop("gsi_refresh", None)
  return cfg


__all__ = ["g1_carrybox_smp_env_cfg"]
