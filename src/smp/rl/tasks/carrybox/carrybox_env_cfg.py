"""G1 carry-box task with SMP guidance."""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.events import gsi_box_refresh, gsi_box_reset, init_smp_box_state
from smp.rl.rewards import task_smp_box_product
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
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_box_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          carrybox_mdp.robot_to_box,
          0.15,
          {
            "command_name": "carrybox",
            "robot_name": "robot",
            "box_name": "box",
            "pos_err_scale": 2.0,
          },
        ),
        (
          carrybox_mdp.hands_to_box,
          0.8,
          {
            "robot_name": "robot",
            "box_name": "box",
            "lateral_offset": 0.18,
            "vertical_offset": 0.03,
            "pos_err_scale": 8.0,
          },
        ),
        (
          carrybox_mdp.carried_box_to_goal,
          1.0,
          {
            "command_name": "carrybox",
            "robot_name": "robot",
            "box_name": "box",
            "lateral_offset": 0.18,
            "vertical_offset": 0.03,
            "pos_err_scale": 8.0,
            "min_height": 0.45,
            "height_gate_scale": 14.0,
            "goal_err_scale": 1.8,
          },
        ),
      ),
      "ws": 4,
      "box_name": "box",
    },
  )


  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].func = init_smp_box_state
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_carrybox.pt"
  )
  cfg.events["init_smp_state"].params["box_name"] = "box"
  cfg.events["gsi_reset"].func = gsi_box_reset
  cfg.events["gsi_reset"].params = {"box_name": "box"}
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

  return cfg


__all__ = ["g1_carrybox_smp_env_cfg"]
