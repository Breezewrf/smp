"""G1 and X2 getup tasks with SMP guidance."""

from __future__ import annotations

import os
from collections.abc import Callable

import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as _get_g1_spec
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import g1_smp_env_cfg, x2_smp_env_cfg
from smp.rl.rewards import body_orientation_l2, stand_still, task_smp_product
from smp.rl.tasks.getup import mdp
from smp.robots.x2 import X2_GETUP_HOME, get_x2_spec_with_body_collisions

# Matches the existing ``head_collision`` geom on ``torso_link`` in g1.xml.
HEAD_POS_IN_TORSO: tuple[float, float, float] = (0.0, 0.0, 0.43)
# The fixed X2 head body's origin is near its centre; offset the reward site to
# the top of its collision cylinder.
HEAD_POS_IN_X2_HEAD: tuple[float, float, float] = (0.0, 0.0, 0.08)

DEFAULT_X2_GETUP_CKPT = "datasets/pretrain_ckpt/pretrained_getup_x2.pt"


def x2_getup_ckpt_path() -> str:
  """Resolve the X2 getup prior, allowing direct use of pretraining runs."""
  return os.environ.get("SMP_X2_GETUP_CKPT", DEFAULT_X2_GETUP_CKPT)


def get_g1_spec_with_head() -> mujoco.MjSpec:  # type: ignore[attr-defined]
  """Stock G1 spec with a massless ``head`` site on ``torso_link``."""
  spec = _get_g1_spec()
  torso = spec.body("torso_link")
  if not any(s.name == "head" for s in torso.sites):
    torso.add_site(name="head", pos=HEAD_POS_IN_TORSO)
  return spec


def get_x2_getup_spec() -> mujoco.MjSpec:
  """Fixed-head X2 spec with full-body terrain contact and a head site."""
  spec = get_x2_spec_with_body_collisions()
  head = spec.body("head_pitch_link")
  if not any(site.name == "head" for site in head.sites):
    head.add_site(name="head", pos=HEAD_POS_IN_X2_HEAD)
  return spec


def _getup_smp_env_cfg(
  cfg: ManagerBasedRlEnvCfg,
  ckpt_path: str,
  spec_fn: Callable[[], mujoco.MjSpec],
) -> ManagerBasedRlEnvCfg:
  """Add robot-independent getup events, rewards, and terminations."""

  # --- Scene ---------------------------------------------------------------
  cfg.scene.entities["robot"].spec_fn = spec_fn

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = ckpt_path
  cfg.events["init_smp_state"].params["gsi_max_head_height"] = 0.9
  gsi_reset_cfg = cfg.events.pop("gsi_reset")
  cfg.events["record_success_by_initial_head_height"] = EventTermCfg(
    func=mdp.record_success_by_initial_head_height,
    mode="reset",
    params={"bin_edges": (0.3, 0.5, 0.7, 0.9)},
  )
  cfg.events["gsi_reset"] = gsi_reset_cfg
  cfg.events["reset_stand_counter"] = EventTermCfg(
    func=mdp.reset_stand_counter, mode="reset"
  )

  # --- Metrics -------------------------------------------------------------
  cfg.metrics["getup_success"] = MetricsTermCfg(
    func=mdp.episode_success,
    reduce="last",
    params={"head_height": 1.2, "max_speed": 0.5, "hold_steps": 25},
  )

  # --- Rewards -------------------------------------------------------------
  # task = 0.7·upward_velocity + 0.3·head_height, gated by SMP.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.upward_velocity,
          0.7,
          {
            "target_velocity": 0.25,
            "head_height_threshold": 0.9,
            "scale": 100.0,
          },
        ),
        (mdp.track_head_height, 0.3, {"target_height": 1.1, "scale": 1.0}),
        (
          body_orientation_l2,
          -0.1,
          {},
        ),
        (
          stand_still,
          -0.01,
          {},
        ),
      ),
    },
  )

  # --- Terminations --------------------------------------------------------
  cfg.terminations.pop("self_collision", None)
  cfg.terminations["smp_too_low"] = TerminationTermCfg(
    func=mdp.smp_too_low,
    params={"threshold": 0.02, "ws": 6.0, "grace_steps": 5},
  )

  cfg.terminations["stood_up"] = TerminationTermCfg(
    func=mdp.stood_up,
    time_out=True,
    params={"head_height": 1.2, "max_speed": 0.5, "hold_steps": 25},
  )

  cfg.episode_length_s = 5

  return cfg


def g1_getup_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 getup environment."""
  cfg = _getup_smp_env_cfg(
    g1_smp_env_cfg(play=play),
    "datasets/pretrain_ckpt/pretrained_getup_f2s2.pt",
    get_g1_spec_with_head,
  )
  if play:
    cfg.auto_reset = False
    cfg.episode_length_s = int(1e9)
  return cfg


def x2_getup_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the fixed-head, 29-DoF X2 getup environment."""
  cfg = _getup_smp_env_cfg(
    x2_smp_env_cfg(play=play),
    x2_getup_ckpt_path(),
    get_x2_getup_spec,
  )
  cfg.scene.entities["robot"].init_state = X2_GETUP_HOME
  cfg.sim.nconmax = 64
  if play:
    cfg.auto_reset = False
    cfg.episode_length_s = int(1e9)
  return cfg
