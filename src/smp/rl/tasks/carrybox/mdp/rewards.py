"""Carry-box reward components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from smp.rl.tasks.carrybox.mdp.commands import CarryBoxCommand


def _command(env: "ManagerBasedRlEnv", command_name: str) -> "CarryBoxCommand":
  return env.command_manager.get_term(command_name)  # type: ignore[return-value]


def box_to_goal(
  env: "ManagerBasedRlEnv",
  command_name: str,
  box_name: str = "box",
  pos_err_scale: float = 1.5,
) -> torch.Tensor:
  """Exponential XY distance reward for moving the box to the goal."""
  cmd = _command(env, command_name)
  box = env.scene[box_name]
  err = torch.norm(cmd.target_pos_w[:, :2] - box.data.root_link_pos_w[:, :2], dim=-1)
  return torch.exp(-pos_err_scale * err)


def box_progress(
  env: "ManagerBasedRlEnv",
  command_name: str,
  box_name: str = "box",
) -> torch.Tensor:
  """Normalized projected progress from box start to goal."""
  cmd = _command(env, command_name)
  box = env.scene[box_name]
  return cmd._progress_fraction(box.data.root_link_pos_w)  # noqa: SLF001


def robot_to_box(
  env: "ManagerBasedRlEnv",
  command_name: str,
  robot_name: str = "robot",
  box_name: str = "box",
  pos_err_scale: float = 2.0,
) -> torch.Tensor:
  """Encourage the robot to stay close enough to influence the box."""
  del command_name
  robot = env.scene[robot_name]
  box = env.scene[box_name]
  err = torch.norm(box.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=-1)
  return torch.exp(-pos_err_scale * err)


def hands_to_box(
  env: "ManagerBasedRlEnv",
  robot_name: str = "robot",
  box_name: str = "box",
  left_hand_body: str = "left_wrist_yaw_link",
  right_hand_body: str = "right_wrist_yaw_link",
  lateral_offset: float = 0.18,
  vertical_offset: float = 0.03,
  pos_err_scale: float = 8.0,
) -> torch.Tensor:
  """Encourage both wrists to approach opposite sides of the box."""
  robot = env.scene[robot_name]
  box = env.scene[box_name]
  cache_key = (robot_name, left_hand_body, right_hand_body)
  hand_id_cache = getattr(env, "_carrybox_hand_id_cache", {})
  if cache_key not in hand_id_cache:
    hand_id_cache[cache_key] = robot.find_bodies(
      [left_hand_body, right_hand_body],
      preserve_order=True,
    )[0]
    env._carrybox_hand_id_cache = hand_id_cache  # type: ignore[attr-defined]
  hand_ids = hand_id_cache[cache_key]
  left_hand = robot.data.body_link_pos_w[:, hand_ids[0]]
  right_hand = robot.data.body_link_pos_w[:, hand_ids[1]]

  box_pos = box.data.root_link_pos_w
  box_quat = box.data.root_link_quat_w
  left_target_local = torch.zeros(env.num_envs, 3, device=env.device)
  right_target_local = torch.zeros(env.num_envs, 3, device=env.device)
  left_target_local[:, 1] = lateral_offset
  right_target_local[:, 1] = -lateral_offset
  left_target_local[:, 2] = vertical_offset
  right_target_local[:, 2] = vertical_offset
  left_target = box_pos + quat_apply(box_quat, left_target_local)
  right_target = box_pos + quat_apply(box_quat, right_target_local)

  left_err = torch.norm(left_hand - left_target, dim=-1)
  right_err = torch.norm(right_hand - right_target, dim=-1)
  return torch.exp(-pos_err_scale * 0.5 * (left_err + right_err))


def box_upright(
  env: "ManagerBasedRlEnv",
  box_name: str = "box",
  tilt_err_scale: float = 2.0,
) -> torch.Tensor:
  """Reward the box remaining upright."""
  box = env.scene[box_name]
  up_local = torch.zeros(env.num_envs, 3, device=env.device)
  up_local[:, 2] = 1.0
  up_w = quat_apply(box.data.root_link_quat_w, up_local)
  tilt_err = 1.0 - up_w[:, 2].clamp(-1.0, 1.0)
  return torch.exp(-tilt_err_scale * tilt_err)


def box_height(
  env: "ManagerBasedRlEnv",
  box_name: str = "box",
  target_height: float = 0.18,
  height_err_scale: float = 20.0,
) -> torch.Tensor:
  """Keep the box near its nominal carry/ground-contact height."""
  box = env.scene[box_name]
  box_height_above_origin = box.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  err = torch.abs(box_height_above_origin - target_height)
  return torch.exp(-height_err_scale * err)


def box_robot_distance_penalty(
  env: "ManagerBasedRlEnv",
  robot_name: str = "robot",
  box_name: str = "box",
  max_distance: float = 1.5,
) -> torch.Tensor:
  """Quadratic penalty when the robot loses the box."""
  robot = env.scene[robot_name]
  box = env.scene[box_name]
  dist = torch.norm(box.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=-1)
  return torch.square((dist - max_distance).clamp_min(0.0))


def box_speed_penalty(
  env: "ManagerBasedRlEnv",
  box_name: str = "box",
  max_speed: float = 3.0,
) -> torch.Tensor:
  """Quadratic penalty only for very large box speeds."""
  box = env.scene[box_name]
  speed = torch.norm(box.data.root_link_lin_vel_w, dim=-1)
  return torch.square((speed - max_speed).clamp_min(0.0))
