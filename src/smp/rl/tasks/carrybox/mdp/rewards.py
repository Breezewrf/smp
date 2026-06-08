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


def _box_height_above_origin(env: "ManagerBasedRlEnv", box_name: str) -> torch.Tensor:
  box = env.scene[box_name]
  return box.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def _body_positions(
  env: "ManagerBasedRlEnv",
  robot_name: str,
  body_names: tuple[str, ...],
  cache_attr: str,
) -> torch.Tensor:
  robot = env.scene[robot_name]
  cache = getattr(env, cache_attr, {})
  cache_key = (robot_name, body_names)
  if cache_key not in cache:
    cache[cache_key] = robot.find_bodies(list(body_names), preserve_order=True)[0]
    setattr(env, cache_attr, cache)
  return robot.data.body_link_pos_w[:, cache[cache_key]]


def _hands_to_box_score(
  env: "ManagerBasedRlEnv",
  robot_name: str,
  box_name: str,
  left_hand_body: str,
  right_hand_body: str,
  lateral_offset: float,
  vertical_offset: float,
  pos_err_scale: float,
) -> torch.Tensor:
  hand_pos = _body_positions(
    env,
    robot_name,
    (left_hand_body, right_hand_body),
    "_carrybox_hand_id_cache",
  )
  left_hand = hand_pos[:, 0]
  right_hand = hand_pos[:, 1]

  box = env.scene[box_name]
  box_pos = box.data.root_link_pos_w
  box_quat = box.data.root_link_quat_w
  left_target_local = box_pos.new_zeros((env.num_envs, 3))
  right_target_local = box_pos.new_zeros((env.num_envs, 3))
  left_target_local[:, 1] = lateral_offset
  right_target_local[:, 1] = -lateral_offset
  left_target_local[:, 2] = vertical_offset
  right_target_local[:, 2] = vertical_offset
  left_target = box_pos + quat_apply(box_quat, left_target_local)
  right_target = box_pos + quat_apply(box_quat, right_target_local)

  left_err = torch.norm(left_hand - left_target, dim=-1)
  right_err = torch.norm(right_hand - right_target, dim=-1)
  return torch.exp(-pos_err_scale * 0.5 * (left_err + right_err))


def _held_lift_memory(
  env: "ManagerBasedRlEnv",
  held_score: torch.Tensor | None = None,
  score_threshold: float = 0.45,
) -> torch.Tensor:
  memory = getattr(env, "_carrybox_has_held_lift", None)
  if (
    not isinstance(memory, torch.Tensor)
    or memory.shape != (env.num_envs,)
    or memory.device != torch.device(env.device)
  ):
    memory = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._carrybox_has_held_lift = memory  # type: ignore[attr-defined]

  if held_score is not None:
    memory |= held_score > score_threshold
  return memory.float()


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
  return _hands_to_box_score(
    env,
    robot_name,
    box_name,
    left_hand_body,
    right_hand_body,
    lateral_offset,
    vertical_offset,
    pos_err_scale,
  )


def held_box_lift(
  env: "ManagerBasedRlEnv",
  robot_name: str = "robot",
  box_name: str = "box",
  left_hand_body: str = "left_wrist_yaw_link",
  right_hand_body: str = "right_wrist_yaw_link",
  lateral_offset: float = 0.18,
  vertical_offset: float = 0.03,
  pos_err_scale: float = 8.0,
  min_height: float = 0.45,
  height_gate_scale: float = 14.0,
  memory_threshold: float = 0.45,
) -> torch.Tensor:
  """Reward lifting the box while both hands are near the side targets."""
  hand_score = _hands_to_box_score(
    env,
    robot_name,
    box_name,
    left_hand_body,
    right_hand_body,
    lateral_offset,
    vertical_offset,
    pos_err_scale,
  )
  height = _box_height_above_origin(env, box_name)
  lift_score = torch.sigmoid((height - min_height) * height_gate_scale)
  held_score = hand_score * lift_score
  _held_lift_memory(env, held_score=held_score, score_threshold=memory_threshold)
  return held_score


def carried_box_progress(
  env: "ManagerBasedRlEnv",
  command_name: str,
  robot_name: str = "robot",
  box_name: str = "box",
  left_hand_body: str = "left_wrist_yaw_link",
  right_hand_body: str = "right_wrist_yaw_link",
  lateral_offset: float = 0.18,
  vertical_offset: float = 0.03,
  pos_err_scale: float = 8.0,
  min_height: float = 0.45,
  height_gate_scale: float = 14.0,
) -> torch.Tensor:
  """Only credit XY progress while the box is currently held and lifted."""
  cmd = _command(env, command_name)
  box = env.scene[box_name]
  progress = cmd._progress_fraction(box.data.root_link_pos_w)  # noqa: SLF001
  held = held_box_lift(
    env,
    robot_name=robot_name,
    box_name=box_name,
    left_hand_body=left_hand_body,
    right_hand_body=right_hand_body,
    lateral_offset=lateral_offset,
    vertical_offset=vertical_offset,
    pos_err_scale=pos_err_scale,
    min_height=min_height,
    height_gate_scale=height_gate_scale,
  )
  return progress * held


def carried_box_to_goal(
  env: "ManagerBasedRlEnv",
  command_name: str,
  robot_name: str = "robot",
  box_name: str = "box",
  left_hand_body: str = "left_wrist_yaw_link",
  right_hand_body: str = "right_wrist_yaw_link",
  lateral_offset: float = 0.18,
  vertical_offset: float = 0.03,
  pos_err_scale: float = 8.0,
  min_height: float = 0.45,
  height_gate_scale: float = 14.0,
  goal_err_scale: float = 1.8,
) -> torch.Tensor:
  """Reward getting near the goal only while the box is held and lifted."""
  cmd = _command(env, command_name)
  box = env.scene[box_name]
  goal_err = torch.norm(cmd.target_pos_w[:, :2] - box.data.root_link_pos_w[:, :2], dim=-1)
  goal_score = torch.exp(-goal_err_scale * goal_err)
  held = held_box_lift(
    env,
    robot_name=robot_name,
    box_name=box_name,
    left_hand_body=left_hand_body,
    right_hand_body=right_hand_body,
    lateral_offset=lateral_offset,
    vertical_offset=vertical_offset,
    pos_err_scale=pos_err_scale,
    min_height=min_height,
    height_gate_scale=height_gate_scale,
  )
  return goal_score * held


def place_box_at_goal(
  env: "ManagerBasedRlEnv",
  command_name: str,
  box_name: str = "box",
  place_height: float = 0.18,
  goal_err_scale: float = 3.0,
  height_err_scale: float = 12.0,
  speed_err_scale: float = 1.0,
) -> torch.Tensor:
  """Reward placing the previously lifted box near the target on the ground."""
  cmd = _command(env, command_name)
  box = env.scene[box_name]
  goal_err = torch.norm(cmd.target_pos_w[:, :2] - box.data.root_link_pos_w[:, :2], dim=-1)
  height_err = torch.abs(_box_height_above_origin(env, box_name) - place_height)
  speed = torch.norm(box.data.root_link_lin_vel_w, dim=-1)
  goal_score = torch.exp(-goal_err_scale * goal_err)
  height_score = torch.exp(-height_err_scale * height_err)
  settle_score = torch.exp(-speed_err_scale * speed)
  return goal_score * height_score * settle_score * _held_lift_memory(env)


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
  err = torch.abs(_box_height_above_origin(env, box_name) - target_height)
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


def ungrasped_box_motion_penalty(
  env: "ManagerBasedRlEnv",
  robot_name: str = "robot",
  box_name: str = "box",
  left_hand_body: str = "left_wrist_yaw_link",
  right_hand_body: str = "right_wrist_yaw_link",
  lateral_offset: float = 0.18,
  vertical_offset: float = 0.03,
  pos_err_scale: float = 8.0,
  speed_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize moving the box when the hands are not controlling it."""
  hand_score = _hands_to_box_score(
    env,
    robot_name,
    box_name,
    left_hand_body,
    right_hand_body,
    lateral_offset,
    vertical_offset,
    pos_err_scale,
  )
  box = env.scene[box_name]
  speed = torch.norm(box.data.root_link_lin_vel_w, dim=-1)
  active_speed = (speed - speed_threshold).clamp_min(0.0)
  return torch.square(active_speed) * (1.0 - hand_score).clamp(0.0, 1.0)


def feet_to_box_penalty(
  env: "ManagerBasedRlEnv",
  robot_name: str = "robot",
  box_name: str = "box",
  left_foot_body: str = "left_ankle_roll_link",
  right_foot_body: str = "right_ankle_roll_link",
  left_hand_body: str = "left_wrist_yaw_link",
  right_hand_body: str = "right_wrist_yaw_link",
  lateral_offset: float = 0.18,
  vertical_offset: float = 0.03,
  hand_pos_err_scale: float = 8.0,
  margin: float = 0.28,
  speed_threshold: float = 0.05,
) -> torch.Tensor:
  """Approximate anti-kick penalty using ankle proximity during box motion."""
  foot_pos = _body_positions(
    env,
    robot_name,
    (left_foot_body, right_foot_body),
    "_carrybox_foot_id_cache",
  )
  box = env.scene[box_name]
  dist = torch.norm(foot_pos[:, :, :2] - box.data.root_link_pos_w[:, None, :2], dim=-1)
  min_dist = torch.min(dist, dim=-1).values
  foot_score = torch.square(((margin - min_dist) / margin).clamp_min(0.0))
  speed = torch.norm(box.data.root_link_lin_vel_w, dim=-1)
  active_speed = (speed - speed_threshold).clamp_min(0.0)
  hand_score = _hands_to_box_score(
    env,
    robot_name,
    box_name,
    left_hand_body,
    right_hand_body,
    lateral_offset,
    vertical_offset,
    hand_pos_err_scale,
  )
  return foot_score * torch.square(active_speed) * (1.0 - hand_score).clamp(0.0, 1.0)
