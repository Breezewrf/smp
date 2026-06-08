"""Carry-box observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def box_velocity_b(
  env: "ManagerBasedRlEnv",
  box_name: str = "box",
  robot_name: str = "robot",
) -> torch.Tensor:
  """Box linear/angular velocity rotated into the robot base frame."""
  robot = env.scene[robot_name]
  box = env.scene[box_name]
  inv_base = quat_inv(robot.data.root_link_quat_w)
  lin_b = quat_apply(inv_base, box.data.root_link_lin_vel_w)
  ang_b = quat_apply(inv_base, box.data.root_link_ang_vel_w)
  return torch.cat([lin_b, ang_b], dim=-1)
