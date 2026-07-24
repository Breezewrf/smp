"""Robot-specific assets and motion feature layouts."""

from smp.robots.g1 import G1_EE_BODY_NAMES, G1_JOINT_NAMES
from smp.robots.x2 import (
  X2_ACTION_SCALE,
  X2_CSV_JOINT_NAMES,
  X2_EE_BODY_NAMES,
  X2_GETUP_HOME,
  X2_JOINT_NAMES,
  get_x2_robot_cfg,
)

__all__ = [
  "G1_EE_BODY_NAMES",
  "G1_JOINT_NAMES",
  "X2_ACTION_SCALE",
  "X2_CSV_JOINT_NAMES",
  "X2_EE_BODY_NAMES",
  "X2_GETUP_HOME",
  "X2_JOINT_NAMES",
  "get_x2_robot_cfg",
]
