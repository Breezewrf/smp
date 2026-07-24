"""Agibot X2 asset configuration for SMP training."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

X2_CSV_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_pitch_joint",
  "waist_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
  "head_yaw_joint",
  "head_pitch_joint",
)

# The deployed head is fixed. The RL asset removes these final two joints from
# the model, and the motion pipeline drops the corresponding CSV columns.
X2_JOINT_NAMES: tuple[str, ...] = X2_CSV_JOINT_NAMES[:-2]

X2_EE_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "head_pitch_link",
  "left_wrist_roll_link",
  "right_wrist_roll_link",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
X2_XML_PATH = _PROJECT_ROOT / "assets" / "robots" / "x2" / "x2_ultra.xml"


def _build_x2_spec(*, foot_contacts_only: bool) -> mujoco.MjSpec:
  """Load X2, fix its head, and configure collision geometry."""
  if not X2_XML_PATH.is_file():
    raise FileNotFoundError(f"X2 MJCF not found: {X2_XML_PATH}")
  spec = mujoco.MjSpec.from_file(str(X2_XML_PATH))

  for actuator in list(spec.actuators):
    spec.delete(actuator)

  fixed_head_joints = {"head_yaw_joint", "head_pitch_joint"}
  for sensor in list(spec.sensors):
    if sensor.objname in fixed_head_joints:
      spec.delete(sensor)
  for joint_name in fixed_head_joints:
    spec.delete(spec.joint(joint_name))

  # Manager terms require unique names for collision randomization.
  for body in spec.bodies:
    for index, geom in enumerate(body.geoms):
      if not geom.name:
        kind = "collision" if geom.contype else "visual"
        geom.name = f"{body.name}_{kind}_{index}"
      is_foot_contact = body.name in {
        "left_ankle_roll_link",
        "right_ankle_roll_link",
      }
      if geom.contype:
        if foot_contacts_only and not is_foot_contact:
          geom.contype = 0
          geom.conaffinity = 0
        elif not foot_contacts_only:
          # Keep terrain contact while preventing overlapping robot meshes from
          # colliding with one another during getup.
          geom.conaffinity = 0

  # Scene attachment intentionally takes its timestep from SimulationCfg.
  spec.option.timestep = mujoco.MjOption().timestep
  return spec


def get_x2_spec() -> mujoco.MjSpec:
  """Return the fixed-head X2 locomotion spec with foot-only contacts."""
  return _build_x2_spec(foot_contacts_only=True)


def get_x2_spec_with_body_collisions() -> mujoco.MjSpec:
  """Return fixed-head X2 with all body geoms able to contact terrain."""
  return _build_x2_spec(foot_contacts_only=False)


_HIP_PITCH_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint"),
  stiffness=40.0,
  damping=4.0,
  effort_limit=120.0,
)
_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",),
  stiffness=30.0,
  damping=3.0,
  effort_limit=120.0,
)
_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=80.0,
  damping=8.0,
  effort_limit=120.0,
)
_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=40.0,
  damping=4.0,
  effort_limit=36.0,
)
_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=20.0,
  damping=2.0,
  effort_limit=24.0,
)
_WAIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=20.0,
  damping=4.0,
  effort_limit=120.0,
)
_WAIST_PITCH_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=20.0,
  damping=4.0,
  effort_limit=48.0,
)
_SHOULDER_PITCH_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"),
  stiffness=20.0,
  damping=2.0,
  effort_limit=36.0,
)
_SHOULDER_YAW_ELBOW_WRIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_yaw_joint",
  ),
  stiffness=20.0,
  damping=2.0,
  effort_limit=24.0,
)
_WRIST_PITCH_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_roll_joint"),
  stiffness=20.0,
  damping=2.0,
  effort_limit=4.8,
)

X2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _HIP_PITCH_ROLL,
    _HIP_YAW,
    _KNEE,
    _ANKLE_PITCH,
    _ANKLE_ROLL,
    _WAIST_YAW,
    _WAIST_PITCH_ROLL,
    _SHOULDER_PITCH_ROLL,
    _SHOULDER_YAW_ELBOW_WRIST_YAW,
    _WRIST_PITCH_ROLL,
  ),
  soft_joint_pos_limit_factor=0.9,
)

X2_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.68),
  joint_pos={
    ".*_hip_pitch_joint": -0.3,
    ".*_knee_joint": 0.7,
    ".*_ankle_pitch_joint": -0.18,
    "waist_pitch_joint": 0.2,
    ".*_shoulder_pitch_joint": 0.15,
    ".*_elbow_joint": -1.0,
    "left_wrist_yaw_joint": -0.8,
    "right_wrist_yaw_joint": 0.8,
  },
  joint_vel={".*": 0.0},
)

# Symmetric getup action offset selected from reference-motion statistics and
# validated in the full-body-contact X2 model. The elevated root places the
# foot contact spheres on the floor without an initial drop.
X2_GETUP_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.694),
  joint_pos={
    ".*_hip_pitch_joint": -0.25,
    ".*_knee_joint": 0.60,
    ".*_ankle_pitch_joint": 0.05,
    "waist_pitch_joint": 0.30,
    ".*_shoulder_pitch_joint": 0.196,
  },
  joint_vel={".*": 0.0},
)


def get_x2_robot_cfg() -> EntityCfg:
  """Return a fresh 29-DoF X2 configuration with a physically fixed head."""
  return EntityCfg(
    init_state=X2_HOME,
    spec_fn=get_x2_spec,
    articulation=X2_ARTICULATION,
    sort_actuators=True,
  )


X2_ACTION_SCALE: dict[str, float] = {}
for actuator in X2_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  assert actuator.effort_limit is not None
  scale = 0.25 * actuator.effort_limit / actuator.stiffness
  for name_expr in actuator.target_names_expr:
    X2_ACTION_SCALE[name_expr] = scale
