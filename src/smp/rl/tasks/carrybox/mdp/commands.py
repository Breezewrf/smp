"""Carry-box command: reset box start pose and expose box/goal offsets."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from smp.rl.carrybox_stages import CARRYBOX_STAGE_IDS, CARRYBOX_STAGE_NAMES

if TYPE_CHECKING:
  import viser
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _xy_world_to_local(vec_w: torch.Tensor, heading_w: torch.Tensor) -> torch.Tensor:
  cos_h = torch.cos(heading_w)
  sin_h = torch.sin(heading_w)
  x_w, y_w = vec_w[..., 0], vec_w[..., 1]
  return torch.stack([cos_h * x_w + sin_h * y_w, -sin_h * x_w + cos_h * y_w], dim=-1)


class CarryBoxCommand(CommandTerm):
  """Episode command for moving a free box from a start pose to a goal pose."""

  cfg: CarryBoxCommandCfg

  def __init__(self, cfg: CarryBoxCommandCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.robot_name]
    self.box: Entity = env.scene[cfg.box_name]

    self.start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    # [box_rel_robot_b(3), goal_rel_box_b(3), progress, goal_dist]
    self.command_b = torch.zeros(self.num_envs, 8, device=self.device)
    self.reset_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)

    self.metrics["box_goal_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["robot_box_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["box_progress"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["box_height"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["has_lifted"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["stage_pickup"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["stage_carry"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["stage_place"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["gsi_stage_fallback"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["pickup_gate"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["carry_gate"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["place_gate"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["hand_score"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["lift_score"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["held_score"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["pickup_task"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["carry_task"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["place_task"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["stage_task"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["placed_at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

    self._gui_enabled: viser.GuiCheckboxHandle | None = None
    self._gui_distance: viser.GuiSliderHandle | None = None
    self._gui_angle: viser.GuiSliderHandle | None = None
    self._gui_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  def _update_metrics(self) -> None:
    box_pos = self.box.data.root_link_pos_w
    goal_err = torch.norm(self.target_pos_w[:, :2] - box_pos[:, :2], dim=-1)
    robot_err = torch.norm(box_pos[:, :2] - self.robot.data.root_link_pos_w[:, :2], dim=-1)
    progress = self._progress_fraction(box_pos)
    at_goal = (goal_err < self.cfg.success_threshold).float()
    height = box_pos[:, 2] - self._env.scene.env_origins[:, 2]
    height_ok = torch.abs(height - self.cfg.place_height) < self.cfg.place_height_threshold
    speed_ok = torch.norm(self.box.data.root_link_lin_vel_w, dim=-1) < self.cfg.place_speed_threshold
    lift_memory = getattr(self._env, "_carrybox_has_held_lift", None)
    has_lifted = (
      lift_memory
      if isinstance(lift_memory, torch.Tensor) and lift_memory.shape == (self.num_envs,)
      else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    )
    gsi_fallback = getattr(self._env, "_carrybox_gsi_stage_fallback", None)
    if (
      not isinstance(gsi_fallback, torch.Tensor)
      or gsi_fallback.shape != (self.num_envs,)
      or gsi_fallback.device != torch.device(self.device)
    ):
      gsi_fallback = torch.zeros(self.num_envs, device=self.device)
    placed = (goal_err < self.cfg.success_threshold) & height_ok & speed_ok & has_lifted
    placed_at_goal = placed.float()
    self.episode_success = torch.maximum(self.episode_success, placed_at_goal)

    self.metrics["box_goal_error"] = goal_err
    self.metrics["robot_box_error"] = robot_err
    self.metrics["box_progress"] = progress
    self.metrics["box_height"] = height
    self.metrics["has_lifted"] = has_lifted.float()
    self.metrics["stage_pickup"] = (self.reset_stage == CARRYBOX_STAGE_IDS["pickup"]).float()
    self.metrics["stage_carry"] = (self.reset_stage == CARRYBOX_STAGE_IDS["carry"]).float()
    self.metrics["stage_place"] = (self.reset_stage == CARRYBOX_STAGE_IDS["place"]).float()
    self.metrics["gsi_stage_fallback"] = gsi_fallback.clone()
    for metric_name in (
      "pickup_gate",
      "carry_gate",
      "place_gate",
      "hand_score",
      "lift_score",
      "held_score",
      "pickup_task",
      "carry_task",
      "place_task",
      "stage_task",
    ):
      metric = getattr(self._env, f"_carrybox_{metric_name}", None)
      if (
        not isinstance(metric, torch.Tensor)
        or metric.shape != (self.num_envs,)
        or metric.device != torch.device(self.device)
      ):
        metric = torch.zeros(self.num_envs, device=self.device)
      self.metrics[metric_name] = metric.clone()
    self.metrics["at_goal"] = at_goal
    self.metrics["placed_at_goal"] = placed_at_goal
    self.metrics["episode_success"] = self.episode_success

  def compute_success(self) -> torch.Tensor:
    return self.metrics["placed_at_goal"] > 0.5

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = int(env_ids.numel())
    origins = self._env.scene.env_origins[env_ids]
    self.episode_success[env_ids] = 0.0
    self.reset_stage[env_ids] = self._sample_reset_stage(env_ids)

    if self.cfg.fixed_start_pos is not None:
      start_pos = torch.tensor(
        self.cfg.fixed_start_pos,
        dtype=torch.float32,
        device=self.device,
      ).expand(n, -1) + origins
      quat = torch.zeros(n, 4, device=self.device)
      quat[:, 0] = 1.0
      pose = torch.cat([start_pos, quat], dim=-1)
      velocity = torch.zeros(n, 6, device=self.device)
      self.box.write_root_link_pose_to_sim(pose, env_ids=env_ids)
      self.box.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
    else:
      start_pos = self.box.data.root_link_pos_w[env_ids].clone()

    if self.cfg.fixed_goal_offset is not None:
      goal_offset = torch.tensor(
        self.cfg.fixed_goal_offset,
        dtype=torch.float32,
        device=self.device,
      ).expand(n, -1)
      target_pos = start_pos + goal_offset
      target_pos[:, 2] = origins[:, 2] + self.cfg.goal_height
    elif self.cfg.goal_mode == "relative":
      dist_min, dist_max = self._stage_goal_distance_range(env_ids)
      dist = dist_min + torch.rand(n, device=self.device) * (dist_max - dist_min)
      angle_min, angle_max = self._stage_goal_angle_range(env_ids)
      angle = angle_min + torch.rand(n, device=self.device) * (angle_max - angle_min)
      no_angle_randomization = angle_max <= angle_min
      if torch.any(no_angle_randomization):
        angle[no_angle_randomization] = angle_min[no_angle_randomization]
      goal_local = torch.zeros(n, 3, device=self.device)
      goal_local[:, 0] = dist * torch.cos(angle)
      goal_local[:, 1] = dist * torch.sin(angle)
      goal_local[:, 2] = self.cfg.goal_height
      target_pos = start_pos + goal_local
      target_pos[:, 2] = origins[:, 2] + self.cfg.goal_height
    else:
      gr = self.cfg.target_pose_range
      target_local = torch.empty(n, 3, device=self.device)
      target_local[:, 0].uniform_(gr.x[0], gr.x[1])
      target_local[:, 1].uniform_(gr.y[0], gr.y[1])
      target_local[:, 2].uniform_(gr.z[0], gr.z[1])
      target_pos = target_local + origins

    self.start_pos_w[env_ids] = start_pos
    self.target_pos_w[env_ids] = target_pos
    lift_memory = getattr(self._env, "_carrybox_has_held_lift", None)
    if (
      not isinstance(lift_memory, torch.Tensor)
      or lift_memory.shape != (self.num_envs,)
      or lift_memory.device != torch.device(self.device)
    ):
      lift_memory = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
      self._env._carrybox_has_held_lift = lift_memory  # type: ignore[attr-defined]
    lift_memory[env_ids] = self.reset_stage[env_ids] != CARRYBOX_STAGE_IDS["pickup"]

  def _sample_reset_stage(self, env_ids: torch.Tensor) -> torch.Tensor:
    stage = getattr(self._env, "_carrybox_reset_stage", None)
    if (
      isinstance(stage, torch.Tensor)
      and stage.shape == (self.num_envs,)
      and stage.device == torch.device(self.device)
    ):
      return stage[env_ids].long().clamp(0, len(CARRYBOX_STAGE_NAMES) - 1)

    weights = torch.tensor(
      self.cfg.reset_stage_weights,
      dtype=torch.float32,
      device=self.device,
    )
    if torch.any(weights < 0.0) or weights.sum() <= 0.0:
      msg = (
        "reset_stage_weights must be non-negative with positive sum, "
        f"got {self.cfg.reset_stage_weights}."
      )
      raise ValueError(msg)
    return torch.multinomial(weights / weights.sum(), env_ids.numel(), replacement=True)

  def _stage_goal_distance_range(
    self,
    env_ids: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(env_ids.numel())
    stage = self.reset_stage[env_ids]
    min_dist = torch.empty(n, device=self.device)
    max_dist = torch.empty(n, device=self.device)
    ranges = (
      self.cfg.pickup_goal_distance,
      self.cfg.carry_goal_distance,
      self.cfg.place_goal_distance,
    )
    for stage_id, distance_range in enumerate(ranges):
      mask = stage == stage_id
      min_dist[mask] = distance_range[0]
      max_dist[mask] = distance_range[1]
    return min_dist, max_dist

  def _stage_goal_angle_range(
    self,
    env_ids: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(env_ids.numel())
    stage = self.reset_stage[env_ids]
    min_angle = torch.empty(n, device=self.device)
    max_angle = torch.empty(n, device=self.device)
    ranges = (
      self.cfg.pickup_goal_angle,
      self.cfg.carry_goal_angle,
      self.cfg.place_goal_angle,
    )
    default_angle = self.cfg.relative_goal_range.angle
    if default_angle is None:
      default_angle = (0.0, 0.0)
    for stage_id, angle_range in enumerate(ranges):
      selected = default_angle if angle_range is None else angle_range
      mask = stage == stage_id
      min_angle[mask] = selected[0]
      max_angle[mask] = selected[1]
    return min_angle, max_angle

  def _progress_fraction(self, box_pos_w: torch.Tensor) -> torch.Tensor:
    start_xy = self.start_pos_w[:, :2]
    target_xy = self.target_pos_w[:, :2]
    goal_vec = target_xy - start_xy
    goal_dist = torch.norm(goal_vec, dim=-1).clamp_min(1e-6)
    goal_dir = goal_vec / goal_dist.unsqueeze(-1)
    progress = ((box_pos_w[:, :2] - start_xy) * goal_dir).sum(dim=-1)
    return (progress / goal_dist).clamp(0.0, 1.0)

  def _update_command(self) -> None:
    robot_pos = self.robot.data.root_link_pos_w
    box_pos = self.box.data.root_link_pos_w
    heading_w = self.robot.data.heading_w

    self.command_b[:, 0:2] = _xy_world_to_local(box_pos[:, :2] - robot_pos[:, :2], heading_w)
    self.command_b[:, 2] = box_pos[:, 2] - robot_pos[:, 2]
    self.command_b[:, 3:5] = _xy_world_to_local(self.target_pos_w[:, :2] - box_pos[:, :2], heading_w)
    self.command_b[:, 5] = self.target_pos_w[:, 2] - box_pos[:, 2]
    self.command_b[:, 6] = self._progress_fraction(box_pos)
    self.command_b[:, 7] = torch.norm(self.target_pos_w[:, :2] - box_pos[:, :2], dim=-1)

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    with server.gui.add_folder(name.capitalize()):
      self._gui_enabled = server.gui.add_checkbox("Enable", initial_value=False)
      self._gui_distance = server.gui.add_slider(
        "goal_distance",
        min=0.0,
        max=float(self.cfg.relative_goal_range.distance[1]),
        step=0.1,
        initial_value=float(sum(self.cfg.relative_goal_range.distance) * 0.5),
      )
      self._gui_angle = server.gui.add_slider(
        "goal_angle (rad)",
        min=-math.pi,
        max=math.pi,
        step=0.05,
        initial_value=0.0,
      )
    self._gui_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._gui_enabled is None or not self._gui_enabled.value:
      return
    assert self._gui_get_env_idx is not None
    assert self._gui_distance is not None
    assert self._gui_angle is not None
    idx = self._gui_get_env_idx()
    distance = float(self._gui_distance.value)
    angle = float(self._gui_angle.value)
    box_pos = self.box.data.root_link_pos_w[idx]
    self.start_pos_w[idx] = box_pos
    self.target_pos_w[idx, 0] = box_pos[0] + distance * math.cos(angle)
    self.target_pos_w[idx, 1] = box_pos[1] + distance * math.sin(angle)
    self.target_pos_w[idx, 2] = self._env.scene.env_origins[idx, 2] + self.cfg.goal_height
    self._update_command()

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    start = self.start_pos_w.cpu().numpy()
    target = self.target_pos_w.cpu().numpy()
    box = self.box.data.root_link_pos_w.cpu().numpy()
    for batch in env_indices:
      visualizer.add_sphere(
        target[batch],
        radius=self.cfg.viz.target_radius,
        color=self.cfg.viz.target_color,
        label=f"carrybox_goal_{batch}",
      )
      visualizer.add_sphere(
        start[batch],
        radius=self.cfg.viz.start_radius,
        color=self.cfg.viz.start_color,
        label=f"carrybox_start_{batch}",
      )
      visualizer.add_arrow(
        box[batch],
        target[batch],
        color=self.cfg.viz.arrow_color,
        width=0.01,
      )


@dataclass(kw_only=True)
class CarryBoxCommandCfg(CommandTermCfg):
  robot_name: str = "robot"
  box_name: str = "box"
  success_threshold: float = 0.25
  place_height: float = 0.18
  place_height_threshold: float = 0.08
  place_speed_threshold: float = 0.35
  goal_mode: Literal["relative", "absolute"] = "relative"
  goal_height: float = 0.18
  fixed_start_pos: tuple[float, float, float] | None = None
  fixed_goal_offset: tuple[float, float, float] | None = None
  reset_stage_weights: tuple[float, float, float] = (0.45, 0.35, 0.20)
  pickup_goal_distance: tuple[float, float] = (0.35, 0.70)
  carry_goal_distance: tuple[float, float] = (0.60, 1.20)
  place_goal_distance: tuple[float, float] = (0.00, 0.20)
  pickup_goal_angle: tuple[float, float] | None = None
  carry_goal_angle: tuple[float, float] | None = None
  place_goal_angle: tuple[float, float] | None = (-0.20, 0.20)

  @dataclass
  class RelativeGoalRangeCfg:
    distance: tuple[float, float] = (1.2, 2.4)
    angle: tuple[float, float] | None = (-0.6, 0.6)

  @dataclass
  class TargetPoseRangeCfg:
    x: tuple[float, float] = (1.6, 2.6)
    y: tuple[float, float] = (-0.8, 0.8)
    z: tuple[float, float] = (0.18, 0.18)

  @dataclass
  class VizCfg:
    target_radius: float = 0.12
    start_radius: float = 0.08
    target_color: tuple[float, float, float, float] = (0.1, 0.35, 1.0, 0.45)
    start_color: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.35)
    arrow_color: tuple[float, float, float, float] = (0.2, 0.8, 1.0, 0.5)

  relative_goal_range: RelativeGoalRangeCfg = field(default_factory=RelativeGoalRangeCfg)
  target_pose_range: TargetPoseRangeCfg = field(default_factory=TargetPoseRangeCfg)
  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: "ManagerBasedRlEnv") -> CarryBoxCommand:
    return CarryBoxCommand(self, env)
