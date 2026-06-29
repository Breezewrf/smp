"""MuJoCo keyboard playback for SMP velocity policies."""

from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.keys import (
  KEY_1,
  KEY_2,
  KEY_A,
  KEY_D,
  KEY_E,
  KEY_H,
  KEY_Q,
  KEY_S,
  KEY_W,
  KEY_Z,
)

import smp.rl.tasks  # noqa: F401  # registers Smp-* tasks


@dataclass(frozen=True)
class Sim2SimKeyboardConfig:
  task_id: str = "Smp-Velocity-G1"
  checkpoint_file: str = ""
  """Policy checkpoint. Empty selects latest model_*.pt under the task log dir."""
  device: str | None = None
  num_envs: int = 1
  frame_rate: float = 60.0
  log_root: str = "logs/rsl_rl"
  command_name: str = "twist"
  cmd_step_lin: float = 0.1
  cmd_step_yaw: float = 0.1
  lin_vel_x_limit: float = 3.0
  lin_vel_y_limit: float = 1.0
  ang_vel_z_limit: float = 1.5


def _checkpoint_step(path: Path) -> int:
  match = re.search(r"model_(\d+)\.pt$", path.name)
  return int(match.group(1)) if match else -1


def _latest_checkpoint(log_root: Path, experiment_name: str) -> Path:
  candidates = sorted(
    (log_root / experiment_name).glob("**/model_*.pt"),
    key=lambda p: (_checkpoint_step(p), p.stat().st_mtime),
  )
  if not candidates:
    raise FileNotFoundError(
      f"No model_*.pt checkpoint found under {log_root / experiment_name}"
    )
  return candidates[-1]


class KeyboardVelocityController:
  """Thread-safe velocity command state updated by keyboard callbacks."""

  def __init__(
    self,
    step_lin: float,
    step_yaw: float,
    x_limit: float,
    y_limit: float,
    yaw_limit: float,
  ) -> None:
    self._lock = threading.Lock()
    self._cmd = [0.0, 0.0, 0.0]
    self.step_lin = max(1e-4, float(step_lin))
    self.step_yaw = max(1e-4, float(step_yaw))
    self.x_limit = float(abs(x_limit))
    self.y_limit = float(abs(y_limit))
    self.yaw_limit = float(abs(yaw_limit))

  def _clip(self) -> None:
    self._cmd[0] = max(-self.x_limit, min(self.x_limit, self._cmd[0]))
    self._cmd[1] = max(-self.y_limit, min(self.y_limit, self._cmd[1]))
    self._cmd[2] = max(-self.yaw_limit, min(self.yaw_limit, self._cmd[2]))

  def _print_state(self, prefix: str = "") -> None:
    print(
      f"[CMD] {prefix}"
      f"vx={self._cmd[0]:+.2f}, vy={self._cmd[1]:+.2f}, wz={self._cmd[2]:+.2f}"
    )

  def print_help(self) -> None:
    print("[INFO] Keyboard controls:")
    print("  W/S : increase/decrease forward velocity vx")
    print("  A/D : increase/decrease lateral velocity vy")
    print("  Q/E : increase/decrease yaw rate wz")
    print("  Z   : zero velocity command")
    print("  1/2 : decrease/increase command step")
    print("  H   : print this help")

  def on_key(self, key: int) -> None:
    with self._lock:
      if key == KEY_W:
        self._cmd[0] += self.step_lin
      elif key == KEY_S:
        self._cmd[0] -= self.step_lin
      elif key == KEY_A:
        self._cmd[1] += self.step_lin
      elif key == KEY_D:
        self._cmd[1] -= self.step_lin
      elif key == KEY_Q:
        self._cmd[2] += self.step_yaw
      elif key == KEY_E:
        self._cmd[2] -= self.step_yaw
      elif key == KEY_Z:
        self._cmd = [0.0, 0.0, 0.0]
      elif key == KEY_1:
        self.step_lin = max(0.01, self.step_lin * 0.8)
        self.step_yaw = max(0.01, self.step_yaw * 0.8)
        print(f"[CMD] step: lin={self.step_lin:.3f}, yaw={self.step_yaw:.3f}")
        return
      elif key == KEY_2:
        self.step_lin = min(1.0, self.step_lin * 1.25)
        self.step_yaw = min(1.0, self.step_yaw * 1.25)
        print(f"[CMD] step: lin={self.step_lin:.3f}, yaw={self.step_yaw:.3f}")
        return
      elif key == KEY_H:
        self.print_help()
        return
      else:
        return

      self._clip()
      self._print_state()

  def command_tensor(self, ref: torch.Tensor) -> torch.Tensor:
    with self._lock:
      return torch.tensor(self._cmd, dtype=ref.dtype, device=ref.device)


class KeyboardCommandPolicy:
  """Writes the keyboard velocity command before querying the actor."""

  def __init__(self, actor_policy, command_term, controller: KeyboardVelocityController):
    self.actor_policy = actor_policy
    self.command_term = command_term
    self.controller = controller

  def __call__(self, obs):
    cmd = self.controller.command_tensor(self.command_term.vel_command_b)
    self.command_term.vel_command_b[:, :] = cmd.unsqueeze(0)
    self.command_term.is_standing_env[:] = False
    return self.actor_policy(obs)


def _make_keyboard_driven(env_cfg, command_name: str) -> None:
  command_cfg = env_cfg.commands.get(command_name)
  if command_cfg is None:
    raise ValueError(f"Task does not expose command '{command_name}'.")
  required = (
    "heading_command",
    "rel_heading_envs",
    "rel_standing_envs",
    "resampling_time_range",
    "ranges",
  )
  if not all(hasattr(command_cfg, name) for name in required):
    raise ValueError(f"Command '{command_name}' is not a compatible velocity command.")

  command_cfg.heading_command = False
  command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_standing_envs = 0.0
  command_cfg.resampling_time_range = (1e9, 1e9)
  if getattr(command_cfg.ranges, "heading", None) is not None:
    command_cfg.ranges.heading = None


def run(cfg: Sim2SimKeyboardConfig) -> None:
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(cfg.task_id, play=True)
  agent_cfg = load_rl_cfg(cfg.task_id)
  env_cfg.scene.num_envs = max(1, int(cfg.num_envs))
  _make_keyboard_driven(env_cfg, cfg.command_name)

  checkpoint = (
    Path(cfg.checkpoint_file).expanduser()
    if cfg.checkpoint_file
    else _latest_checkpoint(Path(cfg.log_root), agent_cfg.experiment_name)
  )
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
  print(f"[INFO] Loading checkpoint: {checkpoint}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  actor_policy = runner.get_inference_policy(device=device)

  command_term = env.command_manager.get_term(cfg.command_name)
  if not all(hasattr(command_term, name) for name in ("vel_command_b", "is_standing_env")):
    raise RuntimeError(f"Command term '{cfg.command_name}' is not keyboard-compatible.")

  controller = KeyboardVelocityController(
    step_lin=cfg.cmd_step_lin,
    step_yaw=cfg.cmd_step_yaw,
    x_limit=cfg.lin_vel_x_limit,
    y_limit=cfg.lin_vel_y_limit,
    yaw_limit=cfg.ang_vel_z_limit,
  )
  controller.print_help()
  policy = KeyboardCommandPolicy(actor_policy, command_term, controller)

  viewer = NativeMujocoViewer(
    vec_env,
    policy,
    frame_rate=cfg.frame_rate,
    key_callback=controller.on_key,
  )
  viewer.run()
  vec_env.close()


def main() -> None:
  cfg = tyro.cli(Sim2SimKeyboardConfig)
  run(cfg)


if __name__ == "__main__":
  main()
