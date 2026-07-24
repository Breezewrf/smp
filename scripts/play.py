"""Play wrapper with support for non-resetting task visualization."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts import play as mjlab_play

import smp.rl.tasks  # noqa: F401  # registers Smp-* tasks in the mjlab registry


class _ViewerPlayEnv(ManagerBasedRlEnv):
  """Allow viewers to keep stepping terminal states when auto-reset is disabled."""

  def step(self, action: torch.Tensor):
    if not self.cfg.auto_reset:
      self._manual_reset_pending.zero_()
    return super().step(action)


def main() -> None:
  mjlab_play.ManagerBasedRlEnv = _ViewerPlayEnv
  mjlab_play.main()


if __name__ == "__main__":
  main()
