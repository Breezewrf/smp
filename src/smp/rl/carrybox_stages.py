"""Shared carry-box stage identifiers.

Keep this outside the task package so generic RL events can use the constants
without importing task registration code.
"""

from __future__ import annotations

from typing import Literal

CarryBoxStage = Literal["pickup", "carry", "place"]

STAGE_PICKUP = 0
STAGE_CARRY = 1
STAGE_PLACE = 2

CARRYBOX_STAGE_NAMES: tuple[CarryBoxStage, ...] = ("pickup", "carry", "place")
CARRYBOX_STAGE_IDS: dict[CarryBoxStage, int] = {
  "pickup": STAGE_PICKUP,
  "carry": STAGE_CARRY,
  "place": STAGE_PLACE,
}

