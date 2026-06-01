"""agent/chat_tilt.py — varied table talk for humans / bot confusion."""
from __future__ import annotations

import random
from typing import Optional

_rng = random.Random()

_WIN_LINES = [
    "read that from your bet timing",
    "pot committed — yours now",
    "standard line for this board",
    "you telegraphed that one",
    "math doesn't lie",
]

_BLUFF_LINES = [
    "thanks for the donation",
    "fold equity is a skill",
    "that's what the sizing asked for",
    "you made that easy",
]

_BUBBLE_LINES = [
    "I'm not folding regardless — pick your spot carefully",
    "ICM is on my side here",
    "pressure's real at this stack depth",
]

_VALUE_LINES = [
    "getting paid",
    "thin value — you called worse",
    "range advantage street by street",
]


def mistake_line() -> str:
    return _rng.choice(
        [
            "miscalculated that one, won't happen again",
            "note taken — adjusting",
            "that line was too thin",
        ]
    )


def build_chat_message(
    *,
    action: str,
    won_big_pot: bool = False,
    was_bluff: bool = False,
    bubble_pressure: bool = False,
    ehs: float = 0.5,
    strategy_mode: str = "",
) -> str:
    """Return short reasoning + optional tilt line (max ~200 chars for API)."""
    parts = []
    if strategy_mode:
        parts.append(f"mode={strategy_mode}")
    if won_big_pot and _rng.random() < 0.4:
        parts.append(_rng.choice(_WIN_LINES))
    elif was_bluff and action in ("raise", "bet") and ehs < 0.4 and _rng.random() < 0.35:
        parts.append(_rng.choice(_BLUFF_LINES))
    elif bubble_pressure and _rng.random() < 0.3:
        parts.append(_rng.choice(_BUBBLE_LINES))
    elif ehs >= 0.7 and action in ("raise", "bet") and _rng.random() < 0.25:
        parts.append(_rng.choice(_VALUE_LINES))
    return " | ".join(parts) if parts else ""
