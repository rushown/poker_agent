"""engine/push_fold.py — short-stack push/fold charts (Nash-inspired)."""
from __future__ import annotations

from typing import Dict, Set

# Hands to shove by position at ≤15 BB (notation sets)
_PUSH_15: Dict[str, Set[str]] = {
    "UTG": {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs"},
    "MP": {"AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AKo", "AQs", "AQo", "AJs", "KQs"},
    "CO": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "AKs", "AKo", "AQs", "AQo",
        "AJs", "ATs", "KQs", "KQo", "QJs",
    },
    "BTN": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "AKs", "AKo",
        "AQs", "AQo", "AJs", "AJo", "ATs", "KQs", "KQo", "KJs", "QJs", "JTs",
        "A5s", "A4s",
    },
    "SB": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "AKs", "AKo",
        "AQs", "AQo", "AJs", "AJo", "ATs", "A9s", "KQs", "KQo", "KJs", "QJs",
        "JTs", "T9s", "A5s", "A4s", "A3s", "A2s",
    },
    "BB": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "AKs", "AKo", "AQs",
        "AQo", "AJs", "ATs", "KQs",
    },
}

# Tighter at ≤8 BB
_PUSH_8: Dict[str, Set[str]] = {
    "UTG": {"AA", "KK", "QQ", "JJ", "AKs", "AKo"},
    "MP": {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs"},
    "CO": {"AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AKo", "AQs", "AQo", "AJs"},
    "BTN": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "AKs", "AKo", "AQs", "AQo",
        "AJs", "ATs", "KQs",
    },
    "SB": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "AKs", "AKo", "AQs",
        "AQo", "AJs", "AJo", "ATs", "KQs", "KJs",
    },
    "BB": {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs"},
}


def should_push(
    notation: str,
    position: str,
    bb_depth: float,
    facing_raise: bool,
) -> bool:
    """True if hand is in push range for stack depth."""
    pos = position.upper() if position.upper() in _PUSH_15 else "MP"
    if bb_depth <= 8:
        base = _PUSH_8.get(pos, set())
    else:
        base = _PUSH_15.get(pos, set())

    if notation in base:
        return True

    # Facing raise — slightly tighter unless very short
    if facing_raise and bb_depth > 10:
        premium = {"AA", "KK", "QQ", "AKs", "AKo"}
        return notation in premium
    return False


def should_call_push(
    notation: str,
    bb_depth: float,
    bubble_factor: float = 1.0,
) -> bool:
    """Call an all-in at short stacks."""
    if bubble_factor > 1.5:
        return notation in {"AA", "KK", "QQ", "AKs", "AKo"}
    if bb_depth <= 8:
        return notation in _PUSH_8.get("BB", set()) | {"AQo", "JJ"}
    return notation in _PUSH_15.get("BB", set()) | {"TT", "AQo"}
