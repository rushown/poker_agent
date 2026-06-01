"""engine/push_fold.py — Nash-equilibrium push/fold ranges by position and stack depth.

Ranges derived from Nash solver output for 6-max NLHE.
Four depth tiers: ≤10bb, ≤15bb, ≤20bb, ≤25bb.
"""
from __future__ import annotations

from typing import Dict, Set

# ── 10 BB push ranges (frequency-correct from Nash solver) ───────────────────
_PUSH_10: Dict[str, Set[str]] = {
    "UTG": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "AKo", "AQo", "AJo", "ATo", "A9o",
        "KQs", "KJs", "KTs", "K9s",
        "KQo", "KJo", "KTo",
        "QJs", "QTs",
        "JTs", "T9s",
    },
    "HJ": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o",
        "KQs", "KJs", "KTs", "K9s", "K8s",
        "KQo", "KJo", "KTo",
        "QJs", "QTs", "Q9s",
        "JTs", "J9s", "T9s", "98s",
    },
    "CO": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s",
        "KQo", "KJo", "KTo", "K9o",
        "QJs", "QTs", "Q9s", "Q8s",
        "JTs", "J9s", "J8s", "T9s", "T8s", "98s",
    },
    "BTN": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o", "A4o", "A3o", "A2o",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s",
        "KQo", "KJo", "KTo", "K9o", "K8o",
        "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s",
        "QJo", "QTo", "Q9o",
        "JTs", "J9s", "J8s", "J7s",
        "JTo", "J9o",
        "T9s", "T8s", "T7s", "98s", "97s", "87s", "76s", "65s",
        "T9o", "98o", "87o", "76o",
    },
    "SB": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o", "A4o", "A3o", "A2o",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s",
        "KQo", "KJo", "KTo", "K9o", "K8o", "K7o",
        "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s", "Q5s",
        "QJo", "QTo", "Q9o", "Q8o",
        "JTs", "J9s", "J8s", "J7s", "J6s",
        "JTo", "J9o", "J8o",
        "T9s", "T8s", "T7s", "T6s", "98s", "97s", "96s", "87s", "86s", "76s", "75s", "65s",
        "T9o", "98o", "87o",
    },
    "BB": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o",
        "KQs", "KJs", "KTs", "K9s",
        "KQo", "KJo", "KTo",
        "QJs", "QTs",
        "JTs",
    },
}

# ── 15 BB push ranges ─────────────────────────────────────────────────────────
_PUSH_15: Dict[str, Set[str]] = {
    "UTG": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s",
        "AKo", "AQo", "AJo", "ATo",
        "KQs", "KJs", "KTs",
        "KQo", "KJo",
        "QJs",
    },
    "HJ": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "AKo", "AQo", "AJo", "ATo", "A9o",
        "KQs", "KJs", "KTs", "K9s",
        "KQo", "KJo", "KTo",
        "QJs", "QTs", "Q9s",
        "JTs", "J9s", "T9s",
    },
    "CO": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o",
        "KQs", "KJs", "KTs", "K9s", "K8s",
        "KQo", "KJo", "KTo",
        "QJs", "QTs", "Q9s", "Q8s",
        "JTs", "J9s", "J8s", "T9s", "T8s", "98s",
    },
    "BTN": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s",
        "KQo", "KJo", "KTo", "K9o",
        "QJs", "QTs", "Q9s", "Q8s", "Q7s",
        "QJo", "QTo",
        "JTs", "J9s", "J8s", "J7s",
        "JTo",
        "T9s", "T8s", "T7s", "98s", "97s", "87s", "76s",
        "T9o", "98o",
    },
    "SB": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o", "A4o", "A3o", "A2o",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s",
        "KQo", "KJo", "KTo", "K9o", "K8o",
        "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s",
        "QJo", "QTo", "Q9o",
        "JTs", "J9s", "J8s", "J7s",
        "JTo", "J9o",
        "T9s", "T8s", "T7s", "98s", "97s", "96s", "87s", "86s", "76s", "65s",
        "T9o", "98o", "87o",
    },
    "BB": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s",
        "AKo", "AQo", "AJo", "ATo", "A9o",
        "KQs", "KJs", "KTs",
        "KQo", "KJo",
        "QJs", "QTs",
        "JTs",
    },
}

# ── 20 BB push ranges ─────────────────────────────────────────────────────────
_PUSH_20: Dict[str, Set[str]] = {
    "UTG": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77",
        "AKs", "AQs", "AJs", "ATs",
        "AKo", "AQo",
        "KQs",
    },
    "HJ": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55",
        "AKs", "AQs", "AJs", "ATs", "A9s",
        "AKo", "AQo", "AJo",
        "KQs", "KJs",
        "KQo",
        "QJs",
    },
    "CO": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s",
        "AKo", "AQo", "AJo", "ATo",
        "KQs", "KJs", "KTs", "K9s",
        "KQo", "KJo",
        "QJs", "QTs", "Q9s",
        "JTs", "J9s", "T9s",
    },
    "BTN": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s",
        "KQo", "KJo", "KTo",
        "QJs", "QTs", "Q9s", "Q8s",
        "QJo",
        "JTs", "J9s", "J8s",
        "T9s", "T8s", "98s", "87s",
        "T9o",
    },
    "SB": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s",
        "KQo", "KJo", "KTo", "K9o",
        "QJs", "QTs", "Q9s", "Q8s", "Q7s",
        "QJo", "QTo",
        "JTs", "J9s", "J8s", "J7s",
        "JTo",
        "T9s", "T8s", "T7s", "98s", "97s", "87s", "76s", "65s",
        "T9o", "98o",
    },
    "BB": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77",
        "AKs", "AQs", "AJs", "ATs", "A9s",
        "AKo", "AQo", "AJo", "ATo",
        "KQs", "KJs",
        "KQo",
        "QJs",
    },
}

# ── 25 BB push ranges ─────────────────────────────────────────────────────────
_PUSH_25: Dict[str, Set[str]] = {
    "UTG": {"AA", "KK", "QQ", "JJ", "AKs", "AKo"},
    "HJ": {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AQs", "AKo"},
    "CO": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88",
        "AKs", "AQs", "AJs", "ATs",
        "AKo", "AQo", "AJo",
        "KQs",
    },
    "BTN": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A5s",
        "AKo", "AQo", "AJo", "ATo", "A9o",
        "KQs", "KJs", "KTs", "K9s",
        "KQo", "KJo",
        "QJs", "QTs",
        "JTs", "T9s",
    },
    "SB": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s",
        "KQo", "KJo", "KTo", "K9o",
        "QJs", "QTs", "Q9s", "Q8s",
        "QJo",
        "JTs", "J9s", "J8s",
        "T9s", "T8s", "98s", "87s", "76s",
        "T9o",
    },
    "BB": {
        "AA", "KK", "QQ", "JJ", "TT", "99",
        "AKs", "AQs", "AJs",
        "AKo", "AQo",
        "KQs",
    },
}

# ── Nash call ranges vs an all-in shove ──────────────────────────────────────
# Keyed by (bb_depth_tier, position) — conservative (ICM-adjusted)
_CALL_VS_PUSH: Dict[str, Set[str]] = {
    "10": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s",
        "AKo", "AQo", "AJo", "ATo", "A9o",
        "KQs", "KJs", "KTs",
        "KQo", "KJo",
        "QJs", "QTs",
        "JTs",
    },
    "15": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77",
        "AKs", "AQs", "AJs", "ATs", "A9s",
        "AKo", "AQo", "AJo", "ATo",
        "KQs", "KJs",
        "KQo",
        "QJs",
    },
    "20": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88",
        "AKs", "AQs", "AJs", "ATs",
        "AKo", "AQo", "AJo",
        "KQs",
        "KQo",
    },
    "25": {
        "AA", "KK", "QQ", "JJ",
        "AKs", "AQs",
        "AKo",
    },
}

_POSITIONS = {"UTG", "HJ", "MP", "CO", "BTN", "SB", "BB"}


def _normalize_pos(position: str) -> str:
    p = position.upper()
    if p not in _POSITIONS:
        return "HJ"
    if p == "MP":
        return "HJ"
    return p


def _push_table(bb_depth: float) -> Dict[str, Set[str]]:
    if bb_depth <= 10:
        return _PUSH_10
    if bb_depth <= 15:
        return _PUSH_15
    if bb_depth <= 20:
        return _PUSH_20
    return _PUSH_25


def should_push(
    notation: str,
    position: str,
    bb_depth: float,
    facing_raise: bool,
) -> bool:
    """True if hand is in Nash push range for this position and stack depth."""
    pos = _normalize_pos(position)
    table = _push_table(bb_depth)
    push_set = table.get(pos, set())

    if notation in push_set:
        # Facing a raise when deep: only jam premiums
        if facing_raise and bb_depth > 15:
            return notation in {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
        return True
    return False


def should_call_push(
    notation: str,
    bb_depth: float,
    bubble_factor: float = 1.0,
) -> bool:
    """True if hand clears Nash call-vs-shove threshold, ICM-adjusted."""
    if bubble_factor > 1.5:
        # Near bubble: only call with clear value
        return notation in {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}

    if bubble_factor > 1.2:
        # Moderate ICM pressure — tighten by one tier
        tier = "25" if bb_depth > 20 else ("20" if bb_depth > 15 else "15")
    else:
        if bb_depth <= 10:
            tier = "10"
        elif bb_depth <= 15:
            tier = "15"
        elif bb_depth <= 20:
            tier = "20"
        else:
            tier = "25"

    return notation in _CALL_VS_PUSH.get(tier, set())
