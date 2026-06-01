"""agent/preflop_ranges.py — frequency-based open and 3-bet ranges."""
from __future__ import annotations

import random
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.adaptive_memory import StrategyTuning

# Open-raise frequencies by position (0.0 = always fold, 1.0 = always raise)
# UTG: 88+, AJo+, ATs+, KQo+, KJs+ (tight value-heavy)
_UTG_OPEN: Dict[str, float] = {
    "AA": 1.0, "KK": 1.0, "QQ": 1.0, "JJ": 1.0, "TT": 1.0,
    "99": 1.0, "88": 1.0,
    "AKs": 1.0, "AQs": 1.0, "AJs": 1.0, "ATs": 1.0, "KQs": 1.0, "KJs": 1.0,
    "AKo": 1.0, "AQo": 1.0, "AJo": 1.0, "KQo": 1.0,
    "T9s": 0.0, "QJs": 0.0,
}

_MP_OPEN: Dict[str, float] = {
    **_UTG_OPEN,
    "99": 1.0, "88": 0.85, "77": 0.6, "AJo": 0.5, "ATo": 0.0,
    "KQo": 0.55, "QJs": 0.7, "JTs": 0.65, "A9s": 0.75,
}

_CO_OPEN: Dict[str, float] = {
    **_MP_OPEN,
    "77": 0.9, "66": 0.75, "55": 0.55, "ATo": 0.45, "KJo": 0.5,
    "QJo": 0.35, "T9s": 0.8, "98s": 0.7, "87s": 0.65, "A8s": 0.7,
    "A5s": 0.55, "A4s": 0.45,
}

_BTN_OPEN: Dict[str, float] = {
    **_CO_OPEN,
    "44": 0.7, "33": 0.6, "22": 0.5, "A7s": 0.75, "A6s": 0.7,
    "A3s": 0.55, "A2s": 0.5, "KTo": 0.55, "QTo": 0.45, "JTo": 0.5,
    "T9o": 0.35, "98o": 0.3, "76s": 0.75, "65s": 0.7, "54s": 0.65,
    "A9o": 0.4,
}

_SB_OPEN: Dict[str, float] = {
    **_CO_OPEN,
    "A9o": 0.55, "KTo": 0.65, "QTo": 0.5,
}

_BB_DEFEND: Dict[str, float] = {
    "AA": 1.0, "KK": 1.0, "QQ": 1.0, "JJ": 1.0, "TT": 1.0,
    "99": 0.95, "88": 0.9, "77": 0.85, "66": 0.75, "55": 0.65,
    "AKs": 1.0, "AQs": 1.0, "AJs": 0.95, "ATs": 0.9, "KQs": 0.9,
    "AKo": 1.0, "AQo": 0.95, "AJo": 0.85, "KQo": 0.8, "QJs": 0.85,
    "JTs": 0.8, "T9s": 0.75, "98s": 0.7, "87s": 0.65, "76s": 0.6,
    "A5s": 0.7, "A4s": 0.65, "A3s": 0.6, "A2s": 0.55,
}

OPEN_RANGES: Dict[str, Dict[str, float]] = {
    "UTG": _UTG_OPEN,
    "MP": _MP_OPEN,
    "CO": _CO_OPEN,
    "BTN": _BTN_OPEN,
    "SB": _SB_OPEN,
    "BB": _BB_DEFEND,
}

# 3-bet frequencies when facing a raise (simplified)
_THREE_BET: Dict[str, float] = {
    "AA": 1.0, "KK": 1.0, "QQ": 1.0, "JJ": 0.9, "TT": 0.55,
    "AKs": 1.0, "AKo": 0.85, "AQs": 0.75, "AQo": 0.4, "AJs": 0.45,
    "KQs": 0.35, "99": 0.4, "88": 0.25,
    "A5s": 0.35, "A4s": 0.3,  # bluff 3-bets IP
}

# 4-bet value / bluff
_FOUR_BET: Dict[str, float] = {
    "AA": 1.0, "KK": 1.0, "QQ": 0.85, "AKs": 0.9, "AKo": 0.7,
}


def open_frequency(notation: str, position: str) -> float:
    pos = position.upper() if position.upper() in OPEN_RANGES else "MP"
    return OPEN_RANGES.get(pos, {}).get(notation, 0.0)


def should_open(
    notation: str,
    position: str,
    rng: Optional[random.Random] = None,
    tuning: Optional["StrategyTuning"] = None,
) -> bool:
    from agent.techniques import scale_open_frequency

    freq = open_frequency(notation, position)
    if tuning is not None:
        freq = scale_open_frequency(freq, tuning, position)
    if freq <= 0.0:
        return False
    if freq >= 1.0:
        return True
    r = rng or random
    return r.random() < freq


def should_3bet(
    notation: str,
    position: str,
    rng: Optional[random.Random] = None,
    tuning: Optional["StrategyTuning"] = None,
) -> bool:
    freq = _THREE_BET.get(notation, 0.0)
    if position in ("BTN", "CO", "SB") and notation in ("A5s", "A4s", "A3s"):
        freq = min(1.0, freq + 0.15)
    if tuning is not None:
        freq = min(1.0, freq * tuning.preflop_aggression)
    if freq <= 0.0:
        return False
    if freq >= 1.0:
        return True
    r = rng or random
    return r.random() < freq


def should_4bet(notation: str, rng: Optional[random.Random] = None) -> bool:
    freq = _FOUR_BET.get(notation, 0.0)
    if freq <= 0.0:
        return False
    if freq >= 1.0:
        return True
    r = rng or random
    return r.random() < freq


def open_raise_size_bb(position: str, bb_size: float) -> float:
    """Target open size in chips (TO-amount preflop ≈ total bet)."""
    mult = {
        "UTG": 2.5, "MP": 2.5, "CO": 2.3, "BTN": 2.2, "SB": 2.5, "BB": 2.5,
    }.get(position.upper(), 2.5)
    return bb_size * mult


def three_bet_size(
    position: str,
    facing_raise_size: float,
    bb_size: float,
    stack: float,
) -> float:
    """Positional 3-bet TO-amount."""
    pos = position.upper()
    if pos == "SB":
        target = max(facing_raise_size * 3.5, bb_size * 11)
    elif pos == "BTN":
        target = max(facing_raise_size * 3.0, bb_size * 9)
    elif pos in ("CO", "MP"):
        target = max(facing_raise_size * 2.9, bb_size * 9.5)
    else:
        target = max(facing_raise_size * 3.0, bb_size * 10)
    return min(stack, target)


def four_bet_size(facing_raise_size: float, bb_size: float, stack: float) -> float:
    return min(stack, max(facing_raise_size * 2.5, bb_size * 22))
