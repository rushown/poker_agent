"""agent/preflop_ranges.py — solver-calibrated preflop ranges for 6-max NLHE.

Ranges reflect GTO Wizard / PioSOLVER findings at 100bb effective.
Frequencies are open probabilities (1.0 = always, 0.0 = never/fold).
"""
from __future__ import annotations

import random
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.adaptive_memory import StrategyTuning

# ── Open-raise frequencies by position ───────────────────────────────────────
# UTG: ~15% range (tight, value-heavy but includes speculative suited hands)
_UTG_OPEN: Dict[str, float] = {
    # Pairs
    "AA": 1.0, "KK": 1.0, "QQ": 1.0, "JJ": 1.0, "TT": 1.0, "99": 1.0,
    "88": 1.0, "77": 1.0, "66": 0.9, "55": 0.7, "44": 0.5, "33": 0.3, "22": 0.2,
    # Suited aces
    "AKs": 1.0, "AQs": 1.0, "AJs": 1.0, "ATs": 1.0,
    "A9s": 0.8, "A8s": 0.7, "A7s": 0.6, "A6s": 0.5, "A5s": 0.6, "A4s": 0.5, "A3s": 0.4, "A2s": 0.3,
    # Offsuit aces
    "AKo": 1.0, "AQo": 1.0, "AJo": 0.7, "ATo": 0.3,
    # Suited kings
    "KQs": 1.0, "KJs": 1.0, "KTs": 0.9, "K9s": 0.5, "K8s": 0.3,
    # Offsuit kings
    "KQo": 0.8, "KJo": 0.5, "KTo": 0.2,
    # Suited queens
    "QJs": 0.8, "QTs": 0.6, "Q9s": 0.3,
    # Offsuit queens
    "QJo": 0.2,
    # Suited connectors
    "JTs": 0.7, "T9s": 0.5, "98s": 0.4, "87s": 0.3, "76s": 0.2,
    "J9s": 0.3,
}

# MP: ~19% range
_MP_OPEN: Dict[str, float] = {
    **_UTG_OPEN,
    "33": 0.5, "22": 0.4,
    "A9s": 0.9, "A8s": 0.8, "A7s": 0.7, "A6s": 0.6, "A5s": 0.7, "A4s": 0.6, "A3s": 0.5, "A2s": 0.4,
    "AJo": 0.8, "ATo": 0.5, "A9o": 0.2,
    "K9s": 0.7, "K8s": 0.5, "K7s": 0.3,
    "KJo": 0.6, "KTo": 0.4,
    "QTs": 0.7, "Q9s": 0.4, "Q8s": 0.2,
    "QJo": 0.3, "QTo": 0.1,
    "JTs": 0.8, "J9s": 0.5, "T9s": 0.6, "T8s": 0.3, "98s": 0.5, "87s": 0.4, "76s": 0.3, "65s": 0.2,
}

# CO: ~27% range
_CO_OPEN: Dict[str, float] = {
    **_MP_OPEN,
    "44": 0.8, "33": 0.7, "22": 0.6,
    "A9s": 1.0, "A8s": 0.9, "A7s": 0.8, "A6s": 0.7, "A5s": 0.9, "A4s": 0.8, "A3s": 0.7, "A2s": 0.6,
    "ATo": 0.7, "A9o": 0.5, "A8o": 0.3,
    "K8s": 0.6, "K7s": 0.5, "K6s": 0.4, "K5s": 0.3,
    "KTo": 0.6, "K9o": 0.5,
    "Q9s": 0.6, "Q8s": 0.4, "Q7s": 0.2,
    "QJo": 0.5, "QTo": 0.3, "Q9o": 0.1,
    "JTs": 0.9, "J9s": 0.7, "J8s": 0.5, "J7s": 0.2,
    "JTo": 0.3,
    "T9s": 0.8, "T8s": 0.6, "T7s": 0.3,
    "98s": 0.7, "97s": 0.4, "87s": 0.6, "86s": 0.3, "76s": 0.5, "75s": 0.2, "65s": 0.4, "54s": 0.2,
    "T9o": 0.2,
}

# BTN: ~45% range — widest opens, max positional advantage
_BTN_OPEN: Dict[str, float] = {
    **_CO_OPEN,
    "44": 0.9, "33": 0.8, "22": 0.7,
    "A9o": 0.7, "A8o": 0.5, "A7o": 0.4, "A6o": 0.3, "A5o": 0.4, "A4o": 0.3, "A3o": 0.2, "A2o": 0.2,
    "K6s": 0.6, "K5s": 0.5, "K4s": 0.3, "K3s": 0.2, "K2s": 0.1,
    "K9o": 0.6, "K8o": 0.4, "K7o": 0.25,
    "Q8s": 0.6, "Q7s": 0.4, "Q6s": 0.3, "Q5s": 0.2,
    "QJo": 0.7, "QTo": 0.5, "Q9o": 0.35, "Q8o": 0.15,
    "J8s": 0.6, "J7s": 0.45, "J6s": 0.2,
    "JTo": 0.55, "J9o": 0.35,
    "T8s": 0.7, "T7s": 0.55, "T6s": 0.3,
    "T9o": 0.45, "T8o": 0.25,
    "97s": 0.6, "96s": 0.3, "86s": 0.5, "85s": 0.2, "75s": 0.4, "74s": 0.1, "64s": 0.2, "53s": 0.2, "43s": 0.1,
    "98o": 0.2, "87o": 0.1, "76o": 0.1, "65o": 0.1,
}

# SB: ~38% (tighter than BTN — OOP vs BB)
_SB_OPEN: Dict[str, float] = {
    **_CO_OPEN,
    "44": 0.9, "33": 0.8, "22": 0.7,
    "A9o": 0.6, "A8o": 0.4, "A7o": 0.3, "A6o": 0.2, "A5o": 0.3, "A4o": 0.2,
    "K7s": 0.6, "K6s": 0.5, "K5s": 0.4, "K4s": 0.2,
    "K9o": 0.5, "K8o": 0.3, "K7o": 0.2,
    "Q8s": 0.5, "Q7s": 0.3,
    "QJo": 0.6, "QTo": 0.4, "Q9o": 0.2,
    "J8s": 0.5, "J7s": 0.3,
    "JTo": 0.4, "J9o": 0.2,
    "T8s": 0.6, "T7s": 0.4,
    "T9o": 0.3,
    "97s": 0.5, "86s": 0.4, "75s": 0.3, "64s": 0.2, "53s": 0.2,
    "98o": 0.2,
}

# BB: defense frequencies vs single raise (not opens — BB calls/3bets)
_BB_DEFEND: Dict[str, float] = {
    "AA": 1.0, "KK": 1.0, "QQ": 1.0, "JJ": 1.0, "TT": 1.0,
    "99": 1.0, "88": 0.95, "77": 0.9, "66": 0.85, "55": 0.75, "44": 0.65, "33": 0.55, "22": 0.45,
    "AKs": 1.0, "AQs": 1.0, "AJs": 1.0, "ATs": 0.95, "A9s": 0.9, "A8s": 0.85,
    "A7s": 0.8, "A6s": 0.75, "A5s": 0.8, "A4s": 0.75, "A3s": 0.7, "A2s": 0.65,
    "AKo": 1.0, "AQo": 0.95, "AJo": 0.85, "ATo": 0.75, "A9o": 0.65, "A8o": 0.55,
    "KQs": 0.95, "KJs": 0.9, "KTs": 0.85, "K9s": 0.75, "K8s": 0.65, "K7s": 0.55,
    "KQo": 0.85, "KJo": 0.75, "KTo": 0.60, "K9o": 0.45,
    "QJs": 0.9, "QTs": 0.85, "Q9s": 0.75, "Q8s": 0.65, "Q7s": 0.5,
    "QJo": 0.7, "QTo": 0.55, "Q9o": 0.4,
    "JTs": 0.85, "J9s": 0.8, "J8s": 0.7, "J7s": 0.55,
    "JTo": 0.6, "J9o": 0.45,
    "T9s": 0.8, "T8s": 0.75, "T7s": 0.6,
    "T9o": 0.5, "T8o": 0.35,
    "98s": 0.75, "97s": 0.65, "96s": 0.5,
    "87s": 0.7, "86s": 0.55, "76s": 0.65, "75s": 0.5, "65s": 0.6, "54s": 0.5, "43s": 0.35,
}

OPEN_RANGES: Dict[str, Dict[str, float]] = {
    "UTG": _UTG_OPEN,
    "MP": _MP_OPEN,
    "CO": _CO_OPEN,
    "BTN": _BTN_OPEN,
    "SB": _SB_OPEN,
    "BB": _BB_DEFEND,
}

# ── 3-bet ranges (value + bluff) ─────────────────────────────────────────────
# Bluff 3-bets are polarized: strong blockers (A5s, A4s, A3s) + suited connectors
_THREE_BET_VALUE: Dict[str, float] = {
    "AA": 1.0, "KK": 1.0, "QQ": 1.0, "JJ": 0.9, "TT": 0.6, "99": 0.3,
    "AKs": 1.0, "AKo": 0.9, "AQs": 0.75, "AQo": 0.4, "AJs": 0.5, "KQs": 0.4,
}

_THREE_BET_BLUFF: Dict[str, float] = {
    # Suited aces: block top-pair calls, have equity vs calls
    "A5s": 0.55, "A4s": 0.5, "A3s": 0.4, "A2s": 0.3,
    # Suited connectors: can flop strong draws
    "K9s": 0.3, "K8s": 0.2, "Q9s": 0.2, "J9s": 0.2, "T8s": 0.15,
    "98s": 0.2, "87s": 0.15, "76s": 0.15, "65s": 0.1,
}

# Position multipliers for 3-bet bluffing (more liberal IP)
_THREE_BET_POS_MULT: Dict[str, float] = {
    "UTG": 0.6, "MP": 0.7, "CO": 0.85, "BTN": 1.0, "SB": 0.75, "BB": 0.9,
}

# 4-bet range (always value — we 4bet for value, defend 3bets with calls OOP)
_FOUR_BET: Dict[str, float] = {
    "AA": 1.0, "KK": 1.0, "QQ": 0.85, "JJ": 0.5,
    "AKs": 0.95, "AKo": 0.8,
    "A5s": 0.3, "A4s": 0.25,  # bluff 4bets with nut-flush blockers
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
    r = rng or random
    pos = position.upper() if position.upper() in _THREE_BET_POS_MULT else "MP"
    pos_mult = _THREE_BET_POS_MULT.get(pos, 0.8)

    # Value 3bets
    val_freq = _THREE_BET_VALUE.get(notation, 0.0)
    if tuning is not None:
        val_freq = min(1.0, val_freq * tuning.preflop_aggression)
    if val_freq > 0.0 and r.random() < val_freq:
        return True

    # Bluff 3bets (position-dependent)
    bluff_freq = _THREE_BET_BLUFF.get(notation, 0.0) * pos_mult
    if tuning is not None:
        bluff_freq = min(1.0, bluff_freq * tuning.bluff_frequency)
    if bluff_freq > 0.0 and r.random() < bluff_freq:
        return True

    return False


def should_4bet(notation: str, rng: Optional[random.Random] = None) -> bool:
    freq = _FOUR_BET.get(notation, 0.0)
    if freq <= 0.0:
        return False
    if freq >= 1.0:
        return True
    r = rng or random
    return r.random() < freq


def open_raise_size_bb(position: str, bb_size: float) -> float:
    """Solver-standard open sizes. BTN/SB slightly smaller (2.2-2.3x), others 2.5x."""
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
    """Positional 3-bet sizing: IP smaller (3x), OOP larger (3.5x), min 9bb."""
    pos = position.upper()
    if pos == "SB":
        # OOP, linear range — size up
        target = max(facing_raise_size * 3.5, bb_size * 12)
    elif pos == "BB":
        target = max(facing_raise_size * 3.2, bb_size * 11)
    elif pos == "BTN":
        # IP, can size down slightly
        target = max(facing_raise_size * 3.0, bb_size * 9)
    elif pos in ("CO", "MP"):
        target = max(facing_raise_size * 2.9, bb_size * 10)
    else:  # UTG 3-betting is rare, size for protection
        target = max(facing_raise_size * 3.0, bb_size * 11)
    return min(stack, target)


def four_bet_size(facing_raise_size: float, bb_size: float, stack: float) -> float:
    """Standard 4bet: ~2.5x the 3bet, minimum 22bb."""
    return min(stack, max(facing_raise_size * 2.5, bb_size * 22))


def squeeze_size(
    open_size: float,
    callers: int,
    bb_size: float,
    stack: float,
) -> float:
    """Squeeze sizing: open + callers * dead money + squeeze premium."""
    dead_money = callers * open_size
    target = open_size * 3.5 + dead_money * 0.5
    target = max(target, bb_size * (12 + callers * 3))
    return min(stack, target)
