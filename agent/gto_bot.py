"""agent/gto_bot.py — GTO baseline with adaptive frequency tuning."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from agent.preflop_ranges import (
    four_bet_size,
    open_raise_size_bb,
    should_3bet,
    should_4bet,
    should_open,
    three_bet_size,
)
from agent.techniques import (
    bluff_allowed,
    effective_call_equity,
    should_defend_vs_bet,
    value_bet_fraction,
)
from engine.hand_eval import card_rank, RANKS
from models.adaptive_memory import StrategyTuning

PREMIUM_HANDS = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
STRONG_HANDS = {"TT", "99", "AQs", "AQo", "AJs", "KQs", "ATs"}
POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"]

_rng = random.Random()
_DEFAULT_TUNING = StrategyTuning()


def hand_notation(hole: List[str]) -> str:
    if len(hole) < 2:
        return "??"
    r1 = card_rank(hole[0])
    r2 = card_rank(hole[1])
    s1 = hole[0][1].lower()
    s2 = hole[1][1].lower()
    high = RANKS[max(r1, r2)]
    low = RANKS[min(r1, r2)]
    if r1 == r2:
        return high + high
    suited = "s" if s1 == s2 else "o"
    return high + low + suited


def preflop_action(
    hole: List[str],
    position: str,
    is_facing_raise: bool,
    facing_raise_size: float,
    bb_size: float,
    stack: float,
    pot: float,
    allowed_actions: List[Dict],
    fold_to_3bet_avg: float = 0.5,
    tuning: Optional[StrategyTuning] = None,
) -> Tuple[str, float, str]:
    t = tuning or _DEFAULT_TUNING
    notation = hand_notation(hole)
    pos = position.upper() if position.upper() in POSITIONS else "MP"

    if is_facing_raise:
        raise_count = 2 if facing_raise_size > bb_size * 8 else 1

        if raise_count >= 2 and should_4bet(notation, _rng):
            amt = four_bet_size(facing_raise_size, bb_size, stack)
            return "raise", amt, f"4-bet {notation} from {pos}"

        if should_3bet(notation, pos, _rng, t) or notation in PREMIUM_HANDS:
            amt = three_bet_size(pos, facing_raise_size, bb_size, stack)
            if fold_to_3bet_avg > 0.65 and notation in STRONG_HANDS | PREMIUM_HANDS:
                amt = min(stack, amt * 0.9)
            return "raise", amt, f"3-bet {notation} from {pos}"

        if should_open(notation, pos, _rng, t) or notation in STRONG_HANDS | PREMIUM_HANDS:
            return "call", _call_to(allowed_actions), f"Defend {notation} vs raise"

        return "fold", 0, f"Fold {notation} vs raise from {pos}"

    if should_open(notation, pos, _rng, t):
        amt = min(stack, open_raise_size_bb(pos, bb_size))
        return "raise", amt, f"Open {notation} from {pos}"

    if pos == "BB" and _has_check(allowed_actions):
        return "check", 0, f"Check BB with {notation}"

    return "fold", 0, f"Fold {notation} from {pos}"


def postflop_action(
    ehs: float,
    pot: float,
    call_amount: float,
    stack: float,
    street: str,
    is_in_position: bool,
    allowed_actions: List[Dict],
    board_texture: str = "dry",
    fold_to_cbet: float = 0.45,
    bet_size_mult: float = 0.5,
    tuning: Optional[StrategyTuning] = None,
) -> Tuple[str, float, str]:
    t = tuning or _DEFAULT_TUNING
    can_check = _has_check(allowed_actions)
    min_bet, max_bet = _bet_range(allowed_actions)
    call_to = _call_to(allowed_actions)

    required_equity = effective_call_equity(call_amount, pot, t) if call_amount > 0 else 0.0
    has_bet_to_call = call_amount > 0 and not can_check
    val_frac = value_bet_fraction(ehs, t)
    cbet_mult = bet_size_mult * t.cbet_frequency

    if not has_bet_to_call:
        if ehs >= 0.70:
            bet = min(stack, pot * val_frac)
            bet = _clamp_bet(bet, min_bet, max_bet)
            return "raise", bet, f"Value {ehs:.0%} ({val_frac:.0%} pot)"
        if ehs >= 0.52 and is_in_position:
            bet = min(stack, pot * cbet_mult)
            bet = _clamp_bet(bet, min_bet, max_bet)
            return "raise", bet, f"Thin value {ehs:.0%}"
        if (
            is_in_position
            and board_texture == "dry"
            and bluff_allowed(ehs, street, fold_to_cbet, t)
        ):
            bet = min(stack, pot * 0.33 * t.bluff_frequency)
            bet = _clamp_bet(bet, min_bet, max_bet)
            return "raise", bet, f"Bluff {ehs:.0%} (FtCB={fold_to_cbet:.0%})"
        if ehs >= 0.30 and is_in_position and fold_to_cbet > 0.52:
            bet = min(stack, pot * cbet_mult * 0.9)
            bet = _clamp_bet(bet, min_bet, max_bet)
            return "raise", bet, f"Cbet {ehs:.0%}"
        return "check", 0, f"Check {ehs:.0%}"

    if should_defend_vs_bet(ehs, call_amount, pot, t):
        if ehs >= required_equity + 0.12 and ehs >= 0.78:
            raise_to = min(stack, max(min_bet, call_to + call_amount * 2.0))
            raise_to = _clamp_bet(raise_to, min_bet, max_bet)
            return "raise", raise_to, f"Value raise {ehs:.0%}"
        return "call", call_to, f"Defend {ehs:.0%} (req {required_equity:.0%}, MDF)"

    return "fold", 0, f"Fold {ehs:.0%} < {required_equity:.0%}"


def _has_check(allowed_actions: List[Dict]) -> bool:
    return any(a.get("action") == "check" for a in allowed_actions)


def _bet_range(allowed_actions: List[Dict]) -> Tuple[float, float]:
    for a in allowed_actions:
        if a.get("action") in ("bet", "raise"):
            return float(a.get("minAmount", 0)), float(a.get("maxAmount", 9999999))
    return 0.0, 0.0


def _call_to(allowed_actions: List[Dict]) -> float:
    for a in allowed_actions:
        if a.get("action") == "call":
            if a.get("toAmount") is not None:
                return float(a["toAmount"])
            return float(a.get("amount") or a.get("callAmount") or 0)
    return 0.0


def _clamp_bet(bet: float, min_bet: float, max_bet: float) -> float:
    if min_bet:
        return max(min_bet, min(max_bet, bet))
    return bet
