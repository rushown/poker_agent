"""agent/gto_bot.py — solver-calibrated GTO baseline with board-texture aware sizing."""
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
    bluff_fraction,
    effective_call_equity,
    minimum_defense_frequency,
    should_defend_vs_bet,
    should_overbet,
    value_bet_fraction,
)
from engine.hand_eval import card_rank, RANKS
from models.adaptive_memory import StrategyTuning

PREMIUM_HANDS = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
STRONG_HANDS = {"TT", "99", "AQs", "AQo", "AJs", "KQs", "ATs"}
POSITIONS = ["UTG", "MP", "HJ", "CO", "BTN", "SB", "BB"]

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
        # Detect if we're facing a 3bet (raise size is large relative to bb)
        is_facing_3bet = facing_raise_size > bb_size * 7

        if is_facing_3bet and should_4bet(notation, _rng):
            amt = four_bet_size(facing_raise_size, bb_size, stack)
            return "raise", amt, f"4-bet {notation} from {pos}"

        if should_3bet(notation, pos, _rng, t) or notation in PREMIUM_HANDS:
            amt = three_bet_size(pos, facing_raise_size, bb_size, stack)
            # Size down slightly vs tight opponents who fold more
            if fold_to_3bet_avg > 0.62 and notation in STRONG_HANDS | PREMIUM_HANDS:
                amt = min(stack, amt * 0.88)
            return "raise", amt, f"3-bet {notation} from {pos}"

        # Cold call: medium strength hands with good implied odds
        if _should_cold_call(notation, pos, facing_raise_size, bb_size, stack):
            return "call", _call_to(allowed_actions), f"Call {notation} vs raise"

        return "fold", 0, f"Fold {notation} vs raise from {pos}"

    if should_open(notation, pos, _rng, t):
        amt = min(stack, open_raise_size_bb(pos, bb_size))
        return "raise", amt, f"Open {notation} from {pos}"

    if pos == "BB" and _has_check(allowed_actions):
        return "check", 0, f"Check BB with {notation}"

    return "fold", 0, f"Fold {notation} from {pos}"


def _should_cold_call(
    notation: str,
    position: str,
    facing_raise_size: float,
    bb_size: float,
    stack: float,
) -> bool:
    """Cold-call when hand has implied odds but isn't strong enough to 3-bet."""
    bb_depth = stack / max(1, bb_size)
    if bb_depth < 15:
        return False  # too shallow for implied-odds calls

    ip_positions = {"CO", "BTN", "SB"}
    in_position = position in ip_positions

    # Pocket pairs: set-mine when deep enough (any position)
    pairs = {"22", "33", "44", "55", "66", "77", "88", "99", "TT"}
    if notation in pairs and bb_depth >= 15:
        return True

    # Suited connectors and suited broadways — strong implied odds IP
    suited_calls_ip = {
        "QJs", "JTs", "T9s", "98s", "87s", "76s", "65s",
        "KTs", "KJs", "QTs", "A9s", "ATs", "A8s",
        "K9s", "Q9s", "J9s", "T8s",
    }
    if notation in suited_calls_ip and in_position:
        return True

    # Offsuit broadways IP (have good equity vs 3bet range)
    offsuit_calls_ip = {"AJo", "ATo", "KQo", "KJo", "QJo"}
    if notation in offsuit_calls_ip and position in ("CO", "BTN"):
        return True

    # BB defends wide (closing action, already has 1bb invested)
    if position == "BB":
        bb_defend = {
            "77", "66", "55", "44", "33", "22",
            "AJo", "ATo", "A9o", "A8o", "KJo", "KTo", "QJo", "QTo",
            "KTs", "KJs", "QTs", "QJs", "JTs", "J9s", "T9s", "T8s",
            "98s", "97s", "87s", "86s", "76s", "75s", "65s",
            "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
            "K9s", "K8s", "Q9s", "J8s",
        }
        if notation in bb_defend:
            return True

    return False


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
    opponent_af: float = 1.5,
) -> Tuple[str, float, str]:
    t = tuning or _DEFAULT_TUNING
    can_check = _has_check(allowed_actions)
    min_bet, max_bet = _bet_range(allowed_actions)
    call_to = _call_to(allowed_actions)

    has_bet_to_call = call_amount > 0 and not can_check

    if not has_bet_to_call:
        # ── We can check or bet ───────────────────────────────────────────────

        # Overbet: nuts on polarized texture
        if should_overbet(ehs, street, board_texture, is_in_position, pot, stack):
            overbet = min(stack, pot * 1.6)
            overbet = _clamp_bet(overbet, min_bet, max_bet)
            return "raise", overbet, f"Overbet nuts {ehs:.0%} {board_texture}"

        # Value betting — texture-aware sizing
        val_frac = value_bet_fraction(ehs, t, board_texture)

        if ehs >= 0.65:
            bet = min(stack, pot * val_frac)
            bet = _clamp_bet(bet, min_bet, max_bet)
            size_label = _size_label(val_frac)
            return "raise", bet, f"Value {size_label} {ehs:.0%} ({board_texture})"

        # Thin value — calibrated at 0.60 IP / 0.65 OOP to hit ~45% bet rate
        thin_threshold = 0.60 if is_in_position else 0.65
        if ehs >= thin_threshold:
            thin_frac = 0.40 if board_texture == "dry" else 0.55
            bet = min(stack, pot * thin_frac * t.cbet_frequency)
            bet = _clamp_bet(bet, min_bet, max_bet)
            if bet > 0:
                return "raise", bet, f"Thin value {ehs:.0%} {board_texture}"

        # Cbet / bluff
        b_frac = bluff_fraction(ehs, street, board_texture, fold_to_cbet, is_in_position, t)
        if b_frac > 0.0:
            bet = min(stack, pot * b_frac)
            bet = _clamp_bet(bet, min_bet, max_bet)
            return "raise", bet, f"Bluff {b_frac:.0%}p {ehs:.0%} (FtCB={fold_to_cbet:.0%})"

        # Cbet: only on flop, in position, with high fold equity (>0.62)
        # Turn/river cbets are handled by value/bluff paths above
        if is_in_position and street == "flop" and ehs >= 0.35 and fold_to_cbet > 0.62:
            cbet_frac = _cbet_fraction(ehs, board_texture, street)
            if cbet_frac > 0:
                bet = min(stack, pot * cbet_frac * t.cbet_frequency)
                bet = _clamp_bet(bet, min_bet, max_bet)
                if bet > 0:
                    return "raise", bet, f"Cbet {cbet_frac:.0%}p {ehs:.0%}"

        # OOP probe bet: turn/river with decent equity after checking flop
        if not is_in_position and street in ("turn", "river") and ehs >= 0.45:
            probe_frac = 0.35 if street == "turn" else 0.45
            probe = min(stack, pot * probe_frac)
            probe = _clamp_bet(probe, min_bet, max_bet)
            if probe > 0:
                return "raise", probe, f"Probe OOP {street} {ehs:.0%}"

        return "check", 0, f"Check {ehs:.0%} ({board_texture})"

    # ── We face a bet ─────────────────────────────────────────────────────────
    required_equity = effective_call_equity(call_amount, pot, t)
    mdf = minimum_defense_frequency(call_amount, pot)

    if should_defend_vs_bet(ehs, call_amount, pot, t):
        # Check-raise: strong hand or strong draw OOP vs cbet
        if (
            not is_in_position
            and _should_check_raise(ehs, street, board_texture, opponent_af)
            and min_bet > 0
        ):
            raise_to = min(stack, max(min_bet, call_to + call_amount * 2.2))
            raise_to = _clamp_bet(raise_to, min_bet, max_bet)
            return "raise", raise_to, f"Check-raise {ehs:.0%} {board_texture}"

        # Raise for value IP
        if is_in_position and ehs >= required_equity + 0.14 and ehs >= 0.80:
            raise_to = min(stack, max(min_bet, call_to + call_amount * 2.0))
            raise_to = _clamp_bet(raise_to, min_bet, max_bet)
            return "raise", raise_to, f"Value raise IP {ehs:.0%}"

        return "call", call_to, f"Defend {ehs:.0%} (req {required_equity:.0%} MDF={mdf:.0%})"

    return "fold", 0, f"Fold {ehs:.0%} < {required_equity:.0%}"


def _cbet_fraction(ehs: float, board_texture: str, street: str) -> float:
    """Return standard cbet sizing fraction by texture and street.

    Dry: 1/3 pot (range bet — wide frequency, small size)
    Wet: 2/3 pot (protection + value extraction)
    Monotone: 1/2 pot (draw-heavy — balanced)
    Paired: 1/3-1/2 pot
    """
    if board_texture == "dry":
        return 0.33
    if board_texture in ("wet", "monotone"):
        return 0.66 if ehs >= 0.42 else 0.50
    return 0.40  # paired


def _should_check_raise(
    ehs: float,
    street: str,
    board_texture: str,
    opponent_af: float,
) -> bool:
    """Check-raise with strong hands or draws vs aggressive opponents."""
    if street not in ("flop", "turn"):
        return False
    if board_texture in ("wet", "monotone"):
        return ehs >= 0.60 or (ehs < 0.35 and opponent_af >= 2.5)
    return ehs >= 0.75  # dry: only strong hands


def _size_label(frac: float) -> str:
    if frac <= 0.35:
        return "1/3pot"
    if frac <= 0.55:
        return "1/2pot"
    if frac <= 0.70:
        return "2/3pot"
    if frac <= 0.85:
        return "3/4pot"
    if frac <= 1.05:
        return "pot"
    return "overbet"


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
