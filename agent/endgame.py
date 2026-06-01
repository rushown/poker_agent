"""agent/endgame.py — heads-up / final-table hyper-aggression."""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from agent.gto_bot import hand_notation, _call_to, _has_check, _bet_range, _clamp_bet

_rng = random.Random()
_hand_counter = 0


def _bump_hand() -> int:
    global _hand_counter
    _hand_counter += 1
    return _hand_counter


class EndgameModule:
    """2–3 players left: steal wide, polarized rivers, varied sizing."""

    def __init__(self) -> None:
        self.fold_to_3bet: float = 0.45
        self._size_rotation = [0.33, 0.55, 0.75, 0.90]

    def update_fold_to_3bet(self, rate: float) -> None:
        if rate > 0:
            self.fold_to_3bet = rate

    def decide(
        self,
        hole: List[str],
        position: str,
        stack: float,
        pot: float,
        bb_size: float,
        street: str,
        is_facing_raise: bool,
        call_amount: float,
        ehs: float,
        allowed_actions: List[dict],
        is_in_position: bool,
    ) -> Optional[Tuple[str, float, str]]:
        notation = hand_notation(hole)
        pos = position.upper()
        acts = {a.get("action") for a in allowed_actions}
        min_bet, max_bet = _bet_range(allowed_actions)
        h = _bump_hand()
        size_mult = self._size_rotation[h % len(self._size_rotation)]

        if street == "preflop":
            if is_facing_raise:
                if pos == "BTN" and self.fold_to_3bet >= 0.55 and "raise" in acts:
                    return "raise", min(stack, max(min_bet, pot * 3)), f"[END] BTN 3bet any2 {notation}"
                if ehs >= 0.45 and "call" in acts:
                    return "call", _call_to(allowed_actions), f"[END] call {notation}"
                if "fold" in acts:
                    return "fold", 0, f"[END] fold"
                return None
            if pos in ("BTN", "CO", "SB") and "raise" in acts:
                open_to = min(stack, max(min_bet, bb_size * (2.2 + size_mult)))
                return "raise", _clamp_bet(open_to, min_bet, max_bet), f"[END] steal {notation}"
            if _has_check(allowed_actions):
                return "check", 0, f"[END] check"
            return "fold", 0, f"[END] fold"

        if street == "river":
            if not is_facing_raise and ehs >= 0.55 and "raise" in acts:
                bet = min(stack, pot * 0.5)
                return "raise", _clamp_bet(bet, min_bet, max_bet), f"[END] river value 0.5x"
            if not is_facing_raise and ehs < 0.35 and is_in_position and "raise" in acts:
                bet = min(stack, pot * 1.75)
                return "raise", _clamp_bet(bet, min_bet, max_bet), f"[END] river polar bluff"
            if is_facing_raise and ehs < 0.25 and "fold" in acts:
                return "fold", 0, f"[END] river fold air"

        if not is_facing_raise and ehs >= 0.5 and "raise" in acts:
            geo = min(stack, pot * size_mult)
            return "raise", _clamp_bet(geo, min_bet, max_bet), f"[END] geo bet {size_mult:.0%} pot"

        return None


def test_against_aggressive_bot() -> bool:
    """Sanity: endgame steals on button vs high fold-to-3bet."""
    eg = EndgameModule()
    eg.update_fold_to_3bet(0.60)
    acts = [{"action": "raise", "minAmount": 4, "maxAmount": 200}]
    r = eg.decide(
        ["7c", "2d"], "BTN", 200, 3, 2, "preflop", False, 0, 0.3, acts, True
    )
    return r is not None and r[0] == "raise"


def self_test() -> list:
    errors: list = []
    if not test_against_aggressive_bot():
        errors.append("test_against_aggressive_bot failed")
    return errors
