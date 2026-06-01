"""agent/early_game.py — early-match intimidation (first ~20% of hands)."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from agent.gto_bot import hand_notation, _call_to, _has_check
from agent.preflop_ranges import open_raise_size_bb
from engine.hand_eval import card_rank
from models.opponent_tracker import OpponentTracker

_rng = random.Random()

# Top ~20% of hands (pairs, aces, broadway, suited connectors)
_TOP20 = {
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77",
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A5s", "A4s",
    "AKo", "AQo", "AJo", "ATo",
    "KQs", "KJs", "KTs", "QJs", "QTs", "JTs", "T9s", "98s", "87s", "76s",
    "KQo", "KJo", "QJo",
}


class EarlyGameAggressor:
    """Hyper-aggressive opening phase to shape table image."""

    def __init__(self, tracker: OpponentTracker) -> None:
        self.tracker = tracker
        self.fold_to_aggression: Dict[str, float] = {}
        self.scared_money_ids: List[str] = []

    def in_early_phase(self, hands_played: int, target_hands: int = 500) -> bool:
        return hands_played < max(50, int(target_hands * 0.20))

    def is_top20(self, hole: List[str]) -> bool:
        return hand_notation(hole) in _TOP20

    def is_top30_shove(self, hole: List[str]) -> bool:
        n = hand_notation(hole)
        if n in _TOP20:
            return True
        if len(hole) == 2:
            r1, r2 = card_rank(hole[0]), card_rank(hole[1])
            if max(r1, r2) >= 12:  # K+
                return True
        return False

    def record_reaction(self, agent_id: str, folded: bool) -> None:
        if not agent_id:
            return
        hist = self.tracker.get(agent_id)
        if hist.hands_seen < 1:
            return
        key = agent_id
        prev = self.fold_to_aggression.get(key, 0.5)
        fold_rate = hist.fold_to_3bet if hist.fold_to_3bet_opps > 2 else prev
        self.fold_to_aggression[key] = fold_rate
        if fold_rate > 0.60 and agent_id not in self.scared_money_ids:
            self.scared_money_ids.append(agent_id)

    def decide(
        self,
        hole: List[str],
        position: str,
        stack: float,
        bb_size: float,
        pot: float,
        is_facing_raise: bool,
        facing_raise_size: float,
        allowed_actions: List[dict],
        opponent_ids: List[str],
    ) -> Optional[Tuple[str, float, str]]:
        notation = hand_notation(hole)
        acts = {a.get("action") for a in allowed_actions}

        if is_facing_raise:
            if self.is_top30_shove(hole) and "raise" in acts:
                return "raise", stack, f"[INTIM] 3bet-shove {notation}"
            if notation in _TOP20 and "call" in acts:
                return "call", _call_to(allowed_actions), f"[INTIM] defend {notation}"
            if "fold" in acts:
                return "fold", 0, f"[INTIM] fold {notation}"
            return None

        if self.is_top20(hole) or _rng.random() < 0.35:
            if "raise" in acts:
                # 4–6x open
                base = open_raise_size_bb(position, bb_size)
                size = min(stack, max(base * 1.8, bb_size * 4.5))
                for a in allowed_actions:
                    if a.get("action") in ("raise", "bet"):
                        mn = float(a.get("minAmount", size))
                        mx = float(a.get("maxAmount", stack))
                        size = max(mn, min(mx, size))
                if size >= stack * 0.85 and "raise" in acts:
                    return "raise", stack, f"[INTIM] open-shove {notation}"
                return "raise", size, f"[INTIM] oversized open {notation} {size:.0f}"
            if _has_check(allowed_actions):
                return "check", 0, f"[INTIM] trap {notation}"

        if "fold" in acts:
            return "fold", 0, f"[INTIM] fold marginal {notation}"
        return None
