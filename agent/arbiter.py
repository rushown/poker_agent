"""agent/arbiter.py — lean arbiter for the ADAPTIVE strategy.

Decision flow (every hand):
  1. Calculate EHS
  2. Safety override (AA/KK/QQ — never fold premiums)
  3. Push-fold (≤15BB preflop — Nash tables)
  4. ADAPTIVE strategy (reads strategy_rules.json)
  5. Emergency time-budget fallback
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_tl = threading.local()
_tl.opponent_type = "unknown"
_tl.opponent_stats = None   # OpponentStats for dominant opponent (used by midgame)

def build_chat_message(action: str, **kw) -> str:
    return action.upper()

def mistake_line() -> str:
    return ""
from agent.meta_learner import MetaLearner, NUMBERED_STRATEGIES
from engine.ehs import calculate_ehs, clear_ehs_cache, samples_for_street
from engine.hand_eval import hand_notation
from engine.push_fold import should_call_push, should_push
from models.adaptive_memory import AdaptiveMemory
from models.bot_pattern_detector import BotPatternDetector
from models.opponent_tracker import OpponentTracker, OpponentStats


def _pot_odds_required(call: float, pot: float) -> float:
    return call / (call + pot) if (call + pot) > 0 else 0.0


# Simple mode constants — no external dependency needed
_MODE_GTO = "gto_balanced"
_MODE_ICM = "icm_survival"


@dataclass
class GameContext:
    hole_cards: List[str]
    community_cards: List[str]
    pot: float
    call_amount: float
    stack: float
    bb_size: float
    street: str
    position: str
    is_in_position: bool
    all_stacks: List[float] = field(default_factory=list)
    payouts: Optional[List[float]] = None
    allowed_actions: List[Dict] = field(default_factory=list)
    opponent_ids: List[str] = field(default_factory=list)
    is_facing_raise: bool = False
    facing_raise_size: float = 0.0
    table_id: str = ""
    hand_number: Any = None
    call_to_amount: float = 0.0
    committed_chips: float = 0.0


class StrategyArbiter:
    def __init__(
        self,
        tracker: OpponentTracker,
        adaptive: Optional[AdaptiveMemory] = None,
        meta: Optional[MetaLearner] = None,
        brutal_check: Any = None,
    ):
        self.tracker = tracker
        self.adaptive = adaptive or AdaptiveMemory()
        self.meta = meta or MetaLearner()
        self.brutal = brutal_check
        self.detector = BotPatternDetector()
        self._hand_cache_key: Optional[str] = None
        self._last_ehs: float = 0.5
        self._current_mode: str = _MODE_GTO
        self.match_hands_played: int = 0
        self.target_hands: int = 500
        self.players_remaining: int = 6
        raw = os.getenv("ARENA_STRATEGY", "").strip().upper()
        self._strategy_override: str = raw if raw not in ("DEFAULT", "EXISTING", "") else "ADAPTIVE"

    def set_match_context(self, hands_played: int, target_hands: int = 500, players_remaining: int = 6) -> None:
        self.match_hands_played = hands_played
        self.target_hands = target_hands
        self.players_remaining = max(2, players_remaining)

    def on_new_hand(self, table_id: str, hand_number: Any, stack: float = 0) -> None:
        key = f"{table_id}:{hand_number}"
        if key != self._hand_cache_key:
            clear_ehs_cache()
            self._hand_cache_key = key
        if hand_number is not None:
            self.adaptive.start_hand(table_id, hand_number, stack)

    def decide(self, ctx: GameContext, deadline: float = 0.0) -> Tuple[str, float, str]:
        start = time.monotonic()
        budget = self._time_budget(start, deadline)

        ehs = calculate_ehs(
            ctx.hole_cards, ctx.community_cards,
            samples=samples_for_street(ctx.street),
            street=ctx.street,
            facing_raise=ctx.is_facing_raise or ctx.call_amount > ctx.bb_size,
        )
        self._last_ehs = ehs
        bb_depth = ctx.stack / max(1, ctx.bb_size)

        # 1. Safety override — never fold AA/KK/QQ preflop
        safety = self._safety_override(ctx)
        if safety:
            return self._finalize(ctx, *safety)

        # 2. Push-fold at short stacks
        if bb_depth <= 15 and ctx.street == "preflop":
            pf = self._push_fold(ctx, ehs)
            if pf:
                return self._finalize(ctx, *pf)

        # 3. Emergency time fallback
        if self._over_budget(start, budget):
            return self._finalize(ctx, *self._fast_fallback(ctx, ehs))

        # 4. Classify dominant opponent — inject via thread-local (safe for async multi-table)
        if ctx.opponent_ids:
            dominant_type = "unknown"
            dominant_stats = None
            for oid in ctx.opponent_ids:
                stats = self.tracker.get(oid)
                if stats and stats.hands_seen >= 8:
                    profile = self.detector.classify(stats)
                    dominant_type = profile.bot_type.value
                    dominant_stats = stats
                    break
            _tl.opponent_type = dominant_type
            _tl.opponent_stats = dominant_stats   # used by midgame bluff detection

        # 4. ADAPTIVE strategy decision
        sid = self._strategy_override
        from agent.strategies import route as _sr
        try:
            action, amount, reason = _sr(sid, ctx, ehs, bb_depth)
        except Exception:
            action, amount, reason = self._fast_fallback(ctx, ehs)
        return self._finalize(ctx, action, amount, reason)

    def _finalize(self, ctx: GameContext, action: str, amount: float, chat: str = "") -> Tuple[str, float, str]:
        if ctx.hand_number is not None and ctx.table_id:
            self.adaptive.record_decision(
                table_id=ctx.table_id, hand_number=ctx.hand_number, street=ctx.street,
                action=action, amount=amount, ehs=self._last_ehs, pot=ctx.pot,
                call_amount=ctx.call_amount, position=ctx.position,
                required_equity=_pot_odds_required(ctx.call_amount, ctx.pot),
                notation=hand_notation(ctx.hole_cards), stack=ctx.stack,
            )
        mistake = ""
        if ctx.hand_number and self.adaptive.mistakes.bad_calls > 0:
            last = self.adaptive._pending.get(self.adaptive.hand_key(ctx.table_id, ctx.hand_number), [])
            if last and last[-1].mistake:
                mistake = mistake_line()
        tilt = build_chat_message(
            action=action, won_big_pot=False,
            was_bluff=action in ("raise", "bet") and self._last_ehs < 0.35,
            bubble_pressure=False, ehs=self._last_ehs, strategy_mode=_MODE_GTO,
        )
        safe_chat = tilt.strip() if tilt.strip() else action.upper()
        if mistake:
            safe_chat = f"{safe_chat} {mistake}".strip()
        return action, amount, safe_chat[:200]

    def _safety_override(self, ctx: GameContext) -> Optional[Tuple[str, float, str]]:
        notation = hand_notation(ctx.hole_cards)
        acts = {a.get("action", "").lower() for a in ctx.allowed_actions}
        if ctx.street == "preflop" and notation in ("AA", "KK", "QQ"):
            if ctx.is_facing_raise and "raise" in acts:
                mx = max((float(a.get("maxAmount", ctx.stack)) for a in ctx.allowed_actions
                          if a.get("action") in ("raise", "all-in")), default=ctx.stack)
                return "raise", mx, f"Safety: never fold {notation}"
            if "raise" in acts and not ctx.is_facing_raise:
                for a in ctx.allowed_actions:
                    if a.get("action") == "raise":
                        return "raise", float(a.get("minAmount", ctx.bb_size * 2)), f"Safety: open {notation}"
        return None

    def _push_fold(self, ctx: GameContext, ehs: float) -> Optional[Tuple[str, float, str]]:
        notation = hand_notation(ctx.hole_cards)
        bb_depth = ctx.stack / max(1, ctx.bb_size)
        if ctx.is_facing_raise or ctx.call_amount > ctx.bb_size:
            to_call = ctx.call_to_amount or ctx.call_amount
            if should_call_push(notation, bb_depth, 1.0) or ehs >= 0.58:
                return "call", to_call, f"Push-fold call {notation}"
            if ehs >= 0.52:
                return "call", to_call, f"Push-fold call marginal {notation}"
            return "fold", 0, f"Push-fold fold {notation}"
        if should_push(notation, ctx.position, bb_depth, ctx.is_facing_raise) or ehs >= 0.62:
            return "raise", ctx.stack, f"Push {notation} @ {bb_depth:.1f}BB"
        return "fold", 0, f"Push-fold fold {notation}"

    def _time_budget(self, start: float, deadline: float) -> float:
        return max(0.15, deadline - time.time() - 0.25) if deadline > 0 else 1.4

    def _over_budget(self, start: float, budget: float) -> bool:
        return (time.monotonic() - start) > budget

    def _fast_fallback(self, ctx: GameContext, ehs: float) -> Tuple[str, float, str]:
        acts = {a.get("action", "").lower() for a in ctx.allowed_actions}
        if "check" in acts:
            return "check", 0, f"Timeout check EHS={ehs:.0%}"
        if ehs > 0.5 and "call" in acts:
            return "call", ctx.call_to_amount or ctx.call_amount, "Timeout call"
        return "fold", 0, "Timeout fold"

    def _clamp(self, amount: float, allowed_actions: List[Dict], stack: float) -> float:
        if amount <= 0:
            return 0.0
        for a in allowed_actions:
            if a.get("action") in ("bet", "raise", "all-in"):
                mn = float(a.get("minAmount", 0))
                mx = float(a.get("maxAmount", stack))
                return max(mn, min(mx, min(amount, stack)))
        return min(amount, stack)

    @property
    def current_mode(self) -> str:
        return self._current_mode
