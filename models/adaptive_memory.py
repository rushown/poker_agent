"""models/adaptive_memory.py — self-tuning strategy parameters learned from outcomes."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class StrategyTuning:
    """Live knobs adjusted from session results (1.0 = baseline)."""
    preflop_aggression: float = 1.0       # scales open/3-bet frequencies
    postflop_value_bias: float = 1.0      # value bet sizing
    bluff_frequency: float = 1.0          # bluff / barrel willingness
    exploit_blend: float = 1.0            # weight on exploits vs GTO
    icm_tightness: float = 1.0            # ICM discipline multiplier
    call_threshold_adj: float = 0.0       # added to required equity to call
    cbet_frequency: float = 1.0           # continuation bet rate
    steal_frequency: float = 1.0          # open-raise steal bias

    def clamp(self) -> None:
        self.preflop_aggression = _clamp(self.preflop_aggression, 0.7, 1.35)
        self.postflop_value_bias = _clamp(self.postflop_value_bias, 0.75, 1.4)
        self.bluff_frequency = _clamp(self.bluff_frequency, 0.5, 1.25)
        self.exploit_blend = _clamp(self.exploit_blend, 0.6, 1.4)
        self.icm_tightness = _clamp(self.icm_tightness, 0.85, 1.3)
        self.call_threshold_adj = _clamp(self.call_threshold_adj, -0.08, 0.12)
        self.cbet_frequency = _clamp(self.cbet_frequency, 0.6, 1.35)
        self.steal_frequency = _clamp(self.steal_frequency, 0.7, 1.35)


@dataclass
class DecisionRecord:
    table_id: str
    hand_number: Any
    street: str
    action: str
    amount: float
    ehs: float
    pot: float
    call_amount: float
    position: str
    required_equity: float
    notation: str
    stack: float
    won_hand: Optional[bool] = None
    stack_delta: float = 0.0
    mistake: str = ""


@dataclass
class MistakeStats:
    bad_calls: int = 0
    bad_folds: int = 0
    overbluffs: int = 0
    missed_value: int = 0
    spewy_hands: int = 0
    good_hands: int = 0
    hands_reviewed: int = 0


class AdaptiveMemory:
    """Persists tuning + mistake history across sessions."""

    def __init__(self, state_file: str = ".arena-adaptive-state"):
        self.state_file = state_file
        self.tuning = StrategyTuning()
        self.mistakes = MistakeStats()
        self._pending: Dict[str, List[DecisionRecord]] = {}
        self._stack_at_hand_start: Dict[str, float] = {}
        self.recent_results: List[float] = []
        self.strategy_weights: Dict[str, float] = {
            "intimidation": 1.0,
            "exploitation": 1.0,
            "gto": 1.0,
            "icm": 1.0,
            "endgame": 1.0,
        }
        self._last_strategy_mode: str = "gto"
        self._load()

    def hand_key(self, table_id: str, hand_number: Any) -> str:
        return f"{table_id}:{hand_number}"

    def start_hand(self, table_id: str, hand_number: Any, stack: float) -> None:
        key = self.hand_key(table_id, hand_number)
        self._stack_at_hand_start[key] = stack
        self._pending[key] = []

    def record_decision(
        self,
        table_id: str,
        hand_number: Any,
        street: str,
        action: str,
        amount: float,
        ehs: float,
        pot: float,
        call_amount: float,
        position: str,
        required_equity: float,
        notation: str,
        stack: float,
    ) -> None:
        key = self.hand_key(table_id, hand_number)
        if key not in self._pending:
            self.start_hand(table_id, hand_number, stack)
        self._pending[key].append(
            DecisionRecord(
                table_id=table_id,
                hand_number=hand_number,
                street=street,
                action=action.lower(),
                amount=amount,
                ehs=ehs,
                pot=pot,
                call_amount=call_amount,
                position=position,
                required_equity=required_equity,
                notation=notation,
                stack=stack,
            )
        )

    def finish_hand(
        self,
        table_id: str,
        hand_number: Any,
        stack_end: float,
        won: Optional[bool] = None,
        bb_size: float = 100.0,
    ) -> None:
        key = self.hand_key(table_id, hand_number)
        decisions = self._pending.pop(key, [])
        start_stack = self._stack_at_hand_start.pop(key, stack_end)
        stack_delta = stack_end - start_stack

        self.recent_results.append(stack_delta)
        if len(self.recent_results) > 100:
            self.recent_results = self.recent_results[-100:]

        if not decisions:
            return

        self._analyze_hand(decisions, stack_delta, won, bb_size)
        self.post_hand_review(decisions, stack_delta, won, bb_size)
        self.tuning.clamp()
        self.save()
        if self.mistakes.hands_reviewed % 50 == 0:
            self.export_strategy_report()

    def _analyze_hand(
        self,
        decisions: List[DecisionRecord],
        stack_delta: float,
        won: Optional[bool],
        bb_size: float,
    ) -> None:
        self.mistakes.hands_reviewed += 1
        bb = max(1.0, bb_size)
        spew = stack_delta < -15 * bb

        for d in decisions:
            act = d.action
            if act == "call" and d.ehs < d.required_equity - 0.04:
                d.mistake = "bad_call"
                self.mistakes.bad_calls += 1
                self._learn_from_mistake("bad_call")
            elif act == "fold" and d.call_amount > 0 and d.ehs > d.required_equity + 0.06:
                d.mistake = "bad_fold"
                self.mistakes.bad_folds += 1
                self._learn_from_mistake("bad_fold")
            elif act in ("raise", "bet") and d.ehs < 0.32 and d.street in ("turn", "river"):
                if stack_delta < 0 or won is False:
                    d.mistake = "overbluff"
                    self.mistakes.overbluffs += 1
                    self._learn_from_mistake("overbluff")
            elif act in ("check", "fold") and d.ehs >= 0.72 and d.street == "river":
                if stack_delta <= 0 and won is not True:
                    d.mistake = "missed_value"
                    self.mistakes.missed_value += 1
                    self._learn_from_mistake("missed_value")

        if spew and stack_delta < 0:
            self.mistakes.spewy_hands += 1
            self._learn_from_mistake("spew")
        elif stack_delta > 5 * bb and won is not False:
            self.mistakes.good_hands += 1
            self._learn_from_success()

        if won is False and len(decisions) >= 2:
            # Session running bad — lean slightly more GTO
            wr = self.recent_win_rate()
            if wr < 0.35 and len(self.recent_results) >= 20:
                self.tuning.exploit_blend *= 0.98
                self.tuning.call_threshold_adj += 0.005

    def _learn_from_mistake(self, kind: str) -> None:
        lr = 0.02
        if kind == "bad_call":
            self.tuning.call_threshold_adj += lr
            self.tuning.exploit_blend *= 1.0 - lr
        elif kind == "bad_fold":
            self.tuning.call_threshold_adj -= lr * 0.75
        elif kind == "overbluff":
            self.tuning.bluff_frequency *= 1.0 - lr * 1.5
            self.tuning.cbet_frequency *= 1.0 - lr * 0.5
        elif kind == "missed_value":
            self.tuning.postflop_value_bias *= 1.0 + lr
        elif kind == "spew":
            self.tuning.preflop_aggression *= 1.0 - lr
            self.tuning.bluff_frequency *= 1.0 - lr
            self.tuning.icm_tightness *= 1.0 + lr * 0.5

    def _learn_from_success(self) -> None:
        lr = 0.008
        self.tuning.exploit_blend = min(1.4, self.tuning.exploit_blend * (1.0 + lr))
        if self.mistakes.bad_calls > self.mistakes.good_hands:
            return
        self.tuning.postflop_value_bias = min(1.4, self.tuning.postflop_value_bias * (1.0 + lr * 0.5))

    def set_strategy_mode(self, mode: str) -> None:
        self._last_strategy_mode = mode.lower()

    def post_hand_review(
        self,
        decisions: List[DecisionRecord],
        stack_delta: float,
        won: Optional[bool],
        bb_size: float,
    ) -> None:
        """Adjust strategy line weights from big pot outcomes."""
        bb = max(1.0, bb_size)
        big = abs(stack_delta) >= 8 * bb
        mode_key = self._last_strategy_mode.split("_")[0]
        if mode_key not in self.strategy_weights:
            mode_key = "gto"
        if big and stack_delta > 0:
            self.strategy_weights[mode_key] = min(
                1.5, self.strategy_weights.get(mode_key, 1.0) * 1.05
            )
        elif big and stack_delta < 0:
            for d in decisions:
                if d.mistake:
                    self.strategy_weights[mode_key] = max(
                        0.6, self.strategy_weights.get(mode_key, 1.0) * 0.95
                    )
                    break

    def export_strategy_report(self, path: str = "strategy_report.json") -> None:
        report = {
            "updated": time.time(),
            "weights": self.strategy_weights,
            "tuning": asdict(self.tuning),
            "mistakes": asdict(self.mistakes),
            "recent_win_rate": self.recent_win_rate(),
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(report, f, indent=2)
        os.replace(tmp, path)

    def recent_win_rate(self) -> float:
        if not self.recent_results:
            return 0.5
        wins = sum(1 for r in self.recent_results if r > 0)
        return wins / len(self.recent_results)

    def effective_exploit_blend(self, opponent_confidence: float) -> float:
        base = 0.45 + opponent_confidence * 0.4
        wr = self.recent_win_rate()
        if wr < 0.4:
            base *= 0.85
        elif wr > 0.55:
            base *= 1.1
        return min(1.0, base * self.tuning.exploit_blend)

    def summary(self) -> str:
        t = self.tuning
        m = self.mistakes
        return (
            f"tune preflop={t.preflop_aggression:.2f} val={t.postflop_value_bias:.2f} "
            f"bluff={t.bluff_frequency:.2f} exploit={t.exploit_blend:.2f} "
            f"call_adj={t.call_threshold_adj:+.2f} | "
            f"mistakes calls={m.bad_calls} folds={m.bad_folds} "
            f"bluffs={m.overbluffs} value={m.missed_value} reviewed={m.hands_reviewed}"
        )

    def save(self) -> None:
        data = {
            "tuning": asdict(self.tuning),
            "mistakes": asdict(self.mistakes),
            "recent_results": self.recent_results[-50:],
            "strategy_weights": self.strategy_weights,
        }
        tmp = self.state_file + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.state_file)
        except Exception as e:
            logger.debug(f"adaptive save failed: {e}")

    def _load(self) -> None:
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                data = json.load(f)
            self.tuning = StrategyTuning(**data.get("tuning", {}))
            self.mistakes = MistakeStats(**data.get("mistakes", {}))
            self.recent_results = data.get("recent_results", [])
            self.strategy_weights.update(data.get("strategy_weights", {}))
            self.tuning.clamp()
        except Exception:
            pass


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
