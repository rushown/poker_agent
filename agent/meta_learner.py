"""agent/meta_learner.py — UCB1-bandit strategy selection with bootstrap validation.

Strategy selection uses UCB1 (Upper Confidence Bound) which is theoretically
optimal for exploration-exploitation in a multi-armed bandit setting.
UCB1: select strategy i = argmax(avg_reward_i + sqrt(2 * ln(total_pulls) / pulls_i))
Switching threshold: UCB1 upper bound of best > lower bound of current by 0.5 bb/100.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from agent.brutal_check import bootstrap_improvement_significant
from loguru import logger


# Core meta-strategies for the default arbiter.
CORE_STRATEGIES = ("MANIAC", "TAG", "LAG", "NIT", "EXPLOIT")
# All numbered strategies registered in agent/strategies/.
NUMBERED_STRATEGIES = ("ADAPTIVE",)
# Full pool: META mode uses all; default mode uses CORE only.
STRATEGIES = CORE_STRATEGIES + NUMBERED_STRATEGIES


@dataclass
class StrategyBlockStats:
    hands: int = 0
    chip_delta: float = 0.0
    hand_deltas: List[float] = field(default_factory=list)

    @property
    def bb_per_hand(self) -> float:
        return self.chip_delta / max(1, self.hands)


class MetaLearner:
    """Sliding-window performance per named strategy; pick best every N hands."""

    def __init__(self, state_file: str = ".arena-meta-state", block_size: int = 50):
        self.state_file = state_file
        self.block_size = block_size
        self.active: str = "EXPLOIT"
        self.previous_safe: str = "EXPLOIT"
        self._block: StrategyBlockStats = StrategyBlockStats()
        self._history: Dict[str, Deque[float]] = {s: deque(maxlen=20) for s in STRATEGIES}
        self._hand_deltas_by_strategy: Dict[str, Deque[float]] = {
            s: deque(maxlen=500) for s in STRATEGIES
        }
        self._hands_in_block = 0
        self._last_switch_log: List[Dict] = []
        self._load()

    def select_strategy(
        self,
        table_context: Optional[dict] = None,
        opponent_confidence: float = 0.5,
    ) -> str:
        if table_context and table_context.get("force_strategy"):
            return str(table_context["force_strategy"])
        if table_context and table_context.get("force_tag"):
            return "TAG"
        if opponent_confidence < 0.25 and self.active not in STRATEGIES:
            return self.previous_safe
        return self.active

    def record_outcome(
        self,
        strategy_name: str,
        hand_delta: float,
        hand_winner: bool,
        bb_size: float = 2.0,
    ) -> None:
        name = strategy_name if strategy_name in STRATEGIES else "EXPLOIT"
        self._block.hands += 1
        self._block.chip_delta += hand_delta
        self._block.hand_deltas.append(hand_delta)
        self._hand_deltas_by_strategy[name].append(hand_delta / max(1.0, bb_size))
        self._hands_in_block += 1

        if self._hands_in_block >= self.block_size:
            bbph = self._block.bb_per_hand
            self._history[name].append(bbph)
            logger.info(
                f"Meta block {name}: hands={self._block.hands} bb/hand={bbph:.2f}"
            )
            self._rotate_strategy(name)
            self._block = StrategyBlockStats()
            self._hands_in_block = 0
            self.save()

    def _rotate_strategy(self, last_block_strategy: str) -> None:
        candidate = self._best_strategy_by_history()
        before = list(self._hand_deltas_by_strategy.get(self.active, []))[-100:]
        after = list(self._hand_deltas_by_strategy.get(candidate, []))[-100:]

        # Only force exploration when candidate has zero recorded data (bootstrap requires samples)
        force_explore = candidate != self.active and len(after) == 0
        if force_explore:
            significant, p_val = True, 0.0
        else:
            significant, p_val = validate_performance(before, after)

        old = self.active
        if significant and candidate != self.active:
            self.previous_safe = self.active
            self.active = candidate
            reason = "forced explore" if force_explore else f"bootstrap p={p_val:.3f}"
            logger.info(f"Meta switch {old} -> {candidate} ({reason})")
        else:
            logger.info(
                f"Meta keep {self.active} (candidate {candidate} not significant p={p_val:.3f})"
            )
        self._last_switch_log.append(
            {
                "ts": time.time(),
                "from": old,
                "to": self.active,
                "candidate": candidate,
                "p_value": p_val,
                "significant": significant,
                "last_block": last_block_strategy,
            }
        )

    def _best_strategy_by_history(self) -> str:
        """UCB1 bandit: balance exploration and exploitation.

        UCB1 score = avg_reward + sqrt(2 * ln(total_pulls) / pulls_i)
        Strategies with fewer samples get a bonus to encourage exploration.
        Switch only if best UCB1 upper bound exceeds current strategy's avg by 0.5 bb/100.
        """
        total_pulls = sum(len(self._history[s]) for s in STRATEGIES)
        if total_pulls == 0:
            return self.active

        best_name = self.active
        best_ucb1 = -1e9
        for s in STRATEGIES:
            hist = self._history[s]
            pulls = len(hist)
            if pulls == 0:
                # Unsampled strategy: give it max exploration bonus
                ucb1 = 2.0
            else:
                avg = sum(hist) / pulls
                exploration = math.sqrt(2 * math.log(max(1, total_pulls)) / pulls)
                ucb1 = avg + exploration
            if ucb1 > best_ucb1:
                best_ucb1 = ucb1
                best_name = s

        # Switch only if meaningful improvement (> 0.5 bb/100 over current avg)
        current_hist = self._history[self.active]
        current_avg = sum(current_hist) / max(1, len(current_hist)) if current_hist else -1e9
        if best_name == self.active or best_ucb1 < current_avg + 0.5:
            return self.active
        return best_name

    def validate_performance(
        self,
        before: Optional[List[float]] = None,
        after: Optional[List[float]] = None,
    ) -> Tuple[bool, float]:
        before = before or list(self._hand_deltas_by_strategy.get(self.active, []))
        after = after or list(self._hand_deltas_by_strategy.get(self._best_strategy_by_history(), []))
        return bootstrap_improvement_significant(before, after)

    def reset_match(self) -> None:
        self._block = StrategyBlockStats()
        self._hands_in_block = 0

    def scoreboard(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for s in STRATEGIES:
            deltas = list(self._hand_deltas_by_strategy[s])
            hist = list(self._history[s])
            out[s] = {
                "hands": len(deltas),
                "avg_bb": sum(deltas) / max(1, len(deltas)),
                "blocks": len(hist),
                "block_avg": sum(hist) / max(1, len(hist)) if hist else 0.0,
            }
        return out

    def save(self) -> None:
        data = {
            "active": self.active,
            "previous_safe": self.previous_safe,
            "history": {k: list(v) for k, v in self._history.items()},
            "hand_deltas": {
                k: list(v)[-200:] for k, v in self._hand_deltas_by_strategy.items()
            },
            "switch_log": self._last_switch_log[-20:],
            "updated": time.time(),
        }
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.state_file)

    def _load(self) -> None:
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                data = json.load(f)
            self.active = data.get("active", "EXPLOIT")
            self.previous_safe = data.get("previous_safe", "EXPLOIT")
            for s, vals in data.get("history", {}).items():
                if s in self._history:
                    self._history[s] = deque(vals, maxlen=20)
            for s, vals in data.get("hand_deltas", {}).items():
                if s in self._hand_deltas_by_strategy:
                    self._hand_deltas_by_strategy[s] = deque(vals, maxlen=500)
            self._last_switch_log = data.get("switch_log", [])
        except Exception:
            pass


def validate_performance(
    before: List[float], after: List[float]
) -> Tuple[bool, float]:
    return bootstrap_improvement_significant(before, after)


def self_test() -> List[str]:
    errors: List[str] = []
    m = MetaLearner(state_file="/tmp/meta_self.json", block_size=5)
    for i in range(6):
        m.record_outcome("TAG", 2.0, True, bb_size=2)
    if m.active not in STRATEGIES:
        errors.append("active must be valid strategy")
    sig, _ = validate_performance([0.0] * 20, [4.0] * 20)
    if not sig:
        errors.append("bootstrap should accept clear winner")
    return errors
