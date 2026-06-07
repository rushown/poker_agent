"""agent/performance_dashboard.py — lightweight performance tracking per strategy."""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict


class PerformanceDashboard:
    def __init__(self, *a, **kw):
        self._deltas: Deque[float] = deque(maxlen=500)
        self._by_strategy: Dict[str, Deque[float]] = {}
        self._wins = 0
        self._hands = 0

    def record(self, *a, **kw) -> None:
        pass

    def record_hand(
        self,
        strategy_mode: str = "",
        meta_strategy: str = "",
        stack_delta: float = 0.0,
        bb_size: float = 2.0,
        vpip: bool = False,
        pfr: bool = False,
        won: object = None,
    ) -> None:
        self._hands += 1
        bb_delta = stack_delta / max(1.0, bb_size)
        self._deltas.append(bb_delta)
        key = meta_strategy or strategy_mode or "default"
        if key not in self._by_strategy:
            self._by_strategy[key] = deque(maxlen=200)
        self._by_strategy[key].append(bb_delta)
        if won is True:
            self._wins += 1

    def snapshot(self) -> dict:
        recent = list(self._deltas)
        avg = sum(recent) / len(recent) if recent else 0.0
        by_strat = {
            k: round(sum(v) / len(v), 3) if v else 0.0
            for k, v in self._by_strategy.items()
        }
        return {
            "hands": self._hands,
            "wins": self._wins,
            "win_rate": round(self._wins / max(1, self._hands), 3),
            "avg_bb_per_hand": round(avg, 3),
            "by_strategy": by_strat,
        }

    def summary(self, *a, **kw) -> dict:
        return self.snapshot()
