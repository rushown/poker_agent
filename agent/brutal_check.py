"""agent/brutal_check.py — lightweight self-monitoring for API health and session stats."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


def bootstrap_improvement_significant(*a, **kw) -> bool:
    return False


@dataclass
class BrutalSelfCheck:
    """Tracks API errors, action counts, and hand outcomes for health checks."""

    last_action_at: float = field(default_factory=time.time)

    # Rolling windows
    _api_results: Deque = field(default_factory=lambda: deque(maxlen=100))
    _hand_deltas: Deque = field(default_factory=lambda: deque(maxlen=200))

    # Counters
    _actions_total: int = 0
    _hands_total: int = 0
    _api_400_count: int = 0

    def __post_init__(self) -> None:
        self._api_results = deque(maxlen=100)
        self._hand_deltas = deque(maxlen=200)

    # ── called by runner.py ────────────────────────────────────────────

    def record_action(self, *a, **kw) -> None:
        self._actions_total += 1
        self.last_action_at = time.time()

    def on_action(self, *a, **kw) -> None:
        self.record_action()

    def record(self, *a, **kw) -> None:
        self.record_action()

    def check(self, *a, **kw) -> None:
        pass

    def record_api_call(self, success: bool, status: int = 200) -> None:
        self._api_results.append((success, status, time.time()))
        if not success and status == 400:
            self._api_400_count += 1

    def record_hand(
        self,
        strategy_mode: str = "",
        meta_strategy: str = "",
        chip_delta: float = 0.0,
        bb_size: float = 2.0,
        action_ev: float = 0.0,
        baseline_ev: float = 0.0,
        won: Optional[bool] = None,
    ) -> None:
        self._hands_total += 1
        self._hand_deltas.append(chip_delta / max(1.0, bb_size))
        self.last_action_at = time.time()

    def save(self) -> None:
        pass

    def should_rollback(self) -> bool:
        if len(self._api_results) < 20:
            return False
        recent = list(self._api_results)[-20:]
        fail_rate = sum(1 for ok, _, _ in recent if not ok) / len(recent)
        return fail_rate > 0.50

    def _alert(self, msg: str) -> None:
        from loguru import logger
        logger.warning(f"[BrutalCheck] {msg}")

    # ── properties ───────────────────────────────────────────────────

    @property
    def api_errors_400(self) -> int:
        return self._api_400_count

    # ── health / metrics dicts ────────────────────────────────────────

    def health_dict(self) -> dict:
        errors_last_100 = sum(
            1 for ok, _, _ in self._api_results if not ok
        )
        recent_bbph = 0.0
        if self._hand_deltas:
            recent_bbph = sum(self._hand_deltas) / len(self._hand_deltas)
        return {
            "status": "degraded" if self.should_rollback() else "ok",
            "errors_last_100": errors_last_100,
            "api_400_total": self._api_400_count,
            "actions_total": self._actions_total,
            "hands_total": self._hands_total,
            "recent_bb_per_hand": round(recent_bbph, 3),
        }

    def metrics_dict(self) -> dict:
        return self.health_dict()
