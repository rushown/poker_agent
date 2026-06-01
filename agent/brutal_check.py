"""agent/brutal_check.py — continuous validation, rollback, and honest metrics."""
from __future__ import annotations

import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class HandMetric:
    ts: float
    strategy_mode: str
    meta_strategy: str
    chip_delta: float
    bb_size: float
    action_ev: float = 0.0
    baseline_ev: float = 0.0
    won: Optional[bool] = None

    @property
    def bb100(self) -> float:
        if self.bb_size <= 0:
            return 0.0
        return (self.chip_delta / self.bb_size) * 100


class BrutalSelfCheck:
    """Central instrumentation: health, profitability, rollback."""

    ROLLBACK_DROP_BB100 = 5.0
    ROLLBACK_WINDOW = 200
    INTIMIDATION_HANDS = 50
    INTIMIDATION_MIN_BB100 = 2.0

    def __init__(self, state_file: str = ".arena-brutal-state"):
        self.state_file = state_file
        self.started_at = time.time()
        self.api_errors_400: int = 0
        self.api_errors_total: int = 0
        self.api_calls: int = 0
        self.crashes: int = 0
        self.last_action_at: float = time.time()
        self._hand_metrics: Deque[HandMetric] = deque(maxlen=2000)
        self._bb100_baseline: float = 0.0
        self._champion_bb100: float = 0.0
        self._intimidation_chip_delta: float = 0.0
        self._intimidation_hands: int = 0
        self._intimidation_aborted: bool = False
        self._change_log: List[Dict[str, Any]] = []
        self._alerts: Deque[str] = deque(maxlen=100)
        self._opponent_predictions: Dict[str, List[Tuple[float, bool]]] = {}
        self._underperform_streak: int = 0
        self._safe_config_path = ".arena-safe-config"
        self._load()

    def record_api_call(self, success: bool, status: int = 200) -> None:
        self.api_calls += 1
        if not success:
            self.api_errors_total += 1
            if status == 400:
                self.api_errors_400 += 1
                self._alert(f"API 400 error (total 400s={self.api_errors_400})")

    def record_action(self) -> None:
        self.last_action_at = time.time()

    def record_hand(
        self,
        *,
        strategy_mode: str,
        meta_strategy: str,
        chip_delta: float,
        bb_size: float,
        action_ev: float = 0.0,
        baseline_ev: float = 0.0,
        won: Optional[bool] = None,
    ) -> None:
        self.record_action()
        m = HandMetric(
            ts=time.time(),
            strategy_mode=strategy_mode,
            meta_strategy=meta_strategy,
            chip_delta=chip_delta,
            bb_size=bb_size,
            action_ev=action_ev,
            baseline_ev=baseline_ev,
            won=won,
        )
        self._hand_metrics.append(m)

        if strategy_mode == "intimidation":
            self._intimidation_chip_delta += chip_delta
            self._intimidation_hands += 1
            if (
                self._intimidation_hands >= self.INTIMIDATION_HANDS
                and not self._intimidation_aborted
            ):
                bb100 = self._intimidation_bb100()
                if bb100 < self.INTIMIDATION_MIN_BB100:
                    self._intimidation_aborted = True
                    self._alert(
                        f"INTIMIDATION ABORT: bb/100={bb100:.1f} < {self.INTIMIDATION_MIN_BB100}"
                    )

        if baseline_ev > 0 and action_ev < baseline_ev * 0.85:
            self._underperform_streak += 1
        else:
            self._underperform_streak = 0

        if self._underperform_streak >= 20:
            self._alert("Strategy underperforming baseline for 20+ hands")

        self._check_rollback()
        self.save()

    def intimidation_active(self) -> bool:
        if self._intimidation_aborted:
            return False
        return self._intimidation_hands < self.INTIMIDATION_HANDS

    def _intimidation_bb100(self) -> float:
        if self._intimidation_hands == 0:
            return 0.0
        # approximate bb from recent hands
        recent = [h for h in self._hand_metrics if h.strategy_mode == "intimidation"]
        if not recent:
            return 0.0
        bb = max(1.0, sum(h.bb_size for h in recent) / len(recent))
        return (self._intimidation_chip_delta / bb / self._intimidation_hands) * 100

    def record_fold_prediction(self, agent_id: str, predicted_fold: float, did_fold: bool) -> None:
        hist = self._opponent_predictions.setdefault(agent_id, [])
        hist.append((predicted_fold, did_fold))
        if len(hist) > 100:
            self._opponent_predictions[agent_id] = hist[-100:]

    def opponent_model_accuracy(self, agent_id: str) -> Optional[float]:
        hist = self._opponent_predictions.get(agent_id, [])
        if len(hist) < 10:
            return None
        correct = sum(
            1 for pred, actual in hist[-50:] if (pred > 0.5) == actual
        )
        return correct / min(50, len(hist))

    def rolling_bb100(self, window: int = 200) -> float:
        recent = list(self._hand_metrics)[-window:]
        if not recent:
            return 0.0
        chips = sum(h.chip_delta for h in recent)
        bbs = sum(max(1.0, h.bb_size) for h in recent)
        return (chips / bbs) * 100 if bbs else 0.0

    def _check_rollback(self) -> None:
        if len(self._hand_metrics) < 50:
            return
        current = self.rolling_bb100(self.ROLLBACK_WINDOW)
        if self._bb100_baseline == 0.0:
            self._bb100_baseline = current
            return
        drop = self._bb100_baseline - current
        if drop > self.ROLLBACK_DROP_BB100:
            self._alert(
                f"ROLLBACK TRIGGER: bb/100 dropped {drop:.1f} "
                f"(baseline={self._bb100_baseline:.1f} now={current:.1f})"
            )
            self._bb100_baseline = current

    def should_rollback(self) -> bool:
        if len(self._hand_metrics) < self.ROLLBACK_WINDOW:
            return False
        return (self._bb100_baseline - self.rolling_bb100(self.ROLLBACK_WINDOW)) > (
            self.ROLLBACK_DROP_BB100
        )

    def log_change(self, description: str, diff_ref: str = "") -> None:
        entry = {"ts": time.time(), "description": description, "ref": diff_ref}
        self._change_log.append(entry)
        if len(self._change_log) > 200:
            self._change_log = self._change_log[-200:]
        logger.info(f"[brutal] change: {description}")

    def _alert(self, msg: str) -> None:
        self._alerts.append(f"{time.time():.0f}: {msg}")
        logger.warning(f"[BRUTAL] {msg}")

    def api_error_rate_1m(self, window_calls: Deque[Tuple[float, bool]]) -> float:
        if not window_calls:
            return 0.0
        failures = sum(1 for _, ok in window_calls if not ok)
        return failures / len(window_calls)

    def health_dict(self) -> Dict[str, Any]:
        idle_s = time.time() - self.last_action_at
        recent_400 = self.api_errors_400
        return {
            "status": "ok" if self.api_errors_400 == 0 else "degraded",
            "errors_last_100": recent_400,
            "brutal_check": "ok" if self.api_errors_400 == 0 else "degraded",
            "uptime_s": round(time.time() - self.started_at, 1),
            "api_400_errors": self.api_errors_400,
            "api_error_rate": round(
                self.api_errors_total / max(1, self.api_calls), 4
            ),
            "rolling_bb100": round(self.rolling_bb100(200), 2),
            "intimidation_aborted": self._intimidation_aborted,
            "intimidation_hands": self._intimidation_hands,
            "idle_since_action_s": round(idle_s, 1),
            "underperform_streak": self._underperform_streak,
            "alerts": list(self._alerts)[-10:],
            "should_rollback": self.should_rollback(),
        }

    def metrics_dict(self) -> Dict[str, Any]:
        by_mode: Dict[str, List[float]] = {}
        by_meta: Dict[str, List[float]] = {}
        for h in self._hand_metrics:
            by_mode.setdefault(h.strategy_mode, []).append(h.bb100)
            by_meta.setdefault(h.meta_strategy, []).append(h.bb100)

        def avg(xs: List[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        return {
            "hands_logged": len(self._hand_metrics),
            "rolling_bb100_200": self.rolling_bb100(200),
            "by_strategy_mode": {k: round(avg(v), 2) for k, v in by_mode.items()},
            "by_meta_strategy": {k: round(avg(v), 2) for k, v in by_meta.items()},
            "opponent_model_accuracy": {
                aid: round(acc, 3)
                for aid, acc in (
                    (aid, self.opponent_model_accuracy(aid))
                    for aid in self._opponent_predictions
                )
                if acc is not None
            },
            "change_log": self._change_log[-20:],
        }

    def save(self) -> None:
        data = {
            "api_errors_400": self.api_errors_400,
            "api_errors_total": self.api_errors_total,
            "api_calls": self.api_calls,
            "intimidation_aborted": self._intimidation_aborted,
            "intimidation_hands": self._intimidation_hands,
            "bb100_baseline": self._bb100_baseline,
            "change_log": self._change_log[-50:],
            "hands": [
                {
                    "strategy_mode": h.strategy_mode,
                    "meta_strategy": h.meta_strategy,
                    "chip_delta": h.chip_delta,
                    "bb_size": h.bb_size,
                    "won": h.won,
                }
                for h in list(self._hand_metrics)[-100:]
            ],
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
            self.api_errors_400 = data.get("api_errors_400", 0)
            self.api_errors_total = data.get("api_errors_total", 0)
            self.api_calls = data.get("api_calls", 0)
            self._intimidation_aborted = data.get("intimidation_aborted", False)
            self._intimidation_hands = data.get("intimidation_hands", 0)
            self._bb100_baseline = data.get("bb100_baseline", 0.0)
            self._change_log = data.get("change_log", [])
        except Exception:
            pass


def bootstrap_improvement_significant(
    before: List[float],
    after: List[float],
    n_samples: int = 500,
    p_threshold: float = 0.10,
) -> Tuple[bool, float]:
    """Return (significant, p_value). One-sided: after mean > before mean."""
    if len(before) < 5 or len(after) < 5:
        return False, 1.0
    observed = (sum(after) / len(after)) - (sum(before) / len(before))
    if observed <= 0:
        return False, 1.0
    count_not_improved = 0
    for _ in range(n_samples):
        b = [random.choice(before) for _ in range(len(before))]
        a = [random.choice(after) for _ in range(len(after))]
        if (sum(a) / len(a) - sum(b) / len(b)) <= 0:
            count_not_improved += 1
    p_value = count_not_improved / n_samples
    return p_value < p_threshold, p_value


def self_test() -> List[str]:
    errors: List[str] = []
    bsc = BrutalSelfCheck(state_file="/tmp/brutal_test.json")
    bsc.record_hand(
        strategy_mode="gto",
        meta_strategy="TAG",
        chip_delta=10,
        bb_size=2,
        won=True,
    )
    if bsc.rolling_bb100(10) == 0 and len(bsc._hand_metrics) > 0:
        errors.append("rolling_bb100 should be non-zero after win")
    sig, p = bootstrap_improvement_significant([0.0] * 20, [5.0] * 20)
    if not sig:
        errors.append(f"bootstrap should detect improvement, p={p}")
    return errors
