"""api/error_window.py — rolling API error rate for circuit breaker."""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Tuple


class RollingErrorWindow:
    def __init__(self, window_s: float = 60.0, max_rate: float = 0.10):
        self.window_s = window_s
        self.max_rate = max_rate
        self._events: Deque[Tuple[float, bool]] = deque()

    def record(self, success: bool) -> None:
        now = time.time()
        self._events.append((now, success))
        self._prune(now)

    def _prune(self, now: float) -> None:
        while self._events and self._events[0][0] < now - self.window_s:
            self._events.popleft()

    def error_rate(self) -> float:
        if not self._events:
            return 0.0
        failures = sum(1 for _, ok in self._events if not ok)
        return failures / len(self._events)

    def should_pause(self) -> bool:
        self._prune(time.time())
        return len(self._events) >= 10 and self.error_rate() > self.max_rate
