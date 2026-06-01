"""agent/self_test_runner.py — run all module self_tests before agent starts."""
from __future__ import annotations

import sys
from typing import Callable, List, Tuple

MODULE_TESTS: List[Tuple[str, Callable[[], List[str]]]] = []


def _register_all() -> None:
    global MODULE_TESTS
    if MODULE_TESTS:
        return

    from agent import brutal_check
    from agent import ab_test
    from agent import meta_learner
    from agent import endgame
    from models import bot_pattern_detector
    from api import action_amount

    MODULE_TESTS = [
        ("brutal_check", brutal_check.self_test),
        ("ab_test", ab_test.ABTestRunner().self_test),
        ("meta_learner", meta_learner.self_test),
        ("endgame", endgame.self_test),
        ("bot_pattern_detector", bot_pattern_detector.self_test),
        ("action_amount", action_amount.self_test),
    ]


def run_startup_self_tests() -> None:
    _register_all()
    failures: List[str] = []
    for name, fn in MODULE_TESTS:
        try:
            errs = fn()
            if errs:
                failures.extend([f"{name}: {e}" for e in errs])
        except Exception as e:
            failures.append(f"{name}: exception {e}")
    if failures:
        for f in failures:
            print(f"SELF_TEST FAIL: {f}", file=sys.stderr)
        raise SystemExit(1)
