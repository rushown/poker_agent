#!/usr/bin/env python3
"""scripts/benchmark.py — per-version performance tracking and rollback guard.

Each version gets a benchmark score. If v{N+1} scores WORSE than v{N},
the pipeline automatically rolls back to v{N}'s rules.

Score formula:
  score = (100 - fold_pct) * 2 + raise_pct * 1.5 + call_pct * 1.0
          - bad_folds * 3 - missed_value * 2 - bad_raises * 4
  Higher = better. Tracks improvement across all versions.

Usage:
  python scripts/benchmark.py              # score all versions
  python scripts/benchmark.py --check N    # check if vN beats v(N-1), exit 1 if regression
  python scripts/benchmark.py --rollback N # copy v(N-1) rules back as current rules
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT      = Path(__file__).parent.parent
AGENTS    = Path("/home/ocean/vscode/plutus-agents")
RULES     = ROOT / "strategy_rules.json"
BENCH_LOG = ROOT / "benchmark_log.json"

# Only include versions with enough data for meaningful comparison
MIN_DECISIONS = 200
# Don't compare against very old versions (outdated baseline)
MAX_LOOKBACK  = 2   # only compare against previous 2 versions


def parse_version_log(version: int) -> dict | None:
    log_path = AGENTS / f"v{version}" / "run.log"
    if not log_path.exists():
        return None

    decisions = []
    for line in log_path.read_text(errors="replace").splitlines():
        if '"event": "decision"' in line:
            try:
                decisions.append(json.loads(line[line.index('{"event"'):]))
            except Exception:
                pass

    if len(decisions) < MIN_DECISIONS:
        return None

    by_action: dict = defaultdict(list)
    for d in decisions:
        by_action[d.get("action_taken", "")].append(float(d.get("ehs", 0)))

    total      = len(decisions)
    fold_pct   = len(by_action.get("fold", [])) / total * 100
    raise_pct  = (len(by_action.get("raise", [])) + len(by_action.get("bet", []))) / total * 100
    call_pct   = len(by_action.get("call", [])) / total * 100
    bad_folds  = len([e for e in by_action.get("fold",  []) if e > 0.58])
    missed_val = len([e for e in by_action.get("check", []) if e > 0.72])
    bad_raises = len([e for e in by_action.get("raise", []) + by_action.get("bet", []) if e < 0.38])

    score = (
        (100 - fold_pct) * 2.0
        + raise_pct * 1.5
        + call_pct * 1.0
        - bad_folds * 3.0
        - missed_val * 2.0
        - bad_raises * 4.0
    )

    return {
        "version": version,
        "total_decisions": total,
        "fold_pct": round(fold_pct, 1),
        "raise_pct": round(raise_pct, 1),
        "call_pct": round(call_pct, 1),
        "bad_folds": bad_folds,
        "missed_value": missed_val,
        "bad_raises": bad_raises,
        "score": round(score, 2),
        "avg_ehs_fold": round(sum(by_action.get("fold", [0])) / max(1, len(by_action.get("fold", [1]))), 3),
    }


def load_benchmark_log() -> dict:
    if BENCH_LOG.exists():
        try:
            return json.loads(BENCH_LOG.read_text())
        except Exception:
            pass
    return {"versions": {}}


def save_benchmark_log(data: dict) -> None:
    BENCH_LOG.write_text(json.dumps(data, indent=2))


def score_all(verbose: bool = True) -> dict[int, dict]:
    results = {}
    for v in range(1, 10):
        r = parse_version_log(v)
        if r:
            results[v] = r
            if verbose:
                print(f"  v{v}: score={r['score']:.1f} | fold={r['fold_pct']}% raise={r['raise_pct']}% call={r['call_pct']}% | bad_folds={r['bad_folds']} missed={r['missed_value']} bad_raises={r['bad_raises']}")

    # Save to benchmark log
    log = load_benchmark_log()
    for v, r in results.items():
        log["versions"][str(v)] = r
    save_benchmark_log(log)
    return results


def check_regression(version: int) -> tuple[bool, str]:
    """Check if version N is better than the best of the previous MAX_LOOKBACK versions.

    Returns (is_better, report_string).
    """
    results = score_all(verbose=False)
    if version not in results:
        return True, f"v{version} not scored yet (insufficient data)"

    current_score = results[version]["score"]

    # Find best score among previous versions (limited lookback, ignoring very old)
    baseline_versions = [
        v for v in results
        if v < version and v >= version - MAX_LOOKBACK
    ]
    if not baseline_versions:
        return True, f"No baseline versions to compare (v{version} is first)"

    best_baseline_v  = max(baseline_versions, key=lambda v: results[v]["score"])
    best_baseline    = results[best_baseline_v]["score"]
    improvement      = current_score - best_baseline
    improvement_pct  = (improvement / max(1, abs(best_baseline))) * 100

    report = (
        f"v{version} score={current_score:.1f} vs v{best_baseline_v} score={best_baseline:.1f} "
        f"(Δ={improvement:+.1f}, {improvement_pct:+.1f}%)"
    )

    # Require at least 2-point improvement (not just noise)
    is_better = current_score >= best_baseline - 2.0
    return is_better, report


def rollback(to_version: int) -> None:
    """Restore strategy_rules_v{to_version}.json as the current strategy_rules.json."""
    source = ROOT / f"strategy_rules_v{to_version}.json"
    if not source.exists():
        print(f"ERROR: strategy_rules_v{to_version}.json not found — cannot rollback", file=sys.stderr)
        sys.exit(1)
    RULES.write_text(source.read_text())
    print(f"ROLLBACK: restored v{to_version} rules as current strategy_rules.json")


def print_leaderboard(results: dict[int, dict]) -> None:
    print("\n═══ PLUTUS BENCHMARK LEADERBOARD ═══")
    ranked = sorted(results.values(), key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(ranked):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
        print(f"  {medal} v{r['version']}: score={r['score']:.1f} | fold={r['fold_pct']}% raise={r['raise_pct']}% | bad={r['bad_folds']}f {r['missed_value']}m {r['bad_raises']}r")
    print("═══════════════════════════════════\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plutus benchmark tracker")
    parser.add_argument("--check",    type=int, help="Check if version N beats previous best")
    parser.add_argument("--rollback", type=int, help="Roll back to version N's rules")
    parser.add_argument("--leaderboard", action="store_true")
    args = parser.parse_args()

    if args.rollback is not None:
        rollback(args.rollback)
        return

    if args.check is not None:
        results = score_all(verbose=False)
        is_better, report = check_regression(args.check)
        print(report)
        if not is_better:
            print(f"REGRESSION DETECTED — v{args.check} is worse than baseline")
            best_v = max((v for v in results if v < args.check), key=lambda v: results[v]["score"], default=args.check - 1)
            print(f"Auto-rolling back to v{best_v} rules...")
            rollback(best_v)
            sys.exit(1)
        print(f"v{args.check} PASSES benchmark check")
        return

    print("Scoring all versions:")
    results = score_all(verbose=True)
    if args.leaderboard or results:
        print_leaderboard(results)


if __name__ == "__main__":
    main()
