#!/usr/bin/env python3
"""cross_version_analysis.py — aggregate mistakes across all Plutus versions.

Reads v1, v2, v3, ... run.log files and builds a unified mistake dataset
for the multi-agent DeepSeek pipeline. This is how v4+ learns from ALL
prior history, not just the most recent version.

Usage:
  python scripts/cross_version_analysis.py            # all versions
  python scripts/cross_version_analysis.py --from 2  # v2 onwards
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

AGENTS_DIR = Path("/home/ocean/vscode/plutus-agents")
ROOT       = Path(__file__).parent.parent


def parse_log(log_path: Path) -> tuple[list[dict], list[dict]]:
    decisions, outcomes = [], []
    for line in log_path.read_text(errors="replace").splitlines():
        if '"event": "decision"' in line:
            try:
                decisions.append(json.loads(line[line.index('{"event"'):]))
            except Exception:
                pass
        if '"event": "hand_outcome"' in line:
            try:
                outcomes.append(json.loads(line[line.index('{"event"'):]))
            except Exception:
                pass
    return decisions, outcomes


def version_stats(version: int, decisions: list[dict], outcomes: list[dict]) -> dict:
    by_action: dict = defaultdict(list)
    for d in decisions:
        by_action[d.get("action_taken", "")].append(float(d.get("ehs", 0)))

    total = len(decisions)
    if total == 0:
        return {}

    bad_folds  = [d for d in decisions if d.get("action_taken") == "fold"
                  and float(d.get("ehs", 0)) > 0.58]
    missed_val = [d for d in decisions if d.get("action_taken") == "check"
                  and float(d.get("ehs", 0)) > 0.72]
    bad_raises = [d for d in decisions if d.get("action_taken") in ("raise", "bet")
                  and float(d.get("ehs", 0)) < 0.38]

    bb_vals    = [float(o.get("bb_delta", 0)) for o in outcomes]
    bb100      = round(sum(bb_vals) / len(bb_vals) * 100, 1) if bb_vals else 0

    def avg(lst: list) -> float:
        return round(sum(lst) / len(lst), 3) if lst else 0

    fold_ehs_list = by_action.get("fold", [])
    raise_ehs_list = by_action.get("raise", []) + by_action.get("bet", [])

    return {
        "version": version,
        "total_decisions": total,
        "total_hands": len(outcomes),
        "bb_per_100": bb100,
        "action_pcts": {
            "fold":  round(len(by_action.get("fold", [])) / total * 100, 1),
            "call":  round(len(by_action.get("call", [])) / total * 100, 1),
            "raise": round(len(raise_ehs_list) / total * 100, 1),
        },
        "avg_ehs_fold":  avg(fold_ehs_list),
        "avg_ehs_raise": avg(raise_ehs_list),
        "regret_counts": {
            "bad_folds":    len(bad_folds),
            "missed_value": len(missed_val),
            "bad_raises":   len(bad_raises),
        },
        "bad_folds": [
            {
                "version": version,
                "hand": d.get("hand_number"),
                "ehs": round(float(d.get("ehs", 0)), 3),
                "mode": d.get("strategy_mode", ""),
                "street": d.get("street", "preflop"),
            }
            for d in bad_folds[:15]
        ],
        "missed_value": [
            {
                "version": version,
                "hand": d.get("hand_number"),
                "ehs": round(float(d.get("ehs", 0)), 3),
            }
            for d in missed_val[:10]
        ],
        "bad_raises": [
            {
                "version": version,
                "hand": d.get("hand_number"),
                "ehs": round(float(d.get("ehs", 0)), 3),
            }
            for d in bad_raises[:5]
        ],
    }


def load_rules_for_version(version: int) -> dict:
    """Load the rules that were in effect when version N ran."""
    # v1 ran with strategy_rules_v1.json, v2 ran with strategy_rules_v2.json, etc.
    # The current strategy_rules.json is the rules for the NEXT version.
    candidates = [
        ROOT / f"strategy_rules_v{version}.json",
        ROOT / "strategy_rules.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                return json.loads(c.read_text())
            except Exception:
                pass
    return {}


def aggregate_all_versions(from_version: int = 1) -> dict:
    """Aggregate stats and mistakes from all available versions."""
    all_stats = []
    all_bad_folds = []
    all_missed_val = []
    all_bad_raises = []
    version_rules  = {}

    for version in range(from_version, 10):
        log_path = AGENTS_DIR / f"v{version}" / "run.log"
        if not log_path.exists():
            continue

        print(f"[cross-version] Loading v{version}...", file=sys.stderr)
        decisions, outcomes = parse_log(log_path)
        if len(decisions) < 20:
            print(f"[cross-version] v{version}: only {len(decisions)} decisions — skipping", file=sys.stderr)
            continue

        stats = version_stats(version, decisions, outcomes)
        if stats:
            all_stats.append(stats)
            all_bad_folds.extend(stats.get("bad_folds", []))
            all_missed_val.extend(stats.get("missed_value", []))
            all_bad_raises.extend(stats.get("bad_raises", []))
            version_rules[f"v{version}"] = load_rules_for_version(version)

    # Cross-version trends
    trends = _compute_trends(all_stats)

    return {
        "versions_analyzed": [s["version"] for s in all_stats],
        "per_version_stats": all_stats,
        "cross_version_trends": trends,
        "all_bad_folds":   all_bad_folds[:40],
        "all_missed_value": all_missed_val[:20],
        "all_bad_raises":  all_bad_raises[:15],
        "total_bad_folds":    len(all_bad_folds),
        "total_missed_value": len(all_missed_val),
        "total_bad_raises":   len(all_bad_raises),
        "rules_by_version": version_rules,
    }


def _compute_trends(all_stats: list[dict]) -> dict:
    if len(all_stats) < 2:
        return {}

    versions   = [s["version"] for s in all_stats]
    fold_rates = [s["action_pcts"].get("fold", 0) for s in all_stats]
    call_rates = [s["action_pcts"].get("call", 0) for s in all_stats]
    bb100s     = [s.get("bb_per_100", 0) for s in all_stats]
    bad_folds  = [s["regret_counts"].get("bad_folds", 0) for s in all_stats]

    def trend(vals: list) -> str:
        if len(vals) < 2:
            return "stable"
        delta = vals[-1] - vals[0]
        if delta < -5:
            return f"improving ({vals[0]:.1f} → {vals[-1]:.1f})"
        if delta > 5:
            return f"worsening ({vals[0]:.1f} → {vals[-1]:.1f})"
        return f"stable ({vals[-1]:.1f})"

    fold_improving = len(fold_rates) >= 2 and fold_rates[-1] < fold_rates[-2]

    return {
        "fold_rate_trend":   trend(fold_rates),
        "call_rate_trend":   trend(call_rates),
        "bb100_trend":       trend(bb100s),
        "bad_fold_trend":    trend(bad_folds),
        "fold_rate_by_version":  dict(zip(versions, fold_rates)),
        "bb100_by_version":      dict(zip(versions, bb100s)),
        "bad_folds_by_version":  dict(zip(versions, bad_folds)),
        "fold_rate_improving":   fold_improving,
        "conclusion": (
            "Fold rate is trending down — threshold loosening is working."
            if fold_improving else
            "Fold rate NOT improving — need more aggressive threshold loosening or different approach."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_version", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Output JSON (default: pretty JSON)")
    args = parser.parse_args()

    result = aggregate_all_versions(from_version=args.from_version)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
