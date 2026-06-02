#!/usr/bin/env python3
"""refine_strategy.py — one refinement cycle: parse logs → DeepSeek → update strategy_rules.json.

Usage:
    python refine_strategy.py --log run.log [--min-hands 30]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

RULES_FILE = ROOT / "strategy_rules.json"


# ── log parsing ────────────────────────────────────────────────────────

def parse_log(log_path: str) -> dict:
    """Extract every decision event from run.log into structured stats."""
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    decisions = []
    for line in path.read_text(errors="replace").splitlines():
        if '"event": "decision"' not in line:
            continue
        try:
            idx = line.index('{"event"')
            d = json.loads(line[idx:])
            decisions.append(d)
        except Exception:
            continue

    if not decisions:
        return {"error": "no decision events found in log"}

    # Aggregate stats
    total = len(decisions)
    by_action = defaultdict(list)          # action → list of ehs
    by_street = defaultdict(lambda: defaultdict(int))  # street → action → count
    folds_high_ehs = []                    # EHS when we folded despite decent hand
    calls_bad_odds = []                    # calls where we likely had bad odds
    raises_low_ehs = []                    # raises with low EHS (potential bluff leaks)
    missed_value   = []                    # checked with high EHS when we could bet

    for d in decisions:
        action = d.get("action_taken", "")
        ehs    = float(d.get("ehs", 0))
        street = d.get("strategy_mode", "unknown")
        mode   = d.get("strategy_mode", "")
        amt    = float(d.get("amount", 0))

        by_action[action].append(ehs)
        by_street[street][action] += 1

        if action == "fold" and ehs > 0.55:
            folds_high_ehs.append({"ehs": round(ehs, 3), "street": street})

        if action == "check" and ehs > 0.70:
            missed_value.append({"ehs": round(ehs, 3), "street": street})

        if action in ("raise", "bet") and ehs < 0.38:
            raises_low_ehs.append({"ehs": round(ehs, 3), "street": street, "amount": amt})

    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0

    def dist(lst):
        if not lst: return {}
        buckets = {"<0.40": 0, "0.40-0.55": 0, "0.55-0.70": 0, "0.70-0.85": 0, ">0.85": 0}
        for v in lst:
            if v < 0.40:   buckets["<0.40"] += 1
            elif v < 0.55: buckets["0.40-0.55"] += 1
            elif v < 0.70: buckets["0.55-0.70"] += 1
            elif v < 0.85: buckets["0.70-0.85"] += 1
            else:          buckets[">0.85"] += 1
        return buckets

    def pct(n): return round(n / total * 100, 1)

    stats = {
        "total_decisions": total,
        "action_counts": {k: len(v) for k, v in by_action.items()},
        "action_pcts": {
            "fold":  pct(len(by_action["fold"])),
            "call":  pct(len(by_action["call"])),
            "check": pct(len(by_action["check"])),
            "raise": pct(len(by_action.get("raise", [])) + len(by_action.get("bet", []))),
        },
        "avg_ehs": {
            "when_fold":  avg(by_action["fold"]),
            "when_call":  avg(by_action["call"]),
            "when_check": avg(by_action["check"]),
            "when_raise": avg(by_action.get("raise", []) + by_action.get("bet", [])),
        },
        "ehs_distribution_at_fold": dist(by_action["fold"]),
        "problems": {
            "folds_with_ehs_above_55": folds_high_ehs[:10],
            "checks_with_ehs_above_70": missed_value[:10],
            "raises_with_ehs_below_38": raises_low_ehs[:10],
        },
        "by_mode": {s: dict(v) for s, v in by_street.items()},
    }

    return stats


# ── DeepSeek refinement ────────────────────────────────────────────────

def run_deepseek_refinement(current_rules: dict, log_stats: dict) -> dict:
    """Send current rules + log stats to DeepSeek. Returns revised rules dict."""
    from scripts.ds_query import query  # local helper

    cycle = current_rules.get("cycle", 0)

    prompt = f"""
You are a poker strategy optimizer reviewing a Texas Hold'em bot's performance logs.
Your job: identify exactly where the bot deviated from perfect play and output a revised strategy.

CURRENT RULES (cycle {cycle}):
{json.dumps(current_rules, indent=2)}

LOG STATISTICS FROM THIS CYCLE:
{json.dumps(log_stats, indent=2)}

CORE CONSTRAINTS (never violate):
1. "Only play when you have the hand" — every raise/bet requires EHS >= street minimum
2. No speculative bluffs unless opponent data proves they fold enough (fold_to_cbet > 0.60)
3. Do not lower ANY threshold more than 0.05 per cycle (prevent overcorrection)
4. Do not raise a threshold more than 0.05 per cycle
5. If bluff_enabled is currently false, only set it to true if logs show folded hands > 60%

ANALYSIS REQUIRED:
For each problem in log_stats.problems, cite the specific data and propose a rule change.
For example:
- "folds_with_ehs_above_55: [ehs=0.62, street=intimidation] → open_ip_min_ehs is too high, reduce from 0.60 to 0.57"
- "checks_with_ehs_above_70: [ehs=0.74, street=intimidation] → no_bet_value_min_ehs is too high, reduce from 0.65 to 0.62"

OUTPUT FORMAT — return ONLY valid JSON, no commentary outside the JSON:
{{
  "cycle_analysis": {{
    "problems_found": ["problem 1 with log evidence", "problem 2", ...],
    "what_worked": ["thing 1 that is correct and should not change", ...],
    "key_leaks": "1-2 sentence summary of the biggest leak this cycle"
  }},
  "revised_rules": {{
    ... (complete strategy_rules.json with ONLY the changed numbers — same structure as input)
  }},
  "changes_made": [
    {{"field": "preflop.open_ip_min_ehs", "old": 0.60, "new": 0.57, "reason": "cite log evidence"}},
    ...
  ],
  "convergence": false
}}

Set convergence=true only if NO changes were needed (zero actionable improvements).
If you find speculative or unsafe changes to reject, note them in cycle_analysis.problems_found and do NOT include them in revised_rules.
"""

    print(f"\n[DeepSeek] Sending cycle {cycle} logs for analysis...")
    result_str = query(prompt, model="deepseek-reasoner", no_cache=True, timeout=120)

    # Parse JSON from response
    try:
        # Strip any markdown fences
        clean = result_str.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        return json.loads(clean)
    except Exception as e:
        print(f"[DeepSeek] JSON parse failed: {e}")
        print(f"[DeepSeek] Raw response:\n{result_str[:2000]}")
        return {}


# ── validation ─────────────────────────────────────────────────────────

def validate_rules(old: dict, new: dict) -> tuple[bool, list[str]]:
    """Reject revisions that introduce speculative plays or overcorrections."""
    errors = []
    MAX_DELTA = 0.06

    streets = ["preflop", "flop", "turn", "river"]
    for s in streets:
        old_s = old.get(s, {})
        new_s = new.get(s, {})
        for k, new_val in new_s.items():
            if not isinstance(new_val, (int, float)):
                continue
            old_val = old_s.get(k, new_val)
            delta = abs(new_val - old_val)
            if delta > MAX_DELTA:
                errors.append(f"{s}.{k}: change {old_val}→{new_val} exceeds max delta {MAX_DELTA}")

    # Bluff guard: only allow bluff if fold evidence present
    for s in streets:
        new_s = new.get(s, {})
        old_s = old.get(s, {})
        if new_s.get("bluff_enabled") and not old_s.get("bluff_enabled"):
            errors.append(f"{s}.bluff_enabled: enabled without explicit fold-rate evidence in logs")

    return len(errors) == 0, errors


# ── main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to run.log")
    parser.add_argument("--min-hands", type=int, default=30, help="Minimum hands before refining")
    parser.add_argument("--dry-run", action="store_true", help="Parse logs + show analysis without writing rules")
    args = parser.parse_args()

    # Load current rules
    current_rules = json.loads(RULES_FILE.read_text())
    cycle = current_rules.get("cycle", 0)
    print(f"\n{'='*60}")
    print(f" STRATEGY REFINEMENT — Cycle {cycle}")
    print(f"{'='*60}")

    # Parse log
    print(f"\n[Log] Parsing {args.log}...")
    stats = parse_log(args.log)
    if "error" in stats:
        print(f"[Error] {stats['error']}")
        sys.exit(1)

    total = stats["total_decisions"]
    print(f"[Log] {total} decisions found")
    print(f"[Log] fold={stats['action_pcts']['fold']}%  call={stats['action_pcts']['call']}%  "
          f"check={stats['action_pcts']['check']}%  raise={stats['action_pcts']['raise']}%")
    print(f"[Log] avg EHS: fold={stats['avg_ehs']['when_fold']}  raise={stats['avg_ehs']['when_raise']}")
    print(f"[Log] Problems: {len(stats['problems']['folds_with_ehs_above_55'])} high-EHS folds, "
          f"{len(stats['problems']['checks_with_ehs_above_70'])} missed value spots, "
          f"{len(stats['problems']['raises_with_ehs_below_38'])} low-EHS raises")

    if total < args.min_hands:
        print(f"\n[Skip] Only {total} hands — need {args.min_hands} minimum. Run more hands first.")
        sys.exit(0)

    if args.dry_run:
        print("\n[DryRun] Stats shown above. No rules updated (--dry-run).")
        print(json.dumps(stats, indent=2))
        sys.exit(0)

    # Run DeepSeek
    result = run_deepseek_refinement(current_rules, stats)
    if not result:
        print("[Error] No valid response from DeepSeek.")
        sys.exit(1)

    # Print analysis
    analysis = result.get("cycle_analysis", {})
    print(f"\n[DeepSeek] Key leak: {analysis.get('key_leaks', '(none)')}")
    print(f"[DeepSeek] Problems found: {len(analysis.get('problems_found', []))}")
    for p in analysis.get("problems_found", []):
        print(f"  - {p}")
    print(f"[DeepSeek] Changes proposed: {len(result.get('changes_made', []))}")
    for c in result.get("changes_made", []):
        print(f"  {c['field']}: {c['old']} → {c['new']}  ({c['reason'][:80]})")

    if result.get("convergence"):
        print("\n✅ CONVERGED — no actionable improvements found. Strategy is optimal for this data.")
        sys.exit(0)

    # Validate
    new_rules = result.get("revised_rules", {})
    if not new_rules:
        print("[Error] DeepSeek returned no revised_rules.")
        sys.exit(1)

    ok, validation_errors = validate_rules(current_rules, new_rules)
    if not ok:
        print(f"\n[Validation FAILED] {len(validation_errors)} unsafe changes rejected:")
        for e in validation_errors:
            print(f"  ✗ {e}")
        sys.exit(1)

    # Bump cycle, add notes
    new_rules["cycle"] = cycle + 1
    new_rules["cycle_notes"] = f"Cycle {cycle+1} — {len(result.get('changes_made', []))} changes. " \
                                f"Key leak: {analysis.get('key_leaks', '')[:100]}"

    # Backup old rules
    backup = RULES_FILE.parent / f"strategy_rules_cycle{cycle}.json"
    backup.write_text(json.dumps(current_rules, indent=2))
    print(f"\n[Backup] Old rules saved to {backup.name}")

    # Write new rules
    RULES_FILE.write_text(json.dumps(new_rules, indent=2))
    print(f"[Updated] strategy_rules.json → cycle {cycle+1}")
    print(f"\n{'='*60}")
    print(f" Cycle {cycle} complete. Run the agent again for cycle {cycle+1}.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
