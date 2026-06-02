#!/usr/bin/env python3
"""cycle_engine.py — recursive self-improving poker bot.

Architecture:
  v1 plays 500 hands → full logs collected
  DeepSeek stage 1: opponent profiling (who folds, who calls too much, who 3bets light)
  DeepSeek stage 2: our mistake analysis (folded winners, missed value, over-bluffed)
  DeepSeek stage 3: generates a PROMPT FOR CLAUDE to implement the fixes
  Claude reads DeepSeek's prompt → updates strategy_rules.json
  v2 plays with new rules → repeat

DeepSeek is the researcher + prompt-builder.
Claude is the implementer.
The loop never stops until two cycles produce zero changes.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT   = Path(__file__).parent
RULES  = ROOT / "strategy_rules.json"
AGENTS = Path("/home/ocean/vscode/plutus-agents")
PYTHON = ROOT / ".venv/bin/python"
DS_KEY = os.getenv("DEEPSEEK_API_KEY", "")
if not DS_KEY:
    print("ERROR: DEEPSEEK_API_KEY env var not set. Export it before running.", file=sys.stderr)
    sys.exit(1)
os.environ["DEEPSEEK_API_KEY"] = DS_KEY

sys.path.insert(0, str(ROOT))
from scripts.ds_query import query as ds


def log(msg: str, level: str = "INFO") -> None:
    print(f"{time.strftime('%H:%M:%S')} [{level}] {msg}", flush=True)


def load_rules() -> dict:
    return json.loads(RULES.read_text())


def save_rules(rules: dict) -> None:
    RULES.write_text(json.dumps(rules, indent=2))


# ══════════════════════════════════════════════════════════════════════
#  LOG PARSING
# ══════════════════════════════════════════════════════════════════════

def parse_log(log_path: Path) -> tuple[list[dict], list[dict]]:
    """Returns (decisions, hand_outcomes) from run.log."""
    decisions, outcomes = [], []
    for line in log_path.read_text(errors="replace").splitlines():
        for tag, store in [("decision", decisions), ("hand_outcome", outcomes)]:
            if f'"event": "{tag}"' in line:
                try:
                    d = json.loads(line[line.index('{"event"'):])
                    store.append(d)
                except Exception:
                    pass
    return decisions, outcomes


# ══════════════════════════════════════════════════════════════════════
#  OPPONENT PROFILING
# ══════════════════════════════════════════════════════════════════════

def build_opponent_profiles(decisions: list[dict], outcomes: list[dict]) -> dict:
    """Build per-opponent behavior model from log data."""
    opp_hands: dict = defaultdict(lambda: {
        "seen": 0, "raised_vs_us": 0, "folded": 0, "called": 0,
        "won_vs_us": 0, "stacks": []
    })

    for d in decisions:
        for oid in d.get("opponent_ids", []):
            opp_hands[oid]["seen"] += 1

    for o in outcomes:
        for opp in o.get("opponents", []):
            oid = opp.get("id", "")
            if not oid:
                continue
            opp_hands[oid]["stacks"].append(opp.get("stack", 0))
        for wid in o.get("winner_ids", []):
            if wid and wid in opp_hands:
                opp_hands[wid]["won_vs_us"] += 1

    profiles = {}
    for oid, data in opp_hands.items():
        if data["seen"] < 3:
            continue
        avg_stack = sum(data["stacks"]) / len(data["stacks"]) if data["stacks"] else 0
        profiles[oid] = {
            "hands_seen": data["seen"],
            "win_rate_vs_us": round(data["won_vs_us"] / max(1, data["seen"]), 3),
            "avg_stack": round(avg_stack, 1),
            "label": _classify_opponent(data),
        }
    return profiles


def _classify_opponent(data: dict) -> str:
    win_rate = data["won_vs_us"] / max(1, data["seen"])
    if win_rate > 0.45:
        return "DANGEROUS"
    if win_rate > 0.30:
        return "AVERAGE"
    return "WEAK"


# ══════════════════════════════════════════════════════════════════════
#  REGRET ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def build_regret_table(decisions: list[dict], outcomes: list[dict]) -> dict:
    """Find hands where we made -EV decisions."""
    outcome_map = {str(o.get("hand_number", "")): o for o in outcomes}

    bad_folds, missed_value, bad_raises, good_plays = [], [], [], []

    for d in decisions:
        action = d.get("action_taken", "")
        ehs    = float(d.get("ehs", 0))
        hn     = str(d.get("hand_number", ""))
        mode   = d.get("strategy_mode", "")
        amt    = float(d.get("amount", 0))
        outcome = outcome_map.get(hn, {})
        delta  = float(outcome.get("chip_delta", 0))
        won    = outcome.get("won")
        hole   = outcome.get("hole_cards", [])
        board  = outcome.get("board_cards", [])
        pot    = float(outcome.get("pot", 0))
        po     = amt / (pot + amt) if (pot + amt) > 0 else 0

        rec = {"hand": hn, "action": action, "ehs": round(ehs, 3),
               "delta": delta, "hole": hole, "board": board,
               "pot": round(pot, 1), "mode": mode}

        if action == "fold":
            if ehs > 0.58 and delta < -10:
                bad_folds.append({**rec, "verdict": f"folded EHS={ehs:.2f} and lost {delta:.0f} chips — may have been wrong"})
            elif ehs > 0.55 and won is None:
                bad_folds.append({**rec, "verdict": f"folded decent hand EHS={ehs:.2f} with unknown outcome"})

        elif action == "check" and ehs > 0.72:
            missed_value.append({**rec, "verdict": f"checked EHS={ehs:.2f} — likely missed value bet"})

        elif action in ("raise", "bet") and ehs < 0.38 and delta < -15:
            bad_raises.append({**rec, "verdict": f"raised EHS={ehs:.2f} and lost {delta:.0f} chips — speculative raise"})

        elif action in ("raise", "bet", "call") and (won or delta > 5):
            good_plays.append({**rec, "verdict": f"played EHS={ehs:.2f} → won {delta:.0f} chips ✓"})

    return {
        "bad_folds":    bad_folds[:12],
        "missed_value": missed_value[:12],
        "bad_raises":   bad_raises[:8],
        "good_plays":   good_plays[:8],
        "counts": {
            "bad_folds":    len(bad_folds),
            "missed_value": len(missed_value),
            "bad_raises":   len(bad_raises),
            "good_plays":   len(good_plays),
        }
    }


# ══════════════════════════════════════════════════════════════════════
#  STATS SUMMARY
# ══════════════════════════════════════════════════════════════════════

def build_stats(decisions: list[dict], outcomes: list[dict]) -> dict:
    total = len(decisions)
    if total == 0:
        return {}

    by_action: dict = defaultdict(list)
    for d in decisions:
        by_action[d.get("action_taken", "")].append(float(d.get("ehs", 0)))

    def avg(lst): return round(sum(lst) / len(lst), 3) if lst else 0
    def pct(lst): return round(len(lst) / total * 100, 1)

    total_chips = sum(float(o.get("chip_delta", 0)) for o in outcomes)
    wins = sum(1 for o in outcomes if o.get("won"))
    bb_list = [float(o.get("bb_delta", 0)) for o in outcomes]
    bb100 = round((sum(bb_list) / max(1, len(bb_list))) * 100, 1) if bb_list else 0

    return {
        "total_decisions": total,
        "total_hands": len(outcomes),
        "win_rate": round(wins / max(1, len(outcomes)), 3),
        "bb_per_100": bb100,
        "chip_delta": round(total_chips, 1),
        "action_pcts": {
            "fold":  pct(by_action["fold"]),
            "call":  pct(by_action["call"]),
            "check": pct(by_action["check"]),
            "raise": pct(by_action.get("raise", []) + by_action.get("bet", [])),
        },
        "avg_ehs_by_action": {
            "fold":  avg(by_action["fold"]),
            "call":  avg(by_action["call"]),
            "check": avg(by_action["check"]),
            "raise": avg(by_action.get("raise", []) + by_action.get("bet", [])),
        },
    }


# ══════════════════════════════════════════════════════════════════════
#  DEEPSEEK — STAGE 1: RESEARCH + PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════

STAGE1_PROMPT = """
You are a professional poker analyst and prompt engineer.
Your job: analyse the data below, then write a PRECISE IMPLEMENTATION PROMPT for Claude to update strategy_rules.json.

═══ CURRENT RULES (version {version}) ═══
{rules}

═══ PERFORMANCE STATS ═══
{stats}

═══ OPPONENT PROFILES ═══
{opponents}

═══ REGRET TABLE (our mistakes) ═══
{regrets}

═══ SAMPLE DECISIONS (last 40) ═══
{sample}

YOUR ANALYSIS TASKS:

1. OPPONENT RESEARCH:
   For each opponent profile, determine:
   - Are they calling stations (we should value-bet thinner, never bluff them)?
   - Are they nits (we should steal more, bluff rivers)?
   - Are they aggressive (we should trap, check-raise)?
   State exactly how our strategy should adjust vs each opponent type.

2. MISTAKE ANALYSIS:
   For each entry in bad_folds, missed_value, bad_raises:
   - Cite the specific hand, EHS, and outcome
   - Determine if it was a rule error (threshold wrong) or situational (correct given context)
   - If rule error: state exactly which rule field to change and by how much

3. WHAT WORKED:
   From good_plays, identify what patterns are correctly calibrated and must NOT be changed.

4. GENERATE THE CLAUDE PROMPT:
   Write a precise implementation prompt that Claude will execute to update strategy_rules.json.
   The prompt MUST:
   - List every specific field change with old_value → new_value
   - Cite the evidence from regret_table for each change
   - Include the opponent behavior adjustments as new fields if needed
   - State which rules MUST NOT change and why
   - Specify the opponent_adjustments block to add/update in rules
   - Be machine-executable (Claude should be able to implement it without ambiguity)

Return ONLY valid JSON:
{{
  "version": {version},
  "opponent_research": {{
    "calling_stations": ["id1", "id2"],
    "nits": ["id3"],
    "dangerous": ["id4"],
    "adjustments": "one paragraph on how to exploit each type"
  }},
  "mistake_analysis": [
    {{"hand": "hn", "ehs": 0.0, "error": "rule|situational", "field": "flop.facing_bet_fold_max_ehs", "old": 0.52, "new": 0.48, "evidence": "..."}},
  ],
  "rules_must_not_change": ["field1 — reason", "field2 — reason"],
  "claude_implementation_prompt": "FULL PROMPT TEXT HERE — everything Claude needs to update strategy_rules.json perfectly, including all field changes, opponent adjustments, and validation checks",
  "predicted_bb100_improvement": "+X to +Y BB/100",
  "convergence": false
}}

Set convergence=true only if the mistake_analysis is empty AND good_plays show consistent profitable play.
"""


def run_deepseek_research(version: int, stats: dict, opponents: dict,
                          regrets: dict, decisions: list[dict]) -> dict:
    """Stage 1: DeepSeek does full research + builds Claude prompt."""
    rules = load_rules()
    sample = [
        {"hn": d.get("hand_number"), "a": d.get("action_taken"),
         "ehs": round(float(d.get("ehs", 0)), 3), "mode": d.get("strategy_mode", "")}
        for d in decisions[-40:]
    ]

    prompt = STAGE1_PROMPT.format(
        version=version,
        rules=json.dumps(rules, indent=2),
        stats=json.dumps(stats, indent=2),
        opponents=json.dumps(opponents, indent=2),
        regrets=json.dumps(regrets, indent=2),
        sample=json.dumps(sample, indent=2),
    )

    log("DeepSeek Stage 1: opponent research + mistake analysis + prompt generation...")
    raw = ds(prompt, model="deepseek-reasoner", no_cache=True, timeout=180)
    clean = raw.strip()
    for fence in ("```json", "```"):
        if clean.startswith(fence):
            clean = clean[len(fence):]
    if clean.endswith("```"):
        clean = clean[:-3]
    try:
        return json.loads(clean.strip())
    except Exception as e:
        log(f"DeepSeek Stage 1 parse failed: {e}", "ERROR")
        log(f"Raw (500 chars): {raw[:500]}", "ERROR")
        return {}


# ══════════════════════════════════════════════════════════════════════
#  CLAUDE IMPLEMENTATION (reads DeepSeek's prompt → updates rules)
# ══════════════════════════════════════════════════════════════════════

CLAUDE_EXEC_PROMPT = """
You are implementing strategy rule changes for a poker bot. Execute the following precisely.

{deepseek_prompt}

CURRENT strategy_rules.json:
{current_rules}

VALIDATION CONSTRAINTS (enforce strictly):
- Never change any numeric field by more than 0.07 per cycle
- Never set bluff_enabled=true unless opponent fold data justifies it (>60% fold rate observed)
- Every change must have evidence from the mistake_analysis
- Keep the full JSON structure intact

OUTPUT: Return ONLY valid JSON that is the complete updated strategy_rules.json.
No explanations, no markdown, just the JSON object.
"""


def claude_implements_rules(deepseek_result: dict, version: int) -> dict:
    """Use DeepSeek's generated prompt to update rules (via DeepSeek-chat for speed)."""
    claude_prompt = deepseek_result.get("claude_implementation_prompt", "")
    if not claude_prompt:
        log("No claude_implementation_prompt in DeepSeek output", "WARN")
        return {}

    current = load_rules()
    prompt = CLAUDE_EXEC_PROMPT.format(
        deepseek_prompt=claude_prompt,
        current_rules=json.dumps(current, indent=2),
    )

    log("Executing Claude implementation prompt (via DeepSeek-chat)...")
    raw = ds(prompt, model="deepseek-chat", no_cache=True, timeout=90)
    clean = raw.strip()
    for fence in ("```json", "```"):
        if clean.startswith(fence):
            clean = clean[len(fence):]
    if clean.endswith("```"):
        clean = clean[:-3]
    try:
        return json.loads(clean.strip())
    except Exception as e:
        log(f"Rule implementation parse failed: {e}", "ERROR")
        return {}


# ══════════════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════════════

def merge_pending_fixes(rules: dict) -> dict:
    """Apply any pending fixes from strategy_rules_pending_v3.json if present."""
    pending_file = ROOT / "strategy_rules_pending_v3.json"
    if not pending_file.exists():
        return rules
    try:
        pending = json.loads(pending_file.read_text())
        fixes = pending.get("priority_fixes", [])
        if not fixes:
            return rules
        import copy
        rules = copy.deepcopy(rules)
        for fix in fixes:
            field = fix.get("field", "")
            new_val = fix.get("new")
            if not field or new_val is None:
                continue
            parts = field.split(".")
            if len(parts) == 2:
                section, key = parts
                if section in rules and key in rules[section]:
                    old_val = rules[section][key]
                    delta = abs(new_val - old_val)
                    if delta <= 0.09:
                        rules[section][key] = new_val
                        log(f"[pending fix] {field}: {old_val} → {new_val}  ({fix.get('ev_reason','')[:60]})")
        pending_file.unlink()
        log("Pending v3 fixes applied and cleared.")
    except Exception as e:
        log(f"Pending fix merge failed: {e}", "WARN")
    return rules


def validate_rules(old: dict, new: dict) -> tuple[bool, list[str]]:
    MAX_DELTA = 0.08
    errors = []
    for section in ["preflop", "flop", "turn", "river"]:
        for k, nv in new.get(section, {}).items():
            if not isinstance(nv, (int, float)):
                continue
            ov = old.get(section, {}).get(k, nv)
            if abs(nv - ov) > MAX_DELTA:
                errors.append(f"{section}.{k}: {ov:.3f}→{nv:.3f} exceeds {MAX_DELTA} delta")
        if new.get(section, {}).get("bluff_enabled") and not old.get(section, {}).get("bluff_enabled"):
            errors.append(f"{section}: bluff_enabled added without evidence")
    return len(errors) == 0, errors


# ══════════════════════════════════════════════════════════════════════
#  PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════

def print_banner(version: int) -> None:
    print(f"\n{'═'*64}", flush=True)
    print(f"  CYCLE {version}  →  Plutus-v{version}  |  {time.strftime('%H:%M:%S')}", flush=True)
    print(f"{'═'*64}\n", flush=True)


def print_research(result: dict, version: int) -> None:
    print(f"\n{'─'*64}", flush=True)
    print(f" DeepSeek Research — v{version}", flush=True)
    print(f"{'─'*64}", flush=True)

    opp = result.get("opponent_research", {})
    print(f" Calling stations : {opp.get('calling_stations', [])}", flush=True)
    print(f" Nits             : {opp.get('nits', [])}", flush=True)
    print(f" Dangerous        : {opp.get('dangerous', [])}", flush=True)
    print(f" Opp adjustment   : {opp.get('adjustments','')[:120]}", flush=True)

    mistakes = result.get("mistake_analysis", [])
    print(f"\n Mistakes found   : {len(mistakes)}", flush=True)
    for m in mistakes[:6]:
        chg = f"{m.get('field','')}  {m.get('old','?')} → {m.get('new','?')}"
        print(f"   [{m.get('error','?'):12s}] {chg}", flush=True)
        print(f"              Evidence: {str(m.get('evidence',''))[:80]}", flush=True)

    no_change = result.get("rules_must_not_change", [])
    if no_change:
        print(f"\n Must NOT change  : {len(no_change)} fields", flush=True)
        for nc in no_change[:3]:
            print(f"   • {nc[:80]}", flush=True)

    print(f"\n Predicted lift   : {result.get('predicted_bb100_improvement','?')}", flush=True)
    print(f" Convergence      : {result.get('convergence', False)}", flush=True)
    print(f"{'─'*64}\n", flush=True)


def print_rule_diff(old: dict, new: dict) -> None:
    changed = []
    for section in ["preflop", "flop", "turn", "river"]:
        for k, nv in new.get(section, {}).items():
            ov = old.get(section, {}).get(k)
            if isinstance(nv, (int, float)) and ov is not None and abs(nv - ov) > 0.001:
                changed.append(f"  {section}.{k:40s} {ov:.3f} → {nv:.3f}")
    if changed:
        print(f"\n Rule changes ({len(changed)}):", flush=True)
        for c in changed:
            print(c, flush=True)
    else:
        print("\n No numeric rule changes.", flush=True)


# ══════════════════════════════════════════════════════════════════════
#  AGENT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

def agent_dir(version: int) -> Path:
    return AGENTS / f"v{version}"


def run_log(version: int) -> Path:
    return agent_dir(version) / "run.log"


def register_agent(version: int) -> bool:
    vdir = agent_dir(version)
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / ".env").write_text(
        f"ARENA_COMPETITION_ID=seed_poker_eval_s1\n"
        f"ARENA_BASE_URL=https://arena.dev.fun\n"
        f"ARENA_STRATEGY=ADAPTIVE\n"
        f"AGENT_NAME=Plutus-v{version}\n"
        f"LOG_LEVEL=INFO\nJSON_LOGS=false\n"
        f"HEALTH_PORT={8089 + version}\n"
        f"USE_ASYNC=true\nDECISION_BUDGET_S=1.5\n"
        f"POLL_INTERVAL_S=0.8\nHEARTBEAT_MIN_INTERVAL_S=3600\n"
        f"HEARTBEAT_INTERVAL_S=14400\n"
        f"OWNER_MESSAGE_FILE=.arena-owner-messages.txt\n"
        f"DEEPSEEK_API_KEY={DS_KEY}\n"
    )
    log(f"Registering Plutus-v{version}...")
    r = subprocess.run(
        [str(PYTHON), "-m", "agent.runner",
         "--register", f"Plutus-v{version}", f"v{version} — adaptive, learns every cycle."],
        cwd=str(vdir),
        env={**os.environ, "ARENA_STRATEGY": "ADAPTIVE",
             "AGENT_NAME": f"Plutus-v{version}", "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True, timeout=30,
    )
    out = r.stdout + r.stderr
    if "Registered" in out or "arena_sk_" in out:
        log(f"v{version} registered ✅")
        return True
    log(f"Registration output: {out[:200]}", "WARN")
    return False


def launch_agent(version: int) -> subprocess.Popen:
    vdir = agent_dir(version)
    lf = open(run_log(version), "w")
    log(f"Launching Plutus-v{version}...")
    return subprocess.Popen(
        [str(PYTHON), "-m", "agent.runner"],
        cwd=str(vdir),
        env={**os.environ, "ARENA_STRATEGY": "ADAPTIVE",
             "AGENT_NAME": f"Plutus-v{version}", "PYTHONPATH": str(ROOT)},
        stdout=lf, stderr=lf,
    )


def wait_for_completion(version: int, proc: subprocess.Popen) -> int:
    lpath = run_log(version)
    while proc.poll() is None:
        time.sleep(30)
        n = sum(1 for l in lpath.read_text(errors="replace").splitlines()
                if '"event": "decision"' in l)
        log(f"v{version}: {n} decisions logged...")
    n = sum(1 for l in lpath.read_text(errors="replace").splitlines()
            if '"event": "decision"' in l)
    log(f"v{version} finished. Decisions: {n}")
    return n


def wait_for_v1_live() -> int:
    """Poll v1 run.log until 490+ decisions or process finishes."""
    lpath = run_log(1)
    log("Watching v1 (already running in tmux)...")
    while True:
        n = sum(1 for l in lpath.read_text(errors="replace").splitlines()
                if '"event": "decision"' in l)
        if n >= 490:
            log(f"v1 complete: {n} decisions")
            return n
        if "=== v1 done ===" in lpath.read_text(errors="replace"):
            log(f"v1 process exited: {n} decisions")
            return n
        log(f"v1: {n}/500 — waiting 30s...")
        time.sleep(30)


# ══════════════════════════════════════════════════════════════════════
#  MAIN CYCLE
# ══════════════════════════════════════════════════════════════════════

def run_cycle(start_version: int = 1, max_cycles: int = 10) -> None:
    version = start_version
    consecutive_converged = 0

    while version <= start_version + max_cycles:
        print_banner(version)
        lpath = run_log(version)

        # ── Wait for agent to finish playing ──────────────────────────
        if version == 1 and lpath.exists():
            n = wait_for_v1_live()
        else:
            if not register_agent(version):
                log(f"Registration failed for v{version}. Stopping.", "ERROR")
                break
            proc = launch_agent(version)
            n = wait_for_completion(version, proc)

        if n < 20:
            log(f"Only {n} decisions — insufficient for analysis.", "ERROR")
            break

        # ── Parse logs ─────────────────────────────────────────────────
        log("Parsing decisions and hand outcomes...")
        decisions, outcomes = parse_log(lpath)
        stats     = build_stats(decisions, outcomes)
        opponents = build_opponent_profiles(decisions, outcomes)
        regrets   = build_regret_table(decisions, outcomes)

        log(f"Stats: {n} decisions | {len(outcomes)} hand outcomes | "
            f"bb/100={stats.get('bb_per_100','?')} | "
            f"win_rate={stats.get('win_rate','?')}")
        log(f"Regrets: {regrets['counts']['bad_folds']} bad folds | "
            f"{regrets['counts']['missed_value']} missed value | "
            f"{regrets['counts']['bad_raises']} bad raises")
        log(f"Opponents: {len(opponents)} profiled")

        # ── DeepSeek: research + build Claude prompt ───────────────────
        ds_result = run_deepseek_research(version, stats, opponents, regrets, decisions)
        if not ds_result:
            log("DeepSeek returned no result. Skipping cycle.", "WARN")
            version += 1
            continue

        print_research(ds_result, version)

        # ── Convergence check ──────────────────────────────────────────
        if ds_result.get("convergence"):
            consecutive_converged += 1
            log(f"Convergence signal ({consecutive_converged}/2)")
            if consecutive_converged >= 2:
                print(f"\n{'═'*64}", flush=True)
                print(f"  ✅  CONVERGED after v{version}  —  optimal strategy reached.", flush=True)
                print(f"{'═'*64}\n", flush=True)
                break
        else:
            consecutive_converged = 0

        # ── Claude implements DeepSeek's prompt ────────────────────────
        old_rules = load_rules()
        new_rules = claude_implements_rules(ds_result, version)

        if not new_rules:
            log("Rule implementation failed. Keeping current rules.", "WARN")
        else:
            ok, errs = validate_rules(old_rules, new_rules)
            if not ok:
                log(f"Validation failed ({len(errs)} issues):", "WARN")
                for e in errs:
                    log(f"  ✗ {e}", "WARN")
            else:
                # Backup + save
                (ROOT / f"strategy_rules_v{version}.json").write_text(json.dumps(old_rules, indent=2))
                new_rules = merge_pending_fixes(new_rules)
                new_rules["cycle"] = version + 1
                new_rules["cycle_notes"] = (
                    f"v{version}→v{version+1}: {len(ds_result.get('mistake_analysis',[]))} fixes. "
                    + str(ds_result.get("predicted_bb100_improvement", ""))
                )
                save_rules(new_rules)
                print_rule_diff(old_rules, new_rules)
                log(f"Rules updated → v{version+1} rules ready")

        # ── Summary ────────────────────────────────────────────────────
        print(f"\n{'─'*64}", flush=True)
        print(f" v{version} summary", flush=True)
        print(f"   BB/100      : {stats.get('bb_per_100','?')}", flush=True)
        print(f"   Win rate    : {stats.get('win_rate','?')}", flush=True)
        print(f"   Chip delta  : {stats.get('chip_delta','?')}", flush=True)
        print(f"   Opp types   : {dict((k,v['label']) for k,v in opponents.items())}", flush=True)
        print(f"   Next version: v{version+1}", flush=True)
        print(f"{'─'*64}\n", flush=True)

        version += 1
        time.sleep(5)

    log("Cycle engine stopped.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start",  type=int, default=1)
    p.add_argument("--cycles", type=int, default=10)
    args = p.parse_args()
    run_cycle(start_version=args.start, max_cycles=args.cycles)
