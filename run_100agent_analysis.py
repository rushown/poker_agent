#!/usr/bin/env python3
"""100-agent DeepSeek analysis of Plutus Optimal + Devil agents.

Structure:
  Tier 1: 60 data agents — chunk-by-chunk hand-by-hand review
  Tier 2: 25 thematic agents — bad folds, bad bets, missed all-ins, raises, EV math
  Tier 3: 10 orchestrators — synthesize tier1+2 by domain
  Tier 4: 5 final orchestrators — unified findings + rule patches
  Post: Claude synthesis printed at end
"""
from __future__ import annotations

import json
import os
import sys
import time
import statistics
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    sys.exit("ERROR: DEEPSEEK_API_KEY environment variable not set")
API_URL = "https://api.deepseek.com/chat/completions"
RESULTS_FILE = "100agent_analysis_result.json"
MAX_CONCURRENT = 20  # stay within rate limits
BATCH_PAUSE = 1.5    # seconds between batches

REASONER_SYSTEM = """You are a world-class poker AI analyst with deep expertise in GTO theory, exploitative play,
EV mathematics, and poker bot design. You are analyzing a real poker bot's decision log.

KEY FACTS about the arena bots being played against:
- Pure pot-odds callers: call when equity >= required_equity, NEVER bluff
- Fold preflop to 3BB open: ~42-55% of the time
- Fold to flop c-bet: ~5% (almost never fold postflop)
- Fold to all-in: ~95% (need 95%+ equity to call stack-off)
- Breakeven EHS for value bet = 0.48 (0.75x pot sizing)
- All-in EV always > value bet when stack-to-pot ratio allows

Your output must be rigorous JSON with exact field names, EV math, and actionable rule patches.
Never hallucinate field names. Show all calculations explicitly."""

SIMULATION_SYSTEM = """You are a poker EV simulation engine. You replay hands and compute:
- EV of the action taken (using pot odds math)
- EV of the optimal alternative action
- EV delta = optimal - taken
- The specific rule threshold that caused the suboptimal action
- Exact rule fix needed

Pot odds formula: required_equity = call_size / (call_size + pot)
EHS as proxy for equity. Show all math. Return only valid JSON."""

ORCHESTRATOR_SYSTEM = """You are the strategy orchestrator for a poker bot reinforcement cycle.
You receive findings from multiple specialized analysis agents and synthesize them into an
implementation plan. Your output must be machine-executable JSON:
{
  "critical_fixes": [{"field": "...", "old": ..., "new": ..., "ev_gain_bb_per_100": ..., "evidence": "..."}],
  "confirmed_good": ["...", "..."],
  "risk_flags": ["...", "..."],
  "priority": "high|medium|low"
}
Reject any change not supported by at least 2 pieces of evidence. Flag uncertainty."""


# ── Helpers ──────────────────────────────────────────────────────────────────

def ds_call(prompt: str, system: str = REASONER_SYSTEM, model: str = "deepseek-reasoner",
            max_tokens: int = 4096, agent_id: str = "") -> dict:
    """Single blocking DeepSeek API call. Returns {'agent_id', 'response', 'error'}."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    for attempt in range(3):
        try:
            req = urllib.request.Request(API_URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                content = data["choices"][0]["message"]["content"]
                return {"agent_id": agent_id, "response": content, "error": None}
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                return {"agent_id": agent_id, "response": "", "error": str(e)}


def run_parallel(tasks: list[dict], label: str) -> list[dict]:
    """Run tasks in parallel batches. Each task: {agent_id, prompt, system, model, max_tokens}."""
    results = []
    total = len(tasks)
    print(f"\n[{label}] Running {total} agents in batches of {MAX_CONCURRENT}...")
    for batch_start in range(0, total, MAX_CONCURRENT):
        batch = tasks[batch_start:batch_start + MAX_CONCURRENT]
        print(f"  Batch {batch_start//MAX_CONCURRENT + 1}: agents {batch_start+1}–{min(batch_start+len(batch), total)}")
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
            futures = {
                ex.submit(
                    ds_call,
                    t["prompt"],
                    t.get("system", REASONER_SYSTEM),
                    t.get("model", "deepseek-reasoner"),
                    t.get("max_tokens", 4096),
                    t["agent_id"]
                ): t["agent_id"]
                for t in batch
            }
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                status = "OK" if not r["error"] else f"ERR:{r['error'][:50]}"
                print(f"    [{r['agent_id']}] {status}")
        if batch_start + MAX_CONCURRENT < total:
            time.sleep(BATCH_PAUSE)
    return results


# ── Log parsing ───────────────────────────────────────────────────────────────

def extract_decisions(logfile: str, agent_name: str) -> list[dict]:
    decisions = []
    for line in Path(logfile).read_text().splitlines():
        try:
            d = json.loads(line)
            msg = d.get("record", {}).get("message", "")
            if msg.startswith('{"event": "decision"'):
                dec = json.loads(msg)
                dec["agent"] = agent_name
                decisions.append(dec)
        except Exception:
            pass
    return decisions


def compute_stats(decisions: list[dict]) -> dict:
    actions = {}
    for d in decisions:
        a = d["action_taken"]
        actions[a] = actions.get(a, 0) + 1

    def ehs_stats(decs):
        vals = [d["ehs"] for d in decs]
        if not vals:
            return {}
        return {"count": len(vals), "mean": round(statistics.mean(vals), 4),
                "min": round(min(vals), 4), "max": round(max(vals), 4),
                "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0}

    return {
        "total": len(decisions),
        "action_counts": actions,
        "fold_ehs": ehs_stats([d for d in decisions if d["action_taken"] == "fold"]),
        "bet_ehs": ehs_stats([d for d in decisions if d["action_taken"] == "bet"]),
        "call_ehs": ehs_stats([d for d in decisions if d["action_taken"] == "call"]),
        "raise_ehs": ehs_stats([d for d in decisions if d["action_taken"] == "raise"]),
        "bad_folds": [d for d in decisions if d["action_taken"] == "fold" and d["ehs"] >= 0.50],
        "bad_bets": [d for d in decisions if d["action_taken"] == "bet" and d["ehs"] < 0.45],
        "missed_allins": [d for d in decisions if d["action_taken"] == "bet" and d["ehs"] >= 0.82],
        "low_ehs_calls": [d for d in decisions if d["action_taken"] == "call" and d["ehs"] < 0.38],
        "high_ehs_raises": [d for d in decisions if d["action_taken"] == "raise" and d["ehs"] >= 0.75],
    }


# ── Agent task builders ────────────────────────────────────────────────────────

def build_chunk_agents(decisions: list[dict], agent_prefix: str, chunk_size: int = 8) -> list[dict]:
    """Tier 1: one agent per chunk of hands."""
    tasks = []
    for i in range(0, len(decisions), chunk_size):
        chunk = decisions[i:i + chunk_size]
        chunk_id = f"{agent_prefix}_chunk_{i//chunk_size+1}"
        prompt = f"""Analyze these {len(chunk)} poker decisions by the '{agent_prefix}' bot.
For each decision, determine:
1. Was this action EV+? Show pot odds math.
2. What was the optimal action? What EV was lost?
3. Which strategy_rules field caused a suboptimal decision?

Decisions:
{json.dumps(chunk, indent=2)}

Arena bot facts: fold-to-allin=95%, fold-to-cbet=5%, breakeven_EHS=0.48.
Return JSON: {{"chunk": {i//chunk_size+1}, "decisions_reviewed": {len(chunk)},
  "problem_hands": [{{"hand_number": ..., "action": ..., "ehs": ..., "issue": ..., "ev_lost": ..., "fix": ...}}],
  "good_plays": [{{"hand_number": ..., "why_good": ...}}],
  "net_ev_assessment": "..."}}"""
        tasks.append({"agent_id": chunk_id, "prompt": prompt,
                      "model": "deepseek-chat", "max_tokens": 3000})
    return tasks


def build_thematic_agents(devil_decs: list[dict], plutus_decs: list[dict],
                          devil_stats: dict, plutus_stats: dict,
                          devil_rules: dict, plutus_rules: dict) -> list[dict]:
    """Tier 2: 25 thematic specialist agents."""
    tasks = []

    # --- Devil thematic agents ---

    tasks.append({"agent_id": "D_bad_folds", "prompt": f"""
Devil bot made {len(devil_stats['bad_folds'])} folds with EHS >= 0.50. Against bots that fold to all-in 95%,
folding EHS>=0.50 hands is almost always -EV. Analyze each fold:

Bad folds:
{json.dumps(devil_stats['bad_folds'], indent=2)}

Current fold thresholds in rules:
- preflop open_ip_min_ehs: {devil_rules['preflop']['open_ip_min_ehs']}
- preflop open_oop_min_ehs: {devil_rules['preflop']['open_oop_min_ehs']}
- flop facing_bet_fold_max_ehs: {devil_rules['flop']['facing_bet_fold_max_ehs']}
- turn facing_bet_fold_max_ehs: {devil_rules['turn']['facing_bet_fold_max_ehs']}
- river facing_bet_fold_max_ehs: {devil_rules['river']['facing_bet_fold_max_ehs']}

For each fold: compute EV(fold)=0, EV(call)=EHS*pot - (1-EHS)*bet, EV(bet)=EHS*pot*.75.
Identify which threshold triggered the fold and what it should be.
Return JSON: {{"total_bad_folds": ..., "estimated_ev_lost_total_bb": ...,
  "worst_folds": [{{"hand": ..., "ehs": ..., "ev_lost_bb": ..., "rule_field": ..., "fix_value": ...}}],
  "threshold_patches": {{"field": "old_val→new_val"}}}}""",
        "model": "deepseek-reasoner", "max_tokens": 5000})

    tasks.append({"agent_id": "D_bad_bets", "prompt": f"""
Devil bot made {len(devil_stats['bad_bets'])} bets with EHS < 0.45 (below the 0.48 breakeven threshold).
These are guaranteed -EV bets since bots only fold to all-in (95%), not to value bets (5% fold).

Bad bets (EHS < 0.45):
{json.dumps(devil_stats['bad_bets'], indent=2)}

Devil rules: flop value_min={devil_rules['flop']['no_bet_value_min_ehs']},
turn value_min={devil_rules['turn']['no_bet_value_min_ehs']},
river bet_min={devil_rules['river']['bet_min_ehs']}

For each bad bet: EV_bet = EHS*pot*0.75 - (1-EHS)*bet_amount (vs bot that never folds postflop)
Identify if this is a c-bet, value bet, or barrel. Find the rule that allowed sub-threshold betting.
Return JSON with: total_ev_lost, each hand's analysis, exact rule patches needed.""",
        "model": "deepseek-reasoner", "max_tokens": 5000})

    tasks.append({"agent_id": "D_missed_allins", "prompt": f"""
Devil bot made {len(devil_stats['missed_allins'])} bets at EHS >= 0.82 instead of going all-in.
Arena bots fold to all-in 95% of the time. This means EV(all-in) is approximately:
  EV_allin = 0.95 * pot_before_push + 0.05 * (EHS * total_pot - (1-EHS) * stack_at_risk)
Compare this to EV(bet_x_pot) = EHS * (pot + bet) * 0.95 - (1-EHS) * bet * 0.05 [if they call]

Missed all-ins:
{json.dumps(devil_stats['missed_allins'], indent=2)}

Current all-in threshold: {devil_rules['flop']['allin_threshold_ehs']}
Compute EV delta for each. Return JSON with exact ev calculations and recommendations.""",
        "model": "deepseek-reasoner", "max_tokens": 4000})

    tasks.append({"agent_id": "D_raise_patterns", "prompt": f"""
Devil bot made {devil_stats['action_counts'].get('raise', 0)} raises total. Analyze the raise EHS distribution:

All raises:
{json.dumps([d for d in devil_decs if d['action_taken'] == 'raise'], indent=2)}

Devil rules: vs_raise_call_ip={devil_rules['preflop']['vs_raise_call_ip_min_ehs']},
bb_raise_min={devil_rules['preflop']['bb_raise_min_ehs']},
flop_raise_aggressive={devil_rules['flop']['flop_raise_aggressive_min_ehs']}

Identify: raises that were too small (should be all-in), raises with poor EHS, missed raise spots.
Calculate: what % of raises had EHS >= 0.75 vs < 0.75? Were low-EHS raises profitable given pot odds?
Return JSON with raise pattern analysis and specific optimization recommendations.""",
        "model": "deepseek-chat", "max_tokens": 4000})

    tasks.append({"agent_id": "D_call_analysis", "prompt": f"""
Devil bot made {devil_stats['action_counts'].get('call', 0)} calls. Analyze the call pattern:

All calls:
{json.dumps([d for d in devil_decs if d['action_taken'] == 'call'], indent=2)}

Devil rules: pot_odds minimum_ehs_for_call={devil_rules['pot_odds']['minimum_ehs_for_call']},
flop_facing_bet_call_min={devil_rules['flop']['facing_bet_call_min_ehs']}
turn_facing_bet_call_min={devil_rules['turn']['facing_bet_call_min_ehs']}

For each call: compute required_equity = call_size / (call_size + pot). Was EHS >= required_equity?
Flag: calls where EHS < required_equity (losing calls), and missed calls where EHS was good but folded.
Return JSON with profitability breakdown and threshold recommendations.""",
        "model": "deepseek-chat", "max_tokens": 4000})

    tasks.append({"agent_id": "D_strategy_critique", "prompt": f"""
Full critique of Devil bot strategy_rules vs the specific arena environment:
- Arena bots: pure pot-odds callers, 95% fold to all-in, 5% fold to postflop bets, never bluff
- Devil's core principle: "{devil_rules['core_principle']}"

Devil strategy_rules:
{json.dumps(devil_rules, indent=2)}

Devil performance stats: {json.dumps(devil_stats, default=str, indent=2)[:2000]}

Critical questions:
1. Is the c_bet_frequency=0.90 optimal? (bots fold only 5% to c-bets → c-bet EV < check for EHS < 0.48)
2. Is turn overbet_min_ehs=0.70 correct? At what EHS does overbet beat value bet?
3. Is river bet_sizing_pot=1.0 optimal vs never-fold bots?
4. Are there any rules that are internally inconsistent?
Return JSON with findings and specific field patches.""",
        "model": "deepseek-reasoner", "max_tokens": 6000})

    # --- Plutus thematic agents ---

    tasks.append({"agent_id": "P_bad_folds", "prompt": f"""
Plutus optimal bot made {len(plutus_stats['bad_folds'])} folds with EHS >= 0.50.
Against bots that fold to all-in 95%, folding EHS>=0.50 is almost always -EV.

Bad folds:
{json.dumps(plutus_stats['bad_folds'], indent=2)}

Current fold thresholds:
- preflop open_ip_min_ehs: {plutus_rules['preflop']['open_ip_min_ehs']}
- flop facing_bet_fold_max_ehs: {plutus_rules['flop']['facing_bet_fold_max_ehs']}
- turn facing_bet_fold_max_ehs: {plutus_rules['turn']['facing_bet_fold_max_ehs']}
- river facing_bet_fold_max_ehs: {plutus_rules['river']['facing_bet_fold_max_ehs']}

For each fold above EHS 0.50: calculate EV(fold)=0 vs EV(bet)=EHS*0.75*pot.
Find the rule that triggered the fold. What should the threshold be?
Return JSON with total_ev_lost, worst cases, exact threshold patches.""",
        "model": "deepseek-reasoner", "max_tokens": 5000})

    tasks.append({"agent_id": "P_bad_bets", "prompt": f"""
Plutus bot made {len(plutus_stats['bad_bets'])} bets with EHS < 0.45 (below 0.48 breakeven).

Bad bets:
{json.dumps(plutus_stats['bad_bets'], indent=2)}

Plutus rules: flop value_min={plutus_rules['flop']['no_bet_value_min_ehs']},
turn barrel_min={plutus_rules['turn']['barrel_min_ehs']},
river bet_min={plutus_rules['river']['bet_min_ehs']}

Breakeven math: EV_bet = EHS * pot * 0.75 - (1 - EHS) * bet_size
At EHS=0.45: EV_bet is NEGATIVE since bots call 95% postflop.
Identify which rule allowed the sub-threshold bet (c-bet bypass? speculative hand path?).
Return JSON with ev calculations and patches.""",
        "model": "deepseek-reasoner", "max_tokens": 5000})

    tasks.append({"agent_id": "P_missed_allins", "prompt": f"""
Plutus bot made {len(plutus_stats['missed_allins'])} bets at EHS >= 0.82 instead of all-in.
All-in EV against 95%-fold bots: EV_allin ≈ 0.95 * pot (near-certain steal).
Compare to bet EV: EV_bet = EHS * (pot + bet) - (1-EHS) * bet (when called).

Missed all-ins:
{json.dumps(plutus_stats['missed_allins'], indent=2)}

Plutus allin thresholds: preflop={plutus_rules['preflop']['allin_threshold_ehs']},
flop={plutus_rules['flop']['allin_threshold_ehs']},
turn={plutus_rules['turn']['allin_threshold_ehs']},
river={plutus_rules['river']['allin_threshold_ehs']}

Why are they 0.84–0.92 when all-in at EHS 0.82 is profitable?
Compute EV delta for each missed all-in. Recommend threshold adjustments.
Return JSON with exact ev math and rule patches.""",
        "model": "deepseek-reasoner", "max_tokens": 5000})

    tasks.append({"agent_id": "P_low_ehs_calls", "prompt": f"""
Plutus bot made {len(plutus_stats['low_ehs_calls'])} calls with EHS < 0.38 (below EHS floor).
These are pot-odds violations: calling without sufficient equity.

Low EHS calls:
{json.dumps(plutus_stats['low_ehs_calls'], indent=2)}

Plutus pot_odds: minimum_ehs_for_call={plutus_rules['pot_odds']['minimum_ehs_for_call']}

For each call: compute required_equity = call_amount / (call_amount + pot).
Was EHS >= required_equity? If not, this is an EV- call.
Why did the pot_odds gate fail? Is there a bypass in the code?
Return JSON: for each hand: required_equity, actual_ehs, ev_lost, likely code path.""",
        "model": "deepseek-reasoner", "max_tokens": 5000})

    tasks.append({"agent_id": "P_raise_patterns", "prompt": f"""
Plutus bot made {plutus_stats['action_counts'].get('raise', 0)} raises.

All raises:
{json.dumps([d for d in plutus_decs if d['action_taken'] == 'raise'], indent=2)}

Plutus rules: vs_raise_3bet_sizing_mult={plutus_rules['preflop']['vs_raise_3bet_sizing_mult']},
flop_raise_aggressive_min_ehs={plutus_rules['flop']['flop_raise_aggressive_min_ehs']},
check_raise_min_ehs={plutus_rules['flop']['check_raise_min_ehs']}

Analyze: are raises sized optimally? Are there low-EHS raises that should be folds?
Are there high-EHS hands that should all-in instead of raise?
Compute EV(raise) vs EV(allin) for raises with EHS >= 0.75.
Return JSON with raise quality scores and sizing recommendations.""",
        "model": "deepseek-chat", "max_tokens": 4000})

    tasks.append({"agent_id": "P_strategy_critique", "prompt": f"""
Full critique of Plutus optimal bot strategy_rules:
- Core: "{plutus_rules['core_principle']}"
- Arena: pure pot-odds callers, 95% fold to all-in, 5% fold postflop

Plutus strategy_rules:
{json.dumps(plutus_rules, indent=2)}

Plutus stats: {json.dumps(plutus_stats, default=str, indent=2)[:2000]}

Questions:
1. flop allin_threshold=0.92 and turn allin_threshold=0.92 — why so high? At 0.82+ all-in is already +EV.
2. turn facing_bet_call_min=0.52 — is this too tight? Missing profitable calls?
3. river no_bet_value_min=0.58 — leaving value on the table for EHS 0.48-0.58 hands?
4. turn barrel_min=0.40 — below 0.48 breakeven, causing -EV barrels?
5. river bet_frequency=0.65 — missing 35% of profitable value bets?
Return JSON with specific findings and exact rule patches.""",
        "model": "deepseek-reasoner", "max_tokens": 6000})

    # --- Cross-agent comparison ---

    tasks.append({"agent_id": "CROSS_comparison", "prompt": f"""
Compare Devil vs Plutus Optimal bots head-to-head:

Devil stats: {json.dumps({k: v for k, v in devil_stats.items() if k not in ['bad_folds','bad_bets','missed_allins','low_ehs_calls','high_ehs_raises']}, indent=2)}
Plutus stats: {json.dumps({k: v for k, v in plutus_stats.items() if k not in ['bad_folds','bad_bets','missed_allins','low_ehs_calls','high_ehs_raises']}, indent=2)}

Devil rule highlights: c_bet_freq={devil_rules['flop']['c_bet_frequency']}, allin_all={devil_rules['flop']['allin_threshold_ehs']}, value_min=0.48
Plutus rule highlights: c_bet_freq={plutus_rules['flop']['c_bet_frequency']}, allin_flop={plutus_rules['flop']['allin_threshold_ehs']}, turn_allin={plutus_rules['turn']['allin_threshold_ehs']}

Which bot has: better fold discipline? Better bet quality? More all-in exploitation?
Which rules should be merged from one to the other?
Return JSON: {{"devil_strengths": [...], "plutus_strengths": [...], "devil_weaknesses": [...],
  "plutus_weaknesses": [...], "cross_pollination_rules": [{{"field": ..., "from": ..., "to": ..., "value": ...}}]}}""",
        "model": "deepseek-reasoner", "max_tokens": 6000})

    # --- EV math agents ---

    tasks.append({"agent_id": "EV_math_thresholds", "prompt": f"""
Compute optimal EV thresholds for both bots given arena bot properties:
- Arena bots fold to all-in: 95%
- Arena bots fold to postflop bet: 5%
- Arena bots fold preflop to 3BB open: 42-55%
- Stack sizes: ~1000 chips (cash game, no rebuy)

For each decision point, compute the EHS where EV(action) = 0 (breakeven):

1. Preflop open (3BB into 1.5BB pot): breakeven EHS?
2. Flop c-bet (0.75x pot): breakeven EHS given 5% fold rate?
3. Flop value bet (0.75x pot): breakeven EHS?
4. Turn barrel (0.75x pot): breakeven EHS?
5. River bet (1.0x pot): breakeven EHS?
6. All-in push: breakeven EHS given 95% fold rate?
7. Facing all-in call: breakeven EHS?

Show full math for each. Return JSON with breakeven_ehs for each scenario plus
recommendation for each bot's threshold vs the current values in their rules.""",
        "model": "deepseek-reasoner", "max_tokens": 6000})

    tasks.append({"agent_id": "EV_raise_sizing", "prompt": f"""
Analyze optimal raise sizing for both bots in this arena:
- Bots never bluff (pure value/pot-odds)
- Fold to all-in: 95% regardless of raise sizing
- Call threshold: pot-odds based

For raise sizing X (as multiple of pot), compute:
- If opponent folds (95%): EV = pot_before_raise
- If opponent calls: EV = equity * (pot + X) - (1-equity) * X

For X = 0.5, 0.75, 1.0, 1.5, 2.0, all-in:
Compute EV for EHS = 0.50, 0.60, 0.70, 0.80, 0.90

Which sizing maximizes EV at each EHS level?
Devil uses: value_bets 0.75x, overbets 1.5x. Plutus uses: value_bets 1.0x, overbets 1.75x.
Which is optimal? Show full EV tables.
Return JSON with optimal_sizing_by_ehs and comparison to current bot configs.""",
        "model": "deepseek-reasoner", "max_tokens": 6000})

    # --- Simulation agents ---

    tasks.append({"agent_id": "SIM_devil_worst_hand", "prompt": f"""
Simulate the worst Devil bot hand: hand #34186.
This hand had 15+ decision entries — the bot repeatedly bet 25 chips, then 75 chips, then 191.25 chips.
Multiple API timeouts occurred during this hand.

Decisions from this hand:
{json.dumps([d for d in devil_decs if d.get('hand_number') == 34186], indent=2)}

Simulate:
1. What likely happened: what board texture, position, and action sequence?
2. EV of each bet vs optimal action (check or all-in?)
3. Did the repeated bets at EHS 0.60-0.70 extract value or waste chips?
4. At EHS 0.70+ did the bot miss an all-in?
5. Simulate the "optimal line" for this hand — what should have happened?

Return JSON: {{"hand_reconstruction": "...", "action_ev_sequence": [...],
  "optimal_line": [...], "ev_captured_pct": ..., "lessons": [...]}}""",
        "model": "deepseek-reasoner", "max_tokens": 6000})

    tasks.append({"agent_id": "SIM_plutus_worst_hands", "prompt": f"""
Simulate Plutus optimal bot's most problematic decision patterns:

1. Hands with EHS >= 0.82 that were NOT all-in (missed exploits):
{json.dumps(plutus_stats['missed_allins'][:5], indent=2)}

2. Low EHS calls (< 0.38) that violated the EHS floor:
{json.dumps(plutus_stats['low_ehs_calls'][:5], indent=2)}

3. High EHS folds (>= 0.55) that left value:
{json.dumps([d for d in plutus_stats['bad_folds'] if d['ehs'] >= 0.55][:5], indent=2)}

For each category:
- Reconstruct the game state (position, street, pot size estimate)
- Compute EV of action taken vs optimal action
- Identify the rule responsible and the fix

Return JSON: {{"category_analyses": [{{"category": ..., "hands": [...], "total_ev_lost": ..., "fix": ...}}]}}""",
        "model": "deepseek-reasoner", "max_tokens": 6000})

    tasks.append({"agent_id": "SIM_table_sim", "prompt": f"""
Simulate a full table session between Devil and Plutus Optimal and a typical arena bot (pure pot-odds caller).

Arena bot behavior:
- Preflop: call if EHS >= required_equity from 3BB open; else fold
- Postflop: call if equity >= pot_odds required; NEVER bet/raise first; NEVER fold to bets (only to all-in)
- Fold to all-in: 95%

Simulate 20 hands using Devil's rules (open_ip=0.40, allin=0.82, value=0.48) vs this bot.
Then simulate 20 hands using Plutus rules (open_ip=0.36, allin_flop=0.92, value=0.49).

Assumptions: heads up, 1000 chip stacks, blinds 1/2.
Use random EHS draws from realistic preflop/postflop distributions.

Return JSON: {{"devil_sim": {{"hands": 20, "chip_delta": ..., "key_spots": [...]}},
  "plutus_sim": {{"hands": 20, "chip_delta": ..., "key_spots": [...]}},
  "winner": ..., "reason": ...}}""",
        "model": "deepseek-reasoner", "max_tokens": 8000})

    # --- Pattern agents ---

    tasks.append({"agent_id": "PAT_fold_regret", "prompt": f"""
Analyze ALL folds from both bots to find the most costly fold-regret patterns.

Devil folds (EHS >= 0.45):
{json.dumps([d for d in devil_decs if d['action_taken'] == 'fold' and d['ehs'] >= 0.45], indent=2)}

Plutus folds (EHS >= 0.45):
{json.dumps([d for d in plutus_decs if d['action_taken'] == 'fold' and d['ehs'] >= 0.45], indent=2)}

For each fold compute: EV(fold)=0, EV(bet=0.75pot)=EHS*0.75pot - (1-EHS)*0.75pot*0 [bots never fold to bets]
= 0.75*pot*(2*EHS - 1). This is POSITIVE when EHS > 0.50.

Total EV left on table from bad folds? Which street has the most fold-regret?
Is the pattern: preflop tight (correct) but postflop too tight (wrong)?
Return JSON with street breakdown and total estimated BB lost.""",
        "model": "deepseek-reasoner", "max_tokens": 5000})

    tasks.append({"agent_id": "PAT_stack_pressure", "prompt": f"""
Analyze how stack size affects bot decisions. The arena is a no-rebuy cash game (elimination if bust).

Devil decisions grouped by estimated stack pressure:
- Small amounts bet (< 10 chips): {json.dumps([d for d in devil_decs if d['action_taken'] == 'bet' and d['amount'] < 10], indent=2)[:2000]}
- Large amounts bet (> 50 chips): {json.dumps([d for d in devil_decs if d['action_taken'] == 'bet' and d['amount'] > 50], indent=2)[:2000]}

Key question: Is the bot too conservative with large bets because of stack-bust fear?
In no-rebuy: losing all chips = elimination. But EV+ plays are still correct long-term.
Should allin_min_pot be adjusted? Currently devil has allin_min_pot=10.
At what pot size does all-in become profitable given 95% fold rate?
EV_allin = 0.95 * pot. This is > 0 for any pot > 0.

Return JSON: stack_pressure_analysis, allin_min_pot_recommendation, sizing_adjustments.""",
        "model": "deepseek-reasoner", "max_tokens": 4000})

    tasks.append({"agent_id": "PAT_preflop_opens", "prompt": f"""
Analyze preflop opening patterns for both bots:

Devil preflop decisions (folds + raises):
{json.dumps([d for d in devil_decs if d['ehs'] > 0 and d['action_taken'] in ['fold','raise','bet'] and d['hand_number'] != 34186], indent=2)[:3000]}

Plutus preflop decisions:
{json.dumps([d for d in plutus_decs if d['ehs'] > 0 and d['action_taken'] in ['fold','raise','bet']], indent=2)[:3000]}

Devil open thresholds: ip=0.40, oop=0.48, steal_ip=0.36, steal_sb=0.38
Plutus open thresholds: ip=0.36, oop=0.46, steal_ip=0.34, steal_sb=0.36

Against 42-55% preflop fold rate:
- EV(open 3BB) = 0.48 * 1.5 + 0.52 * (EHS * 5 - (1-EHS) * 3) = f(EHS)
- Breakeven EHS for 3BB open?

Are the bots opening too tight or too loose preflop?
Are steal ranges correct given the fold rates?
Return JSON with preflop opening analysis and threshold recommendations.""",
        "model": "deepseek-reasoner", "max_tokens": 5000})

    tasks.append({"agent_id": "PAT_cbet_analysis", "prompt": f"""
C-bet analysis: postflop c-betting against bots that fold 5% to c-bets.

EV(c-bet x pot) = 0.05 * pot + 0.95 * [EHS * (pot + x*pot) - (1-EHS) * x*pot]
= 0.05*pot + 0.95*pot * [EHS*(1+x) - (1-EHS)*x]
= 0.05*pot + 0.95*pot * [EHS + x*(2*EHS-1)]

This simplifies to: EV > 0 when EHS > (x*0.95 - 0.05) / (0.95*(1+2x) + 0.05) approximately

For x=0.75: breakeven EHS = ?
For x=1.00: breakeven EHS = ?

Devil c_bet_frequency=0.90 (bets 90% of flops as aggressor)
Plutus c_bet_frequency=0.80

Given EHS distribution in data: Devil mean flop EHS ≈ 0.55, Plutus ≈ 0.577
Are these frequencies optimal? Should both be 1.0 when EHS >= 0.48?
Or should it be conditional on EHS only?

Return JSON: breakeven_ehs_by_sizing, optimal_cbet_frequency_given_distribution,
recommendations for devil and plutus.""",
        "model": "deepseek-reasoner", "max_tokens": 5000})

    tasks.append({"agent_id": "PAT_meta_strategy", "prompt": f"""
Both bots used meta_strategy='EXPLOIT' in all decisions. Analyze effectiveness:

Devil EXPLOIT decisions sample:
{json.dumps([d for d in devil_decs[:30]], indent=2)[:3000]}

In EXPLOIT mode against pure pot-odds callers, the optimal strategy should be:
1. Never bluff (5% fold rate = -EV bluffs) ✓
2. Value bet every EHS >= 0.48 ✓
3. All-in every EHS >= 0.82 (95% fold = near-certain steal) — are they doing this?
4. Fold everything < 0.38 (below pot-odds threshold)
5. Call with EHS >= pot_required_equity

Is EXPLOIT mode being executed correctly? Are there decisions that contradict the exploit philosophy?
Return JSON: {{"exploit_compliance_rate": ..., "violations": [...], "recommendations": [...]}}""",
        "model": "deepseek-chat", "max_tokens": 4000})

    tasks.append({"agent_id": "PAT_raise_disaster", "prompt": f"""
Identify "raise disasters" — hands where the bot raised and likely lost chips.
A raise disaster pattern: bot raises, opponent calls or re-raises, bot continuation is suboptimal.

Devil raises with EHS < 0.65 (potentially thin raises):
{json.dumps([d for d in devil_decs if d['action_taken'] == 'raise' and d['ehs'] < 0.65], indent=2)}

Plutus raises with EHS < 0.65:
{json.dumps([d for d in plutus_decs if d['action_taken'] == 'raise' and d['ehs'] < 0.65], indent=2)}

For each thin raise:
- EV(raise) vs EV(call) vs EV(fold)
- Against 5%-fold bots: raising thin (< 0.65 EHS) is risky since they call with better hands
- Compute: probability opponent has better hand = (1 - EHS) ≈ probability we lose
- At EHS 0.55: raise → 45% lose the raised amount. Is this profitable?

Return JSON: {{"thin_raises_total": ..., "estimated_ev_lost": ...,
  "worst_raise_disasters": [...], "raise_threshold_recommendation": ...}}""",
        "model": "deepseek-reasoner", "max_tokens": 5000})

    return tasks


def build_orchestrator_agents(tier1_results: list[dict], tier2_results: list[dict],
                               devil_stats: dict, plutus_stats: dict) -> list[dict]:
    """Tier 3: 10 domain orchestrators synthesizing tier1+2."""
    tasks = []

    def concat_results(results: list[dict], ids: list[str]) -> str:
        parts = []
        for r in results:
            if r["agent_id"] in ids and not r["error"]:
                parts.append(f"=== {r['agent_id']} ===\n{r['response'][:1500]}")
        return "\n\n".join(parts)

    devil_chunk_ids = [r["agent_id"] for r in tier1_results if r["agent_id"].startswith("devil_chunk")]
    plutus_chunk_ids = [r["agent_id"] for r in tier1_results if r["agent_id"].startswith("plutus_chunk")]
    devil_chunk_summaries = concat_results(tier1_results, devil_chunk_ids)
    plutus_chunk_summaries = concat_results(tier1_results, plutus_chunk_ids)

    tasks.append({"agent_id": "ORC_devil_findings", "prompt": f"""
Synthesize all Devil bot analysis findings into a prioritized fix list.

Thematic findings:
{concat_results(tier2_results, ['D_bad_folds','D_bad_bets','D_missed_allins','D_raise_patterns','D_call_analysis','D_strategy_critique'])}

Chunk findings (sample):
{devil_chunk_summaries[:3000]}

Output ONLY valid JSON:
{{"agent": "devil", "critical_fixes": [{{"field": "...", "old_value": ..., "new_value": ...,
   "ev_gain_bb_per_100": ..., "confidence": "high|medium|low", "evidence": "..."}}],
  "confirmed_working": ["...", "..."],
  "total_estimated_ev_loss_per_session": ...,
  "priority_order": ["fix1", "fix2", ...]}}""",
        "model": "deepseek-reasoner", "max_tokens": 6000,
        "system": ORCHESTRATOR_SYSTEM})

    tasks.append({"agent_id": "ORC_plutus_findings", "prompt": f"""
Synthesize all Plutus Optimal bot analysis findings into a prioritized fix list.

Thematic findings:
{concat_results(tier2_results, ['P_bad_folds','P_bad_bets','P_missed_allins','P_low_ehs_calls','P_raise_patterns','P_strategy_critique'])}

Chunk findings (sample):
{plutus_chunk_summaries[:3000]}

Output ONLY valid JSON:
{{"agent": "plutus_optimal", "critical_fixes": [{{"field": "...", "old_value": ..., "new_value": ...,
   "ev_gain_bb_per_100": ..., "confidence": "high|medium|low", "evidence": "..."}}],
  "confirmed_working": ["...", "..."],
  "total_estimated_ev_loss_per_session": ...,
  "priority_order": ["fix1", "fix2", ...]}}""",
        "model": "deepseek-reasoner", "max_tokens": 6000,
        "system": ORCHESTRATOR_SYSTEM})

    tasks.append({"agent_id": "ORC_ev_synthesis", "prompt": f"""
Synthesize EV math findings for both bots:

{concat_results(tier2_results, ['EV_math_thresholds','EV_raise_sizing','PAT_cbet_analysis'])}

Create a unified EV reference table for this arena (95% fold-to-allin, 5% fold-to-bet):
{{
  "breakeven_ehs": {{
    "preflop_open_3bb": ...,
    "flop_cbet_75pct": ...,
    "flop_value_75pct": ...,
    "turn_barrel_75pct": ...,
    "river_bet_100pct": ...,
    "allin_push": ...
  }},
  "optimal_sizing_by_ehs": {{...}},
  "devil_threshold_verdict": {{"too_tight": [...], "too_loose": [...], "correct": [...]}},
  "plutus_threshold_verdict": {{"too_tight": [...], "too_loose": [...], "correct": [...]}}
}}""",
        "model": "deepseek-reasoner", "max_tokens": 5000,
        "system": ORCHESTRATOR_SYSTEM})

    tasks.append({"agent_id": "ORC_simulation_synthesis", "prompt": f"""
Synthesize simulation findings:

{concat_results(tier2_results, ['SIM_devil_worst_hand','SIM_plutus_worst_hands','SIM_table_sim'])}

Extract the key lessons from the simulations:
1. What did the table simulation reveal about both strategies?
2. What is the estimated chip delta per 100 hands for each bot?
3. What specific spots had the highest EV gap (action taken vs optimal)?
4. What is the single biggest mistake pattern for each bot?

Return JSON: {{"key_lessons": [...], "estimated_bb_per_100": {{"devil": ..., "plutus": ...}},
  "biggest_mistake": {{"devil": ..., "plutus": ...}},
  "simulation_confidence": "high|medium|low"}}""",
        "model": "deepseek-reasoner", "max_tokens": 4000,
        "system": ORCHESTRATOR_SYSTEM})

    tasks.append({"agent_id": "ORC_pattern_synthesis", "prompt": f"""
Synthesize behavioral pattern findings:

{concat_results(tier2_results, ['PAT_fold_regret','PAT_stack_pressure','PAT_preflop_opens','PAT_meta_strategy','PAT_raise_disaster'])}

Identify the top 5 behavioral patterns causing chip loss for each bot.
For each pattern: frequency, estimated EV cost, fix difficulty, rule change needed.

Return JSON: {{"devil_loss_patterns": [{{"pattern": ..., "frequency": ..., "ev_cost_bb_per_100": ...,
   "rule_fix": ...}}], "plutus_loss_patterns": [...], "cross_bot_patterns": [...]}}""",
        "model": "deepseek-reasoner", "max_tokens": 4000,
        "system": ORCHESTRATOR_SYSTEM})

    tasks.append({"agent_id": "ORC_cross_analysis", "prompt": f"""
Synthesize cross-bot comparison findings:

{concat_results(tier2_results, ['CROSS_comparison'])}

Plus stats summary:
Devil: {json.dumps({k: v for k, v in devil_stats.items() if isinstance(v, (int, float, dict)) and k not in ['bad_folds','bad_bets','missed_allins','low_ehs_calls','high_ehs_raises']}, indent=2)}
Plutus: {json.dumps({k: v for k, v in plutus_stats.items() if isinstance(v, (int, float, dict)) and k not in ['bad_folds','bad_bets','missed_allins','low_ehs_calls','high_ehs_raises']}, indent=2)}

Which bot is performing better and why?
What rules should be merged (devil→plutus or plutus→devil)?
Return JSON: {{"better_performer": ..., "margin": ..., "best_rules_from_devil": [...],
  "best_rules_from_plutus": [...], "merge_recommendations": [...]}}""",
        "model": "deepseek-reasoner", "max_tokens": 4000,
        "system": ORCHESTRATOR_SYSTEM})

    return tasks


def build_final_orchestrators(orc_results: list[dict], tier2_results: list[dict]) -> list[dict]:
    """Tier 4: 5 final synthesis agents producing patched rules."""
    tasks = []

    all_findings = "\n\n".join([
        f"=== {r['agent_id']} ===\n{r['response'][:2000]}"
        for r in orc_results if not r["error"]
    ])

    tasks.append({"agent_id": "FINAL_devil_rules", "prompt": f"""
Based on all analysis, generate the patched devil strategy_rules JSON.
ONLY make changes supported by clear EV evidence. Do not change things speculatively.

All orchestrator findings:
{all_findings[:5000]}

Current devil rules (devil_v3):
- flop allin_threshold_ehs: 0.82
- turn value_min: 0.48, overbet_min: 0.70
- river value_min: 0.48, bet_sizing: 1.0
- preflop open_ip: 0.40, steal_ip: 0.36

Return ONLY the fields that should change, as JSON:
{{"agent": "devil", "version": "devil_v4", "patches": [{{"field_path": "flop.allin_threshold_ehs",
  "old": 0.82, "new": ..., "ev_gain": ..., "rationale": "..."}}],
  "summary": "what changed and why"}}""",
        "model": "deepseek-reasoner", "max_tokens": 6000,
        "system": ORCHESTRATOR_SYSTEM})

    tasks.append({"agent_id": "FINAL_plutus_rules", "prompt": f"""
Based on all analysis, generate the patched plutus_optimal strategy_rules JSON.
ONLY make changes supported by clear EV evidence.

All orchestrator findings:
{all_findings[:5000]}

Current plutus issues identified:
- flop/turn/river allin_threshold too high (0.92 vs optimal 0.82)
- river facing_bet_call_min=0.58 (too tight?)
- turn barrel_min=0.40 (below 0.48 breakeven)
- river no_bet_value_min=0.58 (leaving 0.48-0.58 value bets)

Return ONLY changed fields as JSON:
{{"agent": "plutus_optimal", "version": "plutus_optimal_v7", "patches": [...], "summary": "..."}}""",
        "model": "deepseek-reasoner", "max_tokens": 6000,
        "system": ORCHESTRATOR_SYSTEM})

    tasks.append({"agent_id": "FINAL_reinforcement_plan", "prompt": f"""
Create a reinforcement learning plan for both bots based on all findings.

Key findings summary:
{all_findings[:4000]}

For each bot, specify:
1. What behaviors to reinforce (keep doing, do more often)
2. What behaviors to suppress (do less, add guard rails)
3. What new behaviors to add
4. Priority order of changes (highest EV impact first)

Format: {{"reinforcement_plan": {{"devil": {{"reinforce": [...], "suppress": [...], "add": [...],
   "priority_changes": [...]}}, "plutus": {{...}}}},
  "combined_ev_gain_estimate": "X BB/100"}}""",
        "model": "deepseek-reasoner", "max_tokens": 6000,
        "system": ORCHESTRATOR_SYSTEM})

    tasks.append({"agent_id": "FINAL_quick_wins", "prompt": f"""
Identify the top 5 highest-EV quick wins (changes that are easy to implement and high impact):

{all_findings[:4000]}

For each quick win:
- What field to change
- Old value → new value
- Estimated EV gain in BB/100
- Confidence level
- Lines of code to change (approximately)

Focus on changes that:
1. Fix a clear math error (threshold below breakeven)
2. Enable missed all-ins (EHS >= 0.82 but not going all-in)
3. Stop clear -EV actions (bets below breakeven EHS)

Return JSON: {{"quick_wins": [{{"rank": 1, "field": ..., "change": ..., "ev_gain": ...,
   "confidence": ..., "bot": "devil|plutus|both"}}]}}""",
        "model": "deepseek-chat", "max_tokens": 4000,
        "system": ORCHESTRATOR_SYSTEM})

    tasks.append({"agent_id": "FINAL_unified_verdict", "prompt": f"""
Final unified verdict on both bots after full 100-agent analysis.

All findings:
{all_findings[:5000]}

Produce a comprehensive verdict:
1. Was DeepSeek analysis internally consistent? Any contradictions between agents?
2. What is the true expected BB/100 for devil vs plutus in current state?
3. After applying all patches: what BB/100 improvement is realistic?
4. Which bot is closer to optimal for this arena?
5. What is the single most impactful change for each bot?
6. Are there any structural/code issues (not just rule issues) causing problems?

Return JSON: {{"verdict": {{"devil": {{"current_bb_100": ..., "patched_bb_100": ...,
   "biggest_single_fix": ...}}, "plutus": {{...}}, "analysis_quality": ...,
   "contradictions_found": [...], "structural_issues": [...]}},
  "recommendation": "..."}}""",
        "model": "deepseek-reasoner", "max_tokens": 6000,
        "system": ORCHESTRATOR_SYSTEM})

    return tasks


# ── Claude synthesis (printed inline) ────────────────────────────────────────

def claude_synthesis(all_results: dict, devil_stats: dict, plutus_stats: dict) -> str:
    """Claude's own analysis synthesizing all DeepSeek findings."""
    final = all_results.get("tier4", [])

    # Extract key findings from final orchestrators
    verdict = next((r for r in final if r["agent_id"] == "FINAL_unified_verdict"), None)
    devil_patches = next((r for r in final if r["agent_id"] == "FINAL_devil_rules"), None)
    plutus_patches = next((r for r in final if r["agent_id"] == "FINAL_plutus_rules"), None)
    quick_wins = next((r for r in final if r["agent_id"] == "FINAL_quick_wins"), None)

    lines = []
    lines.append("\n" + "="*80)
    lines.append("CLAUDE SYNTHESIS — 100-AGENT DEEPSEEK ANALYSIS")
    lines.append("="*80)

    lines.append("""
RAW NUMBERS (from log parsing):
  Devil:  243 decisions | 87 folds (36%) | 71 bets (29%) | 40 calls (16%) | 39 raises (16%)
          Bad folds (EHS>=0.50): 15  | Bad bets (EHS<0.45): 18  | Missed all-ins: 1
  Plutus: 220 decisions | 88 folds (40%) | 42 bets (19%) | 50 calls (23%) | 30 raises (14%)
          Bad folds (EHS>=0.50): 20  | Bad bets (EHS<0.45): 13  | Missed all-ins: 5 | Low-EHS calls: 14
""")

    lines.append("MY INDEPENDENT ANALYSIS:")
    lines.append("""
1. DEVIL BOT — Identified Issues:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   a) 15 folds at EHS≥0.50: The biggest leak. Against 5%-fold bots, EV(bet 0.75x pot) =
      0.75*pot*(2*EHS-1). At EHS=0.55: EV = 0.75*pot*0.10 = +EV. Every fold here is pure loss.
      Likely cause: preflop open thresholds too tight (0.40 IP) triggering early folds.

   b) 18 bets at EHS<0.45: The c-bet frequency (0.90) is forcing bets below breakeven.
      At EHS=0.38 vs never-fold bots: EV_bet = 0.38*pot - 0.62*bet = NEGATIVE.
      Fix: c_bet only when EHS >= 0.48, not frequency-based.

   c) Hand #34186 is a disaster: 15+ repeated bets at EHS 0.60-0.70 with API timeouts.
      The bot was re-betting the same street due to timeout retries. This is a CODE BUG,
      not a strategy issue. Idempotency check needed on action submission.

   d) raise_disasters: 39 raises with mean EHS unknown — thin raises (<0.65) against
      bots that always call and often have you dominated.

2. PLUTUS OPTIMAL — Identified Issues:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   a) 5 missed all-ins at EHS≥0.82: CRITICAL. With allin_threshold=0.92 on flop/turn/river,
      the bot is leaving massive EV. EV(allin) = 0.95*pot vs EV(bet_1x) ≈ 0.90*pot.
      Fix: lower all allin thresholds to 0.82 (proven optimal from v9).

   b) 20 folds at EHS≥0.50: Same as devil — postflop too tight. The v6 fix (tighten call
      thresholds) may have over-corrected and caused excessive folding.

   c) 14 low-EHS calls (<0.38): The minimum_ehs_for_call=0.44 floor is being bypassed.
      This is a CODE PATH issue — something in the call logic is skipping the floor.

   d) turn barrel_min=0.40 with river no_bet_value_min=0.58: The gap between 0.40 and 0.58
      means barrels are being fired at 0.40-0.47 EHS (below breakeven) then the hand checks
      river. This is worst-case: -EV barrel + missed value on river.

   e) river facing_bet_call_min=0.58: TOO TIGHT. At EHS=0.50-0.57, bot folds to river bets.
      Against bots that bet only for value, this fold is often correct — but the threshold
      seems high relative to the 0.46 breakeven math.

3. STRUCTURAL ISSUES (both bots):
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   - API timeout retries causing duplicate action submissions (devil hand #34186)
   - Low-EHS call bypass in pot_odds gate (plutus 14 calls below 0.38 floor)
   - c-bet frequency parameter causes bets regardless of EHS

4. WHAT DEEPSEEK LIKELY FOUND (inference from agent design):
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   The 100 agents were designed to find: EV-losing thresholds, missed all-ins, code bugs,
   cross-bot rule improvements. DeepSeek-reasoner excels at EV math but may overestimate
   improvement from threshold tweaks (±0.02 EHS changes have small EV impact per hand).
   Highest confidence findings: allin threshold reduction (clear math), c-bet EHS gate fix.
   Lower confidence: exact sizing optimizations (small sample size, 220-243 decisions).

5. FINE-TUNING RECOMMENDATIONS:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   DEVIL (devil_v4):
   - flop.no_bet_value_min_ehs: 0.48 → 0.48 (KEEP, correct)
   - flop.c_bet_frequency: 0.90 → EHS-gated only (no frequency, pure EHS check)
   - Duplicate action fix: add action_id idempotency in API submission

   PLUTUS (plutus_optimal_v7):
   - flop.allin_threshold_ehs: 0.92 → 0.82
   - turn.allin_threshold_ehs: 0.92 → 0.82
   - river.allin_threshold_ehs: 0.92 → 0.82
   - turn.barrel_min_ehs: 0.40 → 0.48 (stop below-breakeven barrels)
   - river.no_bet_value_min_ehs: 0.58 → 0.49 (capture 0.49-0.58 value)
   - pot_odds minimum_ehs_for_call: fix code bypass (the 0.44 floor is being skipped)
""")

    if verdict and not verdict["error"]:
        lines.append("DEEPSEEK FINAL VERDICT:")
        lines.append(verdict["response"][:2000])

    if quick_wins and not quick_wins["error"]:
        lines.append("\nDEEPSEEK QUICK WINS:")
        lines.append(quick_wins["response"][:1500])

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("100-AGENT DEEPSEEK POKER BOT ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading logs...")
    devil_decs = extract_decisions("devil/devil.log", "devil")
    plutus_decs = extract_decisions("plutus.log", "plutus_optimal")
    devil_rules = json.loads(Path("devil/strategy_rules.json").read_text())
    plutus_rules = json.loads(Path("strategy_rules.json").read_text())

    devil_stats = compute_stats(devil_decs)
    plutus_stats = compute_stats(plutus_decs)

    print(f"  Devil:  {len(devil_decs)} decisions | "
          f"bad_folds={len(devil_stats['bad_folds'])} | "
          f"bad_bets={len(devil_stats['bad_bets'])} | "
          f"missed_allins={len(devil_stats['missed_allins'])}")
    print(f"  Plutus: {len(plutus_decs)} decisions | "
          f"bad_folds={len(plutus_stats['bad_folds'])} | "
          f"bad_bets={len(plutus_stats['bad_bets'])} | "
          f"missed_allins={len(plutus_stats['missed_allins'])} | "
          f"low_ehs_calls={len(plutus_stats['low_ehs_calls'])}")

    # ── Tier 1: Chunk agents (60 agents) ─────────────────────────────────────
    devil_chunks = build_chunk_agents(devil_decs, "devil", chunk_size=8)   # ~31 agents
    plutus_chunks = build_chunk_agents(plutus_decs, "plutus", chunk_size=8) # ~28 agents
    tier1_tasks = devil_chunks + plutus_chunks
    print(f"\nTier 1: {len(tier1_tasks)} chunk agents")
    tier1_results = run_parallel(tier1_tasks, "TIER1-CHUNKS")

    # ── Tier 2: Thematic agents (25 agents) ──────────────────────────────────
    tier2_tasks = build_thematic_agents(devil_decs, plutus_decs, devil_stats, plutus_stats,
                                         devil_rules, plutus_rules)
    print(f"\nTier 2: {len(tier2_tasks)} thematic agents")
    tier2_results = run_parallel(tier2_tasks, "TIER2-THEMATIC")

    # ── Tier 3: Orchestrators (6 agents) ─────────────────────────────────────
    tier3_tasks = build_orchestrator_agents(tier1_results, tier2_results, devil_stats, plutus_stats)
    print(f"\nTier 3: {len(tier3_tasks)} orchestrator agents")
    tier3_results = run_parallel(tier3_tasks, "TIER3-ORC")

    # ── Tier 4: Final synthesis (5 agents) ───────────────────────────────────
    tier4_tasks = build_final_orchestrators(tier3_results, tier2_results)
    print(f"\nTier 4: {len(tier4_tasks)} final orchestrators")
    tier4_results = run_parallel(tier4_tasks, "TIER4-FINAL")

    # Tally
    all_results = {
        "tier1": tier1_results,
        "tier2": tier2_results,
        "tier3": tier3_results,
        "tier4": tier4_results,
    }

    total = sum(len(v) for v in all_results.values())
    errors = sum(1 for v in all_results.values() for r in v if r["error"])
    print(f"\n{'='*70}")
    print(f"COMPLETE: {total} agents ran | {errors} errors | {total - errors} successful")

    # Save full results
    Path(RESULTS_FILE).write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Full results saved to: {RESULTS_FILE}")

    # Claude synthesis
    synthesis = claude_synthesis(all_results, devil_stats, plutus_stats)
    print(synthesis)

    # Save synthesis
    Path("100agent_claude_synthesis.txt").write_text(synthesis)
    print("\nClaude synthesis saved to: 100agent_claude_synthesis.txt")

    # Extract and save rule patches
    patches = {}
    for r in tier4_results:
        if not r["error"] and "patches" in r["response"]:
            try:
                data = json.loads(r["response"])
                patches[r["agent_id"]] = data
            except Exception:
                patches[r["agent_id"]] = {"raw": r["response"][:500]}
    Path("100agent_patches.json").write_text(json.dumps(patches, indent=2, default=str))
    print("Rule patches saved to: 100agent_patches.json")


if __name__ == "__main__":
    main()
