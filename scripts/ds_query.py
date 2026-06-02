#!/usr/bin/env python3
"""DeepSeek subagent helper — multi-agent, parallel, max-token research engine.

Usage:
  python scripts/ds_query.py "your prompt here"
  python scripts/ds_query.py --reasoner "complex analysis prompt"
  python scripts/ds_query.py --system "custom system" "user prompt"
  python scripts/ds_query.py --no-cache "fresh query"
  python scripts/ds_query.py --max-tokens 8192 "long output"

  # Multi-agent simulation (Python API):
  from scripts.ds_query import multi_agent_pipeline, query_parallel

Returns JSON or plain text to stdout. Errors to stderr. Exit 0 on success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

API_URL   = "https://api.deepseek.com/chat/completions"
CACHE_DIR = Path(".deepseek_cache")
CACHE_TTL = 3600

# Research-grade system prompt — maxes out DeepSeek's knowledge base
RESEARCH_SYSTEM = """You are DeepSeek — a world-class AI research agent with access to your full training knowledge
including academic literature, poker theory (GTO, exploitative play, ICM, Nash equilibrium),
computer science, and game theory. When analyzing poker data:
1. Draw on the full corpus of poker research: Sauce123, PioSOLVER studies, GTO Wizard findings,
   academic papers on opponent modeling and Nash equilibria in HUNL and 6-max NLH.
2. Simulate hands in your reasoning: replay the decision with alternative actions and compute EV delta.
3. Cross-reference statistical patterns against known winning strategy benchmarks.
4. When you cite evidence, be specific: quote EHS thresholds, equity percentages, pot odds math.
5. Never hallucinate field names — only reference fields that exist in the provided JSON.
6. Your output drives an actual poker bot — be rigorous, precise, and maximally helpful."""

# Simulation system prompt — agent that simulates EV of alternative actions
SIMULATION_SYSTEM = """You are a poker EV simulation engine. Given a hand history entry (hole cards, board, EHS,
action taken, outcome), you compute:
  - EV of the action taken
  - EV of the optimal alternative action
  - EV delta (improvement if we had played differently)
  - The specific rule threshold that caused the suboptimal action
Use pot odds formula: required_equity = call_size / (call_size + pot).
Use EHS as a proxy for equity. Show all math explicitly.
Return only valid JSON with no markdown."""

# Orchestrator system prompt — coordinates agent outputs
ORCHESTRATOR_SYSTEM = """You are the strategy orchestrator for a poker bot reinforcement cycle.
You receive research from multiple specialized agents and synthesize it into a single coherent
implementation plan. Your output must be machine-executable — Claude will implement it directly.
Every rule change must have: field name, old value, new value, evidence chain, EV math.
Reject any change not supported by at least two pieces of evidence. Flag uncertainty explicitly."""

DEFAULT_SYSTEM = (
    "You are a fast research assistant. Be concise. "
    "When asked for JSON, return only valid JSON with no markdown fences. "
    "When asked for text, be direct and skip preamble."
)


# ── Cache ─────────────────────────────────────────────────────────────

def _cache_key(prompt: str, model: str, system: str) -> str:
    return hashlib.sha256(f"{model}:{system}:{prompt}".encode()).hexdigest()[:20]


def _cache_get(key: str) -> str | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if time.time() - data.get("ts", 0) > CACHE_TTL:
            path.unlink(missing_ok=True)
            return None
        return data["content"]
    except Exception:
        return None


def _cache_set(key: str, content: str) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        (CACHE_DIR / f"{key}.json").write_text(json.dumps({"ts": time.time(), "content": content}))
    except Exception:
        pass


# ── Core API call ─────────────────────────────────────────────────────

def call_api(
    prompt: str,
    model: str = "deepseek-chat",
    system: str = DEFAULT_SYSTEM,
    timeout: int = 120,
    max_tokens: int = 8192,
    temperature: float = 0.1,
) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read().decode()[:300]}", file=sys.stderr)
            if attempt == 2:
                sys.exit(1)
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Error (attempt {attempt+1}): {e}", file=sys.stderr)
            if attempt == 2:
                sys.exit(1)
            time.sleep(2 ** attempt)
    sys.exit(1)


# ── Single query (public API) ─────────────────────────────────────────

def query(
    prompt: str,
    model: str = "deepseek-chat",
    no_cache: bool = False,
    timeout: int = 120,
    system: str = DEFAULT_SYSTEM,
    max_tokens: int = 8192,
) -> str:
    m = "deepseek-reasoner" if model == "deepseek-reasoner" else "deepseek-chat"
    key = _cache_key(prompt, m, system)
    if not no_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached
    result = call_api(prompt, m, system, timeout=timeout, max_tokens=max_tokens)
    if not no_cache:
        _cache_set(key, result)
    return result


# ── Parallel multi-agent queries ──────────────────────────────────────

def query_parallel(tasks: list[dict], max_workers: int = 4) -> list[str]:
    """Run multiple DeepSeek queries in parallel.

    Each task dict: {prompt, model, system, timeout, max_tokens, no_cache}
    Returns list of results in same order as tasks.
    """
    results = [None] * len(tasks)

    def run_task(idx: int, task: dict) -> tuple[int, str]:
        result = query(
            prompt=task["prompt"],
            model=task.get("model", "deepseek-chat"),
            no_cache=task.get("no_cache", True),
            timeout=task.get("timeout", 120),
            system=task.get("system", DEFAULT_SYSTEM),
            max_tokens=task.get("max_tokens", 8192),
        )
        return idx, result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_task, i, t): i for i, t in enumerate(tasks)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results


# ── Opponent profiler system prompt ───────────────────────────────────

OPPONENT_SYSTEM = """You are an opponent modeling expert for poker bots. You analyze behavior patterns
of automated poker agents and build exploit strategies against them. You know:
- pokerbench bots (used in dev.fun arena) use equity-based decision trees
- They call with 'equity covers price' logic (pot odds based)
- They value bet when they have the best hand and want worse to call
- They check/fold weak hands quickly
- They rarely bluff (bots don't have tilt or emotional pressure)
Given observed interaction patterns, prescribe exact counter-strategies with EV math."""


def _infer_opponent_patterns(all_log_files: list[str]) -> dict:
    """Infer opponent tendencies from our decision log.

    Since we don't log opponent actions directly, we infer from:
    - Hands where WE raised and presumably got calls/folds back (hand_result field)
    - EHS distribution of hands we folded vs raised — what thresholds match opponent pressure
    - Frequency of multi-way vs heads-up situations (opponent_ids count)
    """
    import json
    from pathlib import Path
    from collections import defaultdict

    multiway_hands   = 0
    headsup_hands    = 0
    total_decisions  = 0
    raise_outcomes   = []
    fold_with_high_ehs = []
    ehs_when_folded  = []
    ehs_when_raised  = []
    ehs_when_called  = []

    for log_file in all_log_files:
        lp = Path(log_file)
        if not lp.exists():
            continue
        for line in lp.read_text(errors="replace").splitlines():
            if '"event": "decision"' not in line:
                continue
            try:
                d = json.loads(line[line.index('{"event"'):])
            except Exception:
                continue
            ehs    = float(d.get("ehs", 0))
            action = d.get("action_taken", "")
            opps   = d.get("opponent_ids", [])
            result = d.get("hand_result", "")
            total_decisions += 1

            if len(opps) >= 3:
                multiway_hands += 1
            elif len(opps) == 1:
                headsup_hands += 1

            if action == "fold":
                ehs_when_folded.append(ehs)
                if ehs > 0.58:
                    fold_with_high_ehs.append({"ehs": ehs, "opps": len(opps)})
            elif action in ("raise", "bet"):
                ehs_when_raised.append(ehs)
            elif action == "call":
                ehs_when_called.append(ehs)

    def avg(lst): return round(sum(lst)/len(lst), 3) if lst else 0

    return {
        "total_decisions_analyzed": total_decisions,
        "multiway_pct": round(multiway_hands/max(1,total_decisions)*100, 1),
        "headsup_pct":  round(headsup_hands/max(1,total_decisions)*100, 1),
        "avg_ehs_when_folded":  avg(ehs_when_folded),
        "avg_ehs_when_raised":  avg(ehs_when_raised),
        "avg_ehs_when_called":  avg(ehs_when_called),
        "high_ehs_folds":       len(fold_with_high_ehs),
        "high_ehs_fold_samples": fold_with_high_ehs[:10],
        "fold_ehs_distribution": {
            "below_0.35": len([e for e in ehs_when_folded if e < 0.35]),
            "0.35_to_0.50": len([e for e in ehs_when_folded if 0.35 <= e < 0.50]),
            "0.50_to_0.58": len([e for e in ehs_when_folded if 0.50 <= e < 0.58]),
            "above_0.58":  len([e for e in ehs_when_folded if e >= 0.58]),
        },
    }


def multi_agent_pipeline(
    version: int,
    stats: dict,
    rules: dict,
    cross_version_mistakes: list[dict],
    bad_fold_samples: list[dict],
    log_path_str: str = "",
    all_log_files: list[str] | None = None,
) -> dict:
    """Full 4-agent DeepSeek research pipeline for reinforcement cycle.

    Stage 1 (4 parallel agents):
      A — Poker theory researcher: GTO thresholds, optimal sizing
      B — Cross-version mistake analyst: recurring patterns v1→vN
      C — EV simulator: per-hand replay with alternative action math
      D — Opponent profiler: pokerbench bot tendencies + counter-strategies

    Stage 2 (sequential):
      Orchestrator — synthesizes A+B+C+D with 2-agent consensus requirement

    Returns the final orchestrated dict ready for Claude to implement.
    """
    rules_str  = json.dumps(rules, indent=2)
    stats_str  = json.dumps(stats, indent=2)
    cross_str  = json.dumps(cross_version_mistakes[:40], indent=2)
    folds_str  = json.dumps(bad_fold_samples[:10], indent=2)   # smaller batch to avoid truncation

    # Infer opponent patterns from all available logs
    log_files = all_log_files or [
        f"/home/ocean/vscode/plutus-agents/v{v}/run.log" for v in range(1, 7)
    ]
    opp_patterns = _infer_opponent_patterns(log_files)
    opp_str = json.dumps(opp_patterns, indent=2)

    # ── Stage 1A: Poker theory + strong/weak aspect audit ────────────
    theory_prompt = f"""
Research task: optimal EHS thresholds for a 6-max NLH bot vs pokerbench-style bots.

CONTEXT:
- Our bot uses EHS (Effective Hand Strength, range 0-1) as its primary decision metric
- Current performance: fold={stats.get('action_pcts',{}).get('fold','?')}%, raise={stats.get('action_pcts',{}).get('raise','?')}%, call={stats.get('action_pcts',{}).get('call','?')}%
- We play exclusively vs automated pokerbench bots (equity-based, rarely bluff, pot-odds callers)

STRONG ASPECTS OF CURRENT RULES (do not regress these):
- avg_ehs_fold={stats.get('avg_ehs_fold','?')} — folding genuinely weak hands correctly
- avg_ehs_raise={stats.get('avg_ehs_raise','?')} — raising with reasonable equity
- Aggression tiers: all-in ≥0.90, large-bet ≥0.78, medium-bet ≥0.65 (new in v4)
- Connector/broadway see-flop logic (new in v3)
- Pot-odds call layer (new in v4)

WEAK ASPECTS TO RESEARCH:
- bad_folds={stats.get('regret_counts',{}).get('bad_folds','?')} — still folding some profitable hands
- missed_value={stats.get('regret_counts',{}).get('missed_value','?')} — checking when should bet
- Opponent counter-strategy unknown (we don't know call/fold frequencies of pokerbench bots)

RESEARCH QUESTIONS (use full academic knowledge):
1. For 6-max NLH vs equity-based bots: what is the correct open-raise frequency?
   Express as EHS threshold AND as % of hands (e.g. "top 30% of hands = EHS >= 0.48").
2. Pokerbench bots call with pot-odds math. If we bet 75% pot, they need 75/(75+100)=43% equity.
   What is the EHS range of hands we should bet for VALUE vs such a bot (they call with 43%+ equity)?
3. What is the correct river all-in frequency vs a pot-odds bot?
   If bot calls all-ins with EHS >= 50% (breakeven), our EV of all-in = EHS*(2*pot) - (1-EHS)*pot.
   At what EHS does this become +EV vs checking?
4. STRONG/WEAK AUDIT of current rules — for each field in the rules, state if it is:
   a) CORRECT (within GTO range for this game type)
   b) TOO TIGHT (causing missed value or unnecessary folds)
   c) TOO LOOSE (causing bad calls or bad raises)

CURRENT RULES:
{rules_str}

Return JSON:
{{
  "optimal_open_pct": 0.0,
  "optimal_open_ehs": 0.0,
  "value_bet_vs_potodds_bot": {{"min_ehs": 0.0, "sizing_pct_pot": 0.0, "math": ""}},
  "river_allin_breakeven_ehs": 0.0,
  "river_allin_optimal_ehs": 0.0,
  "strong_aspects": ["field: why it is correct"],
  "weak_aspects": [
    {{"field": "section.key", "issue": "too_tight/too_loose", "recommended_value": 0.0, "math": ""}}
  ],
  "bot_vs_bot_exploits": ["specific exploit vs equity-based caller"],
  "sizing_by_ehs": {{
    "ehs_0.90_plus": {{"action": "all-in", "ev_math": ""}},
    "ehs_0.78_to_0.90": {{"action": "large bet", "sizing": "1.2x pot", "ev_math": ""}},
    "ehs_0.65_to_0.78": {{"action": "medium bet", "sizing": "0.75x pot", "ev_math": ""}}
  }}
}}
"""

    # ── Stage 1B: Cross-version pattern + behavior trend analysis ─────
    pattern_prompt = f"""
Analyze ALL historical versions of a poker bot to find patterns, behaviors, and improvement trajectory.

CROSS-VERSION MISTAKE DATA (v1 through v{version}):
{cross_str}

CURRENT VERSION (v{version}) STATS:
{stats_str}

DEEP ANALYSIS TASKS:
1. IMPROVEMENT TRAJECTORY: For each metric (fold_rate, bad_folds, call_rate, raise_rate),
   compute the improvement % from v1 to v{version}. Is the improvement accelerating or plateauing?

2. RECURRING FAILURES: Which exact mistake categories persist across ALL versions?
   These are systematic bugs in our strategy logic, not random variance.

3. BEHAVIORS THAT IMPROVED: List every metric that got better and by how much.
   These represent validated fixes that must NOT be reversed.

4. DIMINISHING RETURNS CHECK: Are the threshold changes (each -0.08 per cycle) hitting a floor?
   If fold_rate was 77%→83%→46%→?, what is the predicted floor and are we near it?

5. NEXT PRIORITY: Given the trajectory, what is the single highest-leverage fix for v{version+1}?

Return JSON:
{{
  "improvement_trajectory": {{
    "fold_rate": {{"v1": 0.0, "current": 0.0, "improvement_pct": 0.0, "trend": "accelerating/plateauing"}},
    "bad_folds": {{"v1": 0, "current": 0, "improvement_pct": 0.0}},
    "call_rate": {{"v1": 0.0, "current": 0.0, "improvement_pct": 0.0}},
    "raise_rate": {{"v1": 0.0, "current": 0.0, "improvement_pct": 0.0}}
  }},
  "recurring_failures": [
    {{"pattern": "", "versions_affected": [], "root_cause_field": "section.key", "ev_cost_per_100_hands": 0.0, "fix": ""}}
  ],
  "validated_improvements": ["what worked and must not regress"],
  "floor_analysis": {{"current_fold_rate": 0.0, "predicted_floor": 0.0, "near_floor": true}},
  "highest_leverage_fix": {{"field": "section.key", "old": 0.0, "new": 0.0, "expected_hands_saved": 0}},
  "priority_order": ["field1 — highest EV fix", "field2", "field3"]
}}
"""

    # ── Stage 1C: EV simulator (batched — 5 hands max to avoid truncation) ──
    folds_batch = bad_fold_samples[:5]   # hard limit to prevent truncation
    folds_batch_str = json.dumps(folds_batch, indent=2)
    sim_prompt = f"""
EV simulation engine. Replay each bad fold with alternative action. Show ALL math explicitly.

BAD FOLDS (max 5 hands — we folded with EHS > 0.58):
{folds_batch_str}

CURRENT RULES (relevant thresholds):
- preflop call_ip: {rules.get('preflop',{}).get('vs_raise_call_ip_min_ehs', '?')}
- preflop call_oop: {rules.get('preflop',{}).get('vs_raise_call_oop_min_ehs', '?')}
- flop call_min: {rules.get('flop',{}).get('facing_bet_call_min_ehs', '?')}
- flop fold_max: {rules.get('flop',{}).get('facing_bet_fold_max_ehs', '?')}
- pot_odds call_margin: {rules.get('pot_odds',{}).get('call_margin', 1.05)}

FOR EACH HAND compute:
  pot = estimated 10BB (use 10 if unknown)
  call_size = estimated 5BB (use 5 if unknown)
  EV_fold = 0 (definition)
  EV_call = EHS * pot - (1-EHS) * call_size
  EV_delta = EV_call - EV_fold
  required_equity = call_size / (call_size + pot)
  pot_odds_call = EHS >= required_equity * 1.05

ALSO simulate river scenarios (no bad fold data needed):
  Scenario 1: EHS=0.92, pot=100, opponent checks to us. EV of all-in vs check.
    EV_allin = 0.40 * (0.92*(200) - 0.08*100) + 0.60 * 100  [assume 40% villain call freq]
    EV_check = EHS * pot = 0.92 * 100  [approximate]
  Scenario 2: EHS=0.78, pot=100. EV of 1.2x pot bet vs check.
  Scenario 3: EHS=0.65, pot=100. EV of 0.75x pot bet vs check.

Return compact JSON (IMPORTANT: keep response under 2000 tokens):
{{
  "fold_simulations": [
    {{"hand": 0, "ehs": 0.0, "ev_fold": 0, "ev_call": 0.0, "ev_delta": 0.0,
      "pot_odds_justified": true, "rule_violated": "section.key", "fix": 0.0}}
  ],
  "river_scenarios": [
    {{"ehs": 0.92, "ev_allin": 0.0, "ev_check": 0.0, "allin_better": true}},
    {{"ehs": 0.78, "ev_large_bet": 0.0, "ev_check": 0.0, "bet_better": true}},
    {{"ehs": 0.65, "ev_med_bet": 0.0, "ev_check": 0.0, "bet_better": true}}
  ],
  "total_ev_recovered_if_fixed": 0.0,
  "recommended_threshold_changes": [
    {{"field": "section.key", "old": 0.0, "new": 0.0, "math": "concise"}}
  ]
}}
"""

    # ── Stage 1D: Opponent profiler ───────────────────────────────────
    opp_prompt = f"""
Profile the opponents our poker bot faces and prescribe counter-strategies.

OBSERVED INTERACTION PATTERNS (inferred from our decision log):
{opp_str}

KNOWN OPPONENT TYPE: devfun arena pokerbench bots (bot_1 through bot_5)
These bots:
- Use pokerbench framework (public benchmark for poker AI evaluation)
- Make decisions based on equity calculations (shown in chat: "equity X% covers price Y%")
- Call when their EHS covers pot odds (pot-odds callers)
- Value bet when they have the best hand and want worse hands to call
- Rarely bluff (bots don't tilt or adapt emotionally)
- Are consistent — they play the same strategy every session

OUR CURRENT RULES:
{rules_str}

COUNTER-STRATEGY ANALYSIS:
1. EXPLOIT POT-ODDS CALLERS:
   If bot calls with EHS >= required_equity (pure pot odds), then:
   - We should OVERBET with strong hands to charge them maximum equity
   - We should bet LESS with medium hands to keep weaker calls in
   - What exact bet sizes (as % of pot) maximize EV vs this calling strategy?

2. EXPLOIT CONSISTENT BOTS:
   Bots don't adjust to our strategy changes. If we always 3-bet with EHS >= 0.67,
   they can't counter-adapt. What is the exploit window?

3. PREFLOP STEAL FREQUENCY:
   We are multi-way {opp_patterns.get('multiway_pct','?')}% of the time.
   In multi-way pots, what EHS threshold should we use vs the pot-odds calling range of 5 bots?

4. OPPONENT FOLD FREQUENCY ESTIMATE:
   From our raises: avg_ehs_when_raised={opp_patterns.get('avg_ehs_when_raised','?')}
   Assuming bots fold hands below their pot-odds threshold, estimate:
   - Their fold-to-raise frequency
   - Whether bluffing becomes +EV (only if fold_freq > 40%)

Return JSON:
{{
  "opponent_profile": {{
    "type": "pot_odds_caller",
    "call_threshold_ehs": 0.0,
    "bluff_frequency": 0.0,
    "estimated_fold_to_raise_pct": 0.0
  }},
  "counter_strategies": [
    {{"vs": "pot_odds_callers", "exploit": "description", "bet_sizing": "X% pot", "ev_math": ""}}
  ],
  "optimal_bet_sizes_vs_these_bots": {{
    "strong_hand_ehs_0.80_plus": {{"sizing": "X% pot", "reason": "charges max equity to callers"}},
    "medium_hand_ehs_0.65_to_0.80": {{"sizing": "X% pot", "reason": ""}},
    "bluff_viable": false,
    "bluff_threshold": "fold_freq must exceed 40% — current estimate X%"
  }},
  "preflop_adjustments": {{
    "multiway_open_ehs": 0.0,
    "headsup_open_ehs": 0.0,
    "steal_frequency_recommendation": ""
  }},
  "rule_changes_from_opponent_analysis": [
    {{"field": "section.key", "old": 0.0, "new": 0.0, "reason": "counter-strategy vs pot-odds bot"}}
  ]
}}
"""

    print(f"[pipeline] Stage 1: launching 4 parallel DeepSeek agents (A=theory, B=patterns, C=simulation, D=opponent)...", file=sys.stderr)
    stage1_results = query_parallel([
        {"prompt": theory_prompt, "model": "deepseek-reasoner", "system": RESEARCH_SYSTEM,   "timeout": 300, "max_tokens": 8192, "no_cache": True},
        {"prompt": pattern_prompt,"model": "deepseek-reasoner", "system": RESEARCH_SYSTEM,   "timeout": 300, "max_tokens": 8192, "no_cache": True},
        {"prompt": sim_prompt,    "model": "deepseek-reasoner", "system": SIMULATION_SYSTEM, "timeout": 240, "max_tokens": 4096, "no_cache": True},
        {"prompt": opp_prompt,    "model": "deepseek-reasoner", "system": OPPONENT_SYSTEM,   "timeout": 240, "max_tokens": 4096, "no_cache": True},
    ], max_workers=4)

    theory_raw, pattern_raw, sim_raw, opp_raw = stage1_results
    print(f"[pipeline] Stage 1 done. Lengths: A={len(theory_raw)} B={len(pattern_raw)} C={len(sim_raw)} D={len(opp_raw)}", file=sys.stderr)

    def _parse_json(raw: str, label: str) -> dict:
        clean = raw.strip()
        for fence in ("```json", "```"):
            if clean.startswith(fence):
                clean = clean[len(fence):]
        clean = clean.rstrip("`").strip()
        try:
            return json.loads(clean)
        except Exception as e:
            print(f"[pipeline] {label} parse failed ({e}) — using raw excerpt", file=sys.stderr)
            return {"_parse_error": str(e), "raw_excerpt": raw[:800]}

    theory_data  = _parse_json(theory_raw,  "A-theory")
    pattern_data = _parse_json(pattern_raw, "B-pattern")
    sim_data     = _parse_json(sim_raw,     "C-simulation")
    opp_data     = _parse_json(opp_raw,     "D-opponent")

    # ── Stage 2: Orchestrator synthesizes all 4 agents ────────────────
    orch_prompt = f"""
You are the strategy orchestrator. Synthesize 4 expert analyses into ONE precise implementation plan.

═══ AGENT A: POKER THEORY + STRONG/WEAK AUDIT ═══
{json.dumps(theory_data, indent=2)[:2000]}

═══ AGENT B: CROSS-VERSION PATTERNS + TRAJECTORY ═══
{json.dumps(pattern_data, indent=2)[:2000]}

═══ AGENT C: EV SIMULATION (hand-by-hand replay) ═══
{json.dumps(sim_data, indent=2)[:1500]}

═══ AGENT D: OPPONENT PROFILE + COUNTER-STRATEGIES ═══
{json.dumps(opp_data, indent=2)[:1500]}

═══ CURRENT RULES (v{version}) ═══
{rules_str}

═══ PERFORMANCE STATS ═══
{stats_str}

SYNTHESIS RULES (enforce strictly):
1. CONFIRMED change = supported by ≥2 of 4 agents. Apply with confidence=high.
2. SINGLE-AGENT change = 1 agent only. Apply only if ev_math is airtight. Mark confidence=medium.
3. REJECT any change where: field doesn't exist in rules, delta > 0.08, direction contradicts regret data.
4. STRONG ASPECTS from Agent A must not regress — flag any change that would hurt a strong aspect.
5. OPPONENT COUNTER-STRATEGY from Agent D overrides generic GTO when agents conflict.
6. Max delta = 0.08 per field per cycle.
7. bluff_enabled=true requires opponent fold_freq > 40% with evidence — Agent D must confirm.

Return ONLY valid JSON (no markdown, no explanations outside JSON):
{{
  "version": {version},
  "strong_aspects_preserved": ["list what v{version} does well that v{version+1} must keep"],
  "mistake_analysis": [
    {{
      "field": "section.key",
      "old": 0.0,
      "new": 0.0,
      "evidence": "cite specific agents and their data",
      "ev_math": "explicit calculation showing EV improvement",
      "supporting_agents": ["A","B","C","D"],
      "confidence": "high/medium/low",
      "fixes": "which specific problem this solves"
    }}
  ],
  "aggression_tiers": {{
    "river_allin_min_ehs": 0.0,
    "large_bet_min_ehs": 0.0,
    "large_bet_pot_mult": 0.0,
    "medium_bet_min_ehs": 0.0,
    "medium_bet_pot_mult": 0.0,
    "flop_raise_aggressive_min_ehs": 0.0,
    "ev_math": "from Agent C simulation"
  }},
  "opponent_exploits_to_implement": [
    {{"description": "exploit name", "code_change": "what to add/change", "ev_gain": ""}}
  ],
  "rules_must_not_change": ["field — reason backed by ≥2 agents"],
  "bluff_verdict": "fold_freq estimate from Agent D, conclusion on bluff_enabled",
  "predicted_bb100_improvement": "+X to +Y",
  "convergence": false,
  "hallucination_risk": "low/medium/high — explain any weak-evidence changes",
  "rejected_changes": [{{"field": "", "reason": ""}}]
}}
"""

    print(f"[pipeline] Stage 2: orchestrator synthesizing all 4 agents...", file=sys.stderr)
    orch_raw = query(
        prompt=orch_prompt,
        model="deepseek-reasoner",
        no_cache=True,
        timeout=360,
        system=ORCHESTRATOR_SYSTEM,
        max_tokens=8192,
    )

    result = _parse_json(orch_raw, "orchestrator")
    result["_stage1"] = {
        "theory":     theory_data,
        "patterns":   pattern_data,
        "simulation": sim_data,
        "opponent":   opp_data,
    }
    result["_opponent_patterns"] = opp_patterns
    return result


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek research engine")
    parser.add_argument("prompt", help="Prompt to send")
    parser.add_argument("--reasoner",   action="store_true", help="Use deepseek-reasoner")
    parser.add_argument("--system",     default=DEFAULT_SYSTEM, help="System prompt override")
    parser.add_argument("--research",   action="store_true", help="Use research-grade system prompt")
    parser.add_argument("--no-cache",   action="store_true", help="Skip cache")
    parser.add_argument("--timeout",    type=int, default=120, help="Timeout seconds")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max output tokens")
    args = parser.parse_args()

    model  = "deepseek-reasoner" if args.reasoner else "deepseek-chat"
    system = RESEARCH_SYSTEM if args.research else args.system
    key    = _cache_key(args.prompt, model, system)

    if not args.no_cache:
        cached = _cache_get(key)
        if cached is not None:
            print(cached)
            return

    result = call_api(args.prompt, model, system, timeout=args.timeout, max_tokens=args.max_tokens)
    if not args.no_cache:
        _cache_set(key, result)
    print(result)


if __name__ == "__main__":
    main()
