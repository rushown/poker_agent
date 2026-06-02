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
    cross_str  = json.dumps(cross_version_mistakes[:30], indent=2)
    folds_str  = json.dumps(bad_fold_samples[:20], indent=2)

    # ── Stage 1A: Poker theory research ──────────────────────────────
    theory_prompt = f"""
Research task: optimal EHS thresholds for a 6-max NLH bot playing vs other bots (not humans).

CONTEXT:
- Our bot uses EHS (Effective Hand Strength) as its primary decision metric
- Current performance: {stats.get('action_pcts', {})} action distribution, fold rate {stats.get('action_pcts', {}).get('fold', '?')}%
- We are playing vs bot opponents on a benchmark platform

RESEARCH QUESTIONS (draw on your full knowledge of poker theory):
1. What is the academically correct EHS threshold for preflop opens in 6-max? What does GTO research say?
2. For postflop: what EHS values separate value bets, checks, and folds on flop/turn/river?
3. For bot vs bot play specifically: how do optimal strategies differ from human-facing GTO?
   (bots don't tilt, don't make emotional decisions — what exploits work vs bots?)
4. What EHS threshold makes a river all-in +EV vs a calling station bot? Show the pot odds math.
5. For connectors (3-4, 5-6) and broadway (Q-K, J-Q): what is the correct preflop open frequency
   in terms of EHS, and why do they have different value than their raw EHS suggests?
6. What betting sizes maximize EV when EHS is 0.80-0.90 vs 0.90+ on each street?

CURRENT RULES TO EVALUATE:
{rules_str}

Return JSON:
{{
  "preflop_optimal_thresholds": {{"open_ip": 0.0, "open_oop": 0.0, "call_ip": 0.0, "call_oop": 0.0, "rationale": ""}},
  "postflop_optimal_thresholds": {{"flop_value": 0.0, "flop_fold": 0.0, "turn_value": 0.0, "river_value": 0.0, "river_allin": 0.0, "rationale": ""}},
  "bot_vs_bot_exploits": ["exploit1", "exploit2", "exploit3"],
  "connector_broadway_value": "explanation of why these hands have implied odds beyond raw EHS",
  "aggressive_sizing_by_ehs": {{
    "ehs_0.90_plus": "sizing recommendation with math",
    "ehs_0.78_to_0.90": "sizing recommendation",
    "ehs_0.65_to_0.78": "sizing recommendation"
  }},
  "current_rules_verdict": "assessment of what is correct and what needs changing"
}}
"""

    # ── Stage 1B: Cross-version mistake pattern analysis ──────────────
    pattern_prompt = f"""
Analyze mistake patterns across multiple versions of a poker bot.

CROSS-VERSION MISTAKE DATA:
{cross_str}

CURRENT VERSION ({version}) STATS:
{stats_str}

TASK: Find PATTERNS across versions, not just individual mistakes.
- Which mistake categories keep recurring across v1, v2, v3?
- Which fixes from previous cycles actually worked (mistake count dropped)?
- Which rules are STILL too tight despite previous loosening?
- Are there systematic biases (always over-folding specific streets? always bad OOP?)?

For each recurring pattern, compute: how many hands did we lose because of it, what is the total EV cost?

Return JSON:
{{
  "recurring_patterns": [
    {{"pattern": "description", "versions_affected": ["v1","v2","v3"], "frequency": 0, "total_ev_cost_chips": 0, "root_cause_field": "section.key", "recommended_fix": "specific change"}}
  ],
  "fixes_that_worked": ["description of what improved between versions"],
  "still_broken": ["what is still wrong despite previous fixes"],
  "priority_order": ["field1 — fix this first because X", "field2 — second because Y"]
}}
"""

    # ── Stage 1C: EV simulator ────────────────────────────────────────
    sim_prompt = f"""
You are an EV simulation engine. Replay these bad fold decisions with alternative actions.

BAD FOLDS (hands where we folded with EHS > 0.58):
{folds_str}

CURRENT RULES:
{rules_str}

For each bad fold, simulate:
1. EV of folding (always 0 — we fold and lose nothing more)
2. EV of calling instead: EV_call = EHS * pot - (1 - EHS) * call_amount
3. EV delta: EV_call - EV_fold
4. Which specific rule caused the fold (which threshold was violated)?

Also simulate the RIVER ALL-IN scenario:
- If EHS = 0.90 and pot = 100, what is EV of all-in vs check?
  EV_allin = P(villain_calls) * [EHS * (pot + their_call) - (1-EHS) * allin_size] + P(villain_folds) * pot
  Assume villain call frequency = 40% (typical bot calling range)

Return JSON:
{{
  "fold_simulations": [
    {{"hand": 0, "ehs": 0.0, "ev_fold": 0, "ev_call": 0.0, "ev_delta": 0.0, "rule_violated": "section.key", "threshold_was": 0.0, "threshold_should_be": 0.0}}
  ],
  "river_allin_ev_analysis": {{
    "ehs_0.90_pot_100": {{"ev_allin": 0.0, "ev_check": 0.0, "verdict": ""}},
    "ehs_0.85_pot_100": {{"ev_allin": 0.0, "ev_check": 0.0, "verdict": ""}},
    "recommended_allin_threshold": 0.0,
    "recommended_large_bet_threshold": 0.0
  }},
  "total_ev_recovered_if_fixed": 0.0
}}
"""

    print(f"[pipeline] Stage 1: launching 3 parallel DeepSeek agents...", file=sys.stderr)
    stage1_results = query_parallel([
        {"prompt": theory_prompt,  "model": "deepseek-reasoner", "system": RESEARCH_SYSTEM,    "timeout": 240, "max_tokens": 8192, "no_cache": True},
        {"prompt": pattern_prompt, "model": "deepseek-reasoner", "system": RESEARCH_SYSTEM,    "timeout": 240, "max_tokens": 8192, "no_cache": True},
        {"prompt": sim_prompt,     "model": "deepseek-reasoner", "system": SIMULATION_SYSTEM,  "timeout": 240, "max_tokens": 8192, "no_cache": True},
    ], max_workers=3)

    theory_raw, pattern_raw, sim_raw = stage1_results
    print(f"[pipeline] Stage 1 complete. Lengths: theory={len(theory_raw)}, pattern={len(pattern_raw)}, sim={len(sim_raw)}", file=sys.stderr)

    def _parse_json(raw: str, label: str) -> dict:
        clean = raw.strip()
        for fence in ("```json", "```"):
            if clean.startswith(fence):
                clean = clean[len(fence):]
        clean = clean.rstrip("`").strip()
        try:
            return json.loads(clean)
        except Exception as e:
            print(f"[pipeline] {label} JSON parse failed: {e}", file=sys.stderr)
            return {"raw": raw[:500]}

    theory_data  = _parse_json(theory_raw,  "theory")
    pattern_data = _parse_json(pattern_raw, "pattern")
    sim_data     = _parse_json(sim_raw,     "simulation")

    # ── Stage 2: Orchestrator synthesizes all three agents ────────────
    orch_prompt = f"""
You are the strategy orchestrator. Synthesize these three expert analyses into ONE implementation plan.

═══ AGENT A: POKER THEORY RESEARCH ═══
{json.dumps(theory_data, indent=2)}

═══ AGENT B: CROSS-VERSION PATTERN ANALYSIS ═══
{json.dumps(pattern_data, indent=2)}

═══ AGENT C: EV SIMULATION RESULTS ═══
{json.dumps(sim_data, indent=2)}

═══ CURRENT RULES (v{version}) ═══
{rules_str}

═══ CURRENT STATS ═══
{stats_str}

SYNTHESIS TASK:
1. A rule change is CONFIRMED if it is supported by at least 2 of the 3 agents.
2. A rule change is SUGGESTED if supported by only 1 agent — include it but mark hallucination_risk=medium.
3. A rule change is REJECTED if no agent supports it, or if ev_math contradicts it.
4. For each confirmed change, write the exact field, old value, new value, and the combined evidence chain.
5. Max delta per field: 0.08 per cycle. Enforce this strictly.
6. For the new aggression tiers (all-in threshold, large bet threshold, medium bet threshold):
   use Agent A's sizing recommendations and Agent C's EV analysis to set exact values.
7. Do NOT recommend bluff_enabled=true — we have no opponent fold rate data.

Return ONLY valid JSON:
{{
  "version": {version},
  "mistake_analysis": [
    {{
      "field": "section.key",
      "old": 0.0,
      "new": 0.0,
      "evidence": "combined from agents A+B+C",
      "ev_math": "explicit math showing EV improvement",
      "supporting_agents": ["A", "B", "C"],
      "confidence": "high/medium/low"
    }}
  ],
  "aggression_tiers": {{
    "river_allin_min_ehs": 0.0,
    "large_bet_min_ehs": 0.0,
    "large_bet_pot_mult": 0.0,
    "medium_bet_min_ehs": 0.0,
    "medium_bet_pot_mult": 0.0,
    "flop_raise_aggressive_min_ehs": 0.0,
    "ev_math": "from simulation showing these thresholds are +EV"
  }},
  "rules_must_not_change": ["field — reason with evidence from at least 2 agents"],
  "bluff_justification": "no opponent fold rate data available — bluff_enabled stays false",
  "claude_implementation_prompt": "FULL exact prompt for Claude to implement all changes",
  "predicted_bb100_improvement": "+X to +Y",
  "convergence": false,
  "hallucination_risk": "low/medium/high",
  "rejected_changes": [{{"field": "x", "reason": "why rejected"}}]
}}
"""

    print(f"[pipeline] Stage 2: orchestrator synthesizing...", file=sys.stderr)
    orch_raw = query(
        prompt=orch_prompt,
        model="deepseek-reasoner",
        no_cache=True,
        timeout=300,
        system=ORCHESTRATOR_SYSTEM,
        max_tokens=8192,
    )

    result = _parse_json(orch_raw, "orchestrator")
    result["_stage1"] = {
        "theory":     theory_data,
        "patterns":   pattern_data,
        "simulation": sim_data,
    }
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
