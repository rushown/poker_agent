#!/usr/bin/env python3
"""8-agent DeepSeek analysis for the S2 2-7 Offsuit Bluff ($2,500) tournament strategy.

Goal: design a dedicated agent that:
  1. Detects 7-2 offsuit in hole cards
  2. Executes an aggressive bluff line to force opponent folds (win with 7-2)
  3. Falls back to optimal strategy for all other hands

Arena bot facts (confirmed from 4000+ decisions):
  - Fold to all-in: 95%
  - Fold to c-bet: 5%
  - Fold preflop to 3BB open: 42-55%
  - Pure pot-odds callers, never bluff
  - No-rebuy cash game (elimination if bust)
"""
from __future__ import annotations
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request, urllib.error

API_KEY = "sk-d6ecb8fa5bae4c4b9ee6db93188ac8d6"
API_URL = "https://api.deepseek.com/chat/completions"

RESEARCH_SYSTEM = """You are DeepSeek — a world-class poker AI analyst specializing in GTO theory,
exploitative strategy, and EV mathematics. You are designing a strategy for a poker bot
participating in a '2-7 Offsuit Bluff' tournament on a digital arena platform.

KEY FACTS you MUST use (confirmed from 4000+ observed decisions):
- Arena bots are PURE POT-ODDS CALLERS: call when equity >= required_equity, NEVER bluff
- Fold preflop to 3BB open: ~42-55% (average ~48%)
- Fold to flop c-bet: ~5% (almost never fold postflop to regular bets)
- Fold to all-in: ~95% (CRITICAL exploit point)
- Consistent behavior — they NEVER adapt or change strategy
- No-rebuy cash game: if stack hits 0, the bot is eliminated

TOURNAMENT FORMAT: '2-7 Offsuit Bluff'
- Win condition: win a pot while holding 7-2 offsuit (worst hand in Texas Hold'em)
- 7-2 offsuit EHS ≈ 0.30-0.32 (bottom 1% of starting hands)
- Prize: $2,500 for winning with this hand
- Strategy: bluff opponents into folding while holding 7-2 offsuit

Be rigorous. Show all EV math explicitly. Return only valid JSON."""

ORCHESTRATOR_SYSTEM = """You are the strategy orchestrator synthesizing 7 expert analyses into
one complete machine-executable strategy for a 2-7 Offsuit Bluff poker tournament agent.

Your output drives actual code. Every rule must have: field name, value, EV math justification.
Reject any change not supported by at least 2 agents. Mark confidence levels explicitly.
Return ONLY valid JSON, no markdown."""


def ds_call(prompt: str, system: str = RESEARCH_SYSTEM, model: str = "deepseek-reasoner",
            max_tokens: int = 6000, agent_id: str = "") -> dict:
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
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
                content = data["choices"][0]["message"]["content"]
                return {"agent_id": agent_id, "response": content, "error": None}
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                return {"agent_id": agent_id, "response": "", "error": str(e)}


def run_parallel(tasks: list[dict]) -> list[dict]:
    results = []
    print(f"  Launching {len(tasks)} parallel agents...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(ds_call, t["prompt"], t.get("system", RESEARCH_SYSTEM),
                      t.get("model", "deepseek-reasoner"), t.get("max_tokens", 6000),
                      t["agent_id"]): t["agent_id"]
            for t in tasks
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            status = "OK" if not r["error"] else f"ERR:{r['error'][:60]}"
            print(f"    [{r['agent_id']}] {status} ({len(r['response'])} chars)")
    return results


# ── Agent prompts ─────────────────────────────────────────────────────────────

AGENT_A = """
AGENT A — 7-2 Offsuit EV Mathematics

Compute the full EV profile for 7-2 offsuit in Texas Hold'em against arena bots.

KNOWN FACTS:
- 7-2 offsuit is the worst starting hand: EHS ≈ 0.31 preflop (bottom 1%)
- Arena bots: fold to all-in 95%, fold to 3BB open 48%, fold to c-bet 5%
- Stack: 1000 chips, blinds 1/2
- No-rebuy: elimination if bust

COMPUTE:

1. EV of going all-in preflop with 7-2 offsuit (heads-up, pot = 3BB):
   EV_allin = 0.95 * pot_before_push + 0.05 * (EHS * total_pot - (1-EHS) * stack)
   = 0.95 * 3 + 0.05 * (0.31 * 1003 - 0.69 * 1000)
   Show exact calculation.

2. EV of 3BB open with 7-2 offsuit (opponent folds 48%):
   EV_open = 0.48 * 1.5 + 0.52 * [EV_postflop]
   Where EV_postflop with 7-2 on random flop ≈ ?

3. EV of 3BB open → full-pot c-bet if called (opponent folds 5% to c-bet, 95% to all-in):
   Two-street strategy: open → if called, go all-in on flop.
   EV = 0.48 * 1.5 [fold pre]
      + 0.52 * (0.95 * (3+3) [fold to allin flop])
      + 0.52 * 0.05 * (0.31 * 12 - 0.69 * 9) [called and we lose]

4. Which is highest EV: (a) open and fold to call, (b) open → allin on flop, (c) open → allin preflop?

5. Frequency of 7-2 offsuit: 4/1326 = 0.3% of hands. Expected # per 100 hands?

6. What bet sizing maximizes fold probability on each street?
   If opponent uses pure pot-odds math to call:
   - Large bet (2x pot): required_equity = 2/(2+1) = 0.67 → opponent folds if equity < 0.67
   - All-in (10x effective pot): required_equity ≈ 0.90+ → opponent folds 95%

Return JSON:
{
  "ev_preflop_allin": 0.0,
  "ev_open_fold": 0.0,
  "ev_open_flop_allin": 0.0,
  "best_strategy_ev": {"strategy": "...", "ev": 0.0},
  "frequency_per_100_hands": 0.0,
  "bet_sizing_fold_probability": {
    "3BB_open": 0.48,
    "flop_allin": 0.95,
    "river_allin": 0.95
  },
  "conclusions": ["key insight 1", "key insight 2"]
}
"""

AGENT_B = """
AGENT B — Preflop Bluff Strategy for 7-2 Offsuit

Design the optimal preflop strategy when holding 7-2 offsuit.

CONSTRAINTS:
- Must WIN the pot (not just survive) to score the tournament prize
- Arena bots fold to 3BB open: 48%, fold to all-in: 95%
- No-rebuy: losing all chips = game over (cannot afford reckless all-ins every hand)
- 7-2 appears ~0.3% of hands → expected 3x per 1000 hands

SCENARIOS:

1. UNOPENED POT (you act first):
   Option A: Open 3BB → c-bet all-in if called
   Option B: Open all-in immediately
   Option C: Open 3BB → fold if called (too risky)

   For each: compute EV given 48% fold preflop, 95% fold to allin

2. FACING A RAISE (opponent opened before you):
   Option A: 3-bet to 9BB → shove flop
   Option B: Shove all-in preflop over their raise
   Option C: Fold (give up on 7-2)

   EV of 3-bet: if opponent folds to 3-bet (what % of the time?), we win their raise

3. IN THE BIG BLIND (opponent opened 3BB):
   Option A: Shove all-in immediately
   Option B: Call → check-jam flop
   Option C: Raise to 9BB → jam flop

4. POSITION EFFECTS:
   - Does being IP vs OOP change the 7-2 bluff strategy significantly?
   - Can we delay the all-in to flop for better fold equity?

5. WHEN TO GIVE UP:
   - If we raise all-in preflop and opponent calls: we show down 7-2 (likely lose ~69%)
   - EV of calling their all-in vs folding? We should never call an all-in with 7-2.

Return JSON:
{
  "unopened_pot_optimal": {"action": "...", "sizing": "...", "ev": 0.0, "math": ""},
  "facing_raise_optimal": {"action": "...", "sizing": "...", "ev": 0.0, "math": ""},
  "bb_defense_optimal": {"action": "...", "sizing": "...", "ev": 0.0, "math": ""},
  "position_adjustment": {"ip_strategy": "...", "oop_strategy": "..."},
  "when_to_give_up": {"condition": "...", "ev_threshold": 0.0},
  "preflop_rules": {
    "always_raise": true,
    "raise_size_bb": 0.0,
    "shove_preflop_if": "...",
    "fold_if": "opponent goes all-in first (never call)"
  }
}
"""

AGENT_C = """
AGENT C — Postflop Bluff Execution for 7-2 Offsuit

Design the optimal postflop strategy when holding 7-2 offsuit and opponent has called preflop.

KNOWN: Opponent called our preflop raise. This means they have reasonable equity (survived pot-odds check).
Arena bot properties: fold to all-in 95%, fold to c-bet only 5%.

KEY INSIGHT: Against 5% c-bet fold rate, c-betting with 7-2 is -EV:
  EV_cbet(x pot) = 0.05 * pot + 0.95 * (0.31 * (pot + x*pot) - 0.69 * x*pot)
  = 0.05*pot + 0.95*pot*(0.31 + x*(0.62-0.69))
  = 0.05*pot + 0.95*pot*(0.31 - 0.07x)
  This approaches 0 as x grows... but wait, we should go ALL-IN not c-bet.

For ALL-IN on flop (pot after preflop ≈ 6BB, stack remaining ≈ 994BB):
  EV_allin_flop = 0.95 * 6 + 0.05 * (0.31 * 1000 - 0.69 * 997)
  Compute this.

BOARD TEXTURE EFFECTS:
1. 7 or 2 on flop (we pair): EHS improves to ~0.55-0.65, not just a bluff anymore
2. All high cards (K-Q-J): 7-2 makes nothing, pure bluff
3. Connected/flushing board: how does board texture affect fold equity?

MULTI-STREET STRATEGY:
If we don't go all-in immediately:
- Flop bet (1x pot) → ~5% fold → EV negative for 7-2
- Turn barrel → another 5% fold
- River shove → 95% fold (if they've called two streets)
- Total fold equity 3-street: 1 - (0.95^3) = 14.3% ... still low

Compare to immediate flop all-in: 95% fold → clear winner.

EXCEPTIONS:
1. If we made a pair (7 or 2 on board): might check to not risk chips
2. If board is monotone and we have no spade/heart: still bluff?
3. SPR considerations: if pot is large relative to remaining stack, go all-in sooner

Return JSON:
{
  "primary_postflop_strategy": "go_all_in_flop",
  "ev_flop_allin": 0.0,
  "ev_cbet_only": 0.0,
  "ev_check_fold": 0.0,
  "board_texture_adjustments": {
    "pair_on_board": {"condition": "7 or 2 on flop/turn", "action": "..."},
    "high_card_board": {"action": "..."},
    "connected_board": {"action": "..."}
  },
  "multi_street_fold_probability": 0.0,
  "recommendation": "...",
  "postflop_rules": {
    "flop_action": "all_in",
    "flop_min_ehs_to_check_instead": 0.0,
    "turn_action_if_called_flop": "...",
    "give_up_if": "..."
  }
}
"""

AGENT_D = """
AGENT D — Tournament Scoring & Stack Preservation for 7-2 Bluff

This is a no-rebuy cash game. Elimination = game over. Design a strategy that:
1. Maximizes 7-2 bluff SUCCESS rate (win the prize)
2. Does NOT risk full stack recklessly
3. Preserves chips for future 7-2 hands

FREQUENCY ANALYSIS:
- 7-2 offsuit appears in 4/1326 = 0.30% of hands
- In a session of 200 hands: expected 0.6 occurrences of 7-2 offsuit
- This is RARE — we may only get one chance

RISK/REWARD ANALYSIS:
Given only ~0-2 opportunities per session:
Option A: Shove all-in preflop every time (95% fold, max prize, 5% bust risk)
Option B: Open raise → flop all-in (same fold probability, 2 streets of risk)
Option C: Open raise → if called, bet 50% pot each street (lower risk, lower fold probability)
Option D: Open raise → if called, check/fold (basically free try, no bluff prize)

For a $2500 prize:
  Expected_value_per_hand = P(win) × $2500 - P(bust_risk) × stack_value

  If stack_value ≈ $X (tournament equity):
  Option A: 0.95 * $2500 - 0.05 * 0.69 * $X
  Option B: 0.95 * $2500 - 0.05 * 0.69 * $X (similar)

  At what stack value does Option A become -EV?

STACK PRESERVATION RULES:
- Never shove all-in with 7-2 if pot < 5BB (waste of all-in equity)
- Always go all-in on FLOP (not preflop) when pot is already large
- Never call all-in with 7-2 (fold to opponent all-in, save chips)

OPTIMAL FREQUENCY:
- Attempt bluff: every time we hold 7-2 offsuit (it's so rare we must try each time)
- Risk: up to 30% of stack per attempt (minimize bust risk)
- If called flop all-in: showdown loss is acceptable (rare event)

Return JSON:
{
  "attempt_every_72": true,
  "max_stack_risk_pct": 0.0,
  "preferred_pot_size_before_allin_bb": 0,
  "optimal_sequence": ["step1", "step2", "step3"],
  "fold_to_opponent_allin": true,
  "ev_per_bluff_attempt": 0.0,
  "ev_per_100_hands": 0.0,
  "tournament_recommendations": ["rec1", "rec2"]
}
"""

AGENT_E = """
AGENT E — Arena Bot Exploit Analysis for 7-2 Bluff

Reverse-engineer the arena bot behavior to find the maximum fold-inducing sequence.

CONFIRMED BOT BEHAVIOR (from 4000+ decisions):
- Fold to ALL-IN: 95% — most reliable fold trigger
- Fold to 3BB preflop open: 48%
- Fold to c-bet: 5% (almost never)
- Call threshold: pure pot-odds math (equity >= call/pot+call)
- Never bluff, never deviate

EXPLOIT ANALYSIS:

1. WHY 95% fold to all-in?
   The bot uses pot-odds math: call_required_equity = all_in_amount / (all_in_amount + pot)
   At 100BB stacks, going all-in = 100BB into 6BB pot = required equity = 100/106 = 94.3%
   Very few hands have 94%+ equity → 95% fold rate makes mathematical sense.

2. OPTIMAL ALL-IN TIMING:
   - Preflop (pot = 3BB): required_equity for call = 1000/1003 = 99.7% → opponent folds 99%+
   - Flop (pot = 6BB): required_equity = 997/1003 = 99.4% → opponent folds 99%+
   - But actual observed fold rate is "only" 95% — why?
     Opponent has very strong hands ~5% (AA, KK, AK = would call with 70%+ equity even large)

3. BEST TRIGGERING SEQUENCE:
   Sequence 1: Just go all-in preflop → 95% fold immediately
   Sequence 2: Open 3BB → opponent folds 48%; if calls, go all-in → opponent folds 95% → total fold = 48% + 52% * 95% = 97.4%

   Sequence 2 is BETTER because:
   - Higher total fold probability (97.4% vs 95%)
   - When called preflop (52%), going all-in flop still gets them 95% of the time
   - Only 52% * 5% = 2.6% showdown risk

4. SIZING DETAILS:
   - Preflop open: 3BB (minimum raise, cheap test)
   - Flop all-in: push full remaining stack (maximizes required_equity for opponent call)
   - Never bet partial on flop with 7-2 (5% fold = waste of chips)

5. POSITION CONSIDERATION:
   - IP: Open 3BB → flop all-in
   - OOP: 3-bet if facing raise → flop all-in OR just open 3BB → flop all-in

Return JSON:
{
  "exploit_sequence": ["preflop_open_3bb", "flop_all_in_push"],
  "total_fold_probability": 0.0,
  "preflop_only_fold_probability": 0.0,
  "sequence_fold_probability": 0.0,
  "showdown_risk_pct": 0.0,
  "ev_sequence": 0.0,
  "ev_preflop_only": 0.0,
  "optimal_sizing": {
    "preflop": "3BB open raise",
    "flop": "all_in (push full stack)"
  },
  "position_adjustment": "minimal — fold equity from all-in is position-independent",
  "confirmed_exploits": ["...", "..."]
}
"""

AGENT_F = """
AGENT F — Hand Detection & Strategy Override Design

Design the code architecture for detecting 7-2 offsuit and overriding strategy.

CODEBASE CONTEXT:
The bot uses Python with GameContext having:
- ctx.hole_cards: list of card strings like ["7h", "2s"] or ["2d", "7c"]
- ctx.street: "preflop" | "flop" | "turn" | "river"
- ctx.call_amount: how much to call (0 = no bet facing)
- ctx.pot: current pot size
- ctx.stack: remaining stack
- ctx.is_in_position: bool
- Returns: (action, amount, reason) tuple
  - action: "fold" | "call" | "bet" | "raise" | "check"
  - amount: float

DETECTION LOGIC (Python):
```python
def _is_72_offsuit(hole_cards):
    if len(hole_cards) != 2:
        return False
    ranks = set(c[0].upper() for c in hole_cards if c)
    suits = [c[1].lower() for c in hole_cards if len(c) >= 2]
    return '7' in ranks and '2' in ranks and len(set(suits)) == 2
```

STRATEGY OVERRIDE:
When 7-2 offsuit detected, override the normal adaptive strategy:

Preflop scenarios:
1. No bet facing (we can open): raise 3BB
2. Facing a raise (call_amount > 0): if pot < stack * 0.30, re-raise all-in; else fold (don't call)
3. Already have community cards: this shouldn't happen at preflop

Postflop scenarios:
1. Flop/turn/river with no bet facing: push all-in
2. Flop/turn/river facing a bet: if we have a made pair (7 or 2 on board), call/raise; else fold
   Exception: if opponent bet is small (< 20% pot), call/push all-in as bluff-raise

MADE HAND EXCEPTION:
If 7 or 2 appears on board, our hand improves:
- Pair of 7s (board has 7): EHS improves to ~0.58-0.65 → play as medium hand
- Pair of 2s (board has 2): EHS improves to ~0.52-0.58 → play as medium hand
- Two pair or trips: EHS > 0.80 → play as strong hand

Design the strategy_override function:

```python
def bluff27_override(ctx, ehs):
    # Returns (action, amount, reason) or None if not 7-2 offsuit
    if not _is_72_offsuit(ctx.hole_cards):
        return None

    # Check if we improved on board
    if _has_board_pair(ctx):
        return None  # Let normal strategy handle made hands

    # Pure bluff mode
    ...
```

Return JSON:
{
  "detection_function": "Python code as string",
  "override_logic": {
    "preflop_no_bet": {"action": "raise", "sizing": "3BB", "reason": "72_bluff_open"},
    "preflop_facing_raise": {"action": "raise_allin", "condition": "call < 0.30 * stack", "fallback": "fold"},
    "postflop_no_bet": {"action": "raise_allin", "reason": "72_bluff_push"},
    "postflop_facing_bet": {"action": "fold", "exception": "if made pair on board"},
    "made_pair_override": {"action": "defer_to_normal_strategy"}
  },
  "ehs_threshold_for_made_hand": 0.52,
  "integration_note": "Inject _is_72_offsuit check at top of each street handler before EHS-based logic"
}
"""

AGENT_G = """
AGENT G — Normal Hand Strategy for 2-7 Bluff Tournament Agent

This agent plays all non-7-2-offsuit hands. The normal strategy should be:
1. Optimal (based on proven plutus_optimal_v7 rules)
2. Conservative enough to preserve chips for 7-2 bluff opportunities
3. Still aggressive enough to generate value

TOURNAMENT CONTEXT:
- Prize is for winning with 7-2 offsuit specifically
- Normal hands should: win chips, preserve stack, avoid busting
- No chip preservation paranoia — play optimal poker on normal hands

STRATEGY RECOMMENDATION:
Base the normal hand strategy on proven plutus_optimal_v7 rules:
- preflop open_ip: 0.36
- all-in threshold: 0.82 (proven optimal)
- value bet: 0.49+
- zero postflop bluffing
- c-bet frequency: 0.80

BLUFF TOURNAMENT SPECIFIC ADJUSTMENTS:
1. Should we be more conservative on marginal hands (to preserve stack for 7-2 shots)?
   Analysis: 7-2 appears only 0.3% of hands. Being conservative costs more EV than it saves.
   Verdict: Play optimal poker, do not over-conserve.

2. Should we exploit the fact that opponents are bots (predictable)?
   Yes: same exploits as optimal (all-in at EHS 0.82, value bet at 0.49+)

3. What about all-in stack commitment?
   7-2 bluff goes all-in on flop (roughly). If we just survived an all-in with normal hand,
   we might be short-stacked. Should strategy adjust?
   Verdict: Short-stack pushes become correct sooner (lower EHS threshold).

RECOMMENDED ADJUSTMENTS TO NORMAL STRATEGY:
- Keep all-in threshold at 0.82 (proven)
- Keep value floor at 0.49
- Add: short-stack shove (stack < 20BB): push all-in at EHS >= 0.62
- No other changes

Return JSON with the complete strategy_rules for normal hands (not 7-2 offsuit):
{
  "normal_hand_strategy": {
    "core_principle": "...",
    "preflop": { ... same as plutus_optimal_v7 ... },
    "flop": { ... },
    "turn": { ... },
    "river": { ... },
    "aggression": { ... },
    "adjustments": { ... },
    "pot_odds": { ... }
  },
  "tournament_specific_adjustments": ["list what changes vs pure optimal"],
  "short_stack_rules": {
    "threshold_bb": 20,
    "shove_ehs": 0.62
  }
}
"""


def main():
    print("=" * 70)
    print("8-AGENT DEEPSEEK ANALYSIS: 2-7 OFFSUIT BLUFF TOURNAMENT")
    print("=" * 70)

    tier1_tasks = [
        {"agent_id": "A_ev_math",       "prompt": AGENT_A, "model": "deepseek-reasoner", "max_tokens": 6000},
        {"agent_id": "B_preflop",       "prompt": AGENT_B, "model": "deepseek-reasoner", "max_tokens": 6000},
        {"agent_id": "C_postflop",      "prompt": AGENT_C, "model": "deepseek-reasoner", "max_tokens": 6000},
        {"agent_id": "D_tournament",    "prompt": AGENT_D, "model": "deepseek-reasoner", "max_tokens": 5000},
        {"agent_id": "E_exploit",       "prompt": AGENT_E, "model": "deepseek-reasoner", "max_tokens": 5000},
        {"agent_id": "F_code_design",   "prompt": AGENT_F, "model": "deepseek-reasoner", "max_tokens": 6000},
        {"agent_id": "G_normal_strat",  "prompt": AGENT_G, "model": "deepseek-reasoner", "max_tokens": 8000},
    ]

    print("\nTier 1: 7 parallel specialist agents...")
    tier1 = run_parallel(tier1_tasks)

    # Save tier1 results
    Path("bluff27_tier1_results.json").write_text(json.dumps(tier1, indent=2, default=str))
    print(f"\nTier 1 complete. Saved to bluff27_tier1_results.json")

    # Assemble orchestrator input
    tier1_text = "\n\n".join([
        f"=== AGENT {r['agent_id'].upper()} ===\n{r['response'][:3000]}"
        for r in tier1 if not r["error"]
    ])

    orch_prompt = f"""
You are the synthesis orchestrator for a 2-7 Offsuit Bluff poker tournament agent.
7 specialist agents have analyzed the optimal strategy. Synthesize into:
1. Complete strategy_rules JSON for the normal (non-7-2) hands
2. Complete 7-2 bluff override rules
3. Final implementation plan

AGENT FINDINGS:
{tier1_text[:15000]}

ARENA BOT CONFIRMED FACTS (override any agent that contradicts these):
- Fold to all-in: 95%
- Fold preflop 3BB open: 48%
- Fold to c-bet: 5% (irrelevant for bluffing)
- Pure pot-odds callers, consistent behavior

SYNTHESIS REQUIREMENTS:
1. 7-2 BLUFF STRATEGY (most important):
   - Preflop: what size raise? When to shove vs open small?
   - Postflop: all-in on flop (confirmed by EV math from agents A+E)
   - When to give up: opponent all-in (fold, save chips)

2. NORMAL HAND STRATEGY:
   - Based on proven plutus_optimal_v7 rules
   - Tournament-specific adjustments only if supported by 2+ agents

3. STRATEGY RULES JSON:
   Generate a complete, valid strategy_rules JSON following EXACTLY this structure
   (same schema as strategy_rules.json with additional bluff_27_override section)

Return ONLY valid JSON:
{{
  "version": "bluff27_v1",
  "cycle": "bluff27_v1",
  "core_principle": "Win with 7-2 offsuit: detect 7-2 → execute aggressive bluff sequence. Normal hands: play optimal EV-based strategy.",
  "bluff_27_override": {{
    "enabled": true,
    "detection": "hole cards contain exactly 7 and 2, different suits",
    "preflop_raise_bb": 3.0,
    "preflop_shove_if_facing_raise": true,
    "flop_action": "all_in",
    "turn_action_if_called_flop": "fold",
    "fold_to_opponent_allin": true,
    "made_pair_ehs_threshold": 0.52,
    "ev_math": "open 3BB (48% fold = +0.72BB) then flop all-in (95% fold = +5.7BB) total fold prob = 97.4%, EV = +5.2BB per attempt",
    "notes": "Do NOT bluff on 3+ streets — go all-in on flop for max fold equity"
  }},
  "preflop": {{ ... copy from plutus_optimal_v7 with any confirmed adjustments ... }},
  "flop": {{ ... }},
  "turn": {{ ... }},
  "river": {{ ... }},
  "aggression": {{ ... }},
  "adjustments": {{ ... }},
  "pot_odds": {{ ... }},
  "short_stack": {{
    "threshold_bb": 20,
    "shove_min_ehs": 0.62,
    "note": "If stack < 20BB, shove with any EHS >= 0.62 (ICM push-fold)"
  }}
}}

IMPORTANT: Fill in ALL fields for preflop/flop/turn/river/aggression/adjustments/pot_odds.
Use plutus_optimal_v7 as the base and only change what agents specifically supported.
"""

    print("\nTier 2: Orchestrator synthesizing...")
    orch_result = ds_call(orch_prompt, ORCHESTRATOR_SYSTEM, "deepseek-reasoner", 8192, "ORCHESTRATOR")

    print(f"  [ORCHESTRATOR] {'OK' if not orch_result['error'] else 'ERROR'} ({len(orch_result['response'])} chars)")

    # Save full result
    full_result = {
        "tier1": tier1,
        "orchestrator": orch_result,
    }
    Path("bluff27_analysis_result.json").write_text(json.dumps(full_result, indent=2, default=str))
    print("\nFull result saved to: bluff27_analysis_result.json")

    # Try to extract the rules JSON
    raw = orch_result["response"]
    # Strip markdown fences if present
    for fence in ("```json", "```"):
        if raw.strip().startswith(fence):
            raw = raw.strip()[len(fence):]
    raw = raw.rstrip("`").strip()

    try:
        rules = json.loads(raw)
        Path("bluff27_strategy_rules_pending.json").write_text(json.dumps(rules, indent=2))
        print("Strategy rules saved to: bluff27_strategy_rules_pending.json")
    except Exception as e:
        print(f"Could not parse rules JSON: {e}")
        print("Raw orchestrator output saved in bluff27_analysis_result.json")
        # Try to extract JSON block
        import re
        json_match = re.search(r'\{[\s\S]+\}', raw)
        if json_match:
            try:
                rules = json.loads(json_match.group())
                Path("bluff27_strategy_rules_pending.json").write_text(json.dumps(rules, indent=2))
                print("Extracted strategy rules saved to: bluff27_strategy_rules_pending.json")
            except Exception:
                rules = None
        else:
            rules = None

    return full_result, orch_result.get("response", "")


if __name__ == "__main__":
    main()
