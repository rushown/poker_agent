#!/usr/bin/env python3
"""10-agent parallel DeepSeek analysis for Plutus_Aggro and Devil bots.
Agents 1-9 run in parallel, then orchestrator synthesizes into exact rule changes.
"""
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scripts.ds_query import query_parallel, query, RESEARCH_SYSTEM, ORCHESTRATOR_SYSTEM

# ── Raw data from log analysis ────────────────────────────────────────────
AGGRO_STATS = {
    "bot": "Plutus_Aggro",
    "bankroll": 15666,
    "hands_played": 424,
    "hands_won": 110,
    "win_rate_pct": round(110/424*100, 1),
    "bankroll_per_hand_net": round((15666-1000)/424, 2),
    "decisions_analyzed": 21,
    "action_dist": {"fold": "57.1%", "call": "23.8%", "raise": "9.5%", "bet": "9.5%"},
    "avg_ehs_by_action": {"fold": 0.421, "call": 0.415, "raise": 0.603, "bet": 0.420},
    "bad_bets_below_048": [
        {"hand": 24357, "action": "bet", "amount": 20.7, "ehs": 0.4752, "opponents": 1, "note": "slightly below 0.48 threshold"},
        {"hand": 24419, "action": "bet", "amount": 2.0, "ehs": 0.3649, "opponents": 1, "note": "WAY below breakeven - pure -EV bet"},
    ],
    "suspicious_folds": [
        {"hand": 24260, "ehs": 0.5463, "opponents": 5, "note": "folding 0.546 in 6-way pot"},
        {"hand": 24585, "ehs": 0.575, "opponents": 4, "note": "folding 0.575 in 5-way pot"},
    ],
    "large_bets_allins": [],  # ZERO all-ins in entire session
    "calls_analysis": [
        {"hand": 23270, "amount": 1.0, "ehs": 0.4125, "opponents": 1, "note": "BB call preflop - reasonable"},
        {"hand": 24228, "amount": 2.0, "ehs": 0.3725, "opponents": 4, "note": "multiway call - borderline"},
        {"hand": 24357, "amount": 33.0, "ehs": 0.485, "opponents": 1, "note": "large call facing a raise"},
        {"hand": 24419, "amount": 2.0, "ehs": 0.3975, "opponents": 5, "note": "6-way pot call"},
        {"hand": 24507, "amount": 2.0, "ehs": 0.4075, "opponents": 5, "note": "6-way pot call"},
    ],
    "key_concern": "ZERO all-ins in entire session despite allin_threshold_ehs=0.88 in rules - all-in gate not firing",
}

DEVIL_STATS = {
    "bot": "Devil",
    "bankroll": 3528,
    "hands_played": 137,
    "hands_won": 33,
    "win_rate_pct": round(33/137*100, 1),
    "bankroll_per_hand_net": round((3528-1000)/137, 2),
    "decisions_analyzed": 33,
    "action_dist": {"fold": "57.6%", "call": "18.2%", "raise": "12.1%", "bet": "12.1%"},
    "avg_ehs_by_action": {"fold": 0.399, "call": 0.362, "raise": 0.492, "bet": 0.673},
    "bad_raises_below_048": [
        {"hand": 24311, "action": "raise", "amount": 6.0, "ehs": 0.4537, "opponents": 4, "note": "raise at 0.45 in 5-way pot"},
        {"hand": 24311, "action": "raise", "amount": 6.0, "ehs": 0.46, "opponents": 4, "note": "raise at 0.46 in 5-way pot - duplicate entry"},
        {"hand": 24362, "action": "raise", "amount": 6.0, "ehs": 0.47, "opponents": 3, "note": "raise at 0.47 in 4-way pot"},
    ],
    "suspicious_folds": [
        {"hand": 24249, "ehs": 0.5238, "opponents": 5, "note": "folding 0.524 in 6-way pot"},
        {"hand": 24558, "ehs": 0.5687, "opponents": 4, "note": "folding 0.569 in 5-way pot"},
        {"hand": 24569, "ehs": 0.5463, "opponents": 4, "note": "folding 0.546 in 5-way pot"},
    ],
    "excellent_plays": [
        {"hand": 24454, "sequence": "call_preflop→bet_10.5_EHS0.50→bet_27_EHS0.75→allin_3544_EHS0.90",
         "note": "Perfect escalation: identified improving hand, escalated to all-in on near-nut - GREAT"},
    ],
    "large_bets_allins": [
        {"hand": 24311, "action": "bet", "amount": 150.0, "ehs": 0.5476, "note": "large overbet at EHS 0.55 - questionable sizing"},
        {"hand": 24454, "action": "bet", "amount": 3544.0, "ehs": 0.8975, "note": "all-in at EHS 0.90 - EXCELLENT"},
    ],
    "calls_analysis": [
        {"hand": 24225, "amount": 2.0, "ehs": 0.3262, "opponents": 4, "note": "very loose preflop call in 5-way"},
        {"hand": 24311, "amount": 33.0, "ehs": 0.4338, "opponents": 2, "note": "calling big raise at 0.43 HU"},
        {"hand": 24444, "amount": 2.0, "ehs": 0.4313, "opponents": 5, "note": "6-way multiway call"},
        {"hand": 24454, "amount": 2.0, "ehs": 0.3425, "opponents": 3, "note": "4-way call at 0.34 - too loose?"},
        {"hand": 24495, "amount": 1.0, "ehs": 0.3225, "opponents": 3, "note": "4-way call at 0.32 - loose"},
        {"hand": 24444, "amount": 2.0, "ehs": 0.4313, "opponents": 5, "note": "duplicate 6-way"},
    ],
    "bankroll_concern": "Devil bankroll 3528 vs Aggro 15666 after proportionally fewer hands - underperforming badly",
}

AGGRO_RULES = json.loads(Path("plutus_aggro/strategy_rules_aggro.json").read_text())
DEVIL_RULES = json.loads(Path("devil/strategy_rules.json").read_text())

KNOWN_BOT_BEHAVIOR = """
CONFIRMED OPPONENT BEHAVIOR (pokerbench bots, from 4000+ decisions across all versions):
- Pure pot-odds callers: call when equity >= required_equity, NEVER bluff
- Fold preflop to 3BB open: ~42-55% of the time (position dependent)
- Fold to flop c-bet: ~5% (almost NEVER fold to flop bets)
- Fold to all-in: ~95% (need 95%+ equity to call stack-off)
- Value bet breakeven vs these bots: EHS 0.48 (at 0.75x pot sizing)
- Below EHS 0.48: checking > betting (bot always calls with better equity)
- All-in EV ALWAYS beats value bet when EHS >= 0.82 (because 95% fold rate to all-in)
- EV_allin = 0.95 * pot (opponent folds) + 0.05 * EHS*2pot = 0.95*pot + 0.10*EHS*pot
- EV_value_bet(0.75x) = EHS*(1.75*pot) - (1-EHS)*(0.75*pot) = pot*(2.5*EHS - 0.75)
- Cross-over: all-in always better when 0.95 + 0.10*EHS > 2.5*EHS - 0.75 => EHS < 0.72 (all-in ALWAYS better!)
"""

# ── Build 9 specialist agent prompts ──────────────────────────────────────

def agent1_ev_aggro():
    return {
        "prompt": f"""You are an EV math specialist analyzing Plutus_Aggro poker bot decisions.

{KNOWN_BOT_BEHAVIOR}

AGGRO PERFORMANCE DATA:
{json.dumps(AGGRO_STATS, indent=2)}

CURRENT AGGRO RULES:
{json.dumps(AGGRO_RULES, indent=2)}

CRITICAL ISSUES TO ANALYZE:

1. BAD BET AT EHS 0.3649 (hand 24419):
   Bot bet 2 chips with EHS=0.3649 vs single opponent. Opponent calls with equity >= pot_odds.
   Compute: If bet=2, pot=~14 (estimated), required_equity_for_opp = 2/(2+14) = 0.125
   Opponent calls (their equity > 12.5%). Our EHS=0.3649 means we win 36.5% vs their range.
   EV_bet = 0.3649*(14+2+2) - (1-0.3649)*(2) = 0.3649*18 - 0.6351*2 = 6.57 - 1.27 = 5.30 (if called)
   BUT: Is our EHS=0.3649 favorable? With multiway callers, EHS typically means we beat ~36% of random hands.
   The REAL question: should we be betting 2 chips when EHS=0.3649 with only 1 opponent left?
   Reference breakeven: EHS must be >= 0.48 vs these bots to make cbet +EV (they fold only 5% of flop bets).

2. BAD BET AT EHS 0.4752 (hand 24357):
   Bot bet 20.7 chips with EHS=0.4752 vs 1 opponent. Opponent stayed (from 5→1 via folds).
   Is EHS=0.4752 above or below breakeven? Breakeven = 0.48.
   EV_bet = 0.4752*(pot+bet) - (1-0.4752)*(bet) [if opponent calls with 95% frequency since fold_to_cbet=5%]

3. ALL-IN GATE NOT FIRING:
   ZERO all-ins in entire session despite having EHS=0.62 raise (hand 24357).
   The rules say allin_threshold_ehs=0.82-0.88. But no all-ins happened.
   WHY: Is the all-in gate in adaptive.py actually checking the right conditions?
   If a bot has EHS=0.621 and bets only to call=33 later (at EHS=0.485), this suggests the hand
   improved from 0.62 to 0.485 (degraded on turn/river). The allin should fire BEFORE degradation.

4. SUSPICIOUS FOLDS (EHS 0.5463 and 0.575 in multiway):
   In a 6-player pot, facing a bet at EHS=0.5463:
   Required equity to call: depends on bet size vs pot. If bet=0.75*pot:
   required = 0.75*pot / (0.75*pot + pot) = 0.75/1.75 = 42.8%
   Our EHS=0.5463 EXCEEDS this. The fold is WRONG if we're facing a single bet.
   BUT: In multiway pots, effective equity decreases. With 5 opponents:
   adjusted_ehs = EHS^N_opponents? No, that's not right.
   Correct multiway adjustment: EHS vs field (not vs one) - need field EHS which is roughly EHS * reduction factor.
   The standard multiway penalty is +0.04 per opponent per our rules.
   With 5 opponents and fold_max = 0.39 base + 5*0.04 = 0.59 - so fold at 0.5463 would be CORRECT if multiway penalty applies to fold threshold.

COMPUTE FOR EACH ISSUE:
a) Is the bot's action correct or wrong given the opponent model?
b) What is the EV lost by the wrong actions?
c) What exact rule change would fix it? Give field name and new value.

Return JSON:
{{
  "bad_bet_1_analysis": {{"hand": 24419, "action_correct": false, "ev_lost": 0.0, "fix_field": "", "fix_value": 0.0}},
  "bad_bet_2_analysis": {{"hand": 24357, "action_correct": false, "ev_lost": 0.0, "fix_field": "", "fix_value": 0.0}},
  "allin_gate_analysis": {{"firing": false, "why_not_firing": "", "ev_lost_per_session": 0.0, "fix": ""}},
  "multiway_fold_analysis": {{"fold_correct_at_0546_vs5opps": true, "fold_correct_at_0575_vs4opps": true, "explanation": ""}},
  "total_ev_leak_aggro": 0.0,
  "recommended_changes": [{{"field": "", "old": 0.0, "new": 0.0, "ev_gain": 0.0}}]
}}""",
        "model": "deepseek-reasoner",
        "system": RESEARCH_SYSTEM,
        "timeout": 300,
        "max_tokens": 8192,
        "no_cache": True,
    }

def agent2_ev_devil():
    return {
        "prompt": f"""You are an EV math specialist analyzing Devil poker bot decisions.

{KNOWN_BOT_BEHAVIOR}

DEVIL PERFORMANCE DATA:
{json.dumps(DEVIL_STATS, indent=2)}

CURRENT DEVIL RULES:
{json.dumps(DEVIL_RULES, indent=2)}

CRITICAL ISSUES TO ANALYZE:

1. BAD RAISES AT EHS 0.45-0.47 (hands 24311 and 24362):
   Devil raised 6 chips (3BB) with EHS=0.4537, 0.46, 0.47 at 5-way and 4-way tables.
   Preflop 3BB open raise. Opponent fold rate to 3BB open = ~47% (confirmed).
   EV_raise(3BB) at EHS=0.46 vs 4 opponents:
   If each folds 47%: P(all_fold) = 0.47^4 = 0.049. Expected folds gain minimal ante.
   If called: we play a pot with EHS=0.46 postflop. Is 0.46 too weak to profitably continue?
   Breakeven for open-raise: need EHS such that postflop EV positive.
   With 0.46 EHS vs 4 callers (they don't fold much preflop to cbets): mostly unfavorable.

2. LARGE BET 150 AT EHS 0.5476 (hand 24311):
   After raise→call→call, Devil bet 150 chips with EHS=0.5476 vs 2 opponents.
   Pot before bet was roughly: 2*3BB + 2*3BB blinds + call = ~33+ chips.
   A 150 chip bet is roughly 4.5x pot overbet at EHS=0.548.
   Against pot-odds callers: they call if their equity >= 150/(150+pot).
   If pot=33: required = 150/183 = 82% equity needed to call. Very few hands call this!
   So this is effectively a semi-bluff large overbet trying to FOLD opponents.
   But wait: opponents fold to all-in 95%, fold to flop bets only 5%. What about large overbets?
   The overbet might get 50-70% fold rate. At 50% fold: EV = 0.5*pot_before - 0.5*lost_if_call.

3. ALL-IN AT EHS 0.8975 (hand 24454) - EXCELLENT:
   Devil bet 3544 chips with EHS=0.8975. This is all-in.
   EV_allin = 0.95 * pot (opponent folds 95%) + 0.05 * 0.8975 * 2*pot = 0.95*pot + 0.0898*pot = 1.04*pot
   EV_value_bet = 0.8975 * 1.75*pot - 0.1025 * 0.75*pot = 1.57*pot - 0.077*pot = 1.49*pot
   Wait - EV_allin < EV_value_bet here? Let me recheck.
   At EHS=0.90: if opponent calls 5% of time, EV_allin = 0.95*pot + 0.05*(0.9*2*pot - 0.1*pot) = 0.95*pot + 0.05*1.7*pot = 1.035*pot
   EV_value_bet(0.75x): EHS*(pot+bet) - (1-EHS)*bet if called = 0.9*1.75*pot - 0.1*0.75*pot = 1.575*pot - 0.075*pot = 1.5*pot (if always called)
   But bot calls only 5% of all-ins! If we value bet, bot calls 95% with weaker hands:
   Actually with value bet (not all-in), bot's call_freq is ~95% (they almost never fold to non-all-in bets).
   Correct analysis: EV_value_bet(0.75x, 95% call) = 0.95*(0.9*1.75*pot - 0.1*0.75*pot) + 0.05*pot = 0.95*1.5*pot + 0.05*pot = 1.475*pot
   EV_allin (95% fold rate) = 0.95*pot + 0.05*(0.9*2*pot - 0.1*pot) = 0.95*pot + 0.05*1.7*pot = 1.035*pot
   CONCLUSION: Value bet BEATS all-in at EHS=0.90 against 95%-folder bots!
   The 3544 all-in was SUBOPTIMAL! Better to value bet 0.75*pot and collect more from the 95% who call!

4. LOOSE PREFLOP CALLS (EHS 0.3262, 0.3425, 0.3225):
   Devil calls preflop at very low EHS values. Are these BB defends or aggressive calls?
   BB defend logic: you're already invested 1BB, so required_equity = call/(call+pot) = 1/(1+3) = 25%.
   At EHS=0.32: just barely covers pot odds (32% > 25%). These might be CORRECT BB defends.
   But: postflop you'll have 0.32 EHS and need to fold to most bets. This creates a leak.

COMPUTE:
a) For each bad raise: is it a preflop open or 3-bet context? What's EV of raise vs fold?
b) For the 150 overbet: what is the estimated EV and should sizing change?
c) Is the 3544 all-in truly suboptimal, or correct given specific pot size?
d) For loose calls: are BB defends at 0.32-0.34 EHS profitable given postflop play?

Return JSON:
{{
  "bad_raises_analysis": [{{"hand": 24311, "preflop_raise_ev": 0.0, "fold_ev": 0.0, "correct_action": "", "ev_lost": 0.0}}],
  "overbet_150_analysis": {{"ehs": 0.5476, "estimated_fold_rate": 0.0, "ev_overbet": 0.0, "ev_standard_bet": 0.0, "correct": false}},
  "allin_3544_analysis": {{"ehs": 0.8975, "ev_allin": 0.0, "ev_value_bet": 0.0, "which_better": "", "explanation": ""}},
  "preflop_loose_calls": {{"bb_defend_justified": true, "ev_long_term": 0.0, "recommendation": ""}},
  "total_ev_leak_devil": 0.0,
  "recommended_changes": [{{"field": "", "old": 0.0, "new": 0.0, "ev_gain": 0.0}}]
}}""",
        "model": "deepseek-reasoner",
        "system": RESEARCH_SYSTEM,
        "timeout": 300,
        "max_tokens": 8192,
        "no_cache": True,
    }

def agent3_allin_gate():
    return {
        "prompt": f"""You are a poker bot code logic expert. Analyze why the all-in gate is not firing in Plutus_Aggro.

{KNOWN_BOT_BEHAVIOR}

OBSERVATION: Plutus_Aggro played 21 decisions in this session with ZERO all-ins.
The rules say: allin_threshold_ehs = 0.82-0.88 in preflop/flop/turn/river sections.
The adaptive.py code has an all-in gate that should fire BEFORE bet_intelligence_decide().
But no all-ins executed despite one raise at EHS=0.6212.

AGGRO RULES (relevant):
{{
  "preflop": {{"allin_threshold_ehs": 0.86}},
  "flop": {{"flop_allin_min_ehs": 0.88, "allin_threshold_ehs": 0.88}},
  "turn": {{"allin_threshold_ehs": 0.86}},
  "river": {{"allin_threshold_ehs": 0.85}},
  "aggression": {{"all_in_push_min_ehs": 0.82, "allin_min_pot": 10}}
}}

DEVIL RULES (same structure, DOES fire all-in):
{{
  "flop": {{"flop_allin_min_ehs": 0.85}},
  "aggression": {{"all_in_push_min_ehs": 0.82, "allin_min_pot": 10}}
}}

FACT: Devil executed an all-in at EHS=0.8975 (CORRECT). Aggro never did.

ANALYSIS REQUIRED:
1. Why would Aggro not reach EHS >= 0.88 threshold in any of its hands?
   - Hands played suggest EHS max was 0.6212 on raises. No hand reached 0.88+.
   - This means either: (a) Aggro is getting good enough hands but threshold is too high,
     OR (b) Aggro's sessions are cutting off before the best hands arrive.
   - OR (c) The all-in gate fires only on first action, and if Aggro calls preflop (rather than raises), it never gets to act first postflop in an all-in position.

2. Threshold difference between bots:
   - Aggro flop_allin_min_ehs = 0.88 vs Devil = 0.85
   - Aggro all_in_push_min_ehs = 0.82 vs Devil = 0.82 (same)
   - But Devil fired at 0.8975, Aggro never fired at any EHS

3. CRITICAL HYPOTHESIS: The issue is that Aggro's flop_allin threshold (0.88) is HIGHER than Devil's (0.85), but the real issue is whether hands even reach that strength in Aggro's session.
   In the 21 decisions visible, EHS ranges are mostly 0.32-0.62.
   None reached 0.82+. Is this normal variance or a systematic issue?

4. EV impact of not going all-in:
   If we missed 2-3 all-in opportunities per session at EHS=0.85:
   EV_allin_missed = 0.95 * pot * 2-3 hands = ~2-3 * pot * 0.95
   With avg pot = 20BB: lost 38-57 BB per session from missed all-ins.

5. THRESHOLD REDUCTION PROPOSAL:
   To increase all-in frequency: should we lower flop_allin_min_ehs from 0.88 to 0.82?
   At EHS=0.82: EV_allin = 0.95*pot + 0.05*(0.82*2*pot - 0.18*pot) = 0.95*pot + 0.074*pot = 1.024*pot
   EV_value_bet(0.75x): 0.82*1.75*pot - 0.18*0.75*pot = 1.435*pot - 0.135*pot = 1.3*pot (if called 95%)
   Value bet STILL BEATS all-in at EHS=0.82! So lowering all-in threshold doesn't help.

   The REAL optimal: all-in threshold should be based on stack depth and pot size, not just EHS.
   When pot is large relative to stack, all-in has higher EV (denies equity to draws).

Produce analysis of the threshold structure and recommend exact changes to make all-in fire more appropriately.
Return JSON:
{{
  "why_aggro_no_allin": "explanation",
  "threshold_analysis": {{
    "current_flop_allin_ehs": 0.88,
    "devil_flop_allin_ehs": 0.85,
    "optimal_allin_ehs_vs_95pct_folders": 0.0,
    "ev_allin_at_082": 0.0,
    "ev_valuebet_at_082": 0.0,
    "which_better_at_082": ""
  }},
  "all_in_vs_value_bet_crossover": {{"crossover_ehs": 0.0, "math": ""}},
  "ev_lost_no_allin": 0.0,
  "recommended_changes": [
    {{"field": "", "old": 0.0, "new": 0.0, "ev_gain": 0.0, "reasoning": ""}}
  ],
  "conclusion": ""
}}""",
        "model": "deepseek-reasoner",
        "system": RESEARCH_SYSTEM,
        "timeout": 300,
        "max_tokens": 8192,
        "no_cache": True,
    }

def agent4_multiway():
    return {
        "prompt": f"""You are a multiway pot strategy expert for poker bots.

{KNOWN_BOT_BEHAVIOR}

MULTIWAY FOLD PROBLEM:
Both Plutus_Aggro and Devil are folding hands with EHS 0.52-0.57 in 5-6 player pots.
These folds may be CORRECT in multiway situations, but we need to verify.

DATA:
- Aggro folds EHS=0.5463 (5 opponents), EHS=0.575 (4 opponents)
- Devil folds EHS=0.5238 (5 opponents), EHS=0.5687 (4 opponents), EHS=0.5463 (4 opponents)

CURRENT RULES:
- flop.facing_bet_fold_max_ehs = 0.39 (base threshold)
- multiway_adjust_call_min_ehs_penalty = 0.04 per opponent
- 5 opponents: fold_max adjusted = 0.39 + 5*0.04 = 0.59
- 4 opponents: fold_max adjusted = 0.39 + 4*0.04 = 0.55

MATH VERIFICATION:
With 5 opponents (6-way pot) and facing a 0.75x pot bet:
required_equity = 0.75/(0.75 + 1) = 42.8% of the pot

BUT: in multiway pots, EHS vs field != EHS vs one opponent.
True equity in N-way pot: your equity share = EHS^(1/N) approximately (geometric mean correction)
OR more accurately: EHS_multiway = EHS_HU^(N-1) which would be very small.

Actually the correct thinking is: EHS = P(beat random hand). In 6-way pot with 5 opponents:
P(win) = P(beat all 5) ≠ EHS^5 (they aren't independent).
Conservative estimate: P(win_6way) ≈ 0.9 * EHS^2.2 (empirical adjustment for typical hand distributions)

At EHS=0.55, N=5: P(win) ≈ 0.55^2.2 ≈ 0.27 (27% equity in 6-way pot)
Required equity at 0.75 pot bet = 42.8%
27% < 42.8%: FOLD IS CORRECT!

At EHS=0.57, N=4: P(win) ≈ 0.57^1.8 ≈ 0.36 (36% equity in 5-way pot)
Required equity = 42.8%
36% < 42.8%: FOLD MIGHT STILL BE CORRECT depending on exact situation.

BUT: The bots are pot-odds callers, not sophisticated. Their range when betting in multiway is:
- If one bot bets, others haven't acted yet
- The bet signals strength, but we don't know if others will fold or call/raise
- In actual play: facing one bet with 5 opponents, pot odds calculation uses CURRENT pot vs call amount

1. Verify the multiway fold calculations using exact math
2. Is the current multiway_adjust_call_min_ehs_penalty of 0.04 per opponent correct?
3. What EHS threshold should trigger a call vs fold when facing a bet in:
   - 6-way pot
   - 5-way pot
   - 4-way pot
   - heads-up
4. Are the bots LOSING EV by over-folding in multiway or are the folds mathematically justified?
5. Should the penalty be per-active-opponent or fixed per-starting-players?

Return JSON:
{{
  "multiway_math": {{
    "ehs_055_6way_true_equity": 0.0,
    "ehs_057_5way_true_equity": 0.0,
    "ehs_052_6way_true_equity": 0.0,
    "required_equity_075_pot_bet": 0.0
  }},
  "fold_correctness": {{
    "aggro_fold_0546_5opps": {{"correct": true, "explanation": ""}},
    "aggro_fold_0575_4opps": {{"correct": true, "explanation": ""}},
    "devil_fold_0524_5opps": {{"correct": true, "explanation": ""}},
    "devil_fold_0569_4opps": {{"correct": true, "explanation": ""}},
    "devil_fold_0546_4opps": {{"correct": true, "explanation": ""}}
  }},
  "optimal_thresholds_by_players": {{
    "6way_call_min_ehs": 0.0,
    "5way_call_min_ehs": 0.0,
    "4way_call_min_ehs": 0.0,
    "3way_call_min_ehs": 0.0,
    "headsup_call_min_ehs": 0.0
  }},
  "current_penalty_assessment": {{"current_per_opp": 0.04, "correct_per_opp": 0.0, "total_ev_impact": 0.0}},
  "recommended_changes": [{{"field": "", "old": 0.0, "new": 0.0, "reasoning": ""}}],
  "conclusion": "are the multiway folds correct or is there EV leakage?"
}}""",
        "model": "deepseek-reasoner",
        "system": RESEARCH_SYSTEM,
        "timeout": 300,
        "max_tokens": 8192,
        "no_cache": True,
    }

def agent5_preflop():
    return {
        "prompt": f"""You are a preflop range and open-raise strategy expert for NL Hold'em bots.

{KNOWN_BOT_BEHAVIOR}

PREFLOP ANALYSIS TASK:
Both bots show suspicious patterns in preflop decisions.

DEVIL LOOSE CALLS (EHS 0.32-0.43):
- call 2BB with EHS=0.3262 vs 4 opponents (5-way)
- call 2BB with EHS=0.3425 vs 3 opponents
- call 1BB (SB complete?) with EHS=0.3225 vs 3 opponents
- call 2BB with EHS=0.4313 vs 5 opponents

DEVIL BAD RAISES (EHS 0.45-0.47):
- raise 6BB with EHS=0.4537 vs 4 opponents
- raise 6BB with EHS=0.46 vs 4 opponents
- raise 6BB with EHS=0.47 vs 3 opponents

AGGRO CALL PATTERNS:
- call 1BB with EHS=0.4125 vs 1 opponent (BB defend, reasonable)
- call 2BB with EHS=0.3725 vs 4 opponents (5-way, borderline)
- call 33 chips with EHS=0.485 vs 1 opponent (calling a 3-bet? large amount)
- call 2BB with EHS=0.3975 vs 5 opponents
- call 2BB with EHS=0.4075 vs 5 opponents

KEY QUESTION: Are the calls at EHS 0.32-0.43 BB special defenses or open mistakes?

BB DEFENSE MATH:
In BB position vs 3BB open, you already have 1BB invested:
required_equity = (3-1) / (3-1 + 3+1.5[antes]) = 2/6.5 = 30.8% (if antes)
Without antes: required = (3-1) / (1+1.5+3) = 2/5.5 = 36.4%
So BB defense at EHS=0.32 is BELOW pot odds!

But wait: in a limped pot (call 2BB preflop), you're calling the blinds from a non-BB seat:
required = 2 / (2 + dead_money)
If dead money from limpers is 4BB: required = 2/6 = 33%. EHS=0.32 is still slightly below!

RAISE TO 6BB ANALYSIS:
When raising 6BB (2x the standard 3BB), you're giving a 2:1 pot odds offer to callers.
At EHS=0.46 vs 4 opponents who call at equity >= pot_odds:
With 4 opponents, most will fold to 6BB (higher than standard).
But your EHS=0.46 means you win 46% of HU situations, less multiway.
Is this raise profitable in terms of fold equity?

Fold equity of 6BB raise: if each opponent folds 60% (higher fold to larger raise):
P(all fold) = 0.60^4 = 0.13 (13% chance to steal)
P(at least one call) = 87%
Expected EV: complicated. But if called HU at EHS=0.46: likely -EV postflop.

COMPUTE:
1. Are Devil's loose preflop calls (0.32-0.43) BB defends or calling raises?
2. What is the correct preflop call floor in BB defense position?
3. Are raises at EHS 0.45-0.47 with 6BB sizing profitable?
4. What is the optimal preflop call threshold for each position/situation?

Return JSON:
{{
  "bb_defense_threshold": {{"correct_min_ehs": 0.0, "math": "", "current_bots_match": false}},
  "loose_call_analysis": [
    {{"ehs": 0.3262, "context": "bb_defend_or_open_call", "ev": 0.0, "correct": false}}
  ],
  "raise_threshold_analysis": {{
    "raise_6bb_at_045_ev": 0.0,
    "raise_3bb_at_045_ev": 0.0,
    "optimal_raise_ehs_multiway": 0.0,
    "optimal_raise_ehs_headsup": 0.0
  }},
  "recommended_preflop_changes": [
    {{"field": "", "old": 0.0, "new": 0.0, "ev_gain": 0.0, "applies_to": "aggro/devil/both"}}
  ],
  "preflop_strengths": ["what preflop is working well for each bot"]
}}""",
        "model": "deepseek-reasoner",
        "system": RESEARCH_SYSTEM,
        "timeout": 300,
        "max_tokens": 8192,
        "no_cache": True,
    }

def agent6_bet_sizing():
    return {
        "prompt": f"""You are a bet sizing optimization expert for poker bots vs pot-odds callers.

{KNOWN_BOT_BEHAVIOR}

SIZING ANALYSIS TASK:
Analyze the bet sizing decisions across both bots and find optimal sizes.

DEVIL SIZING EXAMPLES:
- bet 10.5 chips at EHS=0.4957 (hand 24454, flop): roughly 0.75x pot (good)
- bet 27 chips at EHS=0.752 (hand 24454, turn): what % of pot is this?
- all-in 3544 chips at EHS=0.8975 (hand 24454, river): all-in
- bet 150 chips at EHS=0.5476 (hand 24311): massive overbet vs 2 opps

AGGRO SIZING EXAMPLES:
- bet 20.7 chips at EHS=0.4752: standard ~0.75x (but below breakeven EHS)
- bet 2.0 chips at EHS=0.3649: min bet or 0.14x pot? Too small!

KEY QUESTION: Against 95%-folder-to-allin bots:
When is overbet (1.5x pot) better than standard (0.75x pot) better than all-in?

EV MATH BY SIZING (opponent call rates vs pokerbench bots):
- Standard 0.75x: callers at EHS >= 43% (they call 57% of hands? or just those with equity?)
  Actually: bots call when THEIR equity >= required_equity = 0.75/(0.75+1) = 42.8%
  So they fold hands with equity < 42.8%, call with >= 42.8%.
  vs N opponents, each with ~50% equity: roughly 50% will have > 42.8%, so ~50% fold rate per opponent.

- Overbet 1.5x: required_equity = 1.5/2.5 = 60%.
  More opponents will fold (those with 43-60% equity now fold).
  Approximately 70% fold rate per opponent to 1.5x overbet.

- All-in: 95% fold rate (confirmed). Almost everyone folds.

EV per unit of OUR EHS:
At EHS=0.70 (strong hand):
- Standard bet (0.75x): fold_rate=50%.
  EV = 0.5*pot + 0.5*(0.70*(1.75*pot) - 0.30*(0.75*pot)) = 0.5*pot + 0.5*(1.225*pot - 0.225*pot) = 0.5*pot + 0.5*pot = pot
- Overbet (1.5x): fold_rate=70%.
  EV = 0.7*pot + 0.3*(0.70*2.5*pot - 0.30*1.5*pot) = 0.7*pot + 0.3*(1.75*pot - 0.45*pot) = 0.7*pot + 0.3*1.3*pot = 0.7*pot + 0.39*pot = 1.09*pot
- All-in: 95% fold.
  EV = 0.95*pot + 0.05*(0.70*2*pot) = 0.95*pot + 0.07*pot = 1.02*pot

So at EHS=0.70: Overbet (1.09) > Standard (1.00) > All-in (1.02)?
Wait, overbet slightly beats all-in at EHS=0.70.

At EHS=0.85 (very strong):
- Standard: EV = 0.5*pot + 0.5*(0.85*1.75*pot - 0.15*0.75*pot) = 0.5 + 0.5*(1.4875 - 0.1125) = 0.5 + 0.6875 = 1.1875*pot
- Overbet: EV = 0.7*pot + 0.3*(0.85*2.5*pot - 0.15*1.5*pot) = 0.7 + 0.3*(2.125 - 0.225) = 0.7 + 0.57 = 1.27*pot
- All-in: EV = 0.95*pot + 0.05*(0.85*2*pot) = 0.95 + 0.085 = 1.035*pot

At EHS=0.85: Overbet (1.27) > Standard (1.19) > All-in (1.035)?
The standard analysis suggests VALUE BETTING beats all-in at these EHS levels vs 95%-folders.

BUT: What if the opponent's calling range has high equity against us?
The 5% that call all-in likely have EHS_vs_us = 0.85 too (they only call with near-nut).
So our win rate vs their calling range might be MUCH LOWER than 0.85!

COMPUTE:
1. What is the optimal bet size at each EHS range given confirmed opponent tendencies?
2. Is Devil's 150 chip overbet at EHS=0.548 justified?
3. What sizing policy should each bot adopt?

Return JSON:
{{
  "ev_by_sizing": {{
    "ehs_045_to_055": {{"optimal_sizing": "X% pot", "ev": 0.0, "allin_ev": 0.0, "why": ""}},
    "ehs_055_to_070": {{"optimal_sizing": "", "ev": 0.0, "allin_ev": 0.0}},
    "ehs_070_to_085": {{"optimal_sizing": "", "ev": 0.0, "allin_ev": 0.0}},
    "ehs_085_plus": {{"optimal_sizing": "", "ev": 0.0, "allin_ev": 0.0}}
  }},
  "devil_150_overbet_at_055": {{"ev": 0.0, "optimal_ev": 0.0, "verdict": "", "optimal_size": ""}},
  "bet_vs_allin_crossover_ehs": 0.0,
  "recommended_sizing_rules": [
    {{"ehs_range": "0.45-0.55", "sizing": "0.75x pot", "field_changes": [{{"field": "", "new_value": 0.0}}]}}
  ],
  "aggro_min_bet_fix": {{"issue": "bets 2 chips at EHS=0.365 (min bet)", "fix": "", "optimal_sizing_here": ""}}
}}""",
        "model": "deepseek-reasoner",
        "system": RESEARCH_SYSTEM,
        "timeout": 300,
        "max_tokens": 8192,
        "no_cache": True,
    }

def agent7_raise_threshold():
    return {
        "prompt": f"""You are a raise threshold and aggression calibration expert.

{KNOWN_BOT_BEHAVIOR}

RAISE THRESHOLD ANALYSIS:
Both bots have different raise thresholds vs observed behavior.

DEVIL RAISES:
- raise 6BB at EHS=0.4537, 0.46, 0.47 (below 0.48 breakeven)
- raise 6BB at EHS=0.585 (this is fine, above breakeven)

AGGRO RAISES:
- raise 6BB at EHS=0.6212 (this is fine)
- raise 6BB at EHS=0.585 (fine)

CURRENT RAISE THRESHOLDS IN RULES:
DEVIL:
- preflop: open_ip_min_ehs=0.32, open_oop_min_ehs=0.42
- bb_raise_min_ehs=0.60
- vs_raise_3bet_min_ehs=0.62

AGGRO:
- preflop: open_ip_min_ehs=0.36, open_oop_min_ehs=0.46
- bb_raise_min_ehs=0.60

THE PROBLEM: Devil's open_ip_min_ehs=0.32 allows raising with EHS as low as 0.32!
In the data, raises appear at 0.45-0.47 which is in the range [0.32, 0.60].
These raises are at tables with 4-5 opponents (multiway).

QUESTION: Should raises have a MULTIWAY PENALTY similar to calls?
If opening at EHS=0.46 in 5-way pot:
- Need to beat 4 opponents postflop
- Your effective equity in multiway = much lower than 0.46
- Profitable open only if fold equity compensates

FOLD EQUITY CALCULATION for 3BB open vs 4 opponents:
Each opponent (pokerbench bot) folds to 3BB open ~47% of time.
P(everyone folds) = 0.47^4 = 4.9%
P(at least one call) = 95.1%
Average callers when called: ~1-2 opponents (some fold, some call)

EV_open = P(all_fold)*steal + P(called)*EV_when_called
EV when called at EHS=0.46 in 2-way pot: hard to compute without knowing postflop play quality.
But we know: postflop c-bet works 5% of time. So mostly calling to showdown.
EV_showdown at EHS=0.46 = ~0. So EV_called ≈ 0 (breakeven).
EV_open = 0.049 * 1.5BB + 0.951 * 0 ≈ 0.074BB positive but tiny.

AT EHS=0.46: Open-raise EV ≈ +0.07BB (barely positive)
The REAL risk: if called and hit a strong board, you'll have lost preflop plus be in tough spot.
Standard threshold: open only if EV > 0.5BB to justify variance.

RECOMMENDATION QUESTION:
1. Should Devil's open_ip_min_ehs be raised from 0.32 to 0.40+?
2. Should raises at EHS < 0.48 (below value bet breakeven) be allowed if positional?
3. What exact raise thresholds optimize EV across all streets?

Current best-confirmed threshold: EHS >= 0.48 is breakeven for value bets.
For raises (which have added fold equity): threshold can be slightly lower, ~0.40-0.44 IP only.

Return JSON:
{{
  "raise_threshold_analysis": {{
    "devil_ip_raise_at_045_ev": {{"ev": 0.0, "justified": false}},
    "devil_oop_raise_at_045_ev": {{"ev": 0.0, "justified": false}},
    "optimal_ip_open_ehs": 0.0,
    "optimal_oop_open_ehs": 0.0,
    "multiway_penalty_for_raises": {{"add_per_opponent": 0.0, "reasoning": ""}}
  }},
  "recommended_changes": [
    {{"bot": "devil", "field": "preflop.open_ip_min_ehs", "old": 0.32, "new": 0.0, "ev_gain": 0.0}},
    {{"bot": "aggro", "field": "preflop.open_ip_min_ehs", "old": 0.36, "new": 0.0, "ev_gain": 0.0}}
  ],
  "what_to_preserve": ["list of raise rules that are working correctly for each bot"]
}}""",
        "model": "deepseek-reasoner",
        "system": RESEARCH_SYSTEM,
        "timeout": 300,
        "max_tokens": 8192,
        "no_cache": True,
    }

def agent8_fold_threshold():
    return {
        "prompt": f"""You are a fold threshold calibration expert for poker bots.

{KNOWN_BOT_BEHAVIOR}

FOLD THRESHOLD ANALYSIS:
Both bots fold at high rates (57-58% of decisions) with mixed quality.

GOOD FOLDS (EHS below threshold, correct):
- Aggro folds at 0.316, 0.319, 0.371, 0.387, 0.397, 0.408, 0.409, 0.415, 0.414, 0.386 (all reasonable)
- Devil folds at 0.273, 0.333, 0.341, 0.352, 0.355, 0.362, 0.365, 0.365, 0.378, 0.382, 0.383, 0.392 (reasonable)

QUESTIONABLE FOLDS (EHS > 0.42 in non-multiway situations):
- Aggro: hand 24317 fold EHS=0.3975 (2 opponents) - possibly correct
- Aggro: hand 24544 fold EHS=0.4088 (3 opponents) - borderline
- Aggro: hand 24561 fold EHS=0.475 (4 opponents) - multiway, might be OK
- Devil: hand 24311, fold at EHS=0.4338 then same hand calls 33 chips at 0.4338 - WAIT, was this the same decision logged twice?
  Actually: Devil called 33 at 0.4338, THEN later bet 150 at 0.5476. No fold here - ignore.

SUSPICIOUS FOLDS (verified problematic):
- Aggro: fold EHS=0.5463 (5 opponents), fold EHS=0.575 (4 opponents)
- Devil: fold EHS=0.5238 (5 opponents), fold EHS=0.5687 (4 opponents), fold EHS=0.5463 (4 opponents)

CURRENT FOLD THRESHOLDS:
- flop.facing_bet_fold_max_ehs = 0.39 (aggro) / 0.39 (devil)
- turn.facing_bet_fold_max_ehs = 0.38
- river.facing_bet_fold_max_ehs = 0.44 (aggro) / 0.42 (devil)
- multiway_adjust_call_min_ehs_penalty = 0.04 per opponent

THE SUSPICIOUS FOLDS QUESTION:
With 5 opponents and base fold_max=0.39:
Adjusted fold_max = 0.39 + 5*0.04 = 0.59

So fold at EHS=0.5463 (< 0.59) is CORRECT by the current rules!
Similarly, fold at EHS=0.575 with 4 opponents: 0.39 + 4*0.04 = 0.55.
EHS=0.575 > 0.55, so this should be a CALL, not a fold!

FINDING: The fold at EHS=0.575 vs 4 opponents is WRONG by the rules!
The multiway penalty suggests calling at 0.575 with 4 opponents (threshold 0.55).
Something is overriding the rules, OR the opponent count used is different from what we see.

INVESTIGATE:
1. Is the multiway penalty being applied to fold_max or call_min? They might have different formulas.
2. The fold_max_ehs rule: does it mean "fold when EHS <= fold_max" or "fold unless EHS > fold_max + penalty"?
3. Are folds at high EHS actually CORRECT when the opponent bet sizing is considered?
   If opponent bets 2x pot: required_equity = 2/(2+1) = 66.7%. At EHS=0.575: wrong to call!
4. What are the correct fold thresholds for each street?

ALSO ANALYZE:
- Both bots fold ~57% of all decisions. Is this too tight or too loose?
- The pokerbench context: with 5 strong opponents, you need EHS >> 0.5 to profitably continue.
- Historical best (v9): very wide calls improved performance. Are we too tight now?

Return JSON:
{{
  "fold_rate_assessment": {{"current_pct_aggro": 57.1, "current_pct_devil": 57.6, "optimal_pct": 0.0, "too_tight_or_loose": ""}},
  "multiway_penalty_bug": {{"exists": true, "evidence": "", "how_to_fix": ""}},
  "suspicious_fold_verification": [
    {{"hand": "aggro_24585", "ehs": 0.575, "opps": 4, "threshold_by_rules": 0.55, "fold_correct": false}}
  ],
  "fold_threshold_recommendations": {{
    "flop_fold_max_base": {{"current": 0.39, "recommended": 0.0}},
    "turn_fold_max_base": {{"current": 0.38, "recommended": 0.0}},
    "river_fold_max_base": {{"current": 0.44, "recommended": 0.0}},
    "multiway_penalty_per_opp": {{"current": 0.04, "recommended": 0.0}}
  }},
  "recommended_changes": [{{"field": "", "old": 0.0, "new": 0.0, "ev_gain": 0.0}}]
}}""",
        "model": "deepseek-reasoner",
        "system": RESEARCH_SYSTEM,
        "timeout": 300,
        "max_tokens": 8192,
        "no_cache": True,
    }

def agent9_cross_learning():
    return {
        "prompt": f"""You are a cross-strategy analysis expert. Compare two poker bots and identify what each does better.

{KNOWN_BOT_BEHAVIOR}

BOT COMPARISON:

DEVIL (bankroll 3528 after 137 hands, 18.4 BB/hand net, win_rate 24.1%):
STRENGTHS:
- Excellent all-in escalation: hand 24454 perfectly escalated call→bet_10→bet_27→allin_3544 as hand improved
- The 3544 all-in at EHS=0.8975 is exactly what should happen with a near-nut hand
- More aggressive: raises in 4 of 33 decisions vs aggro's 2 in 21 decisions
- Wider value betting range: bet at EHS=0.4957, 0.752, 0.8975

WEAKNESSES:
- Lower bankroll than Aggro (3528 vs 15666) despite both starting at 1000
- Bad raises at EHS=0.45-0.47 (below breakeven)
- Very loose preflop calls (0.32, 0.34, 0.32)
- Devil lost chips: 1000 initial + won hands = should be higher. Something is draining chips.
- C-bet frequency = 1.0 (always c-bets) vs Aggro's 0.80 - may be over-betting weak hands

AGGRO (bankroll 15666 after 424 hands, 34.8 BB/hand net, win_rate 26.0%):
STRENGTHS:
- Better bankroll management: 34.8 BB/hand vs Devil's 18.4 BB/hand
- Tighter preflop: only open IP at 0.36+ vs Devil's 0.32
- No raises below 0.58 EHS (vs Devil's 0.45-0.47 raises)
- Higher win rate (26% vs 24.1%)

WEAKNESSES:
- ZERO all-ins in entire session - missing the key EV driver
- Two -EV bets below breakeven threshold
- Bad bet at EHS=0.365 (way below 0.48)
- No large pot wins in sample

CROSS-LEARNING ANALYSIS:
What should Aggro learn from Devil:
1. The all-in escalation pattern (hand 24454) - identify improving hands and escalate
2. Devil's value bet at EHS=0.752 shows it's willing to size up with medium-strong hands

What should Devil learn from Aggro:
1. Preflop discipline - don't call at 0.32 EHS
2. Don't raise with EHS < 0.48 - the fold equity doesn't justify it at these EHS levels
3. Aggro's 0.80 c-bet frequency vs Devil's 1.0 - selective c-betting based on board texture

DEVIL'S BANKROLL PROBLEM:
Devil started at 3528 with 137 hands. That's 2528 chips profit from 1000 starting.
But many large bets/calls at low EHS (0.32, 0.34, 0.32) are likely causing chip bleed.
At EHS=0.32 preflop call (2BB), then miss flop (probably): lose 2BB about 68% of the time.
Over 6 such calls: 6 * 2BB * 0.68 = 8.16BB lost on loose preflop calls alone.

IDENTIFY:
1. The 5 specific things Devil does RIGHT that Aggro should add
2. The 5 specific things Aggro does RIGHT that Devil should add
3. The 3 biggest EV differences between the bots
4. Proposed hybrid strategy that takes best of both

Return JSON:
{{
  "devil_strengths_aggro_should_adopt": [
    {{"strength": "", "how_to_add": "", "ev_gain_aggro": 0.0}}
  ],
  "aggro_strengths_devil_should_adopt": [
    {{"strength": "", "how_to_add": "", "ev_gain_devil": 0.0}}
  ],
  "biggest_ev_differences": [
    {{"metric": "", "aggro_value": 0.0, "devil_value": 0.0, "ev_gap": 0.0}}
  ],
  "why_devil_bankroll_low": {{"explanation": "", "main_chip_drain": ""}},
  "hybrid_strategy_changes": {{
    "aggro": [{{"field": "", "direction": "", "new_value": 0.0}}],
    "devil": [{{"field": "", "direction": "", "new_value": 0.0}}]
  }},
  "performance_projection": {{
    "aggro_current_bb100": 0.0,
    "devil_current_bb100": 0.0,
    "aggro_after_fixes_bb100": 0.0,
    "devil_after_fixes_bb100": 0.0
  }}
}}""",
        "model": "deepseek-reasoner",
        "system": RESEARCH_SYSTEM,
        "timeout": 300,
        "max_tokens": 8192,
        "no_cache": True,
    }

def run_pipeline():
    print("[10-agent analysis] Building tasks...", file=sys.stderr)

    tasks = [
        agent1_ev_aggro(),    # 0: EV analysis for Aggro
        agent2_ev_devil(),    # 1: EV analysis for Devil
        agent3_allin_gate(),  # 2: All-in gate analysis
        agent4_multiway(),    # 3: Multiway strategy
        agent5_preflop(),     # 4: Preflop analysis
        agent6_bet_sizing(),  # 5: Bet sizing
        agent7_raise_threshold(),  # 6: Raise thresholds
        agent8_fold_threshold(),   # 7: Fold thresholds
        agent9_cross_learning(),   # 8: Cross-bot learning
    ]

    print(f"[10-agent analysis] Launching {len(tasks)} parallel agents...", file=sys.stderr)
    results = query_parallel(tasks, max_workers=9)

    def safe_json(raw, label):
        clean = raw.strip()
        for fence in ("```json", "```"):
            if clean.startswith(fence):
                clean = clean[len(fence):]
        clean = clean.rstrip("`").strip()
        try:
            return json.loads(clean)
        except Exception as e:
            print(f"[pipeline] {label} parse failed: {e}", file=sys.stderr)
            return {"_raw": raw[:2000], "_parse_error": str(e), "_label": label}

    agent_outputs = {
        "agent1_ev_aggro": safe_json(results[0], "agent1"),
        "agent2_ev_devil": safe_json(results[1], "agent2"),
        "agent3_allin_gate": safe_json(results[2], "agent3"),
        "agent4_multiway": safe_json(results[3], "agent4"),
        "agent5_preflop": safe_json(results[4], "agent5"),
        "agent6_bet_sizing": safe_json(results[5], "agent6"),
        "agent7_raise_threshold": safe_json(results[6], "agent7"),
        "agent8_fold_threshold": safe_json(results[7], "agent8"),
        "agent9_cross_learning": safe_json(results[8], "agent9"),
    }

    print("[10-agent analysis] All 9 agents complete. Running orchestrator...", file=sys.stderr)

    def trunc(d, n=2000):
        s = json.dumps(d, indent=2)
        return s[:n] if len(s) > n else s

    orch_prompt = f"""You are the master strategy orchestrator. Synthesize 9 expert analyses into exact rule changes for BOTH Plutus_Aggro and Devil poker bots.

CONFIRMED OPPONENT BEHAVIOR:
{KNOWN_BOT_BEHAVIOR}

PERFORMANCE SUMMARY:
- Plutus_Aggro: bankroll 15666 after 424 hands (34.8 BB/hand net, 26% win rate)
- Devil: bankroll 3528 after 137 hands (18.4 BB/hand net, 24.1% win rate)
- Both bots: 57% fold rate, mostly folds at low EHS, some suspicious folds 0.52-0.57 in multiway

═══ AGENT 1: EV MATH FOR AGGRO ═══
{trunc(agent_outputs["agent1_ev_aggro"])}

═══ AGENT 2: EV MATH FOR DEVIL ═══
{trunc(agent_outputs["agent2_ev_devil"])}

═══ AGENT 3: ALL-IN GATE ANALYSIS ═══
{trunc(agent_outputs["agent3_allin_gate"])}

═══ AGENT 4: MULTIWAY STRATEGY ═══
{trunc(agent_outputs["agent4_multiway"])}

═══ AGENT 5: PREFLOP ANALYSIS ═══
{trunc(agent_outputs["agent5_preflop"])}

═══ AGENT 6: BET SIZING ═══
{trunc(agent_outputs["agent6_bet_sizing"])}

═══ AGENT 7: RAISE THRESHOLDS ═══
{trunc(agent_outputs["agent7_raise_threshold"])}

═══ AGENT 8: FOLD THRESHOLDS ═══
{trunc(agent_outputs["agent8_fold_threshold"])}

═══ AGENT 9: CROSS-BOT LEARNING ═══
{trunc(agent_outputs["agent9_cross_learning"])}

═══ CURRENT AGGRO RULES ═══
{json.dumps(AGGRO_RULES, indent=2)[:2000]}

═══ CURRENT DEVIL RULES ═══
{json.dumps(DEVIL_RULES, indent=2)[:2000]}

SYNTHESIS REQUIREMENTS:
1. Each rule change must have: field path, old value, new value, evidence (cite agents), EV math
2. Require ≥2 agents to agree before marking confidence=high
3. MAX delta = 0.08 per field per cycle
4. NEVER enable bluffs (confirmed -EV at 5% flop fold rate)
5. Fix the all-in gate for BOTH bots to fire more frequently where math supports
6. Fix Devil's loose preflop calls
7. Fix both bots' below-breakeven bets
8. Preserve what works: Aggro's bankroll management, Devil's all-in escalation pattern

Return ONLY valid JSON (no markdown):
{{
  "synthesis_summary": "one paragraph explanation",
  "aggro_rule_changes": [
    {{
      "field": "section.key",
      "old": 0.0,
      "new": 0.0,
      "evidence_agents": [1, 2, 3],
      "ev_math": "explicit formula and result",
      "confidence": "high/medium",
      "what_it_fixes": "description"
    }}
  ],
  "devil_rule_changes": [
    {{
      "field": "section.key",
      "old": 0.0,
      "new": 0.0,
      "evidence_agents": [1, 2],
      "ev_math": "",
      "confidence": "high/medium",
      "what_it_fixes": ""
    }}
  ],
  "shared_insights": ["insight that applies to both bots"],
  "aggro_strengths_to_preserve": ["list"],
  "devil_strengths_to_preserve": ["list"],
  "predicted_improvement": {{
    "aggro_bb100_current": 0.0,
    "aggro_bb100_projected": 0.0,
    "devil_bb100_current": 0.0,
    "devil_bb100_projected": 0.0
  }},
  "rejected_changes": [{{"field": "", "reason": ""}}]
}}"""

    orch_raw = query(
        prompt=orch_prompt,
        model="deepseek-reasoner",
        no_cache=True,
        timeout=360,
        system=ORCHESTRATOR_SYSTEM,
        max_tokens=8192,
    )

    orchestrator = safe_json(orch_raw, "orchestrator")

    final = {
        "metadata": {
            "aggro_bankroll": AGGRO_STATS["bankroll"],
            "devil_bankroll": DEVIL_STATS["bankroll"],
            "aggro_hands": AGGRO_STATS["hands_played"],
            "devil_hands": DEVIL_STATS["hands_played"],
        },
        "agent_outputs": agent_outputs,
        "orchestrator": orchestrator,
    }

    out_path = Path("devil_aggro_analysis_v2.json")
    out_path.write_text(json.dumps(final, indent=2))
    print(f"[10-agent analysis] Results saved to {out_path}", file=sys.stderr)
    print(json.dumps(final, indent=2))
    return final


if __name__ == "__main__":
    run_pipeline()
