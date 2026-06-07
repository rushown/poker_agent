#!/usr/bin/env python3
"""
8-agent DeepSeek analysis: per-table decision quality, weak allins, escalation vs allin,
opponent exploitation, and scoring simulation for Plutus / Devil / Aggro agents.
"""
import json, re, sys, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scripts.ds_query import query_parallel, query, RESEARCH_SYSTEM, ORCHESTRATOR_SYSTEM, SIMULATION_SYSTEM

# ── Data extraction ──────────────────────────────────────────────────────────

def extract_decisions(filepath, agent_name):
    decisions = []
    try:
        with open(filepath) as f:
            for line in f:
                try:
                    outer = json.loads(line.strip())
                    text = outer.get('text','')
                    m = re.search(r'- (\{.*\})', text)
                    if m:
                        inner = json.loads(m.group(1))
                        if inner.get('event') == 'decision':
                            inner['agent'] = agent_name
                            decisions.append(inner)
                except: pass
    except FileNotFoundError:
        print(f"[warn] {filepath} not found")
    return decisions

all_decisions = []
for name, path in [
    ('plutus',       'plutus.log'),
    ('devil',        'devil/devil.log'),
    ('plutus_aggro', 'plutus_aggro/plutus_aggro.log'),
    ('brutal',       'brutal/brutal.log'),
]:
    d = extract_decisions(path, name)
    all_decisions.extend(d)
    print(f"[data] {name}: {len(d)} decisions")

# Group by table
tables = collections.defaultdict(list)
for d in all_decisions:
    tables[f"{d['agent']}::{d['table_id']}"].append(d)
for k in tables:
    tables[k].sort(key=lambda x: x['ts'])

def table_seq(d):
    key = f"{d['agent']}::{d['table_id']}"
    return [{'a': x['action_taken'], 'e': round(x['ehs'],3), 'amt': x['amount']}
            for x in tables.get(key, [])]

# ── Pattern extraction ───────────────────────────────────────────────────────

# 1. Weak allins: EHS < 0.72, committed large chips (>50)
weak_allins = []
for d in all_decisions:
    if d.get('ehs',0) < 0.72 and d.get('amount',0) > 50 and d.get('action_taken') in ('raise','bet','call'):
        weak_allins.append({
            'agent': d['agent'], 'hand': d['hand_number'], 'table_id': d['table_id'],
            'ehs': d['ehs'], 'action': d['action_taken'], 'amount': d['amount'],
            'n_opp': len(d.get('opponent_ids',[])),
            'seq': table_seq(d),
        })

# 2. Missed allins: EHS >= 0.82 but used small raise/bet (< 500)
missed_allins = []
for d in all_decisions:
    if d.get('ehs',0) >= 0.82 and d.get('action_taken') in ('raise','bet') and d.get('amount',0) < 500:
        missed_allins.append({
            'agent': d['agent'], 'hand': d['hand_number'], 'table_id': d['table_id'],
            'ehs': d['ehs'], 'action': d['action_taken'], 'amount': d['amount'],
            'n_opp': len(d.get('opponent_ids',[])),
            'seq': table_seq(d),
        })

# 3. Escalation patterns: multi-bet tables where peak EHS 0.60-0.82 (two-pair / flush territory)
#    - these are cases where escalation happened but should agent have gone allin earlier?
escalations = []
for key, seq in tables.items():
    bets = [x for x in seq if x.get('action_taken') in ('raise','bet')]
    if len(bets) >= 2:
        max_ehs = max(x['ehs'] for x in seq)
        max_amount = max(x['amount'] for x in bets)
        # peak in two-pair / strong-but-not-nut territory
        if 0.60 < max_ehs < 0.92 and max_amount > 10:
            escalations.append({
                'agent': seq[0]['agent'], 'table_id': seq[0]['table_id'],
                'hand': seq[0]['hand_number'], 'peak_ehs': round(max_ehs,3),
                'max_bet': max_amount, 'n_bets': len(bets),
                'seq': [{'a': x['action_taken'], 'e': round(x['ehs'],3), 'amt': x['amount']} for x in seq],
            })

# 4. High EHS folds (EHS >= 0.60 and folded)
high_folds = []
for d in all_decisions:
    if d.get('ehs',0) >= 0.60 and d.get('action_taken') == 'fold':
        high_folds.append({
            'agent': d['agent'], 'hand': d['hand_number'],
            'ehs': d['ehs'], 'n_opp': len(d.get('opponent_ids',[])),
            'seq': table_seq(d),
        })

# 5. Two-pair territory bets (EHS 0.55-0.72): right sizing?
two_pair_bets = []
for d in all_decisions:
    if 0.55 <= d.get('ehs',0) < 0.72 and d.get('action_taken') in ('raise','bet') and d.get('amount',0) > 5:
        two_pair_bets.append({
            'agent': d['agent'], 'hand': d['hand_number'],
            'ehs': d['ehs'], 'action': d['action_taken'], 'amount': d['amount'],
            'n_opp': len(d.get('opponent_ids',[])),
            'seq': table_seq(d),
        })

# 6. Good hands that called opponent big bets (EHS 0.55-0.80, calling > 30)
called_large_with_medium = []
for d in all_decisions:
    if 0.55 <= d.get('ehs',0) < 0.80 and d.get('action_taken') == 'call' and d.get('amount',0) > 30:
        called_large_with_medium.append({
            'agent': d['agent'], 'hand': d['hand_number'],
            'ehs': d['ehs'], 'amount': d['amount'],
            'n_opp': len(d.get('opponent_ids',[])),
            'seq': table_seq(d),
        })

# Per-agent stats
agent_stats = {}
for name in ['plutus','devil','plutus_aggro','brutal']:
    d = [x for x in all_decisions if x['agent']==name]
    if not d: continue
    actions = dict(collections.Counter(x['action_taken'] for x in d))
    ehs = [x['ehs'] for x in d]
    large = [x for x in d if x.get('amount',0) > 200]
    agent_stats[name] = {
        'decisions': len(d),
        'actions': actions,
        'avg_ehs': round(sum(ehs)/len(ehs), 3),
        'ehs_dist': {
            '<0.40': sum(1 for e in ehs if e < 0.40),
            '0.40-0.55': sum(1 for e in ehs if 0.40 <= e < 0.55),
            '0.55-0.72': sum(1 for e in ehs if 0.55 <= e < 0.72),
            '0.72-0.82': sum(1 for e in ehs if 0.72 <= e < 0.82),
            '>=0.82': sum(1 for e in ehs if e >= 0.82),
        },
        'large_bets_gt200': len(large),
        'large_bet_ehs': [round(x['ehs'],3) for x in large],
    }

print(f"\n[data] Patterns: weak_allins={len(weak_allins)}, missed_allins={len(missed_allins)}, "
      f"escalations={len(escalations)}, high_folds={len(high_folds)}, "
      f"two_pair_bets={len(two_pair_bets)}, called_large_medium={len(called_large_with_medium)}")

# ── Known bot behavior ───────────────────────────────────────────────────────

BOT_FACTS = """CONFIRMED OPPONENT BEHAVIOR (pokerbench bots, from 4000+ decisions):
- Pure pot-odds callers: call when equity >= required_equity. NEVER bluff.
- Fold preflop to 3BB open: 42-55% of the time
- Fold to flop c-bet: ~5% (almost never fold postflop bets)
- Fold to all-in: ~95% (need 95%+ equity to call stack-off)
- Value bet breakeven vs these bots: EHS 0.48 at 0.75x pot
- All-in EV formula: EV_allin = 0.95*pot + 0.05*(EHS*2*pot - (1-EHS)*pot)
- Value bet EV: EV_value(0.75x) = EHS*(1.75*pot) - (1-EHS)*(0.75*pot) = pot*(2.5*EHS - 0.75)
- Cross-over point: all-in ALWAYS beats value bet when EHS < 0.72
  Proof: 0.95 + 0.10*EHS > 2.5*EHS - 0.75 => 1.70 > 2.40*EHS => EHS < 0.708
- EHS two-pair range: ~0.55-0.72 (strong hand vs 1 opp, medium vs 3+ opps)
- Opponents WILL call 3x pot bets at EHS >= 0.43, 5x pot at EHS >= 0.33"""

# ── Build 8 agent prompts ────────────────────────────────────────────────────

wa_json   = json.dumps(weak_allins[:25], indent=2)
ma_json   = json.dumps(missed_allins, indent=2)
esc_json  = json.dumps(escalations[:25], indent=2)
hf_json   = json.dumps(high_folds[:20], indent=2)
tp_json   = json.dumps(two_pair_bets[:20], indent=2)
cl_json   = json.dumps(called_large_with_medium[:20], indent=2)
stats_json= json.dumps(agent_stats, indent=2)
rules     = json.loads(Path('strategy_rules.json').read_text())
rules_json= json.dumps(rules, indent=2)

# ── Agent A: Per-table replay — what alternative moves would score ────────────
AGENT_A = f"""You are a poker hand replayer. Analyze these decision sequences from a poker bot.
Each entry = one table session: a sequence of actions with EHS (equity hand strength 0-1) and amount.

{BOT_FACTS}

TASK: For each sequence, compute:
1. What the agent actually did (action + amount + EHS at each street)
2. EV_actual: estimate chips won/lost based on the action taken
3. EV_optimal: what the OPTIMAL action would have been and its EV
4. Delta_EV: improvement if optimal was taken
5. Score impact: how many BB/100 this decision pattern costs or gains

Focus on sequences where the agent:
- Committed large chips (amount > 50) with EHS < 0.72 (two-pair / marginal territory)
- OR had EHS >= 0.75 but chose escalated raises instead of all-in
- OR folded with EHS >= 0.60

Use this formula for scoring:
- Starting stack = 1000 chips (100BB), blinds 5/10
- EV_bet(size) = EHS * (pot + size) - (1-EHS) * size  [if called, bot calls with EHS >= size/(pot+size)]
- EV_allin = 0.95 * pot + 0.05 * (EHS * 2*pot - 0.05*pot)  [95% fold to allin]
- EV_fold = 0

For each table sequence, determine the BEST sequence of actions and estimate the EV difference in chips.

WEAK ALLIN DATA (large bets with EHS < 0.72):
{wa_json[:4000]}

Return JSON:
{{
  "table_replays": [
    {{
      "hand": 0,
      "agent": "",
      "actual_sequence": "described",
      "actual_ev_estimate": 0.0,
      "optimal_sequence": "what should have been done",
      "optimal_ev_estimate": 0.0,
      "delta_ev_chips": 0.0,
      "delta_bb100": 0.0,
      "key_mistake": "description of the critical decision error",
      "root_cause": "which strategy rule caused this"
    }}
  ],
  "total_estimated_ev_loss_chips": 0.0,
  "most_costly_mistake_pattern": "",
  "fix_recommendation": ""
}}"""

# ── Agent B: Weak allin forensics — two pair / marginal hands going huge ─────
AGENT_B = f"""Forensic analysis of questionable large bets by a poker bot.
The bot committed large amounts (>50 chips) with EHS < 0.72 (two-pair / flush-draw territory).

{BOT_FACTS}

EHS interpretation guide:
- EHS 0.55-0.65 = two pair / weak flush draw / top pair weak kicker (MEDIUM hand)
- EHS 0.65-0.72 = strong two pair / flush draw with pair / overpair (MEDIUM-STRONG)
- EHS 0.72-0.82 = top two pair / set on safe board / strong flush (STRONG)
- EHS 0.82+ = near-nut: nut flush / straight / top set (VERY STRONG — allin territory)

CRITICAL: Against these bots with 95% fold-to-allin:
- Two-pair (EHS 0.55-0.72): VALUE BET is BETTER than allin (bot calls EV math favors smaller bet)
- But if bot called our raise/bet and we STILL have EHS < 0.72: BOT HAS US BEAT
  This means our two-pair is behind — bot has a SET or better
- When bot calls our bet with EHS 0.55-0.65 range: we are likely losing

WEAK ALLIN DATA:
{wa_json[:4000]}

HIGH EHS FOLDS (EHS >= 0.60 but agent folded):
{hf_json[:2000]}

For each case:
1. Identify: is the EHS consistent with two-pair? flush draw? or something else?
2. Given the sequence, was the opponent applying pressure (we called THEIR bet) or were we betting?
3. At the EHS and bet size shown — what is the required_equity for the call?
   required_equity = amount / (amount + estimated_pot)
4. Was this a profitable call/bet or did opponent have us dominated?
5. What SHOULD the agent have done at that EHS?

Return JSON:
{{
  "weak_allin_analysis": [
    {{
      "hand": 0,
      "agent": "",
      "ehs_interpretation": "two_pair/flush_draw/overpair/etc",
      "sequence_summary": "",
      "was_agent_betting_or_calling": "betting/calling",
      "opponent_likely_holding": "set/nut_flush/straight/better_two_pair",
      "required_equity_for_call": 0.0,
      "verdict": "profitable/breaking_even/losing",
      "correct_action": "fold/check/smaller_bet/value_bet_correct_size",
      "ev_cost": 0.0
    }}
  ],
  "two_pair_strategy_recommendation": {{
    "ehs_0.55_to_0.65": "what to do with two pair weak",
    "ehs_0.65_to_0.72": "what to do with two pair strong",
    "key_rule": "when two pair should FOLD to large bets"
  }},
  "total_chips_wasted_on_weak_allins": 0.0,
  "critical_finding": ""
}}"""

# ── Agent C: Escalation analysis — should these have been immediate allins? ──
AGENT_C = f"""Analyze escalation sequences in a poker bot — cases where the agent made multiple
bets/raises on the same hand instead of going all-in immediately.

{BOT_FACTS}

KEY QUESTION: Given the 95% fold-to-allin rate, is escalation EVER correct?

MATH:
At pot=P, stack=S (say 1000):
- EV_immediate_allin = 0.95*P + 0.05*(EHS * (P+S) - (1-EHS)*S)
- EV_escalate_2_streets = EV[bet 0.75P, get called, then bet again bigger]
  = [if opponent calls 0.75P]: EHS_flop * 2 * EV_bet_turn - (1-EHS_flop)*0.75P
  [complex to compute — but approximately: you extract bet+bet_2 if win vs just pot if allin-fold]

For a bot with 95% fold-to-allin:
- Escalation EXTRACTS MORE VALUE when opponent calls (they have a good hand too)
  Because you keep betting and opponent keeps calling
- Escalation LOSES MORE when opponent has a better hand (you get trapped)
- Immediate allin EXTRACTS: 0.95*pot (fold equity) + tiny amount if called

ESCALATION SEQUENCES:
{esc_json[:5000]}

For each:
1. Was the escalation profitable? (EHS stayed strong or got stronger = escalation good)
2. Did EHS DECLINE during escalation? (opponent improving board = we were LOSING, got trapped)
3. Was there a point where EHS crossed below 0.50 but agent kept betting? (CRITICAL ERROR)
4. At what EHS point should the agent have gone allin vs continued escalating?
5. What was the final outcome likely? (EHS at end of sequence)

Return JSON:
{{
  "escalation_analysis": [
    {{
      "hand": 0,
      "agent": "",
      "ehs_trajectory": "ascending/descending/volatile",
      "peak_ehs": 0.0,
      "ehs_at_end": 0.0,
      "escalation_verdict": "correct/too_aggressive/missed_allin/got_trapped",
      "trap_detected": false,
      "trap_description": "if trapped: what happened",
      "allin_point": "which action in sequence should have been allin",
      "ev_impact": 0.0
    }}
  ],
  "escalation_rule_recommendation": {{
    "when_to_escalate": "EHS >= X on ascending board",
    "when_to_allin_immediately": "EHS >= X with pot >= Y",
    "when_to_stop_escalating": "if EHS drops below X or opponent raises",
    "trap_detection_rule": "fold if EHS was X but dropped below Y after call"
  }},
  "escalation_vs_allin_verdict": "",
  "patterns_found": []
}}"""

# ── Agent D: Opponent exploitation of our patterns ───────────────────────────
AGENT_D = f"""Analyze how pokerbench opponents exploited the poker bot's decision patterns.

{BOT_FACTS}

Note: We don't have direct opponent action logs. Infer opponent exploitation from:
1. Cases where agent's EHS DECLINED after calling opponent bets (opponent had stronger hand)
2. Cases where agent committed chips with EHS < 0.50 (below breakeven)
3. Cases where agent folded AFTER committing significant chips (opponent bluffed? or agent fold-equity loss)
4. Large amounts CALLED at medium EHS (opponent value-bet into us, we called, likely lost)

AGENT STATS (action distribution and EHS by action):
{stats_json}

CALLED LARGE WITH MEDIUM EHS (EHS 0.55-0.80, called > 30 chips):
{cl_json[:3000]}

ESCALATION SEQUENCES (opponent may have been trapping):
{esc_json[:3000]}

HIGH EHS FOLDS (opponent forced us to fold strong hands):
{hf_json[:2000]}

For each pattern:
1. What opponent strategy exploited this?
2. How many chips did the exploitation cost?
3. What counter-strategy should we implement?
4. Which specific rule in our strategy_rules.json should change?

Return JSON:
{{
  "exploitation_patterns": [
    {{
      "pattern": "description",
      "how_opponent_exploited": "",
      "chips_cost_estimate": 0.0,
      "hands_affected": 0,
      "counter_strategy": "",
      "rule_change": {{"field": "", "old_value": 0.0, "new_value": 0.0}}
    }}
  ],
  "most_exploited_scenario": "",
  "opponent_strategy_detected": {{
    "type": "pot_odds_caller/value_bettor/trap_setter",
    "key_behavior": "",
    "our_weakness_they_exploited": ""
  }},
  "total_chips_lost_to_exploitation": 0.0,
  "defensive_adjustments": [
    {{"situation": "", "action": "", "threshold": 0.0, "ev_math": ""}}
  ]
}}"""

# ── Agent E: Two-pair EHS 0.55-0.72 optimal sizing ───────────────────────────
AGENT_E = f"""Calculate optimal bet sizing for two-pair and medium-strong hands (EHS 0.55-0.72)
against pokerbench pot-odds callers.

{BOT_FACTS}

SIZING MATH:
For a bet of size B into pot P:
- Pot odds for opponent: required_equity = B/(B+P)
- If bot calls when equity >= B/(B+P):
  - EHS 0.43 covers B/(B+P) = 0.43, so bot calls if B/(B+P) <= 0.43, i.e., B <= 0.754*P
  - So betting 75% pot: bots with EHS >= 0.43 call
- EV of bet size B: EHS*(P+B) - (1-EHS)*B = EHS*P + EHS*B - B + EHS*B = EHS*P + B*(2*EHS-1)
  Wait: if bot calls with EHS >= B/(B+P), then conditional on call, bot_ehs >= B/(B+P)
  Our EV_bet = prob_call * (EHS*(P+B) - (1-EHS)*B) + prob_fold * P
  prob_fold = fraction of bots with EHS < B/(B+P)

For our EHS=0.65 (two pair):
- If bet 0.5*P: bots with EHS >= 0.33 call (67%+ of bots call), EV = ...
- If bet 0.75*P: bots with EHS >= 0.43 call (57%+ call), EV = ...
- If bet 1.0*P: bots with EHS >= 0.50 call (50%+ call), EV = ...
- If allin (5x): bots with EHS >= 0.83 call (only 5% call), EV = 0.95*P + tiny

Compute for EHS = [0.55, 0.60, 0.65, 0.70] x Bet_sizes = [0.5P, 0.75P, 1.0P, 1.5P, 2.0P, allin]
Show which combination maximizes EV.

TWO-PAIR BETS OBSERVED:
{tp_json[:3000]}

Return JSON:
{{
  "sizing_matrix": {{
    "ehs_0.55": {{
      "bet_0.5P": {{"ev": 0.0, "prob_call": 0.0}},
      "bet_0.75P": {{"ev": 0.0, "prob_call": 0.0}},
      "bet_1.0P": {{"ev": 0.0, "prob_call": 0.0}},
      "bet_1.5P": {{"ev": 0.0, "prob_call": 0.0}},
      "allin": {{"ev": 0.0, "prob_call": 0.05}},
      "optimal_size": "0.75P",
      "optimal_ev": 0.0
    }},
    "ehs_0.60": {{}},
    "ehs_0.65": {{}},
    "ehs_0.70": {{}}
  }},
  "optimal_sizing_rules": [
    {{"ehs_range": "0.55-0.60", "optimal_bet": "X% pot", "reason": "", "ev_math": ""}},
    {{"ehs_range": "0.60-0.65", "optimal_bet": "X% pot", "reason": "", "ev_math": ""}},
    {{"ehs_range": "0.65-0.72", "optimal_bet": "X% pot", "reason": "", "ev_math": ""}}
  ],
  "allin_threshold_for_two_pair_territory": 0.0,
  "two_pair_vs_set_detection": "how to adjust sizing when likely behind",
  "sizing_rule_changes": [
    {{"field": "section.key", "old": 0.0, "new": 0.0, "ev_gain_per_100_hands": 0.0}}
  ]
}}"""

# ── Agent F: Escalation vs Immediate Allin — when does each win? ─────────────
AGENT_F = f"""Compare escalation strategy vs immediate all-in for strong hands (EHS 0.72-0.92).

{BOT_FACTS}

CORE QUESTION: At EHS 0.72-0.92, when should the agent:
A) Go all-in immediately (captures 95% fold equity = 0.95*pot)
B) Escalate: bet medium, then bet bigger next street, then all-in
   (captures more when called on multiple streets, but risks trap)

EV MODEL:
- stack S = 1000, pot = P
- Immediate allin: EV = 0.95*P + 0.05*(EHS*2000 - (1-EHS)*1000) (simplified)
  At EHS=0.82: EV_allin = 0.95P + 0.05*(0.82*2000 - 0.18*1000) = 0.95P + 0.05*(1640-180) = 0.95P + 73
- 2-street escalation (bet 0.75P, get called, bet 2P):
  - Prob call flop (opponent EHS >= 0.43) = ~57%
  - If called: pot becomes 2.5P, then bet 2P all-in
  - If fold to 2P allin: get P + 0.75P + 2P = 3.75P? No wait...
  Work through the math for EHS = [0.72, 0.78, 0.82, 0.88, 0.92]

ESCALATION SEQUENCES WITH HIGH PEAK EHS:
{json.dumps([x for x in escalations if x['peak_ehs'] >= 0.72][:15], indent=2)[:3000]}

MISSED ALLINS (EHS >= 0.82, small raise):
{ma_json}

For each EHS tier, compute:
1. EV(immediate allin) vs EV(escalate 2 streets) vs EV(escalate 3 streets)
2. At what stack-to-pot ratio does immediate allin beat escalation?
3. Cases where escalation was clearly right vs clearly wrong

Return JSON:
{{
  "ev_comparison": [
    {{
      "ehs": 0.82,
      "pot": 50,
      "stack": 1000,
      "ev_immediate_allin": 0.0,
      "ev_2street_escalate": 0.0,
      "ev_3street_escalate": 0.0,
      "optimal_strategy": "allin/2street/3street",
      "reason": ""
    }}
  ],
  "escalation_vs_allin_rules": [
    {{
      "ehs_range": "0.72-0.78",
      "stack_to_pot": ">10",
      "optimal": "escalate or allin",
      "ev_math": "",
      "rule_change": {{"field": "section.key", "old": 0.0, "new": 0.0}}
    }},
    {{
      "ehs_range": "0.78-0.82",
      "optimal": "",
      "ev_math": ""
    }},
    {{
      "ehs_range": "0.82-0.92",
      "optimal": "",
      "ev_math": ""
    }}
  ],
  "missed_allin_cases_verdict": [
    {{"hand": 0, "ehs": 0.0, "amount": 0.0, "should_have_been_allin": true, "ev_lost": 0.0}}
  ],
  "allin_threshold_recommendation": 0.0,
  "escalation_allowed_up_to_ehs": 0.0
}}"""

# ── Agent G: Cross-agent comparison — Plutus vs Devil vs Aggro ───────────────
AGENT_G = f"""Compare the decision quality across 3 different poker agent versions
(Plutus, Devil, Plutus_Aggro) to identify which strategy performs best and why.

{BOT_FACTS}

AGENT STATS:
{stats_json}

WEAK ALLINS BY AGENT:
{json.dumps(collections.Counter(x['agent'] for x in weak_allins), indent=2)}

HIGH EHS FOLDS BY AGENT:
{json.dumps(collections.Counter(x['agent'] for x in high_folds), indent=2)}

LARGE BETS (>200) BY AGENT WITH EHS:
{json.dumps({name: [round(x['ehs'],3) for x in all_decisions if x['agent']==name and x.get('amount',0)>200]
             for name in ['plutus','devil','plutus_aggro']}, indent=2)}

ESCALATION TABLES BY AGENT:
{json.dumps(collections.Counter(x['agent'] for x in escalations), indent=2)}

KEY METRICS TO COMPARE:
1. EHS distribution when betting large (>50 chips) — which agent bets biggest with weakest hands?
2. Fold rate at medium EHS (0.50-0.70) — which agent folds too much profitable equity?
3. Allin discipline — which agent goes allin correctly (EHS >= 0.82)?
4. Two-pair bet sizing — which agent extracts most value in EHS 0.55-0.72 range?
5. Trap detection — which agent stops escalating when hand deteriorates?

Return JSON:
{{
  "agent_comparison": {{
    "plutus": {{
      "strengths": [],
      "weaknesses": [],
      "avg_bet_ehs": 0.0,
      "allin_quality_score": 0,
      "two_pair_handling": "good/ok/poor"
    }},
    "devil": {{}},
    "plutus_aggro": {{}}
  }},
  "best_agent_overall": "",
  "best_agent_for_allins": "",
  "best_agent_for_two_pair": "",
  "worst_patterns": [
    {{"agent": "", "pattern": "", "frequency": 0, "cost_estimate": 0.0}}
  ],
  "recommended_merged_strategy": {{
    "take_from_plutus": [],
    "take_from_devil": [],
    "take_from_aggro": [],
    "discard": []
  }},
  "concrete_rule_changes": [
    {{"field": "section.key", "agent_example": "", "old": 0.0, "new": 0.0, "evidence": ""}}
  ]
}}"""

# ── Agent H: Scoring simulation — what would hands have scored? ──────────────
AGENT_H = f"""Simulate what scoring (BB/100) our poker bot should have achieved vs what mistakes cost.

{BOT_FACTS}

ACTUAL PERFORMANCE DATA:
- Plutus: 705 decisions, EHS mean=0.466, large_bets(>200)=30
- Devil: 516 decisions, EHS mean=0.455, large_bets(>200)=12
- Plutus_Aggro: 336 decisions, EHS mean=0.446, large_bets(>200)=8

MEMORY (from previous versions):
- v9 achieved: 507 BB/100 raw (best ever)
- v11 current: ~510 BB/100 raw (theoretical)
- Current agents target: 420-510 BB/100

MISTAKE COUNTS:
- Weak allins (EHS<0.72, amount>50): {len(weak_allins)} cases across all agents
- High EHS folds (EHS>=0.60): {len(high_folds)} cases
- Escalation cases (multi-bet, peak EHS 0.60-0.92): {len(escalations)} cases
- Bad large calls (EHS<0.38, amount>5): {len([d for d in all_decisions if d.get('ehs',0)<0.38 and d.get('action_taken')=='call' and d.get('amount',0)>5])}

For each mistake type, estimate:
- Average chips lost per occurrence
- BB/100 cost (1 BB = 10 chips, per 100 hands)
- What BB/100 would be if mistakes were eliminated

ALSO simulate 3 specific decision improvements:
1. "Allin upgrade": convert all EHS >= 0.82 raises/bets to immediate allin → how many extra chips?
2. "Fold upgrade": never fold with EHS >= 0.55 unless facing all-in (EHS < 0.82) → chips saved?
3. "Sizing fix": replace all EHS 0.55-0.72 bets >100 chips with 0.75x pot value bet → ev change?

Return JSON:
{{
  "current_estimated_bb100": {{
    "plutus": 0.0,
    "devil": 0.0,
    "plutus_aggro": 0.0
  }},
  "mistake_costs": [
    {{"mistake": "weak_allins", "count": {len(weak_allins)}, "avg_chips_lost": 0.0, "bb100_cost": 0.0}},
    {{"mistake": "high_ehs_folds", "count": {len(high_folds)}, "avg_chips_lost": 0.0, "bb100_cost": 0.0}},
    {{"mistake": "bad_escalation", "count": {len(escalations)}, "avg_chips_lost": 0.0, "bb100_cost": 0.0}}
  ],
  "simulated_improvements": [
    {{"upgrade": "allin_upgrade", "extra_bb100": 0.0, "math": ""}},
    {{"upgrade": "fold_upgrade", "extra_bb100": 0.0, "math": ""}},
    {{"upgrade": "sizing_fix", "extra_bb100": 0.0, "math": ""}}
  ],
  "target_bb100_achievable": 0.0,
  "top_3_fixes_by_ev": [
    {{"fix": "", "ev_gain_bb100": 0.0, "implementation": "specific rule change"}}
  ]
}}"""

# ── Run 8 agents in parallel ─────────────────────────────────────────────────
tasks = [
    {"prompt": AGENT_A, "model": "deepseek-reasoner", "system": RESEARCH_SYSTEM,   "timeout": 300, "max_tokens": 8192, "no_cache": True},
    {"prompt": AGENT_B, "model": "deepseek-reasoner", "system": RESEARCH_SYSTEM,   "timeout": 300, "max_tokens": 8192, "no_cache": True},
    {"prompt": AGENT_C, "model": "deepseek-reasoner", "system": SIMULATION_SYSTEM, "timeout": 300, "max_tokens": 8192, "no_cache": True},
    {"prompt": AGENT_D, "model": "deepseek-reasoner", "system": RESEARCH_SYSTEM,   "timeout": 300, "max_tokens": 8192, "no_cache": True},
    {"prompt": AGENT_E, "model": "deepseek-reasoner", "system": SIMULATION_SYSTEM, "timeout": 300, "max_tokens": 8192, "no_cache": True},
    {"prompt": AGENT_F, "model": "deepseek-reasoner", "system": SIMULATION_SYSTEM, "timeout": 300, "max_tokens": 8192, "no_cache": True},
    {"prompt": AGENT_G, "model": "deepseek-reasoner", "system": RESEARCH_SYSTEM,   "timeout": 300, "max_tokens": 8192, "no_cache": True},
    {"prompt": AGENT_H, "model": "deepseek-reasoner", "system": SIMULATION_SYSTEM, "timeout": 300, "max_tokens": 8192, "no_cache": True},
]

print(f"\n[pipeline] Launching 8 parallel DeepSeek-Reasoner agents...")
print("[pipeline] A=per-table replay, B=weak allins forensics, C=escalation analysis")
print("[pipeline] D=opponent exploitation, E=two-pair sizing, F=escalation vs allin")
print("[pipeline] G=cross-agent comparison, H=scoring simulation")

results = query_parallel(tasks, max_workers=8)

labels = ['A_table_replay', 'B_weak_allins', 'C_escalation', 'D_opponent_exploit',
          'E_two_pair_sizing', 'F_escalate_vs_allin', 'G_cross_agent', 'H_scoring']

# Parse JSON from each agent
def parse_json(raw, label):
    clean = raw.strip()
    for fence in ("```json", "```"):
        if clean.startswith(fence):
            clean = clean[len(fence):]
    clean = clean.rstrip("`").strip()
    try:
        return json.loads(clean)
    except Exception as e:
        print(f"[warn] {label} not valid JSON ({e}), storing as text", file=sys.stderr)
        return {"_raw": raw, "_parse_error": str(e), "_label": label}

parsed = {}
for label, raw in zip(labels, results):
    parsed[label] = parse_json(raw, label)
    print(f"[result] {label}: {len(raw)} chars")

# ── Orchestrator synthesis ────────────────────────────────────────────────────
def safe_trim(d, max_chars=2500):
    s = json.dumps(d, indent=2)
    return s[:max_chars] if len(s) > max_chars else s

orch_prompt = f"""You are the strategy orchestrator. Synthesize 8 expert analyses into ONE definitive
set of actionable rule changes for the poker bot.

═══ A: PER-TABLE REPLAY + EV CALCULATIONS ═══
{safe_trim(parsed['A_table_replay'])}

═══ B: WEAK ALLIN FORENSICS (EHS<0.72 large bets) ═══
{safe_trim(parsed['B_weak_allins'])}

═══ C: ESCALATION PATTERN ANALYSIS ═══
{safe_trim(parsed['C_escalation'])}

═══ D: OPPONENT EXPLOITATION PATTERNS ═══
{safe_trim(parsed['D_opponent_exploit'])}

═══ E: TWO-PAIR OPTIMAL SIZING (EHS 0.55-0.72) ═══
{safe_trim(parsed['E_two_pair_sizing'])}

═══ F: ESCALATION VS IMMEDIATE ALLIN ═══
{safe_trim(parsed['F_escalate_vs_allin'])}

═══ G: CROSS-AGENT COMPARISON ═══
{safe_trim(parsed['G_cross_agent'])}

═══ H: SCORING SIMULATION ═══
{safe_trim(parsed['H_scoring'])}

═══ CURRENT RULES ═══
{rules_json[:2000]}

{BOT_FACTS}

SYNTHESIS REQUIREMENTS:
1. For each rule change: field name, old value, new value, EV math, supporting agents (cite A-H)
2. Must answer:
   a) Were any allins done on hands that were too weak (two pair vs strong opponents)?
   b) Were escalated raises extracting maximum value or leaving chips behind vs direct allin?
   c) How did opponents exploit our patterns and what rule prevents it?
3. Reject any change not backed by ≥2 agents
4. Max change per field: 0.08

Return JSON:
{{
  "critical_findings": [
    {{
      "finding": "Were weak allins (two pair vs bots) profitable or losing?",
      "verdict": "",
      "evidence": "",
      "chips_impact": 0.0
    }},
    {{
      "finding": "Escalated raises vs immediate allin — which extracted more value?",
      "verdict": "",
      "evidence": "",
      "optimal_rule": ""
    }},
    {{
      "finding": "How did opponents exploit our strategy?",
      "verdict": "",
      "counter_strategy": ""
    }}
  ],
  "rule_changes": [
    {{
      "field": "section.key",
      "old": 0.0,
      "new": 0.0,
      "ev_gain_bb100": 0.0,
      "evidence": "cite agents A-H",
      "math": "explicit",
      "confidence": "high/medium"
    }}
  ],
  "two_pair_protocol": {{
    "ehs_0.55_to_0.65": {{"action": "", "sizing": "", "fold_to_opponent_bet_above": 0.0}},
    "ehs_0.65_to_0.72": {{"action": "", "sizing": "", "allin_if_pot_gte": 0.0}},
    "ehs_0.72_to_0.82": {{"action": "", "sizing": "", "allin_if_pot_gte": 0.0}}
  }},
  "escalation_vs_allin_protocol": {{
    "immediate_allin_above_ehs": 0.0,
    "escalate_when": "conditions for multi-street escalation",
    "stop_escalating_when": "EHS drops below X or opponent raises"
  }},
  "predicted_bb100_gain": "+X to +Y",
  "top_priority_fix": {{
    "description": "",
    "field": "",
    "old": 0.0,
    "new": 0.0,
    "reason": ""
  }}
}}"""

print("\n[pipeline] Stage 2: Orchestrator synthesizing all 8 agents...")
orch_raw = query(
    prompt=orch_prompt,
    model="deepseek-reasoner",
    no_cache=True,
    timeout=360,
    system=ORCHESTRATOR_SYSTEM,
    max_tokens=8192,
)

final = parse_json(orch_raw, "orchestrator")
final["_agents"] = parsed
final["_data_summary"] = {
    "total_decisions": len(all_decisions),
    "weak_allins": len(weak_allins),
    "missed_allins": len(missed_allins),
    "escalations": len(escalations),
    "high_ehs_folds": len(high_folds),
    "agent_stats": agent_stats,
}

output_path = "log_analysis_result.json"
Path(output_path).write_text(json.dumps(final, indent=2))
print(f"\n[done] Full result saved to {output_path}")
print(f"[done] Orchestrator output: {len(orch_raw)} chars")

# Print summary
print("\n" + "="*60)
print("CRITICAL FINDINGS:")
for cf in final.get("critical_findings", []):
    print(f"\n>>> {cf.get('finding','')}")
    print(f"    VERDICT: {cf.get('verdict', cf.get('counter_strategy',''))}")
    if cf.get('chips_impact'):
        print(f"    CHIPS: {cf.get('chips_impact')}")

print("\nTOP PRIORITY FIX:")
tp = final.get("top_priority_fix", {})
print(f"  {tp.get('description','')}")
print(f"  {tp.get('field','')} : {tp.get('old','')} → {tp.get('new','')}")
print(f"  Reason: {tp.get('reason','')}")

print("\nRULE CHANGES:")
for rc in final.get("rule_changes", [])[:10]:
    print(f"  {rc.get('field','')}: {rc.get('old','')} → {rc.get('new','')} "
          f"(+{rc.get('ev_gain_bb100','?')} BB/100, confidence={rc.get('confidence','?')})")

print(f"\nPredicted improvement: {final.get('predicted_bb100_gain','?')}")
print("="*60)
