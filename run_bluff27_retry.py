#!/usr/bin/env python3
"""Retry failed agents A, B, C, E from the bluff27 analysis."""
import json, time
from pathlib import Path
import urllib.request

API_KEY = "sk-d6ecb8fa5bae4c4b9ee6db93188ac8d6"
API_URL = "https://api.deepseek.com/chat/completions"

RESEARCH_SYSTEM = """You are DeepSeek — a world-class poker AI analyst.
KEY FACTS (confirmed from 4000+ observations):
- Arena bots: fold to ALL-IN 95%, fold to 3BB open 48%, fold to c-bet 5%
- Pure pot-odds callers, never bluff, consistent behavior
- No-rebuy cash game (elimination if bust)
- Stack ~1000 chips, blinds 1/2

Show all EV math. Return ONLY valid JSON."""


def ds_call(prompt, agent_id, model="deepseek-chat", max_tokens=4096):
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": RESEARCH_SYSTEM},
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
                print(f"[{agent_id}] OK ({len(content)} chars)")
                return content
        except Exception as e:
            print(f"[{agent_id}] attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(5)
    return ""


A_PROMPT = """Compute EV for 7-2 offsuit bluff strategy vs arena bots (fold-to-allin=95%, fold-preflop=48%, fold-cbet=5%).
Stack=1000 chips, blinds=1/2.

Calculate:
1. EV_open_3BB = 0.48 * 1 + 0.52 * EV_postflop (preflop pot=3BB, opponents in=1.5BB, hero=1.5BB extra)
2. EV_flop_allin after called preflop: pot=6BB, stack=997. EV = 0.95*6 + 0.05*(0.31*1003 - 0.69*997)
3. EV_preflop_allin: EV = 0.95*3 + 0.05*(0.31*1003 - 0.69*1000)
4. Best 2-street strategy: open 3BB then flop-allin = EV?
   = 0.48*1 + 0.52*(0.95*6 + 0.05*(0.31*1003 - 0.69*997))
5. Frequency: 7-2 offsuit appears 4/1326 = 0.302% of hands. Per 100 hands: 0.302

Return JSON:
{"ev_preflop_allin": -X, "ev_open_fold": X, "ev_open_then_flop_allin": X, "best_strategy": "open_3bb_then_flop_allin", "frequency_per_100_hands": 0.302, "conclusions": ["..."], "math_shown": true}"""

B_PROMPT = """Design optimal preflop strategy for 7-2 offsuit bluff in this arena:
- Fold-to-3BB-open: 48%, fold-to-allin: 95%
- Stack 1000 chips, blinds 1/2

Scenarios:
1. Unopened pot: open 3BB (48% fold), if called → flop all-in (95% fold). Total fold = 97.4%
2. Facing a raise: 3-bet all-in preflop OR 3-bet small then flop-allin. Which is better EV?
3. In BB facing 3BB raise: shove or fold? EV_shove = 0.95*(3+1) + 0.05*(0.31*1003 - 0.69*1000)
4. Never call with 7-2 offsuit (EHS=0.31 = always losing call)

Return JSON:
{"unopened_optimal": {"action": "raise_3bb", "then": "flop_allin"},
 "facing_raise_optimal": {"action": "reraise_allin", "ev_math": "..."},
 "bb_facing_raise": {"action": "shove_allin", "ev": "..."},
 "never_call": true,
 "preflop_rules": {"always_raise": true, "raise_size_bb": 3.0, "shove_if_facing_allin": false, "fold_to_opponent_allin": true}}"""

C_PROMPT = """Design optimal postflop strategy for 7-2 offsuit bluff.
Opponent called our preflop 3BB raise. Pot = 6BB. Stack remaining = 994BB.
Arena bots: fold to all-in 95%, fold to c-bet only 5%.

Key EV calculation:
EV_flop_allin = 0.95 * 6 + 0.05 * (0.31 * 1000 - 0.69 * 994) = ?
EV_cbet_1xpot = 0.05 * 6 + 0.95 * (0.31 * 12 - 0.69 * 6) = ?
EV_check_fold = 0 (give up)

Board texture:
- If 7 or 2 appears on board: EHS improves to ~0.60, switch to value bet mode
- Otherwise: pure bluff, go all-in

Return JSON:
{"ev_flop_allin": X, "ev_cbet_only": X, "ev_check_fold": 0,
 "primary_action": "all_in", "made_pair_threshold": 0.55,
 "postflop_rules": {
   "flop_no_bet": "all_in", "flop_facing_bet": "fold_unless_made_hand",
   "turn_if_called_flop": "fold", "river_action": "fold"}}"""

E_PROMPT = """Analyze arena bot exploit for 7-2 offsuit bluff sequence.
Bot facts: fold-to-allin=95%, fold-to-3BB-open=48%, fold-to-cbet=5%

Calculate fold probabilities:
Sequence 1: Preflop all-in → 95% fold immediately
Sequence 2: Open 3BB (48% fold) → flop all-in (95% fold if called) = total fold = 0.48 + 0.52*0.95 = 97.4%
Sequence 3: 3 streets c-bet = (1 - 0.95^3) = 14.3% fold (weak)

EV comparison (pot=3BB at open, stack=1000):
Seq1: EV = 0.95*1.5 + 0.05*(0.31*1003 - 0.69*1000) = ?
Seq2: EV = 0.48*1 + 0.52*(0.95*6 + 0.05*(0.31*1003 - 0.69*994)) = ?

Return JSON:
{"sequence1_fold_prob": 0.95, "sequence2_fold_prob": 0.974,
 "sequence1_ev": X, "sequence2_ev": X,
 "optimal_sequence": ["open_3bb_preflop", "all_in_on_flop"],
 "never_multi_street_cbet": true,
 "key_insight": "sequence2 has higher fold probability AND more EV than preflop shove"}"""


def main():
    results = {}

    print("Running Agent A (EV math)...")
    results["A_ev_math"] = ds_call(A_PROMPT, "A", "deepseek-chat", 3000)
    time.sleep(2)

    print("Running Agent B (preflop)...")
    results["B_preflop"] = ds_call(B_PROMPT, "B", "deepseek-chat", 3000)
    time.sleep(2)

    print("Running Agent C (postflop)...")
    results["C_postflop"] = ds_call(C_PROMPT, "C", "deepseek-chat", 3000)
    time.sleep(2)

    print("Running Agent E (exploit)...")
    results["E_exploit"] = ds_call(E_PROMPT, "E", "deepseek-chat", 3000)
    time.sleep(2)

    Path("bluff27_retry_results.json").write_text(json.dumps(results, indent=2))
    print("\nRetry results saved to bluff27_retry_results.json")

    for k, v in results.items():
        print(f"\n--- {k} ---")
        print(v[:500])


if __name__ == "__main__":
    main()
