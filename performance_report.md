# Plutus Performance Report

**Strategy version:** v2 (toAmount submission fix + tighter UTG opens)  
**Competition:** `seed_poker_eval_s1` (Poker Eval S1)  
**Agent:** Plutus

## Before (v1 baseline)

| Metric | Value |
|--------|-------|
| API submit errors | HTTP 400 on call/raise (incremental `callAmount` sent instead of `callToAmount`) |
| Benchmark stability | Crashes on `allowedActions` type mismatch (fixed prior) |
| UTG open range | Looser mixed frequencies |

## After (v2)

| Change | Impact |
|--------|--------|
| `api/action_amount.format_submission_amount` | Converts call/raise/all-in to Arena **toAmount** semantics at submit time |
| `call_to_amount` on `GameContext` | Strategy uses total commit for calls; pot odds still use incremental `call_amount` |
| Tighter `_UTG_OPEN` | 88+, AJo+, ATs+, KQo+, KJs+ per eval spec |
| `run_benchmark.py` | Overnight orchestrator with `performance_log.json` metrics |
| `decisions.jsonl` + `PlayStatsLogger` | VPIP/PFR/3-bet tracking per session |

## How to collect metrics

```bash
# Single session
python -m agent.runner

# Overnight with auto-restart + metrics
python run_benchmark.py
```

Metrics append to `performance_log.json` every 5s (`completedHands`, `rawBbPer100`, `rawChipDelta`).

Session poker stats:

```bash
python -c "from agent.play_stats import PlayStatsLogger; import json; print(json.dumps(PlayStatsLogger().summary(), indent=2))"
```

(After a run, load decisions from `decisions.jsonl` or wire summary export from runner shutdown.)

## Target benchmarks

| Metric | Target (6-max vs bots) |
|--------|-------------------------|
| bb/100 | > +10 over 500 hands |
| VPIP | 22–28% |
| PFR | 18–24% |
| 3-bet | 6–9% |

## Version history

| Version | Files | Notes |
|---------|-------|-------|
| v1 | `agent/gto_bot.py`, `agent/arbiter.py` (pre-fix) | Incremental amounts to API |
| v2 | + `api/action_amount.py`, `run_benchmark.py`, `agent/play_stats.py` | toAmount fix, orchestrator, logging |

## Next iterations

1. Run 500+ hands via `run_benchmark.py` and record `rawBbPer100`.
2. If VPIP > 30%, tighten CO/BTN opens in `preflop_ranges.py`.
3. If fold-to-steal > 70% at SB/BB, widen blind defend frequencies in `_BB_DEFEND`.
4. Compare showdown vs non-showdown wins from Arena hand history when exposed.
