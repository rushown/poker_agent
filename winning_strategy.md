# Plutus v4 — Brutal Self-Check Champion

## Philosophy

Plutus does not assume it is winning. Every subsystem reports measurable truth:

| Question | Instrument |
|----------|------------|
| Is the bot running? | `GET /health` — 400 count, idle watchdog, uptime |
| Is strategy X profitable? | `rolling_bb100(200)` per mode in `.arena-brutal-state` |
| Are opponent models right? | `bot_validation` accuracy every 20 hands |
| Is meta-learner switching wisely? | Bootstrap p-value on hand deltas; keep prior if p≥0.1 |
| Did a change hurt? | Auto rollback if bb/100 drops >5 over 200 hands |

## Architecture

```
runner.py
  ├── brutal_check.py      # metrics, intimidation abort, rollback alerts
  ├── meta_learner.py      # MANIAC/TAG/LAG/NIT/EXPLOIT + bootstrap switch
  ├── strategy_modes.py    # INTIMIDATION → EXPLOIT → ICM → ENDGAME
  ├── ab_test.py           # champion vs challenger (200 hands)
  ├── self_test_runner.py  # refuse start if any module self_test fails
  └── decision_log.py      # JSON lines in plutus.log
```

## Phases (validated)

1. **Intimidation** — first 50 hands or 20% of match. Aborts if bb/100 < +2 → forces TAG via meta.
2. **Exploitation** — bot_pattern_detector with `validate_classifications()` every 20 hands.
3. **Meta-learner** — 50-hand blocks; bootstrap before switching strategy.
4. **ICM** — bubble_factor > 1.5 → survival unless stealing from nits.

## API correctness

`build_action_payload()` → `callToAmount`, `minRaiseTo`, `allInToAmount`. Unit tests in `TestActionPayload`.

## Operations

```bash
python -m agent.runner          # self_tests run first
python supervise.py             # health poll every 5s
python auto_train.py            # 500 hands; revert if bb/100 < 5
curl localhost:8080/health
curl localhost:8080/metrics
```

## Files

- `plutus.log` — JSON decisions
- `performance_history.csv` — auto_train rows
- `strategy_report.json` — weights every 50 hands
- `ab_tests.json` — A/B results
- `ISSUE.md` — written on brutal failure

## Commit tag

`feat: tournament-ready champion with brutal self-check`
