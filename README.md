# Plutus — dev.fun Arena Poker Agent

A production-ready 6-max No-Limit Texas Hold'em agent for the [dev.fun Arena](https://arena.dev.fun) ($50K prize pool). GTO-calibrated EHS decision making, position-aware preflop ranges, board-texture-aware c-betting, and opponent archetype exploitation.

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env          # fill ARENA_API_KEY + ARENA_AGENT_ID
python -m agent.runner

# Docker
docker build -t plutus . && docker run --env-file .env plutus
```

---

## Strategy (pvp_v3_universal)

All thresholds live in `strategy_rules.json` — hot-reloaded, no restart needed.

### Preflop — position-aware open ranges

| Position | Min EHS | Approx top % of hands |
|----------|---------|-----------------------|
| BTN      | 0.50    | ~45%                  |
| CO       | 0.55    | ~33%                  |
| HJ / MP  | 0.60    | ~22%                  |
| UTG      | 0.64    | ~13%                  |
| SB       | 0.52    | ~40%                  |
| BB defend| 0.45    | ~55%                  |

3-bet: EHS ≥ 0.72. Push/fold (Nash charts): stack ≤ 15 BB.

### Postflop — three-tier betting system

**Flop** (fixed the passive check-down bug):
| EHS range | Action |
|-----------|--------|
| ≥ 0.55    | Value bet — 33% pot dry / 75% pot wet board |
| 0.33–0.55 | C-bet at 60% frequency (draws, overcards, pairs) |
| ≤ 0.28    | Pure bluff at 30% frequency |
| else      | Check |

**Turn:**
| EHS range | Action |
|-----------|--------|
| ≥ 0.55    | Value bet 75% pot |
| 0.40–0.55 | Barrel at 52% frequency (semi-bluffs) |
| ≤ 0.22    | Bluff at 20% |
| else      | Check |

**River:**
| EHS range | Action |
|-----------|--------|
| ≥ 0.58    | Value bet 75% pot |
| ≤ 0.16    | Bluff at 12% |
| else      | Check-fold |

### Opponent Exploitation

Tracks VPIP/PFR/AF/3bet/fold-to-cbet per opponent, classifies archetype, adjusts automatically:

| Archetype     | Exploit |
|---------------|---------|
| Fish (VPIP>40%) | Value bet wider, size up 1.25×, eliminate bluffs |
| Nit (VPIP<18%) | Bluff 1.5× more, steal frequently |
| Maniac (AF>4)  | Trap (check strong hands), no river bluffs |
| Calling station| Pure value, maximum sizing, zero bluffs |
| LAG            | Tighten slightly, trap premiums |

---

## Architecture

```
agent/
  runner.py              — Async poll loop, multi-table, deadlines
  arbiter.py             — Decision router (safety → push/fold → adaptive)
  strategies/
    adaptive.py          — Rule executor (reads strategy_rules.json)
    base.py              — Helpers: pot odds, sizing, position

engine/
  ehs.py                 — Monte Carlo EHS, 400–800 samples, LRU cached
  hand_eval.py           — 7-card evaluator (pure Python)
  icm.py                 — ICM EV + bubble factor
  push_fold.py           — Nash push/fold at ≤15 BB

models/
  opponent_tracker.py    — Persistent VPIP/PFR/AF/cbet stats per agent
  adaptive_memory.py     — Self-tuning + mistake logging
  bot_pattern_detector.py— Archetype classifier

api/
  arena_client.py        — HTTP (sync+async), circuit breaker, retry
  state_parser.py        — Arena JSON → GameContext
  hand_buffer.py         — Accumulates all actions per hand
  hand_processor.py      — Full-hand stat replay after hand ends

config/
  settings.py            — Pydantic settings (.env)
```

---

## Decision Flow

```
poll → parse GameContext
  1. Safety override: AA/KK/QQ → always 3bet/call
  2. Push/fold: stack ≤ 15BB → Nash charts
  3. Classify opponent archetype → adjust thresholds
  4. EHS via Monte Carlo (street-calibrated samples)
  5. adaptive.py:
       preflop → position-aware open / 3bet / fold
       flop    → value / c-bet / bluff / check (board texture sizing)
       turn    → value / barrel / bluff / check
       river   → value / bluff / check-fold
  6. POST /texas/action with reasoning
```

---

## EHS Samples

| Street  | Samples | Approx variance |
|---------|---------|-----------------|
| Preflop | 400     | ±2.5%           |
| Flop    | 600     | ±2.0%           |
| Turn    | 800     | ±1.7%           |
| River   | 800     | ±1.7%           |

Cached per (hole, board) key; cleared on each new hand.

---

## Environment Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `ARENA_API_KEY` | — | Auto-saved after register |
| `ARENA_AGENT_ID` | — | Auto-saved after register |
| `ARENA_COMPETITION_ID` | `seed_poker_eval_s1` | Competition to join |
| `ARENA_INVITE_CODE` | — | Faucet invite for wallet |
| `ARENA_BASE_URL` | `https://arena.dev.fun` | API host |
| `AGENT_NAME` | `Plutus` | Display name |
| `DECISION_BUDGET_S` | `1.5` | Per-table timeout (s) |
| `POLL_INTERVAL_S` | `0.8` | Idle poll interval (s) |
| `HEALTH_PORT` | `8080` | Health check port |
| `JSON_LOGS` | `false` | Structured JSON logs |
| `USE_ASYNC` | `true` | Multi-table concurrency |

---

## State Files

| File | Contents |
|------|----------|
| `.arena-credentials` | API key + agent ID (chmod 600) |
| `.arena-poker-state` | Opponent stats |
| `.arena-adaptive-state` | Self-tuning params |
| `decisions.jsonl` | Full decision log |
| `strategy_rules.json` | All thresholds (hot-reloaded) |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `reasoning is required` | Auto-filled on every submit |
| HTTP 503 / 429 | Client retries with `Retry-After`; poll ≥ 550ms |
| Circuit breaker open | Resets on next success |
| `phase=waiting_user` | Benchmark idle — polls until hand dealt |
| 402 wallet empty | Set `ARENA_INVITE_CODE` |

---

## Tests

```bash
pytest tests/ -v
```

---

## References

- [Arena](https://arena.dev.fun) · [Docs](https://docs.dev.fun)
- EHS: Chen & Ankenman, *The Mathematics of Poker*
- GTO thresholds calibrated via DeepSeek-R1 analysis against Libratus/Pluribus research
