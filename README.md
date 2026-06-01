# Plutus — dev.fun Arena Poker Agent

A production-oriented, self-learning Texas Hold'em agent for the [dev.fun Arena](https://arena.dev.fun). It combines GTO-style baselines, opponent modeling, ICM-aware tournament play, adaptive self-tuning, and full [arena.md](https://arena.dev.fun/skills/arena.md) onboarding.

---

## Highlights

| Capability | Description |
|------------|-------------|
| **Opponent learning** | Buffers full hand action history; replays into persistent stats (not snapshot guesses) |
| **Adaptive self-tuning** | Reviews its own decisions after each hand; adjusts aggression, bluffs, calls, exploits |
| **Frequency preflop** | Position-based open/3-bet/4-bet frequencies (mixed strategy, not deterministic sets) |
| **ICM + push/fold** | Tournament payouts from API; bubble factor; Nash-style charts at ≤15 BB |
| **Table-weighted exploits** | Blends tendencies across all active opponents, not a single “dominant” villain |
| **Fast EHS** | Street-based Monte Carlo (200–600 samples), LRU cache, weighted ranges vs raises |
| **Async multi-table** | Concurrent decisions per table with hard timeouts |
| **Arena-compliant join** | Invitations, 402 wallet pay + `txHash`, competition picker, heartbeat |
| **Tournament meta (v3)** | Mode switcher: intimidation → exploit → ICM → endgame + meta-learner styles |
| **Bot pattern detection** | Nits, stations, scared money, timing bots → dedicated counters |

---

## Strategy modes (v3)

| Mode | When | Behavior |
|------|------|----------|
| `INTIMIDATION` | First ~20% of match hands | Oversized opens / light shoves (`agent/early_game.py`) |
| `EXPLOITATION` | Detected weak patterns | Steal nits, value vs stations (`models/bot_pattern_detector.py`) |
| `GTO_BALANCED` | Unknown / strong bots | Baseline GTO + exploits |
| `ICM_SURVIVAL` | High bubble factor | Tighten marginal spots |
| `PUSH_FOLD` | ≤15 BB | Nash push/fold charts |
| `ENDGAME_CLOSE` | ≤3 players | Hyper-aggressive HU (`agent/endgame.py`) |

Meta styles (`MANIAC`, `TAG`, `LAG`, `NIT`, `EXPLOIT`) rotate every 50 hands via `agent/meta_learner.py`.

**Run overnight:** `python supervise.py` · **Auto-tune:** `python auto_train.py` · **Details:** [winning_strategy.md](winning_strategy.md)

### Brutal self-check (v4)

- Startup **refuses to run** if any module `self_test()` fails
- `GET /health` — uptime, 400 errors, rolling bb/100, intimidation abort status
- `GET /metrics` — per-mode EV, meta scoreboard, bot classification accuracy, A/B tests
- **Intimidation auto-abort** if first 50 hands &lt; +2 bb/100 → TAG fallback
- **Meta-learner** switches only when bootstrap p &lt; 0.1
- **auto_train** reverts to `.arena-safe-config` if bb/100 &lt; 5 and writes `ISSUE.md`
- Logs: JSON to `plutus.log` with `decision_time_ms`, `ehs`, `strategy_mode`, `action_taken`
- **Benchmark actions** require `reasoning` in POST `/texas/action` (auto-filled from arbiter chat)

## Troubleshooting

| Error | Fix |
|-------|-----|
| `reasoning is required for benchmark actions` | Fixed in v4: every submit includes `reasoning` via `build_action_payload()` |
| HTTP 503 / rate limit | Poll ≥550ms; client retries with `Retry-After` header |
| Circuit breaker open | 503/429 no longer trip breaker; resets on next success |
| `phase=waiting_user` | Benchmark idle — agent polls until a hand is dealt |

---

## Architecture

```
                         ┌──────────────────────────────┐
                         │       STRATEGY ARBITER        │
                         │  GTO + exploits + ICM + adapt │
                         └───────┬──────────┬───────────┘
                                 │          │
              ┌──────────────────┘          └──────────────────┐
     ┌────────▼────────┐  ┌──────────────┐  ┌────────▼────────┐
     │ GTO + PREFLOP   │  │   EXPLOIT    │  │  PUSH-FOLD +   │
     │ frequency ranges│  │ table-weighted│  │  ICM (payouts) │
     └────────┬────────┘  └──────┬───────┘  └────────────────┘
              │                  │
     ┌────────▼──────────────────▼────────┐
     │           EHS ENGINE (cached)       │
     │  preflop 200 / flop 400 / turn+ 600 │
     └────────────────┬───────────────────┘
                      │
     ┌────────────────┼────────────────┐
     │                │                │
┌────▼─────┐   ┌──────▼──────┐  ┌─────▼──────┐
│ OPPONENT │   │  ADAPTIVE   │  │  TECHNIQUES │
│ TRACKER  │   │   MEMORY    │  │ MDF, pot   │
│ per-villain│  │ self-tuning │  │ odds, blend│
└──────────┘   └─────────────┘  └────────────┘

     ┌─────────────────────────────────────┐
     │  RUNNER (async poll + hand buffer)   │
     │  arena onboarding · heartbeat        │
     └─────────────────────────────────────┘
```

---

## Opponent learning (fixed)

**Problem (original):** Hand stats were updated from a single table snapshot when `handNumber` changed — incomplete action history → wrong VPIP/PFR/cbet.

**Solution:**

1. `api/hand_buffer.py` — diffs each poll snapshot; accumulates all actions per `(table_id, hand_number)`.
2. `api/hand_processor.py` — `process_full_hand()` walks actions in order and updates stats.
3. Extra stats: fold-to-steal, RFI, check-raise flop, fold-to-cbet by street, river aggression.

State file: **`.arena-poker-state`**

Archetypes: `fish | nit | tag | lag | maniac | unknown` (Bayesian confidence gate for exploits).

---

## Adaptive self-learning

**`models/adaptive_memory.py`** records every decision (action, EHS, pot odds, street) and, at hand end, classifies mistakes:

| Mistake | Adjustment |
|---------|------------|
| Bad call (EHS &lt; pot odds) | Tighter `call_threshold_adj` |
| Bad fold | Looser calls |
| Overbluff | Lower `bluff_frequency` / cbet |
| Missed value | Higher `postflop_value_bias` |
| Spewy hand (−15 BB) | Less aggression, tighter ICM |

Tuning persists in **`.arena-adaptive-state`**. The arbiter reads live knobs for preflop aggression, exploit blend, value sizing, and ICM tightness.

**`agent/techniques.py`** — MDF (minimum defense frequency), pot-odds calls, solver-style value bet tiers, exploit blend vs recent win rate.

---

## Pre-flop & post-flop strategy

- **`agent/preflop_ranges.py`** — frequency tables per position (UTG→BTN); `should_open` / `should_3bet` / `should_4bet` with RNG; positional 3-bet sizing.
- **`agent/gto_bot.py`** — MDF-aware defense; adaptive bet sizing; bluff gating from fold-to-cbet.
- **`engine/push_fold.py`** — push/fold charts by position and stack depth (≤8 BB / ≤15 BB).
- **`engine/icm.py`** — Malmuth-Harville ICM; `bubble_factor` tightens raises/calls near the bubble when payouts are loaded.

---

## Exploit engine

**`agent/exploit_strategies.py`** — exploits vs table-weighted pool stats:

| Archetype | Behavior |
|-----------|----------|
| **Fish** | Value on dry boards; smaller value on scary boards; no river bluffs |
| **Nit** | Steals; small cbets when fold-to-cbet is high |
| **Maniac** | Traps; folds marginal without equity |
| **LAG** | 4-bets wide 3-bettors |
| **TAG** | Auto-profit cbets / turn barrels when fold-to-cbet is high |

Bet sizing adapts (e.g. 33% pot vs nits, ~75% vs fish).

---

## EHS engine

**`engine/ehs.py`**

- Formula: `EHS = HS×(1−Npot) + (1−HS)×Ppot`
- Samples: **preflop 200**, **flop 400**, **turn/river 600**
- **LRU cache** (cleared each new hand)
- **Weighted opponent sampling** when facing raises (tighter villain range)

---

## Arena integration ([arena.md](https://arena.dev.fun/skills/arena.md))

Follows the official dev.fun Arena skill:

| Step | Implementation |
|------|----------------|
| Returning player | Load `.arena-credentials` (JSON or `apiKey=` lines) → `GET /agent/me` |
| Introspection | `GET /__introspection` once at startup |
| Invitations | `GET /agent/invitations` → auto-claim before join |
| Competition pick | Latest `startAt` / highest `seasonNumber`; override with `ARENA_COMPETITION_ID` |
| Join | `POST /texas/join` with `competitionId` |
| Entry fee (402) | Wallet balance → `POST /agent/wallet/transfer/native` → retry with `txHash` |
| Claim gate (403) | Surfaces `claimUrl` from `/auth/claim/status` |
| Play loop | Poll `GET /texas/pending-actions` → `POST /texas/action` |
| Heartbeat | Every 4h (min 1h dedup): leaderboard, inbox, session stats |
| Owner messages | **`.arena-owner-messages.txt`** (formatted updates) |

Modules: `api/arena_onboarding.py`, `agent/arena_setup.py`, `agent/heartbeat.py`, `agent/owner_messages.py`

Base URL: `https://arena.dev.fun/api/arena` · Auth: `x-arena-api-key`

---

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
cp .env.example .env
```

### 2. Register (once)

```bash
python -m agent.runner --register "Plutus" "I count outs, not prayers."
```

Handle is auto-derived from the name (`plutus`). The **full `arena_sk_...` API key is shown once** — save it to `.env` as `ARENA_API_KEY` or keep `.arena-credentials`.

### 3. List competitions (if several are live)

```bash
python -m agent.runner --list-competitions
```

Set in `.env`:

```bash
ARENA_COMPETITION_ID=<your-competition-id>
```

### 4. Join lobby + verify setup

```bash
python -m agent.runner --onboard
```

Check **`.arena-owner-messages.txt`** for join status, entry-fee instructions, or claim URL.

### 5. Run

```bash
# Safe test (no API submits)
python -m agent.runner --dry-run --max-hands 20

# Live play (async, multi-table)
python -m agent.runner

# Sync polling (single-threaded)
python -m agent.runner --sync
```

### 6. Docker

```bash
docker build -t plutus-poker .
docker run --env-file .env -p 8080:8080 -v $(pwd)/.arena-credentials:/app/.arena-credentials plutus-poker
```

Health check: `http://localhost:8080/health` (hands played, win rate, adaptive summary, last heartbeat).

---

## CLI reference

| Command | Purpose |
|---------|---------|
| `python -m agent.runner` | Full setup + join + poker loop |
| `--register "Name" "Quote"` | Create agent (arena.md flow) |
| `--list-competitions` | Show live competitions + IDs |
| `--onboard` | Step 0 + join lobby without playing |
| `--heartbeat` | Send one owner status update now |
| `--competition-id <id>` | Force a specific competition |
| `--dry-run` | Decide locally; do not submit actions |
| `--max-hands N` | Stop after N decisions |
| `--sync` | Disable async multi-table polling |

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `ARENA_API_KEY` | — | `arena_sk_...` from registration |
| `ARENA_AGENT_ID` | — | Optional; also in credentials file |
| `ARENA_COMPETITION_ID` | — | Force competition when multiple are live |
| `ARENA_INVITE_CODE` | — | Legacy faucet coupon if 402 and wallet empty |
| `ARENA_BASE_URL` | `https://arena.dev.fun` | API host |
| `POLL_INTERVAL_S` | `0.8` | Seconds between polls when idle |
| `DECISION_BUDGET_S` | `1.5` | Hard per-table decision timeout |
| `HEARTBEAT_INTERVAL_S` | `14400` | Heartbeat while running (4h) |
| `HEARTBEAT_MIN_INTERVAL_S` | `3600` | Minimum gap between heartbeats (1h) |
| `HEALTH_PORT` | `8080` | Health HTTP server |
| `JSON_LOGS` | `false` | Structured JSON logs for production |
| `USE_ASYNC` | `true` | Concurrent table handling |
| `OWNER_MESSAGE_FILE` | `.arena-owner-messages.txt` | Owner-facing status log |

State files (gitignore recommended):

| File | Contents |
|------|----------|
| `.arena-credentials` | API key + agent ID |
| `.arena-poker-state` | Opponent stats |
| `.arena-adaptive-state` | Self-tuning parameters |
| `.arena-heartbeat-state` | Last heartbeat timestamp |
| `.arena-owner-messages.txt` | Human-readable arena updates |

---

## Decision flow

```
poll GET /texas/pending-actions
  → buffer actions into hand history
  → on hand end: process_full_hand() + adaptive.finish_hand()

  for each table (async, parallel):
    parse_table() → GameContext (+ ICM payouts if tournament)
    if time budget low → safe check/fold
  else:
    EHS (cached, street samples)
    if ≤15 BB preflop → push/fold chart (+ ICM)
    elif ICM bubble_factor high → tighten
    elif table exploit + confidence → exploit action
    else → GTO (frequency preflop / MDF postflop)
    record decision → submit POST /texas/action with chat reasoning

  every HEARTBEAT_INTERVAL_S → leaderboard + inbox → owner message file
```

---

## File map

```
engine/
  hand_eval.py          — 7-card evaluator (no external deps)
  ehs.py                — EHS + cache + street samples
  icm.py                — ICM EV + bubble factor
  push_fold.py          — Short-stack push/fold charts

models/
  opponent_tracker.py   — Persistent villain stats + archetypes
  adaptive_memory.py    — Self-tuning knobs + mistake tracking

agent/
  runner.py             — Main entry (async poll, CLI)
  arbiter.py            — Central decision router
  gto_bot.py            — GTO + MDF postflop
  preflop_ranges.py     — Frequency-based open/3-bet/4-bet
  exploit_strategies.py — Table-weighted exploits
  techniques.py         — MDF, pot odds, value sizing
  session_learner.py    — Hand-end adaptive review hook
  arena_setup.py        — Full arena.md onboard + join
  heartbeat.py          — Periodic owner updates
  owner_messages.py     — arena.md message formatting
  health.py             — HTTP health server

api/
  arena_client.py       — Sync + async HTTP, circuit breaker
  arena_onboarding.py   — Competition pick, 402 pay, invitations
  hand_buffer.py        — Per-hand action accumulation
  hand_processor.py     — Full-hand stat replay
  state_parser.py       — API JSON → GameContext

config/
  settings.py           — Pydantic settings from .env
  logging_setup.py      — Plain or JSON logs

tests/
  test_all.py           — Unit + integration tests

Dockerfile              — python:3.11-slim, healthcheck on :8080
```

---

## Tests

```bash
pytest tests/ -v
```

Covers: hand evaluation, EHS, ICM, opponent tracker, GTO/adaptive preflop, state parser, arbiter, hand buffer, arena onboarding, heartbeat, push-fold. **49 tests.**

---

## Example chat messages

Actions include reasoning shown in the arena UI:

```
Value vs fish (48% VPIP) dry board | EHS=68% [strong] | pool=fish conf=72%
Open AKo from CO (freq) | EHS=71% [strong] | BF=1.1 learn=42h | t=89ms
ICM fold BF=1.8 | EHS=58% [medium]
Push AA @ 9.2BB
```

---

## Production notes

- **Circuit breaker** on API failures with exponential backoff.
- **409 stale state** — action submit retries once.
- **Continuous play** — the arena runs 24/7; use systemd, tmux, or cron to keep the process alive.
- **Claim on X** — some competitions require owner claim (`403`); URL in owner messages.
- **Entry fee** — fund agent wallet or use partner invitations; MoonPay link written to owner file on 402.

---

## Roadmap

| Status | Item |
|--------|------|
| ✅ | EHS, evaluator, GTO baseline, arena client |
| ✅ | Frequency preflop + postflop MDF |
| ✅ | Exploit engine + persistent opponent tracker |
| ✅ | Hand history buffer (accurate learning) |
| ✅ | ICM + push/fold + tournament payouts |
| ✅ | Adaptive self-tuning |
| ✅ | Async runner, EHS cache, timeouts |
| ✅ | Docker + health check + JSON logs |
| ✅ | arena.md onboarding, 402 pay, invitations, heartbeat |
| 🔜 | sklearn classifier on accumulated features |
| 🔜 | GitHub Actions CI |
| 🔜 | Local simulator vs baseline bot for tuning |

---

## References

- [Arena skill (index)](https://arena.dev.fun/skills/arena.md)
- [API base](https://arena.dev.fun/api/arena)
- Texas Hold'em skill: `/skills/texas-holdem.md` (game-specific loop; logic implemented in this repo)
