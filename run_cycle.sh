#!/usr/bin/env bash
# run_cycle.sh — full recursive improvement cycle.
#
# Usage:
#   bash run_cycle.sh                    # run forever until convergence
#   bash run_cycle.sh --cycles 3         # run exactly 3 cycles
#   bash run_cycle.sh --hands 100        # 100 hands per cycle (default: 150)
#   bash run_cycle.sh --dry-run          # parse + show DeepSeek analysis, don't update rules
#
# Each cycle:
#   1. Register fresh agent (first cycle only) or reuse existing
#   2. Run ADAPTIVE strategy for N hands
#   3. Feed logs to DeepSeek → get revised strategy_rules.json
#   4. Repeat with new rules
set -e

BASE=/home/ocean/vscode/poker_agent
PYTHON=$BASE/.venv/bin/python
AGENT_DIR=$BASE/.adaptive_agent
# Set DEEPSEEK_API_KEY in your environment before running:
#   export DEEPSEEK_API_KEY=your_key_here
if [ -z "$DEEPSEEK_API_KEY" ]; then
  echo "ERROR: DEEPSEEK_API_KEY env var not set. Export it before running."
  exit 1
fi

MAX_CYCLES=99
HANDS_PER_CYCLE=150
DRY_RUN=0

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --cycles)   MAX_CYCLES=$2;      shift 2 ;;
    --hands)    HANDS_PER_CYCLE=$2; shift 2 ;;
    --dry-run)  DRY_RUN=1;          shift ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

mkdir -p "$AGENT_DIR"

# ── Step 0: Register if not already registered ──────────────────────────
if ! grep -q "ARENA_API_KEY=arena_sk_" "$AGENT_DIR/.env" 2>/dev/null; then
  echo "========================================"
  echo " Registering Plutus-Adaptive agent..."
  echo "========================================"

  cat > "$AGENT_DIR/.env" << EOF
ARENA_COMPETITION_ID=seed_poker_eval_s1
ARENA_BASE_URL=https://arena.dev.fun
ARENA_STRATEGY=ADAPTIVE
AGENT_NAME=Plutus-Adaptive
LOG_LEVEL=INFO
JSON_LOGS=false
HEALTH_PORT=8090
USE_ASYNC=true
DECISION_BUDGET_S=1.5
POLL_INTERVAL_S=0.8
HEARTBEAT_MIN_INTERVAL_S=3600
HEARTBEAT_INTERVAL_S=14400
OWNER_MESSAGE_FILE=.arena-owner-messages.txt
DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY
EOF

  cd "$AGENT_DIR"
  ARENA_STRATEGY=ADAPTIVE AGENT_NAME=Plutus-Adaptive PYTHONPATH=$BASE \
    $PYTHON -m agent.runner --register "Plutus-Adaptive" "I learn from every hand." 2>&1 | \
    grep -E 'Registered|Saved|✅|ERROR|apiKey'

  echo ""
fi

CYCLE=$(python3 -c "import json; print(json.load(open('$BASE/strategy_rules.json')).get('cycle', 0))")
echo "========================================"
echo " Starting from cycle $CYCLE"
echo " Max cycles: $MAX_CYCLES | Hands/cycle: $HANDS_PER_CYCLE"
echo "========================================"

# ── Main loop ────────────────────────────────────────────────────────────
for i in $(seq 1 $MAX_CYCLES); do
  CYCLE=$(python3 -c "import json; print(json.load(open('$BASE/strategy_rules.json')).get('cycle', 0))")
  LOGFILE="$AGENT_DIR/cycle_${CYCLE}.log"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " CYCLE $CYCLE — playing $HANDS_PER_CYCLE hands"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # ── Step 1: Run agent ──────────────────────────────────────────────
  cd "$AGENT_DIR"
  echo "[Run] Starting agent (logs → $LOGFILE)..."
  ARENA_STRATEGY=ADAPTIVE AGENT_NAME=Plutus-Adaptive PYTHONPATH=$BASE \
    DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY \
    $PYTHON -m agent.runner --max-hands $HANDS_PER_CYCLE \
    2>&1 | tee "$LOGFILE" | grep -E '✓|ERROR|WARNING|Strategy|benchmark|phase|BB|hands' || true

  echo "[Run] Done. Parsing results..."

  # Quick stats from log
  HANDS_PLAYED=$(grep -c '"event": "decision"' "$LOGFILE" 2>/dev/null || echo 0)
  echo "[Stats] Decisions logged: $HANDS_PLAYED"

  if [ "$HANDS_PLAYED" -lt 20 ]; then
    echo "[Warning] Very few hands played ($HANDS_PLAYED). Check agent connectivity."
  fi

  # ── Step 2: Refine strategy ────────────────────────────────────────
  cd "$BASE"
  echo ""
  echo "[Refine] Sending logs to DeepSeek for cycle $CYCLE analysis..."

  REFINE_ARGS="--log $LOGFILE --min-hands 20"
  if [ "$DRY_RUN" -eq 1 ]; then
    REFINE_ARGS="$REFINE_ARGS --dry-run"
  fi

  DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY $PYTHON refine_strategy.py $REFINE_ARGS

  # Check if converged
  CONVERGED=$(python3 -c "
import json, sys
r = json.load(open('$BASE/strategy_rules.json'))
notes = r.get('cycle_notes','')
print('yes' if 'converged' in notes.lower() else 'no')
" 2>/dev/null || echo "no")

  if [ "$CONVERGED" = "yes" ]; then
    echo ""
    echo "🏆 STRATEGY CONVERGED after cycle $CYCLE — optimal play achieved."
    break
  fi

  NEW_CYCLE=$(python3 -c "import json; print(json.load(open('$BASE/strategy_rules.json')).get('cycle', 0))")
  echo ""
  echo "[Cycle $CYCLE complete] → now on cycle $NEW_CYCLE"
  echo "Current rules:"
  python3 -c "
import json
r = json.load(open('$BASE/strategy_rules.json'))
for s in ['preflop','flop','turn','river']:
    sec = r.get(s, {})
    vals = {k: v for k,v in sec.items() if isinstance(v, (int,float))}
    print(f'  {s}: {vals}')
"

  # Small pause between cycles
  sleep 5
done

echo ""
echo "All cycles done. Final strategy:"
cat "$BASE/strategy_rules.json"
