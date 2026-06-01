#!/usr/bin/env bash
# Monitor agent.runner for stability. Exit 1 on hard failures, 0 after clean window.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${ROOT}/run_monitor.log"
DURATION_S="${1:-600}"
INTERVAL_S="${2:-30}"
END=$((SECONDS + DURATION_S))
FAIL=0

while (( SECONDS < END )); do
  if ! pgrep -f "python -m agent.runner" >/dev/null 2>&1; then
    echo "[monitor] agent process not running"
    exit 1
  fi
  if grep -qE "Submit failed (400|500|503)|Poll error (400|500|503)|Traceback|Circuit breaker open" "$LOG" 2>/dev/null; then
    echo "[monitor] errors in log:"
    grep -E "Submit failed (400|500|503)|Poll error (400|500|503)|Traceback|Circuit breaker open" "$LOG" | tail -5
    exit 1
  fi
  HEALTH=$(curl -sf --max-time 3 "http://127.0.0.1:8080/health" 2>/dev/null || echo "")
  if [[ -z "$HEALTH" ]]; then
    echo "[monitor] health endpoint unreachable"
    FAIL=1
  else
  STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'), d.get('hands_played',0), d.get('api_400_errors',0))")
    echo "[monitor] $(date +%H:%M:%S) health: $STATUS"
    ST=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status'))")
    HANDS=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('hands_played',0))")
    E400=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('api_400_errors',0))")
    if [[ "$ST" != "ok" ]] || [[ "${E400:-0}" -gt 0 ]]; then
      echo "[monitor] unhealthy: status=$ST api_400_errors=$E400"
      FAIL=1
    fi
    if [[ "${HANDS:-0}" -lt 1 ]]; then
      echo "[monitor] waiting for hands_played > 0"
    fi
  fi
  sleep "$INTERVAL_S"
done

if [[ "$FAIL" -eq 1 ]]; then
  exit 1
fi
HEALTH=$(curl -sf --max-time 3 "http://127.0.0.1:8080/health")
HANDS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('hands_played',0))")
if [[ "${HANDS:-0}" -lt 1 ]]; then
  echo "[monitor] completed window but hands_played=0"
  exit 1
fi
echo "[monitor] ${DURATION_S}s clean — agent stable"
exit 0
