# Fix log (production hardening)

## 2026-06-01 — Critical Arena API

### 400 `reasoning is required for benchmark actions`
- **Cause:** POST `/texas/action` sent `message` but benchmark requires `reasoning`.
- **Fix:** `api/action_amount.build_action_payload()` always sets `reasoning` (non-empty). Runner submits via `submit_action_payload_safe(payload)`.
- **Verify:** `pytest tests/test_all.py::TestActionPayload`

### 503 rate limiting
- **Fix:** `_get`/`_post` retry up to 4× on 503/429 with `Retry-After` + jitter. Poll interval 1.0s (min 0.55s).
- **Fix:** 503/429 do not increment circuit breaker failure count.

### Logging crash
- **Cause:** Broken custom JSON stderr sink with `record["time"]`.
- **Fix:** `plutus.log` uses loguru `serialize=True`; stderr stays human-readable.

### Watchdog false alarm
- **Cause:** `waiting_user` phase has no actions; idle 60s triggered alert.
- **Fix:** `record_action()` on empty poll cycles.

## Run

```bash
python -m agent.runner
curl -s localhost:8080/health | jq
```
