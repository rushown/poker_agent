# Running Plutus on a VPS (or free cloud)

This guide runs `python -m agent.runner` **24/7** on a small Linux VM so the dev.fun Arena benchmark keeps playing hands while you sleep.

## What you need

| Item | Notes |
|------|--------|
| Linux VM | 1 vCPU, 512MB–1GB RAM is enough |
| Python 3.11+ | 3.12/3.14 works if dependencies install |
| Outbound HTTPS | To `arena.dev.fun` |
| Arena credentials | `ARENA_API_KEY`, `ARENA_AGENT_ID`, competition id |

Secrets live in `.env` on the server only — never commit them.

---

## 1. Pick a host (free / cheap options)

| Provider | Free tier | Good for |
|----------|-----------|----------|
| [Oracle Cloud Always Free](https://www.oracle.com/cloud/free/) | ARM Ampere VM (24GB RAM total cap) | Best long-term free VPS |
| [Google Cloud](https://cloud.google.com/free) | e2-micro (limited hours) | Short trials |
| [Fly.io](https://fly.io) | Small shared VMs | Container deploy (`Dockerfile` included) |
| [Railway](https://railway.app) | Trial credits | Quick git deploy |
| [Render](https://render.com) | Free web services (sleep on idle) | Not ideal for 24/7 polling |
| [Hetzner](https://www.hetzner.com/cloud) | ~€4/mo CX22 | Reliable paid default |

**Recommendation:** Oracle Always Free ARM Ubuntu 22.04/24.04, or Hetzner if you want zero surprise limits.

---

## 2. Server setup (Ubuntu/Debian)

SSH into the VM, then:

```bash
sudo apt update && sudo apt install -y git python3 python3-venv python3-pip curl

git clone <your-repo-url> poker_agent
cd poker_agent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from the example:

```bash
cp .env.example .env
chmod 600 .env
nano .env   # paste ARENA_API_KEY, ARENA_AGENT_ID, ARENA_COMPETITION_ID
```

Required variables:

```env
ARENA_API_KEY=arena_sk_...
ARENA_AGENT_ID=cmp...
ARENA_COMPETITION_ID=seed_poker_eval_s1
ARENA_BASE_URL=https://arena.dev.fun
```

First-time Arena registration (only once):

```bash
source .venv/bin/activate
python -m agent.runner --register "Plutus" "I count outs, not prayers."
# Copy api key from output into .env, then:
python -m agent.runner --onboard
```

---

## 3. Run in production

### Option A — systemd (recommended)

```bash
sudo tee /etc/systemd/system/plutus.service << 'EOF'
[Unit]
Description=Plutus Arena Poker Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/poker_agent
Environment=PATH=/home/ubuntu/poker_agent/.venv/bin
ExecStart=/home/ubuntu/poker_agent/.venv/bin/python -m agent.runner
Restart=always
RestartSec=10
StandardOutput=append:/home/ubuntu/poker_agent/run_monitor.log
StandardError=append:/home/ubuntu/poker_agent/run_monitor.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now plutus
sudo systemctl status plutus
```

Health check from the same machine:

```bash
curl -s http://127.0.0.1:8080/health | python3 -m json.tool
```

### Option B — tmux (quick manual)

```bash
cd ~/poker_agent && source .venv/bin/activate
tmux new -s plutus
python -m agent.runner 2>&1 | tee -a run_monitor.log
# Detach: Ctrl-b then d
```

### Option C — built-in supervisor

Restarts the agent if it exits or health fails:

```bash
source .venv/bin/activate
python supervise.py
```

### Option D — Docker

```bash
docker build -t plutus .
docker run -d --name plutus \
  --env-file .env \
  -p 8080:8080 \
  --restart unless-stopped \
  plutus
```

---

## 4. Verify stability

After start, within a few minutes you should see `✓ FOLD`, `✓ CALL`, etc. in `run_monitor.log`.

```bash
# 10-minute stability check (no 400/503/tracebacks, health ok)
chmod +x scripts/stability_monitor.sh
./scripts/stability_monitor.sh 600 30
```

Benchmark progress:

```bash
source .venv/bin/activate
python -c "
from config.settings import settings
from api.arena_client import ArenaClient
c = ArenaClient(settings.arena_api_key, settings.arena_agent_id, settings.arena_base_url)
m = c.get_benchmark_status(settings.arena_competition_id).get('match', {})
print('hands', m.get('completedHands'), '/', m.get('targetHands'))
print('rawBbPer100', m.get('rawBbPer100'))
"
```

---

## 5. Logs and metrics

| Path | Purpose |
|------|---------|
| `run_monitor.log` | Main process log |
| `plutus.log` | Structured JSON log (if enabled) |
| `decisions.jsonl` | Per-decision analytics |
| `http://127.0.0.1:8080/health` | Liveness + hands/errors |
| `http://127.0.0.1:8080/metrics` | Dashboard snapshot |

Rotate logs on small disks:

```bash
# logrotate example: /etc/logrotate.d/plutus
/home/ubuntu/poker_agent/run_monitor.log {
    weekly
    rotate 4
    compress
    copytruncate
}
```

---

## 6. Firewall / exposure

- **Do not** expose port 8080 to the public internet unless you add auth.
- Default: bind health to localhost only (current behavior).
- For remote health checks, use SSH tunnel:  
  `ssh -L 8080:127.0.0.1:8080 user@vps`

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `reasoning is required` (400) | Update repo — `build_action_payload` always sends `reasoning` |
| Fractional raise rejected (400) | Update repo — amounts are integer chips + clamped to `minRaiseTo` |
| Poll timeouts / idle watchdog | Update repo — polls use sync httpx + benchmark fallback |
| `Circuit breaker open` | Arena rate limit; waits and retries; increase `poll_interval_s` if persistent |
| Agent exits immediately | Run `python -m agent.self_test_runner`; check `.env` and API key |
| No hands | Match `phase=waiting_user` — normal between bursts; check benchmark status |

---

## 8. Overnight benchmark

```bash
source .venv/bin/activate
python run_benchmark.py    # agent + status polling
# or
python auto_train.py       # hands target + optional config revert
```

---

## Quick reference

```bash
source .venv/bin/activate
python -m agent.runner              # production loop
curl -s localhost:8080/health | python3 -m json.tool
sudo systemctl restart plutus     # if using systemd
```
