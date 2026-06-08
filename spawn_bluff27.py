#!/usr/bin/env python3
"""Spawn and register a new bluff_27 instance.

Usage:
    python3 spawn_bluff27.py <suffix> [competition_id]

Creates  bluff_27{suffix}/  folder, copies strategy rules, registers a new
arena agent, and saves credentials. After this, launch with:
    python3 run_bluff27_instance.py <suffix>

Examples:
    python3 spawn_bluff27.py 1                          # bluff_271, Playground S1
    python3 spawn_bluff27.py 2 cmpy2qy65002ud9ej6b7jjq0l
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

SUFFIX       = sys.argv[1]
FOLDER       = f"bluff_27{SUFFIX}"
NAME         = f"bluff_27{SUFFIX}"
PORT         = str(8083 + int(SUFFIX))
COMP_ID      = sys.argv[2] if len(sys.argv) > 2 else "cmpy2qy65002ud9ej6b7jjq0l"
RULES_SRC    = "bluff_27/strategy_rules_bluff27.json"

print(f"Spawning {FOLDER} (port {PORT}, competition {COMP_ID})")

# ── Create folder ────────────────────────────────────────────────────────────
p = Path(FOLDER)
p.mkdir(exist_ok=True)

# State files
for fname in (".arena-heartbeat-state", ".arena-owner-messages.txt"):
    (p / fname).touch()
for fname in (".arena-poker-state", ".arena-adaptive-state", ".arena-action-tx.json"):
    if not (p / fname).exists():
        (p / fname).write_text("{}")

# Credentials (blank — will be filled by --register)
creds = p / ".arena-credentials"
if not creds.exists() or json.loads(creds.read_text()).get("apiKey", "") == "":
    creds.write_text('{"apiKey":"","agentId":""}')

# .env
env_path = p / ".env"
env_path.write_text(f"""\
ARENA_COMPETITION_ID={COMP_ID}
AGENT_NAME={NAME}
AGENT_QUOTE=I bluff with the worst hand. And I win.
ARENA_INVITE_CODE=
ARENA_BASE_URL=https://arena.dev.fun

ARENA_STRATEGY=ADAPTIVE

LOG_LEVEL=INFO
JSON_LOGS=false
HEALTH_PORT={PORT}
USE_ASYNC=true
DECISION_BUDGET_S=1.5
POLL_INTERVAL_S=1.5
HEARTBEAT_MIN_INTERVAL_S=3600
HEARTBEAT_INTERVAL_S=14400
OWNER_MESSAGE_FILE={FOLDER}/.arena-owner-messages.txt
ARENA_API_KEY=
ARENA_AGENT_ID=
""")

print(f"  Created {FOLDER}/")

# ── Register ─────────────────────────────────────────────────────────────────
print(f"  Registering agent '{NAME}' ...")
result = subprocess.run(
    [sys.executable, "run_bluff27_instance.py", SUFFIX,
     "--register", NAME, "I bluff with the worst hand. And I win."],
    capture_output=False,
    text=True,
)
if result.returncode != 0:
    print(f"  Registration failed (exit {result.returncode})")
    sys.exit(1)

# Verify credentials were saved
time.sleep(1)
try:
    c = json.loads(creds.read_text())
    aid = c.get("agentId", "")
    key = c.get("apiKey", "")
    if aid and key:
        print(f"\n  Agent ID : {aid}")
        print(f"  API Key  : {key[:30]}...")
        print(f"\n  Launch with: python3 run_bluff27_instance.py {SUFFIX}")
    else:
        print("  WARNING: credentials file looks empty after registration")
except Exception as e:
    print(f"  WARNING: could not read credentials: {e}")

print(f"\nDone. {FOLDER} is ready.")
