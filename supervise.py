#!/usr/bin/env python3
"""supervise.py — restart Plutus when health check fails or process exits."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
HEALTH_URL = "http://127.0.0.1:8080/health"
AGENT_CMD = [sys.executable, "-m", "agent.runner"]
CHECK_INTERVAL_S = 5
STARTUP_GRACE_S = 20


def healthy() -> bool:
    try:
        r = httpx.get(HEALTH_URL, timeout=3.0)
        if r.status_code != 200:
            return False
        data = r.json()
        return data.get("status") == "ok"
    except Exception:
        return False


def main() -> None:
    proc: subprocess.Popen | None = None
    started_at = 0.0
    restarts = 0

    while True:
        if proc is None or proc.poll() is not None:
            code = proc.returncode if proc else None
            if proc:
                print(f"[supervise] agent exited code={code}; restart #{restarts + 1}")
            proc = subprocess.Popen(AGENT_CMD, cwd=str(ROOT))
            started_at = time.time()
            restarts += 1
            time.sleep(STARTUP_GRACE_S)
            continue

        if time.time() - started_at > STARTUP_GRACE_S and not healthy():
            print("[supervise] unhealthy — terminating agent")
            proc.terminate()
            try:
                proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                proc.kill()
            proc = None
            time.sleep(3)
            continue

        time.sleep(CHECK_INTERVAL_S)


if __name__ == "__main__":
    main()
