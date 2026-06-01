#!/usr/bin/env python3
"""auto_train.py — brutal improvement loop with rollback gates."""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from api.arena_client import ArenaClient

load_dotenv()
ROOT = Path(__file__).resolve().parent
HISTORY = ROOT / "performance_history.csv"
CHAMPION = ROOT / ".arena-champion-state"
SAFE = ROOT / ".arena-safe-config"
ISSUE = ROOT / "ISSUE.md"
TARGET_HANDS = int(os.environ.get("AUTO_TRAIN_HANDS", "500"))
BB100_FAIL = 5.0
BB100_CHAMPION = 25.0


def run_supervised_block(hands: int) -> int:
    proc = subprocess.Popen(
        [sys.executable, "-m", "agent.runner", "--max-hands", str(hands)],
        cwd=str(ROOT),
    )
    return proc.wait()


def fetch_metrics(client: ArenaClient, competition_id: str) -> dict:
    st = client.get_benchmark_status(competition_id)
    m = st.get("match") or st
    return {
        "bb100": float(m.get("rawBbPer100") or 0),
        "hands": int(m.get("completedHands") or 0),
        "chip_delta": float(m.get("rawChipDelta") or 0),
    }


def append_history(row: dict) -> None:
    new_file = not HISTORY.exists()
    with HISTORY.open("a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "strategy_used",
                "hands",
                "bb100",
                "win_rate",
                "vpip",
                "pfr",
                "aggression",
                "tournament_standing",
            ],
        )
        if new_file:
            w.writeheader()
        w.writerow(row)


def revert_safe_config() -> None:
    src = ROOT / ".arena-adaptive-state"
    if SAFE.exists():
        shutil.copy(SAFE, src)
        print("[auto_train] reverted to .arena-safe-config")
    brutal = ROOT / ".arena-brutal-state"
    if brutal.exists():
        brutal.unlink()


def write_issue(bb100: float, metrics: dict) -> None:
    ISSUE.write_text(
        f"# Performance regression\n\n"
        f"- bb/100: {bb100}\n"
        f"- metrics: {json.dumps(metrics, indent=2)}\n"
        f"- action: reverted to safe config\n"
        f"- time: {time.ctime()}\n"
    )


def main() -> None:
    competition_id = os.environ.get("ARENA_COMPETITION_ID", "")
    client = ArenaClient(
        api_key=os.environ.get("ARENA_API_KEY", ""),
        agent_id=os.environ.get("ARENA_AGENT_ID", ""),
    )
    client.load_credentials()

    adaptive = ROOT / ".arena-adaptive-state"
    if adaptive.exists() and not SAFE.exists():
        shutil.copy(adaptive, SAFE)

    print(f"[auto_train] running {TARGET_HANDS} hands")
    code = run_supervised_block(TARGET_HANDS)
    if code != 0:
        print(f"[auto_train] agent exited {code}")
        write_issue(-999, {"exit_code": code})
        revert_safe_config()
        sys.exit(1)

    metrics = fetch_metrics(client, competition_id)
    bb100 = metrics["bb100"]
    append_history(
        {
            "timestamp": time.time(),
            "strategy_used": "meta",
            "hands": metrics["hands"],
            "bb100": bb100,
            "win_rate": "",
            "vpip": "",
            "pfr": "",
            "aggression": "",
            "tournament_standing": metrics["chip_delta"],
        }
    )
    print(f"[auto_train] bb/100={bb100} hands={metrics['hands']}")

    if bb100 < BB100_FAIL:
        write_issue(bb100, metrics)
        revert_safe_config()
        sys.exit(2)

    if bb100 >= BB100_CHAMPION and adaptive.exists():
        shutil.copy(adaptive, CHAMPION)
        print("[auto_train] champion checkpoint saved")

    report = ROOT / "strategy_report.json"
    if report.exists():
        print(report.read_text())


if __name__ == "__main__":
    main()
