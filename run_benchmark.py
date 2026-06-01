#!/usr/bin/env python3
"""
run_benchmark.py — overnight benchmark orchestrator for Plutus.

Starts agent.runner, polls Arena benchmark status every 5s (min 2s between API calls),
logs metrics to performance_log.json, restarts agent on crash or when match completes.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from loguru import logger

from api.arena_client import ArenaClient, ArenaAPIError

load_dotenv()

PERF_LOG = Path("performance_log.json")
MIN_API_INTERVAL_S = 2.0
METRICS_INTERVAL_S = 5.0
AGENT_CMD = [sys.executable, "-m", "agent.runner"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_perf_log() -> list:
    if PERF_LOG.exists():
        try:
            return json.loads(PERF_LOG.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def append_perf(entry: Dict[str, Any]) -> None:
    rows = load_perf_log()
    rows.append(entry)
    PERF_LOG.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def extract_match(status: Dict[str, Any]) -> Dict[str, Any]:
    return status.get("match") or status.get("data", {}).get("match") or status


def match_finished(match: Dict[str, Any]) -> bool:
    st = str(match.get("status", "")).lower()
    phase = str(match.get("phase", "")).lower()
    if st in ("completed", "complete", "ended", "finished", "cancelled"):
        return True
    if phase in ("complete", "completed", "ended"):
        return True
    target = int(match.get("targetHands") or 0)
    done = int(match.get("completedHands") or 0)
    return target > 0 and done >= target


def metrics_row(match: Dict[str, Any], agent_pid: Optional[int]) -> Dict[str, Any]:
    return {
        "ts": utc_now(),
        "agent_pid": agent_pid,
        "match_id": match.get("id"),
        "status": match.get("status"),
        "phase": match.get("phase"),
        "completed_hands": match.get("completedHands"),
        "target_hands": match.get("targetHands"),
        "raw_chip_delta": match.get("rawChipDelta"),
        "raw_bb_per_100": match.get("rawBbPer100"),
        "adjusted_bb_per_100": match.get("adjustedBbPer100"),
        "hands_won": match.get("handsWon"),
    }


def start_agent(competition_id: str) -> subprocess.Popen:
    env = os.environ.copy()
    if competition_id:
        env["ARENA_COMPETITION_ID"] = competition_id
    logger.info(f"Starting agent: {' '.join(AGENT_CMD)}")
    return subprocess.Popen(
        AGENT_CMD,
        cwd=str(Path(__file__).resolve().parent),
        env=env,
    )


def restart_benchmark(client: ArenaClient, competition_id: str) -> None:
    try:
        client.start_benchmark(competition_id)
        logger.info("Started new benchmark match")
    except ArenaAPIError as e:
        logger.error(f"benchmark/start failed: {e.status} {e.body}")


def main() -> None:
    competition_id = os.environ.get("ARENA_COMPETITION_ID", "")
    client = ArenaClient(
        api_key=os.environ.get("ARENA_API_KEY", ""),
        agent_id=os.environ.get("ARENA_AGENT_ID", ""),
        base_url=os.environ.get("ARENA_BASE_URL", "https://arena.dev.fun"),
    )
    client.load_credentials()

    agent_proc: Optional[subprocess.Popen] = None
    last_api = 0.0
    last_metrics = 0.0
    running = True

    def _stop(*_args: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    agent_proc = start_agent(competition_id)

    while running:
        now = time.time()

        if agent_proc and agent_proc.poll() is not None:
            code = agent_proc.returncode
            logger.warning(f"Agent exited with code {code}; restarting in 3s")
            append_perf({"ts": utc_now(), "event": "agent_exit", "code": code})
            time.sleep(3)
            agent_proc = start_agent(competition_id)
            last_metrics = 0.0

        if now - last_api >= MIN_API_INTERVAL_S:
            last_api = now
            try:
                status = client.get_benchmark_status(competition_id)
                match = extract_match(status)
                if now - last_metrics >= METRICS_INTERVAL_S:
                    last_metrics = now
                    row = metrics_row(match, agent_proc.pid if agent_proc else None)
                    append_perf(row)
                    logger.info(
                        f"Match {row.get('match_id')}: "
                        f"hands={row.get('completed_hands')}/{row.get('target_hands')} "
                        f"bb/100={row.get('raw_bb_per_100')} "
                        f"chips={row.get('raw_chip_delta')}"
                    )
                if match_finished(match):
                    logger.info("Match finished; starting new benchmark")
                    append_perf({"ts": utc_now(), "event": "match_finished", **match})
                    restart_benchmark(client, competition_id)
                    if agent_proc:
                        agent_proc.terminate()
                        try:
                            agent_proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            agent_proc.kill()
                    time.sleep(2)
                    agent_proc = start_agent(competition_id)
            except ArenaAPIError as e:
                logger.warning(f"Status poll failed: {e.status}")

        time.sleep(0.5)

    if agent_proc and agent_proc.poll() is None:
        agent_proc.terminate()
        try:
            agent_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            agent_proc.kill()
    logger.info("Orchestrator stopped")


if __name__ == "__main__":
    main()
