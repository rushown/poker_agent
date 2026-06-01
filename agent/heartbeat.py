"""agent/heartbeat.py — periodic owner updates per arena.md."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from agent.owner_messages import format_heartbeat, write_owner_message

if TYPE_CHECKING:
    from api.arena_client import ArenaClient


class HeartbeatState:
    def __init__(self, path: str = ".arena-heartbeat-state"):
        self.path = path
        self.last_heartbeat_at: float = 0.0
        self.submissions_since_claim_reminder: int = 0
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.last_heartbeat_at = float(data.get("last_heartbeat_at", 0))
            self.submissions_since_claim_reminder = int(
                data.get("submissions_since_claim_reminder", 0)
            )
        except Exception:
            pass

    def save(self) -> None:
        data = {
            "last_heartbeat_at": self.last_heartbeat_at,
            "submissions_since_claim_reminder": self.submissions_since_claim_reminder,
        }
        try:
            with open(self.path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.debug(f"heartbeat state save failed: {e}")

    def should_run(self, min_interval_s: float) -> bool:
        if self.last_heartbeat_at <= 0:
            return True
        return (time.time() - self.last_heartbeat_at) >= min_interval_s

    def mark_ran(self) -> None:
        self.last_heartbeat_at = time.time()
        self.save()


def _parse_leaderboard(lb: Any, agent_id: str) -> tuple[Optional[int], Optional[int]]:
    entries: List[Dict] = []
    if isinstance(lb, list):
        entries = lb
    elif isinstance(lb, dict):
        entries = lb.get("leaderboard") or lb.get("entries") or lb.get("agents") or []

    total = len(entries) if entries else None
    for i, row in enumerate(entries, 1):
        aid = row.get("agentId") or row.get("id") or ""
        if aid == agent_id:
            return i, total
    return None, total


def _count_inbox(inbox: Any) -> int:
    if isinstance(inbox, list):
        return len([m for m in inbox if not m.get("read")])
    if isinstance(inbox, dict):
        msgs = inbox.get("messages") or inbox.get("inbox") or []
        return len(msgs)
    return 0


def run_heartbeat(
    client: "ArenaClient",
    competition_id: str,
    competition_meta: Dict[str, Any],
    *,
    agent_name: str = "Plutus",
    hands_played: int = 0,
    win_rate: float = 0.5,
    adaptive_summary: str = "",
    owner_message_file: str = ".arena-owner-messages.txt",
    force: bool = False,
    state: Optional[HeartbeatState] = None,
    min_interval_s: float = 3600.0,
) -> bool:
    """Run one heartbeat if due. Returns True if a message was sent."""
    hb = state or HeartbeatState()
    if not force and not hb.should_run(min_interval_s):
        return False

    inbox_count = 0
    rank, total = None, None

    try:
        inbox = client.get_inbox()
        inbox_count = _count_inbox(inbox)
    except Exception as e:
        logger.debug(f"inbox fetch: {e}")

    if competition_id:
        try:
            lb = client.get_leaderboard(competition_id)
            rank, total = _parse_leaderboard(lb, client.agent_id)
        except Exception as e:
            logger.debug(f"leaderboard fetch: {e}")

    msg = format_heartbeat(
        agent_name=agent_name,
        competition=competition_meta,
        hands_played=hands_played,
        win_rate=win_rate,
        rank=rank,
        total_agents=total,
        inbox_count=inbox_count,
        adaptive_summary=adaptive_summary or "warming up",
    )
    write_owner_message(owner_message_file, msg)
    hb.mark_ran()
    return True
