"""api/hand_buffer.py — accumulate per-hand actions from table snapshots.

The API only exposes partial action history on each poll. We diff successive
snapshots and merge actionsByStreet / actionHistory into a complete hand log.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple


def _action_key(action: Dict[str, Any]) -> str:
    """Stable dedup key for an action record."""
    parts = [
        str(action.get("street") or action.get("round") or ""),
        str(action.get("agentId") or action.get("playerId") or action.get("actor") or ""),
        str(action.get("action") or action.get("type") or ""),
        str(action.get("amount") or action.get("betAmount") or ""),
        str(action.get("sequence") or action.get("seq") or ""),
        str(action.get("timestamp") or action.get("ts") or ""),
    ]
    raw = "|".join(parts)
    if all(p in ("", "0", "None") for p in parts[2:]):
        raw += json.dumps(action, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def _extract_actions_from_snapshot(table: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull all known actions from a table snapshot."""
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def add(a: Dict[str, Any], street: str = "") -> None:
        if not a:
            return
        rec = dict(a)
        if street and not rec.get("street"):
            rec["street"] = street
        k = _action_key(rec)
        if k in seen:
            return
        seen.add(k)
        out.append(rec)

    by_street = table.get("actionsByStreet") or table.get("actions_by_street") or {}
    if isinstance(by_street, dict):
        for street, acts in by_street.items():
            for a in acts or []:
                add(a, str(street).lower())

    for a in table.get("actionHistory") or table.get("actions") or []:
        add(a)

    for a in table.get("recentActions") or table.get("recentEvents") or []:
        ev = a
        if a.get("type") == "ActionTaken" and a.get("summary"):
            s = a["summary"]
            ev = {
                "agentId": s.get("agentId"),
                "action": s.get("action"),
                "amount": s.get("amount") or s.get("toAmount"),
                "street": a.get("street"),
            }
        add(ev, str(a.get("street") or "").lower())

    # Last action on a seat (some API versions)
    for seat in table.get("seats") or table.get("players") or []:
        last = seat.get("lastAction") or seat.get("action")
        if isinstance(last, dict):
            add(last)

    return out


class HandHistoryBuffer:
    """Buffers actions per (table_id, hand_number) until the hand completes."""

    def __init__(self) -> None:
        self._actions: Dict[str, List[Dict[str, Any]]] = {}
        self._seen_keys: Dict[str, Set[str]] = {}
        self._last_snap_hash: Dict[str, str] = {}

    @staticmethod
    def hand_key(table_id: str, hand_number: Any) -> str:
        return f"{table_id}:{hand_number}"

    def ingest(self, table: Dict[str, Any], table_id: str) -> Optional[str]:
        """Append new actions from snapshot. Returns current hand key or None."""
        hand_num = table.get("handNumber") or table.get("handId")
        if hand_num is None:
            return None

        key = self.hand_key(table_id, hand_num)
        if key not in self._actions:
            self._actions[key] = []
            self._seen_keys[key] = set()

        for action in _extract_actions_from_snapshot(table):
            ak = _action_key(action)
            if ak not in self._seen_keys[key]:
                self._seen_keys[key].add(ak)
                self._actions[key].append(action)

        return key

    def pop_completed(self, table_id: str, previous_hand: Any) -> Tuple[str, List[Dict[str, Any]]]:
        """Return and remove buffered actions for a completed hand."""
        key = self.hand_key(table_id, previous_hand)
        actions = self._actions.pop(key, [])
        self._seen_keys.pop(key, None)
        return key, actions

    def get(self, table_id: str, hand_number: Any) -> List[Dict[str, Any]]:
        return list(self._actions.get(self.hand_key(table_id, hand_number), []))

    def prune_stale(self, max_entries: int = 200) -> None:
        if len(self._actions) <= max_entries:
            return
        oldest = sorted(self._actions.keys())[: len(self._actions) - max_entries]
        for k in oldest:
            self._actions.pop(k, None)
            self._seen_keys.pop(k, None)
