"""agent/play_stats.py — session stats for VPIP/PFR and performance reports."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.gto_bot import hand_notation


@dataclass
class HandRecord:
    table_id: str
    hand_number: Any
    position: str
    notation: str
    street: str
    action: str
    amount: float
    vpip: bool = False
    pfr: bool = False
    three_bet: bool = False


class PlayStatsLogger:
    def __init__(self, path: str = "decisions.jsonl"):
        self.path = Path(path)
        self._hands: Dict[str, HandRecord] = {}

    def record_decision(
        self,
        *,
        table_id: str,
        hand_number: Any,
        position: str,
        hole: List[str],
        street: str,
        action: str,
        amount: float,
        is_facing_raise: bool,
    ) -> None:
        if hand_number is None:
            return
        key = f"{table_id}:{hand_number}"
        rec = self._hands.get(key)
        if rec is None:
            rec = HandRecord(
                table_id=table_id,
                hand_number=hand_number,
                position=position,
                notation=hand_notation(hole),
                street=street,
                action=action,
                amount=amount,
            )
            self._hands[key] = rec
        act = action.lower()
        if street == "preflop":
            if act in ("call", "raise", "bet", "all-in"):
                rec.vpip = True
            if act in ("raise", "bet", "all-in"):
                rec.pfr = True
                if is_facing_raise:
                    rec.three_bet = True
        rec.action = act
        rec.amount = amount
        self.append_line(
            {
                "table_id": table_id,
                "hand": hand_number,
                "position": position,
                "notation": rec.notation,
                "street": street,
                "action": act,
                "amount": amount,
            }
        )

    def append_line(self, payload: Dict[str, Any]) -> None:
        payload["ts"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def summary(self) -> Dict[str, Any]:
        hands = list(self._hands.values())
        n = len(hands) or 1
        vpip = sum(1 for h in hands if h.vpip)
        pfr = sum(1 for h in hands if h.pfr)
        three = sum(1 for h in hands if h.three_bet)
        by_pos: Dict[str, Dict[str, int]] = {}
        for h in hands:
            p = h.position or "UNK"
            by_pos.setdefault(p, {"hands": 0, "vpip": 0, "pfr": 0})
            by_pos[p]["hands"] += 1
            if h.vpip:
                by_pos[p]["vpip"] += 1
            if h.pfr:
                by_pos[p]["pfr"] += 1
        return {
            "hands_tracked": len(hands),
            "vpip_pct": round(100 * vpip / n, 1),
            "pfr_pct": round(100 * pfr / n, 1),
            "three_bet_pct": round(100 * three / max(1, vpip), 1),
            "by_position": {
                k: {
                    "vpip_pct": round(100 * v["vpip"] / max(1, v["hands"]), 1),
                    "pfr_pct": round(100 * v["pfr"] / max(1, v["hands"]), 1),
                    "hands": v["hands"],
                }
                for k, v in by_pos.items()
            },
        }
