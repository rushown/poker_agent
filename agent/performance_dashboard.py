"""agent/performance_dashboard.py — per-mode EV / VPIP / PFR logging."""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ModeStats:
    hands: int = 0
    chip_delta: float = 0.0
    vpip: int = 0
    pfr: int = 0
    wins: int = 0


class PerformanceDashboard:
    def __init__(
        self,
        csv_path: str = "performance_history.csv",
        json_path: str = "performance_dashboard.json",
    ):
        self.csv_path = Path(csv_path)
        self.json_path = Path(json_path)
        self.by_mode: Dict[str, ModeStats] = {}
        self._ensure_csv()

    def _ensure_csv(self) -> None:
        if self.csv_path.exists():
            return
        with self.csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "timestamp",
                    "strategy_mode",
                    "meta_strategy",
                    "hands",
                    "bb100",
                    "win_rate",
                    "vpip",
                    "pfr",
                ]
            )

    def record_hand(
        self,
        *,
        strategy_mode: str,
        meta_strategy: str,
        stack_delta: float,
        bb_size: float,
        vpip: bool,
        pfr: bool,
        won: Optional[bool],
    ) -> None:
        for key in (strategy_mode, meta_strategy):
            st = self.by_mode.setdefault(key, ModeStats())
            st.hands += 1
            st.chip_delta += stack_delta
            if vpip:
                st.vpip += 1
            if pfr:
                st.pfr += 1
            if won:
                st.wins += 1

        bb100 = (stack_delta / max(1, bb_size)) * 100
        wr = 1.0 if won else 0.0
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [
                    time.time(),
                    strategy_mode,
                    meta_strategy,
                    1,
                    round(bb100, 2),
                    wr,
                    int(vpip),
                    int(pfr),
                ]
            )
        self._flush_json()

    def _flush_json(self) -> None:
        out = {}
        for mode, st in self.by_mode.items():
            h = max(1, st.hands)
            out[mode] = {
                "hands": st.hands,
                "bb_per_hand": round(st.chip_delta / h, 3),
                "vpip_pct": round(100 * st.vpip / h, 1),
                "pfr_pct": round(100 * st.pfr / h, 1),
                "win_rate": round(st.wins / h, 3),
            }
        tmp = str(self.json_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, self.json_path)

    def snapshot(self) -> dict:
        self._flush_json()
        return json.loads(self.json_path.read_text()) if self.json_path.exists() else {}
