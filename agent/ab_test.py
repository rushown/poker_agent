"""agent/ab_test.py — concurrent strategy A/B tests with promotion after 200 hands."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class ABVariant:
    name: str
    config: Dict[str, float] = field(default_factory=dict)
    hands: int = 0
    chip_delta: float = 0.0

    @property
    def bb_per_hand(self) -> float:
        return self.chip_delta / max(1, self.hands)


class ABTestRunner:
    PROMOTE_AFTER = 200

    def __init__(self, path: str = "ab_tests.json"):
        self.path = path
        self.champion: str = "champion"
        self.challenger: str = "challenger"
        self._variants: Dict[str, ABVariant] = {
            self.champion: ABVariant(name=self.champion, config={}),
            self.challenger: ABVariant(
                name=self.challenger,
                config={"bluff_frequency": 1.1},
            ),
        }
        self._table_assignment: Dict[str, str] = {}
        self._load()

    def assign_table(self, table_id: str) -> str:
        if table_id not in self._table_assignment:
            self._table_assignment[table_id] = (
                self.champion if random.random() < 0.5 else self.challenger
            )
        return self._table_assignment[table_id]

    def record(self, variant: str, chip_delta: float) -> None:
        v = self._variants.get(variant)
        if not v:
            return
        v.hands += 1
        v.chip_delta += chip_delta
        if v.hands >= self.PROMOTE_AFTER:
            self._maybe_promote()

    def _maybe_promote(self) -> None:
        c = self._variants[self.champion]
        ch = self._variants[self.challenger]
        if ch.hands < self.PROMOTE_AFTER:
            return
        if ch.bb_per_hand > c.bb_per_hand + 0.05:
            logger.info(
                f"A/B promote {self.challenger} "
                f"({ch.bb_per_hand:.3f} vs {c.bb_per_hand:.3f} bb/hand)"
            )
            self.champion, self.challenger = self.challenger, self.champion
            ch.hands = 0
            ch.chip_delta = 0.0
            c.hands = 0
            c.chip_delta = 0.0
        else:
            ch.hands = 0
            ch.chip_delta = 0.0
        self.save()

    def config_for(self, variant: str) -> Dict[str, float]:
        return dict(self._variants.get(variant, ABVariant(variant)).config)

    def save(self) -> None:
        data = {
            "champion": self.champion,
            "variants": {
                k: {"hands": v.hands, "chip_delta": v.chip_delta, "config": v.config}
                for k, v in self._variants.items()
            },
            "updated": time.time(),
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.champion = data.get("champion", self.champion)
            for name, v in data.get("variants", {}).items():
                self._variants[name] = ABVariant(
                    name=name,
                    config=v.get("config", {}),
                    hands=v.get("hands", 0),
                    chip_delta=v.get("chip_delta", 0),
                )
        except Exception:
            pass

    def self_test(self) -> List[str]:
        errors: List[str] = []
        r = ABTestRunner(path="/tmp/ab_test.json")
        r.record(r.champion, 10)
        if r._variants[r.champion].hands != 1:
            errors.append("record should increment hands")
        return errors
