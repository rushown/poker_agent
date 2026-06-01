"""models/bot_pattern_detector.py — classify opponent bot patterns for exploitation."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from models.opponent_tracker import OpponentStats


class BotType(str, Enum):
    UNKNOWN = "unknown"
    FIXED_TIMING = "fixed_timing"
    RANGE_STATIC = "range_static"
    GTO_BALANCED = "gto_balanced"
    CALLING_STATION = "calling_station"
    NIT = "nit"
    MANIAC = "maniac"
    SCARED_MONEY = "scared_money"


@dataclass
class BotProfile:
    agent_id: str
    bot_type: BotType = BotType.UNKNOWN
    confidence: float = 0.0
    fold_to_aggression: float = 0.5
    timing_ms_samples: List[float] = field(default_factory=list)
    raise_size_tells: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


class BotPatternDetector:
    """Infer bot archetypes from opponent_tracker stats + optional timing samples."""

    def __init__(self) -> None:
        self._profiles: Dict[str, BotProfile] = {}
        self._timing_by_agent: Dict[str, List[float]] = {}
        self._validation_history: Dict[str, List[float]] = {}

    def record_timing(self, agent_id: str, response_ms: float) -> None:
        if not agent_id:
            return
        samples = self._timing_by_agent.setdefault(agent_id, [])
        samples.append(response_ms)
        if len(samples) > 40:
            self._timing_by_agent[agent_id] = samples[-40:]

    def classify(self, stats: OpponentStats) -> BotProfile:
        aid = stats.agent_id
        profile = self._profiles.get(aid) or BotProfile(agent_id=aid)
        profile.timing_ms_samples = list(self._timing_by_agent.get(aid, []))

        hands = max(1, stats.hands_seen)
        vpip = stats.vpip
        pfr = stats.pfr
        af = stats.aggression_factor
        f3 = stats.fold_to_3bet
        fcb = stats.fold_to_cbet
        fsteal = stats.fold_to_steal

        profile.fold_to_aggression = max(f3, fsteal, fcb)

        # Fixed timing bot
        if len(profile.timing_ms_samples) >= 8:
            stdev = statistics.pstdev(profile.timing_ms_samples)
            mean = statistics.mean(profile.timing_ms_samples)
            if stdev < max(25.0, mean * 0.08):
                profile.bot_type = BotType.FIXED_TIMING
                profile.confidence = 0.75
                profile.notes = f"timing stdev={stdev:.0f}ms"
                self._profiles[aid] = profile
                return profile

        if hands >= 12:
            if vpip >= 0.42 and pfr <= 0.12 and f3 < 0.35:
                profile.bot_type = BotType.CALLING_STATION
                profile.confidence = min(0.95, 0.5 + hands / 80)
                profile.notes = f"VPIP={vpip:.0%} PFR={pfr:.0%}"
            elif vpip <= 0.18 and fsteal >= 0.65:
                profile.bot_type = BotType.NIT
                profile.confidence = min(0.9, 0.45 + hands / 60)
                profile.notes = f"nit fold_steal={fsteal:.0%}"
            elif vpip >= 0.38 and af >= 2.5 and pfr >= 0.28:
                profile.bot_type = BotType.MANIAC
                profile.confidence = 0.7
                profile.notes = f"AF={af:.1f}"
            elif 0.22 <= vpip <= 0.32 and 0.16 <= pfr <= 0.24 and 0.35 <= f3 <= 0.65:
                profile.bot_type = BotType.GTO_BALANCED
                profile.confidence = 0.65
                profile.notes = "balanced frequencies"
            elif f3 >= 0.70 or fsteal >= 0.72:
                profile.bot_type = BotType.SCARED_MONEY
                profile.confidence = min(0.9, 0.5 + f3)
                profile.notes = "folds to aggression"
            elif pfr >= 0.2 and stats.three_bet_pct < 0.08 and af < 1.8:
                profile.bot_type = BotType.RANGE_STATIC
                profile.confidence = 0.6
                profile.notes = "static raise sizing"

        if profile.bot_type == BotType.UNKNOWN and hands >= 8:
            arch = stats.archetype
            arch_map = {
                "nit": BotType.NIT,
                "fish": BotType.CALLING_STATION,
                "maniac": BotType.MANIAC,
                "lag": BotType.GTO_BALANCED,
                "tag": BotType.GTO_BALANCED,
            }
            profile.bot_type = arch_map.get(arch, BotType.UNKNOWN)
            profile.confidence = min(0.5, stats.confidence)

        self._profiles[aid] = profile
        return profile

    def get_profile(self, agent_id: str) -> Optional[BotProfile]:
        return self._profiles.get(agent_id)

    def table_dominant_type(
        self, opponent_ids: List[str], stats_map: Dict[str, OpponentStats]
    ) -> BotType:
        counts: Dict[BotType, int] = {}
        for oid in opponent_ids:
            st = stats_map.get(oid)
            if not st:
                continue
            bt = self.classify(st).bot_type
            counts[bt] = counts.get(bt, 0) + 1
        if not counts:
            return BotType.UNKNOWN
        return max(counts, key=counts.get)

    def validate_classifications(
        self, stats_map: Dict[str, OpponentStats], window: int = 20
    ) -> Dict[str, float]:
        """Compare predicted labels vs recent behavior (VPIP/PFR/fold rates)."""
        accuracy: Dict[str, float] = {}
        for aid, stats in stats_map.items():
            if stats.hands_seen < window:
                continue
            profile = self.get_profile(aid) or self.classify(stats)
            predicted = profile.bot_type
            correct = True
            if predicted == BotType.NIT and stats.vpip > 0.25:
                correct = False
            if predicted == BotType.CALLING_STATION and stats.vpip < 0.30:
                correct = False
            if predicted == BotType.MANIAC and stats.vpip < 0.35:
                correct = False
            if predicted == BotType.SCARED_MONEY and stats.fold_to_3bet < 0.55:
                correct = False
            hist = self._validation_history.setdefault(aid, [])
            hist.append(1.0 if correct else 0.0)
            if len(hist) > window:
                self._validation_history[aid] = hist[-window:]
            accuracy[aid] = sum(self._validation_history[aid]) / len(
                self._validation_history[aid]
            )
        return accuracy


def self_test() -> List[str]:
    errors: List[str] = []
    from models.opponent_tracker import OpponentStats

    det = BotPatternDetector()
    s = OpponentStats(agent_id="x", hands_seen=40, vpip_count=5, pfr_count=4)
    s.fold_to_steal_opps = 12
    s.fold_to_steal_count = 9
    p = det.classify(s)
    if p.bot_type not in (BotType.NIT, BotType.SCARED_MONEY, BotType.UNKNOWN):
        errors.append(f"unexpected type {p.bot_type}")
    return errors
