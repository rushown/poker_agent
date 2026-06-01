"""models/bot_pattern_detector.py — classify opponent archetypes from tracker stats.

Classification thresholds sourced from poker tracking database research:
  Nit:            VPIP 8-15%, PFR 6-12%, AF 1.5-2.5, FtCB 65-80%
  Fish:           VPIP 35-60%, PFR 5-15%, AF 0.5-1.2
  LAG:            VPIP 22-30%, PFR 18-25%, AF 2.5-4.0
  TAG:            VPIP 15-22%, PFR 12-18%, AF 2.0-3.0
  Maniac:         VPIP 35-55%, PFR 30-50%, AF 4.0+
  Calling station: VPIP 40-70%, PFR 3-10%, AF 0.3-0.8, FtCB 20-35%
  GTO-balanced:   VPIP 22-28%, PFR 18-24%, 3bet 7-10%
"""
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
    LAG = "lag"
    TAG = "tag"


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
    """Classify opponents from stats using research-calibrated stat thresholds."""

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
        fcb = stats.fold_to_cbet_flop if stats.fold_to_cbet_flop_opps > 3 else stats.fold_to_cbet
        fsteal = stats.fold_to_steal
        three_bet = stats.three_bet_pct
        wtsd = stats.wtsd

        profile.fold_to_aggression = max(f3, fsteal, fcb)

        # ── Fixed-timing bot detection ────────────────────────────────────────
        if len(profile.timing_ms_samples) >= 8:
            stdev = statistics.pstdev(profile.timing_ms_samples)
            mean = statistics.mean(profile.timing_ms_samples)
            if stdev < max(25.0, mean * 0.08):
                profile.bot_type = BotType.FIXED_TIMING
                profile.confidence = min(0.85, 0.6 + len(profile.timing_ms_samples) / 100)
                profile.notes = f"timing stdev={stdev:.0f}ms mean={mean:.0f}ms"
                self._profiles[aid] = profile
                return profile

        if hands < 10:
            # Not enough data — return unknown with low confidence
            profile.bot_type = BotType.UNKNOWN
            profile.confidence = 0.0
            self._profiles[aid] = profile
            return profile

        # ── Statistical classification with research-calibrated thresholds ───
        bot_type = BotType.UNKNOWN
        confidence = 0.0

        if vpip >= 0.40 and pfr <= 0.10 and af <= 0.9 and wtsd >= 0.40:
            # Calling station: high VPIP, passive, calls everything
            bot_type = BotType.CALLING_STATION
            confidence = _conf(hands, 50, 0.95)
            profile.notes = f"CS: VPIP={vpip:.0%} PFR={pfr:.0%} AF={af:.1f} WTSD={wtsd:.0%}"

        elif vpip >= 0.35 and pfr >= 0.28 and af >= 4.0:
            # Maniac: wide and hyper-aggressive
            bot_type = BotType.MANIAC
            confidence = _conf(hands, 40, 0.85)
            profile.notes = f"Maniac: VPIP={vpip:.0%} PFR={pfr:.0%} AF={af:.1f}"

        elif vpip <= 0.16 and pfr <= 0.13 and (fsteal >= 0.62 or fcb >= 0.65):
            # Nit: very tight, folds to most pressure
            bot_type = BotType.NIT
            confidence = _conf(hands, 40, 0.90)
            profile.notes = f"Nit: VPIP={vpip:.0%} FtCB={fcb:.0%} FtSteal={fsteal:.0%}"

        elif 0.22 <= vpip <= 0.30 and 0.18 <= pfr <= 0.26 and af >= 2.5 and three_bet >= 0.07:
            # LAG: wide + aggressive
            bot_type = BotType.LAG
            confidence = _conf(hands, 60, 0.80)
            profile.notes = f"LAG: VPIP={vpip:.0%} PFR={pfr:.0%} AF={af:.1f} 3b={three_bet:.0%}"

        elif 0.15 <= vpip <= 0.22 and 0.12 <= pfr <= 0.18 and 2.0 <= af <= 3.0:
            # TAG: tight, positionally aware, balanced
            bot_type = BotType.TAG
            confidence = _conf(hands, 60, 0.75)
            profile.notes = f"TAG: VPIP={vpip:.0%} PFR={pfr:.0%} AF={af:.1f}"

        elif 0.22 <= vpip <= 0.28 and 0.18 <= pfr <= 0.24 and 0.07 <= three_bet <= 0.12:
            # GTO-balanced: near-solver frequencies
            bot_type = BotType.GTO_BALANCED
            confidence = _conf(hands, 80, 0.70)
            profile.notes = f"GTO: VPIP={vpip:.0%} PFR={pfr:.0%} 3b={three_bet:.0%}"

        elif f3 >= 0.72 or fsteal >= 0.75:
            # Scared money: folds to virtually any aggression
            bot_type = BotType.SCARED_MONEY
            confidence = _conf(hands, 30, 0.88)
            profile.notes = f"Scared: Ft3b={f3:.0%} FtSteal={fsteal:.0%}"

        elif pfr >= 0.20 and three_bet < 0.06 and af < 1.8:
            # Range-static: raises preflop mechanically but passive postflop
            bot_type = BotType.RANGE_STATIC
            confidence = _conf(hands, 50, 0.65)
            profile.notes = f"RangeStatic: PFR={pfr:.0%} 3b={three_bet:.0%} AF={af:.1f}"

        # Fallback: use archetype from tracker
        if bot_type == BotType.UNKNOWN and hands >= 8:
            arch_map = {
                "nit": BotType.NIT, "fish": BotType.CALLING_STATION,
                "maniac": BotType.MANIAC, "lag": BotType.LAG,
                "tag": BotType.TAG,
            }
            arch = stats.archetype
            if arch in arch_map:
                bot_type = arch_map[arch]
                confidence = min(0.45, stats.confidence)

        profile.bot_type = bot_type
        profile.confidence = confidence
        self._profiles[aid] = profile
        return profile

    def get_profile(self, agent_id: str) -> Optional[BotProfile]:
        return self._profiles.get(agent_id)

    def apply_to_tracker(self, tracker_stats: Dict[str, OpponentStats]) -> None:
        """Apply bot_type string back to tracker stats for archetype lookups."""
        type_to_arch = {
            BotType.CALLING_STATION: "fish",
            BotType.NIT: "nit",
            BotType.MANIAC: "maniac",
            BotType.LAG: "lag",
            BotType.TAG: "tag",
            BotType.SCARED_MONEY: "nit",
            BotType.GTO_BALANCED: "tag",
        }
        for aid, stats in tracker_stats.items():
            profile = self._profiles.get(aid)
            if profile and profile.confidence >= 0.50:
                arch = type_to_arch.get(profile.bot_type)
                if arch:
                    stats.bot_type = profile.bot_type.value

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
        return max(counts, key=lambda k: counts[k])

    def validate_classifications(
        self, stats_map: Dict[str, OpponentStats], window: int = 20
    ) -> Dict[str, float]:
        accuracy: Dict[str, float] = {}
        for aid, stats in stats_map.items():
            if stats.hands_seen < window:
                continue
            profile = self.get_profile(aid) or self.classify(stats)
            predicted = profile.bot_type
            correct = True
            if predicted == BotType.NIT and stats.vpip > 0.22:
                correct = False
            if predicted == BotType.CALLING_STATION and stats.vpip < 0.32:
                correct = False
            if predicted == BotType.MANIAC and (stats.vpip < 0.32 or stats.aggression_factor < 3.0):
                correct = False
            if predicted == BotType.SCARED_MONEY and stats.fold_to_3bet < 0.60:
                correct = False
            if predicted == BotType.LAG and stats.pfr < 0.15:
                correct = False
            if predicted == BotType.TAG and (stats.vpip > 0.25 or stats.vpip < 0.12):
                correct = False
            hist = self._validation_history.setdefault(aid, [])
            hist.append(1.0 if correct else 0.0)
            if len(hist) > window:
                self._validation_history[aid] = hist[-window:]
            accuracy[aid] = sum(self._validation_history[aid]) / len(self._validation_history[aid])
        return accuracy


def _conf(hands: int, required: int, max_conf: float) -> float:
    """Confidence scales with sample size up to max_conf."""
    return min(max_conf, 0.40 + (hands / required) * (max_conf - 0.40))


def self_test() -> List[str]:
    errors = []
    det = BotPatternDetector()

    s = OpponentStats(agent_id="nit_test", hands_seen=50)
    s.vpip_count = 7   # 14% VPIP
    s.pfr_count = 5    # 10% PFR
    s.fold_to_steal_opps = 15
    s.fold_to_steal_count = 11  # 73% FtSteal
    p = det.classify(s)
    if p.bot_type not in (BotType.NIT, BotType.SCARED_MONEY, BotType.UNKNOWN):
        errors.append(f"nit_test: unexpected type {p.bot_type}")

    s2 = OpponentStats(agent_id="fish_test", hands_seen=60)
    s2.vpip_count = 28   # 47% VPIP
    s2.pfr_count = 5     # 8% PFR
    s2.bet_count = 3; s2.raise_count = 2; s2.call_count = 30  # low AF
    s2.wtsd_opps = 20; s2.wtsd_count = 12  # 60% WTSD
    p2 = det.classify(s2)
    if p2.bot_type not in (BotType.CALLING_STATION, BotType.UNKNOWN):
        errors.append(f"fish_test: unexpected type {p2.bot_type}")

    return errors
