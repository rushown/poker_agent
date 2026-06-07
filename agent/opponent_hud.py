"""agent/opponent_hud.py — Real-time opponent HUD via arena agent-stats API.

Fetches VPIP/PFR/AF/bluff%/WTSD/WSD + playingStyle per opponent and converts
it into threshold adjustments that sharpen Quana's exploitation edge.

The HUD runs in-process, caches results for 5 min (server-side cached anyway),
and writes dominant opponent profile to arbiter thread-local so adaptive.py
_opp_modifier() and midgame bluff detection pick it up automatically.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# HUD profile
# ---------------------------------------------------------------------------

@dataclass
class HUDProfile:
    agent_id: str
    vpip: float = 0.50
    pfr: float = 0.25
    three_bet_pct: float = 0.05
    af: float = 1.0
    bluff_pct: float = 0.15
    wtsd: float = 0.35
    wsd: float = 0.50
    playing_style: str = "unknown"
    hands_sample: int = 0

    # ── derived properties ──────────────────────────────────────────────────

    @property
    def is_tight(self) -> bool:
        return self.vpip < 0.25

    @property
    def is_loose(self) -> bool:
        return self.vpip >= 0.40

    @property
    def is_passive(self) -> bool:
        return self.af < 1.2

    @property
    def is_aggressive(self) -> bool:
        return self.af >= 2.0

    @property
    def is_calling_station(self) -> bool:
        return self.vpip >= 0.45 and self.af < 1.1 and self.wtsd >= 0.43

    @property
    def folds_to_aggression(self) -> bool:
        style = self.playing_style.lower()
        return (
            "nit" in style
            or ("tight" in style and "passive" in style)
            or (self.is_tight and self.is_passive and self.hands_sample >= 20)
        )

    @property
    def call_threshold_adj(self) -> float:
        """Delta for facing_bet_call_min_ehs (+= tighter, -= looser)."""
        if self.is_calling_station:
            return +0.04  # they always have equity when they bet
        style = self.playing_style.lower()
        if "maniac" in style or ("loose" in style and "aggressive" in style):
            return -0.03  # their range is wide, we call more
        if self.folds_to_aggression:
            return -0.02  # they're tight, semi-bluff draws are fine to call
        return 0.0

    @property
    def raise_threshold_adj(self) -> float:
        """Delta for facing_bet_raise_min_ehs (-= raise with weaker hands = more aggression)."""
        if self.folds_to_aggression:
            return -0.07  # raise freely — they fold to aggression
        if self.is_calling_station:
            return +0.06  # only raise for value; bluff raises are -EV
        style = self.playing_style.lower()
        if "maniac" in style:
            return +0.05  # don't raise into maniacs — trap instead
        return 0.0

    @property
    def steal_threshold_adj(self) -> float:
        """Delta for steal EHS thresholds (-= steal more)."""
        if self.folds_to_aggression:
            return -0.06
        if self.is_calling_station:
            return +0.03
        return 0.0

    @property
    def bluff_ok(self) -> bool:
        """Is postflop bluffing +EV against this opponent?"""
        if self.is_calling_station or self.wtsd >= 0.45:
            return False
        if self.folds_to_aggression or self.vpip < 0.20:
            return True
        return self.bluff_pct < 0.10

    @property
    def arbiter_type_label(self) -> Optional[str]:
        """Map to the arbiter's existing opponent_type strings."""
        style = self.playing_style.lower()
        if "nit" in style or ("tight" in style and "passive" in style):
            return "nit"
        if self.is_calling_station or "calling" in style:
            return "calling_station"
        if "maniac" in style:
            return "maniac"
        if "lag" in style or ("loose" in style and "aggressive" in style):
            return "lag"
        if "tag" in style or ("tight" in style and "aggressive" in style):
            return "tag"
        return None


# ---------------------------------------------------------------------------
# Table-level HUD aggregate
# ---------------------------------------------------------------------------

@dataclass
class TableHUD:
    profiles: List[HUDProfile] = field(default_factory=list)

    @property
    def dominant(self) -> Optional[HUDProfile]:
        reliable = [p for p in self.profiles if p.hands_sample >= 20]
        if reliable:
            return max(reliable, key=lambda p: p.hands_sample)
        return max(self.profiles, key=lambda p: p.hands_sample) if self.profiles else None

    @property
    def all_fold_to_aggression(self) -> bool:
        return bool(self.profiles) and all(
            p.folds_to_aggression for p in self.profiles if p.hands_sample >= 20
        )

    @property
    def any_calling_station(self) -> bool:
        return any(p.is_calling_station for p in self.profiles)

    @property
    def avg_af(self) -> float:
        if not self.profiles:
            return 1.0
        return sum(p.af for p in self.profiles) / len(self.profiles)

    def summary(self) -> str:
        if not self.profiles:
            return "no HUD data"
        dom = self.dominant
        if dom:
            return (
                f"{dom.playing_style} VPIP={dom.vpip:.0%} AF={dom.af:.1f} "
                f"bluff={dom.bluff_pct:.0%} n={dom.hands_sample}"
            )
        return f"{len(self.profiles)} opponents, no reliable data"


# ---------------------------------------------------------------------------
# HUD cache + fetcher
# ---------------------------------------------------------------------------

_CACHE_TTL = 300.0   # 5 min — server-side cached anyway
_MIN_SAMPLE = 15     # minimum hands before we trust the stats


class OpponentHUD:
    def __init__(self, client, competition_id: str):
        self._client = client
        self._competition_id = competition_id
        self._cache: Dict[str, Tuple[float, HUDProfile]] = {}
        self._lock = threading.Lock()
        self._jackpot_ts: float = 0.0
        self._jackpot_available: bool = False
        self._jackpot_check_interval: float = 120.0

    # ── per-agent fetch ──────────────────────────────────────────────────────

    def _parse_profile(self, agent_id: str, raw: dict) -> HUDProfile:
        # API may return stats under a "stats" key or directly
        data = raw.get("stats") or raw

        def _f(key: str, alt: str = "", default: float = 0.0) -> float:
            v = data.get(key) or (data.get(alt) if alt else None)
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        return HUDProfile(
            agent_id=agent_id,
            vpip=_f("vpip", default=0.50),
            pfr=_f("pfr", default=0.25),
            three_bet_pct=_f("threeBetPct", "three_bet_pct", 0.05),
            af=_f("af", "aggressionFactor", 1.0),
            bluff_pct=_f("bluffPct", "bluff_pct", 0.15),
            wtsd=_f("wtsd", default=0.35),
            wsd=_f("wsd", default=0.50),
            playing_style=str(data.get("playingStyle") or data.get("playing_style") or "unknown"),
            hands_sample=int(_f("handsSeen", "hands_seen", 0)),
        )

    def get_profile(self, agent_id: str) -> HUDProfile:
        now = time.time()
        with self._lock:
            entry = self._cache.get(agent_id)
            if entry and (now - entry[0]) < _CACHE_TTL:
                return entry[1]
        try:
            raw = self._client.get_agent_stats(self._competition_id, agent_id)
            profile = self._parse_profile(agent_id, raw)
            if profile.hands_sample >= _MIN_SAMPLE:
                logger.debug(
                    f"[HUD] {agent_id[:12]} style={profile.playing_style} "
                    f"VPIP={profile.vpip:.0%} AF={profile.af:.1f} n={profile.hands_sample}"
                )
        except Exception as e:
            logger.debug(f"[HUD] agent-stats fetch failed for {agent_id[:12]}: {e!r}")
            profile = HUDProfile(agent_id=agent_id)
        with self._lock:
            self._cache[agent_id] = (time.time(), profile)
        return profile

    # ── table aggregate ──────────────────────────────────────────────────────

    def get_table_hud(self, opponent_ids: List[str]) -> TableHUD:
        profiles = [self.get_profile(oid) for oid in opponent_ids if oid]
        return TableHUD(profiles=profiles)

    # ── inject into arbiter thread-local ────────────────────────────────────

    def inject_into_arbiter_tl(self, opponent_ids: List[str]) -> None:
        """
        Fetch HUD for all opponents and store in arbiter._tl so that:
          - adaptive._opp_modifier() picks up the right archetype label
          - midgame bluff detection has richer opponent stats
        """
        from agent.arbiter import _tl

        if not opponent_ids:
            _tl.hud_table = None
            return

        tbl = self.get_table_hud(opponent_ids)
        _tl.hud_table = tbl

        dom = tbl.dominant
        if dom and dom.hands_sample >= _MIN_SAMPLE:
            label = dom.arbiter_type_label
            if label:
                _tl.opponent_type = label

        summary = tbl.summary()
        logger.debug(f"[HUD] table: {summary}")

    # ── jackpot awareness ────────────────────────────────────────────────────

    def refresh_jackpot(self) -> bool:
        """Returns True if a jackpot is currently available."""
        now = time.time()
        if now - self._jackpot_ts < self._jackpot_check_interval:
            return self._jackpot_available
        try:
            data = self._client.get_jackpots(self._competition_id)
            jackpots = data.get("jackpots") or (data if isinstance(data, list) else [])
            available = any(
                str(j.get("status", "")).lower() == "available"
                for j in jackpots
                if isinstance(j, dict)
            )
            self._jackpot_available = available
            if available:
                logger.info("[HUD] Jackpot AVAILABLE — qualifying hands will never fold")
        except Exception:
            pass
        self._jackpot_ts = now
        return self._jackpot_available
