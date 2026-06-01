"""models/opponent_tracker.py — persistent opponent learning system.

Tracks every agent's actions across all sessions and sessions.
Persists to disk as JSON so learning compounds across games.

Stats tracked per opponent:
  - VPIP  (voluntarily put money in pot)
  - PFR   (pre-flop raise %)
  - AF    (aggression factor: (bets+raises) / calls)
  - 3bet  (re-raise frequency)
  - WTSD  (went to showdown %)
  - cbet  (continuation bet %)
  - fold_to_cbet
  - river_bluff_freq
  - hands_seen
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class OpponentStats:
    agent_id: str
    hands_seen: int = 0
    # Pre-flop
    vpip_count: int = 0          # hands where opp put money in voluntarily
    pfr_count: int = 0           # hands where opp raised pre-flop
    three_bet_opps: int = 0
    three_bet_count: int = 0
    fold_to_3bet_opps: int = 0
    fold_to_3bet_count: int = 0
    # Post-flop aggression
    bet_count: int = 0
    raise_count: int = 0
    call_count: int = 0
    check_count: int = 0
    fold_count: int = 0
    # Continuation bets
    cbet_opps: int = 0           # times opp was PFR and saw flop
    cbet_count: int = 0
    fold_to_cbet_opps: int = 0
    fold_to_cbet_count: int = 0
    fold_to_cbet_flop_opps: int = 0
    fold_to_cbet_flop_count: int = 0
    fold_to_cbet_turn_opps: int = 0
    fold_to_cbet_turn_count: int = 0
    fold_to_steal_opps: int = 0
    fold_to_steal_count: int = 0
    raise_first_in_opps: int = 0
    raise_first_in_count: int = 0
    check_raise_flop_opps: int = 0
    check_raise_flop_count: int = 0
    river_aggression_count: int = 0
    # Showdowns
    wtsd_opps: int = 0           # hands that reached river
    wtsd_count: int = 0          # times opp stayed for showdown
    wsd_count: int = 0           # times opp won at showdown
    # River
    river_bet_count: int = 0
    river_bluff_caught: int = 0  # times opp bet river and lost at SD
    # Meta
    last_seen: float = field(default_factory=time.time)
    chat_messages: List[str] = field(default_factory=list)
    bot_type: str = "unknown"

    # -----------------------------------------------------------------------
    # Derived stats (computed properties)
    # -----------------------------------------------------------------------

    @property
    def vpip(self) -> float:
        return self.vpip_count / max(1, self.hands_seen)

    @property
    def pfr(self) -> float:
        return self.pfr_count / max(1, self.hands_seen)

    @property
    def three_bet_pct(self) -> float:
        return self.three_bet_count / max(1, self.three_bet_opps)

    @property
    def fold_to_3bet(self) -> float:
        return self.fold_to_3bet_count / max(1, self.fold_to_3bet_opps)

    @property
    def aggression_factor(self) -> float:
        passive = max(1, self.call_count)
        return (self.bet_count + self.raise_count) / passive

    @property
    def cbet_pct(self) -> float:
        return self.cbet_count / max(1, self.cbet_opps)

    @property
    def fold_to_cbet(self) -> float:
        return self.fold_to_cbet_count / max(1, self.fold_to_cbet_opps)

    @property
    def fold_to_cbet_flop(self) -> float:
        return self.fold_to_cbet_flop_count / max(1, self.fold_to_cbet_flop_opps)

    @property
    def fold_to_cbet_turn(self) -> float:
        return self.fold_to_cbet_turn_count / max(1, self.fold_to_cbet_turn_opps)

    @property
    def fold_to_steal(self) -> float:
        return self.fold_to_steal_count / max(1, self.fold_to_steal_opps)

    @property
    def raise_first_in_pct(self) -> float:
        return self.raise_first_in_count / max(1, self.raise_first_in_opps)

    @property
    def check_raise_flop_pct(self) -> float:
        return self.check_raise_flop_count / max(1, self.check_raise_flop_opps)

    @property
    def river_aggression(self) -> float:
        return self.river_aggression_count / max(1, self.hands_seen)

    @property
    def wtsd(self) -> float:
        return self.wtsd_count / max(1, self.wtsd_opps)

    @property
    def wsd(self) -> float:
        return self.wsd_count / max(1, max(1, self.wtsd_count))

    @property
    def confidence(self) -> float:
        """Bayesian confidence using dynamic threshold formula.

        Derived from research: threshold = 0.5 + 0.4 * (1 - exp(-sample_size / 100))
        Reaches ~0.75 at 50 hands, ~0.85 at 100 hands, ~0.90 at 200 hands.
        """
        import math
        return min(0.95, 0.5 + 0.4 * (1.0 - math.exp(-self.hands_seen / 100.0)))

    @property
    def archetype(self) -> str:
        """Classify opponent using research-calibrated stat thresholds.

        Thresholds derived from poker database research (minimum 15 hands).
          Nit:             VPIP <16%, PFR <13%
          Fish:            VPIP >35%, PFR <15%, AF <1.2
          Maniac:          VPIP >35%, PFR >28%, AF >4.0
          LAG:             VPIP 22-30%, PFR 18-26%, AF >2.5
          TAG:             VPIP 15-22%, PFR 12-18%, AF 2.0-3.0
        """
        if self.hands_seen < 15:
            return "unknown"
        vpip, pfr, af = self.vpip, self.pfr, self.aggression_factor
        # Calling station / fish: wide + passive
        if vpip > 0.35 and pfr < 0.15 and af < 1.2:
            return "fish"
        # Maniac: wide + hyper-aggressive
        if vpip > 0.35 and pfr > 0.28 and af > 4.0:
            return "maniac"
        # Nit: very tight
        if vpip < 0.16:
            return "nit"
        # LAG: loose-aggressive
        if 0.22 <= vpip <= 0.32 and pfr >= 0.18 and af > 2.5:
            return "lag"
        # TAG: tight-aggressive
        if 0.15 <= vpip <= 0.23 and pfr >= 0.12 and 2.0 <= af <= 3.5:
            return "tag"
        return "unknown"

    def summary(self) -> str:
        return (
            f"[{self.agent_id[:8]}] arch={self.archetype} "
            f"hands={self.hands_seen} conf={self.confidence:.2f} "
            f"VPIP={self.vpip:.0%} PFR={self.pfr:.0%} AF={self.aggression_factor:.1f} "
            f"3b={self.three_bet_pct:.0%} cbet={self.cbet_pct:.0%}"
        )


class OpponentTracker:
    """Persistent opponent stats store. Call update_*() during hand replays
    and save() after each hand to accumulate learning across sessions."""

    def __init__(self, state_file: str = ".arena-poker-state"):
        self.state_file = state_file
        self._stats: Dict[str, OpponentStats] = {}
        self._load()

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                raw = json.load(f)
            for agent_id, data in raw.get("opponents", {}).items():
                s = OpponentStats(agent_id=agent_id)
                for k, v in data.items():
                    if k != "chat_messages":
                        setattr(s, k, v)
                    else:
                        s.chat_messages = v[-50:]  # keep last 50
                self._stats[agent_id] = s
        except Exception:
            pass  # start fresh if corrupt

    def save(self) -> None:
        data = {"opponents": {}}
        for aid, s in self._stats.items():
            d = asdict(s)
            d.pop("agent_id", None)
            data["opponents"][aid] = d
        tmp = self.state_file + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.state_file)
        except Exception:
            pass

    def apply_bot_profiles(self, detector: Any) -> None:
        """Sync bot_type from BotPatternDetector classifications."""
        for aid, s in self._stats.items():
            profile = detector.classify(s)
            s.bot_type = profile.bot_type.value

    # -----------------------------------------------------------------------
    # Access
    # -----------------------------------------------------------------------

    def get(self, agent_id: str) -> OpponentStats:
        if agent_id not in self._stats:
            self._stats[agent_id] = OpponentStats(agent_id=agent_id)
        return self._stats[agent_id]

    def all_archetypes(self) -> Dict[str, str]:
        return {aid: s.archetype for aid, s in self._stats.items()}

    # -----------------------------------------------------------------------
    # Update methods (call these as you process hand history)
    # -----------------------------------------------------------------------

    def record_hand_seen(self, agent_id: str) -> None:
        s = self.get(agent_id)
        s.hands_seen += 1
        s.last_seen = time.time()

    def record_vpip(self, agent_id: str) -> None:
        self.get(agent_id).vpip_count += 1

    def record_pfr(self, agent_id: str) -> None:
        self.get(agent_id).pfr_count += 1

    def record_action(self, agent_id: str, action: str) -> None:
        s = self.get(agent_id)
        action = action.lower()
        if action in ("bet", "raise", "all-in", "allin"):
            s.bet_count += 1 if action == "bet" else 0
            s.raise_count += 1 if action in ("raise", "all-in", "allin") else 0
        elif action == "call":
            s.call_count += 1
        elif action == "check":
            s.check_count += 1
        elif action == "fold":
            s.fold_count += 1

    def record_3bet_opportunity(self, agent_id: str, did_3bet: bool) -> None:
        s = self.get(agent_id)
        s.three_bet_opps += 1
        if did_3bet:
            s.three_bet_count += 1

    def record_fold_to_3bet(self, agent_id: str, did_fold: bool) -> None:
        s = self.get(agent_id)
        s.fold_to_3bet_opps += 1
        if did_fold:
            s.fold_to_3bet_count += 1

    def record_cbet_opportunity(self, agent_id: str, did_cbet: bool) -> None:
        s = self.get(agent_id)
        s.cbet_opps += 1
        if did_cbet:
            s.cbet_count += 1

    def record_fold_to_cbet(
        self, agent_id: str, did_fold: bool, street: str = ""
    ) -> None:
        s = self.get(agent_id)
        s.fold_to_cbet_opps += 1
        if did_fold:
            s.fold_to_cbet_count += 1
        st = street.lower()
        if st == "flop":
            s.fold_to_cbet_flop_opps += 1
            if did_fold:
                s.fold_to_cbet_flop_count += 1
        elif st == "turn":
            s.fold_to_cbet_turn_opps += 1
            if did_fold:
                s.fold_to_cbet_turn_count += 1

    def record_fold_to_steal(self, agent_id: str, did_fold: bool) -> None:
        s = self.get(agent_id)
        s.fold_to_steal_opps += 1
        if did_fold:
            s.fold_to_steal_count += 1

    def record_raise_first_in(self, agent_id: str, did_raise: bool) -> None:
        s = self.get(agent_id)
        s.raise_first_in_opps += 1
        if did_raise:
            s.raise_first_in_count += 1

    def record_check_raise_flop(self, agent_id: str, did_cr: bool) -> None:
        s = self.get(agent_id)
        s.check_raise_flop_opps += 1
        if did_cr:
            s.check_raise_flop_count += 1

    def record_river_aggression(self, agent_id: str) -> None:
        self.get(agent_id).river_aggression_count += 1

    def table_weighted_stats(self, opponent_ids: List[str]) -> "OpponentStats":
        """Blend stats across active opponents weighted by confidence."""
        if not opponent_ids:
            return OpponentStats(agent_id="table_avg")
        total_w = 0.0
        blended = OpponentStats(agent_id="table_avg")
        for oid in opponent_ids:
            s = self.get(oid)
            w = s.confidence * max(1, s.hands_seen)
            if w <= 0:
                continue
            total_w += w
            blended.hands_seen += int(s.hands_seen * w)
            blended.vpip_count += int(s.vpip_count * w)
            blended.pfr_count += int(s.pfr_count * w)
            blended.fold_to_cbet_opps += int(s.fold_to_cbet_opps * w)
            blended.fold_to_cbet_count += int(s.fold_to_cbet_count * w)
            blended.fold_to_cbet_flop_opps += int(s.fold_to_cbet_flop_opps * w)
            blended.fold_to_cbet_flop_count += int(s.fold_to_cbet_flop_count * w)
            blended.three_bet_opps += int(s.three_bet_opps * w)
            blended.three_bet_count += int(s.three_bet_count * w)
            blended.fold_to_3bet_opps += int(s.fold_to_3bet_opps * w)
            blended.fold_to_3bet_count += int(s.fold_to_3bet_count * w)
            blended.bet_count += int(s.bet_count * w)
            blended.raise_count += int(s.raise_count * w)
            blended.call_count += int(s.call_count * w)

        if total_w > 0:
            inv = 1.0 / total_w
            for field_name in (
                "hands_seen", "vpip_count", "pfr_count", "fold_to_cbet_opps",
                "fold_to_cbet_count", "fold_to_cbet_flop_opps", "fold_to_cbet_flop_count",
                "three_bet_opps", "three_bet_count", "fold_to_3bet_opps",
                "fold_to_3bet_count", "bet_count", "raise_count", "call_count",
            ):
                setattr(blended, field_name, int(getattr(blended, field_name) * inv))

        blended.hands_seen = max(1, blended.hands_seen)
        return blended

    def record_wtsd(self, agent_id: str, went: bool, won: Optional[bool] = None) -> None:
        s = self.get(agent_id)
        s.wtsd_opps += 1
        if went:
            s.wtsd_count += 1
            if won is True:
                s.wsd_count += 1

    def record_river_bet(self, agent_id: str, was_bluff: bool = False) -> None:
        s = self.get(agent_id)
        s.river_bet_count += 1
        if was_bluff:
            s.river_bluff_caught += 1

    def record_chat(self, agent_id: str, message: str) -> None:
        s = self.get(agent_id)
        s.chat_messages.append(message)
        if len(s.chat_messages) > 50:
            s.chat_messages = s.chat_messages[-50:]
