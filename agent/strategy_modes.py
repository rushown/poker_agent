"""agent/strategy_modes.py — tournament phase and mode selection."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

from models.bot_pattern_detector import BotPatternDetector, BotType
from models.opponent_tracker import OpponentStats


class StrategyMode(str, Enum):
    INTIMIDATION = "intimidation"
    EXPLOITATION = "exploitation"
    GTO_BALANCED = "gto_balanced"
    ICM_SURVIVAL = "icm_survival"
    PUSH_FOLD = "push_fold"
    ENDGAME_CLOSE = "endgame_close"


@dataclass
class TournamentContext:
    match_hands_played: int = 0
    target_hands: int = 500
    players_remaining: int = 6
    bubble_factor: float = 1.0
    recent_win_rate: float = 0.5
    bb_depth: float = 100.0
    stack: float = 200.0
    avg_stack: float = 200.0


class StrategyModeSelector:
    """Pick active strategy mode from stack, phase, and opponent profiles."""

    def __init__(self, detector: Optional[BotPatternDetector] = None) -> None:
        self.detector = detector or BotPatternDetector()
        self.current_mode: StrategyMode = StrategyMode.GTO_BALANCED
        self._mode_hands: int = 0

    def select(
        self,
        ctx: TournamentContext,
        opponent_ids: List[str],
        stats_by_id: Optional[dict] = None,
        brutal_check: Any = None,
    ) -> StrategyMode:
        stats_map = stats_by_id or {}
        from collections import Counter

        profiles = [
            self.detector.classify(stats_map[oid])
            for oid in opponent_ids
            if oid in stats_map
        ]
        dominant = (
            Counter(p.bot_type for p in profiles).most_common(1)[0][0]
            if profiles
            else BotType.UNKNOWN
        )

        pct_done = ctx.match_hands_played / max(1, ctx.target_hands)
        early_phase = pct_done < 0.20 or ctx.match_hands_played < 50

        if ctx.players_remaining <= 3:
            mode = StrategyMode.ENDGAME_CLOSE
        elif ctx.bb_depth <= 15:
            mode = StrategyMode.PUSH_FOLD
        elif ctx.bubble_factor > 1.55 and ctx.bb_depth < 40:
            mode = StrategyMode.ICM_SURVIVAL
        elif (
            early_phase
            and ctx.bb_depth >= 25
            and (brutal_check is None or brutal_check.intimidation_active())
        ):
            mode = StrategyMode.INTIMIDATION
        elif brutal_check is not None and brutal_check._intimidation_aborted:
            mode = StrategyMode.GTO_BALANCED
        elif dominant in (BotType.NIT, BotType.SCARED_MONEY, BotType.CALLING_STATION, BotType.RANGE_STATIC):
            mode = StrategyMode.EXPLOITATION
        elif dominant == BotType.GTO_BALANCED:
            mode = StrategyMode.GTO_BALANCED
        else:
            mode = StrategyMode.GTO_BALANCED

        if ctx.recent_win_rate < 0.30 and ctx.match_hands_played > 30:
            mode = StrategyMode.GTO_BALANCED
        if ctx.recent_win_rate > 0.62 and ctx.bb_depth > 30 and not early_phase:
            mode = StrategyMode.EXPLOITATION

        if mode != self.current_mode:
            self._mode_hands = 0
        self.current_mode = mode
        self._mode_hands += 1
        return mode
