"""agent/arbiter.py — Strategy Arbiter with mode selection and meta-learning."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.chat_tilt import build_chat_message, mistake_line
from agent.early_game import EarlyGameAggressor
from agent.endgame import EndgameModule
from agent.exploit_strategies import get_exploit_action
from agent.gto_bot import hand_notation, postflop_action, preflop_action
from agent.meta_learner import MetaLearner
from agent.strategy_modes import StrategyMode, StrategyModeSelector, TournamentContext
from agent.strategy_styles import play_style
from agent.techniques import (
    calculate_fold_equity,
    exploit_blend_threshold,
    icm_bubble_threshold,
    induce_bluff,
    pot_odds_required,
)
from engine.ehs import calculate_ehs, clear_ehs_cache, ehs_to_bucket, samples_for_street
from engine.icm import bubble_factor
from engine.push_fold import should_call_push, should_push
from models.adaptive_memory import AdaptiveMemory
from models.bot_pattern_detector import BotPatternDetector, BotType
from models.opponent_tracker import OpponentTracker, OpponentStats


@dataclass
class GameContext:
    hole_cards: List[str]
    community_cards: List[str]
    pot: float
    call_amount: float
    stack: float
    bb_size: float
    street: str
    position: str
    is_in_position: bool
    all_stacks: List[float] = field(default_factory=list)
    payouts: Optional[List[float]] = None
    allowed_actions: List[Dict] = field(default_factory=list)
    opponent_ids: List[str] = field(default_factory=list)
    is_facing_raise: bool = False
    facing_raise_size: float = 0.0
    table_id: str = ""
    hand_number: Any = None
    call_to_amount: float = 0.0
    committed_chips: float = 0.0


class StrategyArbiter:
    def __init__(
        self,
        tracker: OpponentTracker,
        adaptive: Optional[AdaptiveMemory] = None,
        meta: Optional[MetaLearner] = None,
        brutal_check: Any = None,
    ):
        self.tracker = tracker
        self.adaptive = adaptive or AdaptiveMemory()
        self.meta = meta or MetaLearner()
        self.brutal = brutal_check
        self.detector = BotPatternDetector()
        self.mode_selector = StrategyModeSelector(self.detector)
        self.early = EarlyGameAggressor(tracker)
        self.endgame = EndgameModule()
        self._hand_cache_key: Optional[str] = None
        self._last_ehs: float = 0.5
        self._current_mode: StrategyMode = StrategyMode.GTO_BALANCED
        self.match_hands_played: int = 0
        self.target_hands: int = 500
        self.players_remaining: int = 6

    def set_match_context(
        self,
        hands_played: int,
        target_hands: int = 500,
        players_remaining: int = 6,
    ) -> None:
        self.match_hands_played = hands_played
        self.target_hands = target_hands
        self.players_remaining = max(2, players_remaining)

    def on_new_hand(self, table_id: str, hand_number: Any, stack: float = 0) -> None:
        key = f"{table_id}:{hand_number}"
        if key != self._hand_cache_key:
            clear_ehs_cache()
            self._hand_cache_key = key
        if hand_number is not None:
            self.adaptive.start_hand(table_id, hand_number, stack)

    def _stats_map(self, opponent_ids: List[str]) -> Dict[str, OpponentStats]:
        return {oid: self.tracker.get(oid) for oid in opponent_ids}

    def decide(self, ctx: GameContext, deadline: float = 0.0) -> Tuple[str, float, str]:
        start = time.monotonic()
        budget = self._time_budget(start, deadline)
        tuning = self.adaptive.tuning

        samples = samples_for_street(ctx.street)
        ehs = calculate_ehs(
            ctx.hole_cards,
            ctx.community_cards,
            samples=samples,
            street=ctx.street,
            facing_raise=ctx.is_facing_raise or ctx.call_amount > ctx.bb_size,
        )
        self._last_ehs = ehs

        stats_map = self._stats_map(ctx.opponent_ids)
        for oid, st in stats_map.items():
            self.detector.classify(st)
        self.tracker.apply_bot_profiles(self.detector)

        table_stats = self.tracker.table_weighted_stats(ctx.opponent_ids)
        fold_to_3bet = table_stats.fold_to_3bet if table_stats.fold_to_3bet_opps > 3 else 0.5
        fold_to_cbet = (
            table_stats.fold_to_cbet_flop
            if table_stats.fold_to_cbet_flop_opps > 3
            else table_stats.fold_to_cbet
        )
        self.endgame.update_fold_to_3bet(fold_to_3bet)

        bb_depth = ctx.stack / max(1, ctx.bb_size)
        avg_stack = sum(ctx.all_stacks) / max(1, len(ctx.all_stacks))
        bf = 1.0
        if ctx.payouts and len(ctx.payouts) > 1:
            bf = bubble_factor(ctx.stack, ctx.all_stacks, ctx.payouts)

        t_ctx = TournamentContext(
            match_hands_played=self.match_hands_played,
            target_hands=self.target_hands,
            players_remaining=self.players_remaining,
            bubble_factor=bf,
            recent_win_rate=self.adaptive.recent_win_rate(),
            bb_depth=bb_depth,
            stack=ctx.stack,
            avg_stack=avg_stack,
        )
        mode = self.mode_selector.select(
            t_ctx, ctx.opponent_ids, stats_map, brutal_check=self.brutal
        )
        self._current_mode = mode
        self.adaptive.set_strategy_mode(mode.value)

        force_ctx: Dict[str, Any] = {"mode": mode.value}
        if self.brutal and self.brutal._intimidation_aborted:
            force_ctx["force_tag"] = True
        meta_style = self.meta.select_strategy(force_ctx, table_stats.confidence)

        safety = self._safety_override(ctx)
        if safety:
            return self._finalize(ctx, *safety, mode, meta_style)

        if self._over_budget(start, budget):
            return self._finalize(ctx, *self._fast_fallback(ctx, ehs), mode, meta_style)

        # --- Mode-specific fast paths ---
        if mode == StrategyMode.INTIMIDATION and self.early.in_early_phase(
            self.match_hands_played, self.target_hands
        ):
            early = self.early.decide(
                ctx.hole_cards,
                ctx.position,
                ctx.stack,
                ctx.bb_size,
                ctx.pot,
                ctx.is_facing_raise,
                ctx.facing_raise_size,
                ctx.allowed_actions,
                ctx.opponent_ids,
            )
            if early:
                return self._finalize(ctx, *early, mode, meta_style)

        if mode == StrategyMode.ENDGAME_CLOSE:
            eg = self.endgame.decide(
                ctx.hole_cards,
                ctx.position,
                ctx.stack,
                ctx.pot,
                ctx.bb_size,
                ctx.street,
                ctx.is_facing_raise,
                ctx.call_amount,
                ehs,
                ctx.allowed_actions,
                ctx.is_in_position,
            )
            if eg:
                return self._finalize(ctx, *eg, mode, meta_style)

        if bb_depth <= 15 and ctx.street == "preflop":
            pf = self._push_fold(ctx, ehs, bf * tuning.icm_tightness)
            if pf:
                return self._finalize(ctx, *pf, mode, meta_style)

        if mode == StrategyMode.ICM_SURVIVAL and bf > icm_bubble_threshold(1.5, tuning):
            if ctx.is_facing_raise and ehs < 0.55:
                return self._finalize(ctx, "fold", 0, f"ICM survival BF={bf:.1f}", mode, meta_style)

        # --- Meta style layer (MANIAC/TAG/LAG/NIT/EXPLOIT) ---
        if meta_style in ("MANIAC", "TAG", "LAG", "NIT"):
            gto_action, gto_amount, gto_reason = play_style(
                meta_style,
                hole=ctx.hole_cards,
                position=ctx.position,
                street=ctx.street,
                ehs=ehs,
                pot=ctx.pot,
                call_amount=ctx.call_amount,
                stack=ctx.stack,
                bb_size=ctx.bb_size,
                is_facing_raise=ctx.is_facing_raise,
                facing_raise_size=ctx.facing_raise_size,
                is_in_position=ctx.is_in_position,
                allowed_actions=ctx.allowed_actions,
                fold_to_3bet=fold_to_3bet,
                fold_to_cbet=fold_to_cbet,
            )
        elif ctx.street == "preflop":
            gto_action, gto_amount, gto_reason = preflop_action(
                hole=ctx.hole_cards,
                position=ctx.position,
                is_facing_raise=ctx.is_facing_raise,
                facing_raise_size=ctx.facing_raise_size,
                bb_size=ctx.bb_size,
                stack=ctx.stack,
                pot=ctx.pot,
                allowed_actions=ctx.allowed_actions,
                fold_to_3bet_avg=fold_to_3bet,
                tuning=tuning,
            )
        else:
            board_texture = self._board_texture(ctx.community_cards)
            gto_action, gto_amount, gto_reason = postflop_action(
                ehs=ehs,
                pot=ctx.pot,
                call_amount=ctx.call_amount,
                stack=ctx.stack,
                street=ctx.street,
                is_in_position=ctx.is_in_position,
                allowed_actions=ctx.allowed_actions,
                board_texture=board_texture,
                fold_to_cbet=fold_to_cbet if fold_to_cbet_opps_ok(table_stats) else 0.45,
                bet_size_mult=0.33 if table_stats.archetype == "nit" else (
                    0.75 if table_stats.archetype == "fish" else 0.5
                ),
                tuning=tuning,
            )

        fold_eq = calculate_fold_equity(max(fold_to_3bet, table_stats.fold_to_steal))
        can_check = any(a.get("action") == "check" for a in ctx.allowed_actions)
        if induce_bluff(ehs, ctx.street, table_stats.aggression_factor, can_check):
            return self._finalize(
                ctx, "check", 0, f"Induce vs AF={table_stats.aggression_factor:.1f}", mode, meta_style
            )

        # Bot-pattern exploitation boost
        if mode == StrategyMode.EXPLOITATION:
            gto_action, gto_amount, gto_reason = self._apply_bot_exploit(
                ctx, gto_action, gto_amount, gto_reason, ehs, stats_map, fold_eq
            )

        if ctx.payouts and bf > icm_bubble_threshold(1.5, tuning):
            if gto_action in ("raise",) and ehs < 0.72:
                return self._finalize(ctx, "fold", 0, f"ICM fold BF={bf:.1f}", mode, meta_style)
            if gto_action == "call" and ehs < 0.55 and bf > icm_bubble_threshold(1.5, tuning) * 1.2:
                return self._finalize(ctx, "fold", 0, f"ICM marginal fold BF={bf:.1f}", mode, meta_style)

        blend_thresh = exploit_blend_threshold(self.adaptive, table_stats.confidence)
        exploit = get_exploit_action(
            stats=table_stats,
            ehs=ehs,
            pot=ctx.pot,
            stack=ctx.stack,
            street=ctx.street,
            is_in_position=ctx.is_in_position,
            allowed_actions=ctx.allowed_actions,
            community=ctx.community_cards,
        )
        if exploit and table_stats.confidence >= blend_thresh - 0.15:
            amount = self._clamp(exploit.amount, ctx.allowed_actions, ctx.stack)
            chat = (
                f"{exploit.reasoning} | mode={mode.value} meta={meta_style} "
                f"EHS={ehs:.0%} FE={fold_eq:.0%}"
            )
            return self._finalize(ctx, exploit.action, amount, chat, mode, meta_style)

        amount = self._clamp(gto_amount, ctx.allowed_actions, ctx.stack)
        chat = (
            f"{gto_reason} | mode={mode.value} meta={meta_style} "
            f"EHS={ehs:.0%} [{ehs_to_bucket(ehs)}] BF={bf:.1f}"
        )
        return self._finalize(ctx, gto_action, amount, chat, mode, meta_style)

    def _apply_bot_exploit(
        self,
        ctx: GameContext,
        action: str,
        amount: float,
        reason: str,
        ehs: float,
        stats_map: Dict[str, OpponentStats],
        fold_eq: float,
    ) -> Tuple[str, float, str]:
        for oid in ctx.opponent_ids:
            prof = self.detector.get_profile(oid)
            if not prof:
                continue
            if prof.bot_type == BotType.NIT and ctx.street == "preflop" and not ctx.is_facing_raise:
                if ctx.position in ("CO", "BTN", "SB") and ehs >= 0.25:
                    return "raise", min(ctx.stack, ctx.bb_size * 2.5), f"Steal orbit vs nit FE={fold_eq:.0%}"
            if prof.bot_type == BotType.CALLING_STATION and ehs >= 0.55:
                return "raise", min(ctx.stack, ctx.pot * 0.75), "Value station — no bluffs"
            if prof.bot_type == BotType.SCARED_MONEY and fold_eq > 0.6:
                return "raise", min(ctx.stack, ctx.pot * 1.2), "Pressure scared money"
        return action, amount, reason

    def _finalize(
        self,
        ctx: GameContext,
        action: str,
        amount: float,
        chat: str,
        mode: StrategyMode,
        meta_style: str,
    ) -> Tuple[str, float, str]:
        if ctx.hand_number is not None and ctx.table_id:
            req = pot_odds_required(ctx.call_amount, ctx.pot)
            self.adaptive.record_decision(
                table_id=ctx.table_id,
                hand_number=ctx.hand_number,
                street=ctx.street,
                action=action,
                amount=amount,
                ehs=self._last_ehs,
                pot=ctx.pot,
                call_amount=ctx.call_amount,
                position=ctx.position,
                required_equity=req,
                notation=hand_notation(ctx.hole_cards),
                stack=ctx.stack,
            )
        mistake = ""
        if ctx.hand_number and self.adaptive.mistakes.bad_calls > 0:
            last = self.adaptive._pending.get(
                self.adaptive.hand_key(ctx.table_id, ctx.hand_number), []
            )
            if last and last[-1].mistake:
                mistake = mistake_line()
        tilt = build_chat_message(
            action=action,
            won_big_pot=False,
            was_bluff=action in ("raise", "bet") and self._last_ehs < 0.35,
            bubble_pressure=mode == StrategyMode.ICM_SURVIVAL,
            ehs=self._last_ehs,
            strategy_mode=mode.value,
        )
        full_chat = f"{chat} {tilt} {mistake}".strip()[:200]
        return action, amount, full_chat

    def _safety_override(
        self, ctx: GameContext
    ) -> Optional[Tuple[str, float, str]]:
        """Never fold premium preflop; fallback_to_gto guardrails."""
        notation = hand_notation(ctx.hole_cards)
        acts = {a.get("action", "").lower() for a in ctx.allowed_actions}
        if ctx.street == "preflop" and notation in ("AA", "KK", "QQ"):
            if ctx.is_facing_raise and "raise" in acts:
                mx = ctx.stack
                for a in ctx.allowed_actions:
                    if a.get("action") in ("raise", "all-in"):
                        mx = float(a.get("maxAmount", ctx.stack))
                return "raise", mx, f"Safety: never fold {notation}"
            if "raise" in acts and not ctx.is_facing_raise:
                for a in ctx.allowed_actions:
                    if a.get("action") == "raise":
                        return (
                            "raise",
                            float(a.get("minAmount", ctx.bb_size * 2)),
                            f"Safety: open {notation}",
                        )
        return None

    def _push_fold(
        self, ctx: GameContext, ehs: float, icm_mult: float
    ) -> Optional[Tuple[str, float, str]]:
        notation = hand_notation(ctx.hole_cards)
        bb_depth = ctx.stack / max(1, ctx.bb_size)
        bf = icm_mult

        if ctx.is_facing_raise or ctx.call_amount > ctx.bb_size:
            to_call = ctx.call_to_amount or ctx.call_amount
            if should_call_push(notation, bb_depth, bf) or ehs >= 0.58:
                return "call", to_call, f"ICM call {notation}"
            if ehs >= 0.52 and bf <= 1.3:
                return "call", to_call, f"Call shove {notation}"
            return "fold", 0, f"Fold vs shove {notation}"

        if should_push(notation, ctx.position, bb_depth, ctx.is_facing_raise) or ehs >= 0.62:
            return "raise", ctx.stack, f"Push {notation} @ {bb_depth:.1f}BB"
        return "fold", 0, f"Push-fold fold {notation}"

    def _time_budget(self, start: float, deadline: float) -> float:
        if deadline <= 0:
            return 1.4
        return max(0.15, deadline - time.time() - 0.25)

    def _over_budget(self, start: float, budget: float) -> bool:
        return (time.monotonic() - start) > budget

    def _fast_fallback(self, ctx: GameContext, ehs: float) -> Tuple[str, float, str]:
        acts = {a.get("action", "").lower() for a in ctx.allowed_actions}
        if "check" in acts:
            return "check", 0, f"Time budget — check EHS={ehs:.0%}"
        if ehs > 0.5 and "call" in acts:
            return "call", ctx.call_to_amount or ctx.call_amount, "Time budget — call"
        return "fold", 0, "Time budget — fold"

    def _board_texture(self, community: List[str]) -> str:
        if len(community) < 3:
            return "dry"
        from engine.hand_eval import card_rank, card_suit
        suits = [card_suit(c) for c in community]
        ranks = [card_rank(c) for c in community]
        if len(set(suits)) <= 2 or (max(ranks) - min(ranks)) <= 4:
            return "wet"
        return "dry"

    def _clamp(self, amount: float, allowed_actions: List[Dict], stack: float) -> float:
        if amount <= 0:
            return 0.0
        for a in allowed_actions:
            if a.get("action") in ("bet", "raise", "all-in"):
                mn = float(a.get("minAmount", 0))
                mx = float(a.get("maxAmount", stack))
                return max(mn, min(mx, min(amount, stack)))
        return min(amount, stack)

    @property
    def current_mode(self) -> str:
        return self._current_mode.value


def fold_to_cbet_opps_ok(stats: OpponentStats) -> bool:
    return stats.fold_to_cbet_flop_opps > 3 or stats.fold_to_cbet_opps > 5
