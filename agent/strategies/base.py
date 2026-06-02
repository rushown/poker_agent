"""agent/strategies/base.py — shared helpers for S1-S9 strategy modules."""
from __future__ import annotations

from typing import Tuple

from agent.arbiter import GameContext

Decision = Tuple[str, float, str]


def acts(ctx: GameContext) -> set:
    return {a.get("action", "").lower() for a in ctx.allowed_actions}


def has(ctx: GameContext, action: str) -> bool:
    return action in acts(ctx)


def bet_bounds(ctx: GameContext) -> tuple:
    for a in ctx.allowed_actions:
        if a.get("action") in ("bet", "raise", "all-in"):
            return float(a.get("minAmount", 0)), float(a.get("maxAmount", ctx.stack))
    return 0.0, ctx.stack


def clamp(amount: float, ctx: GameContext) -> float:
    if amount <= 0:
        return 0.0
    mn, mx = bet_bounds(ctx)
    if mn == 0.0 and mx == ctx.stack and not any(
        a.get("action") in ("bet", "raise", "all-in") for a in ctx.allowed_actions
    ):
        return 0.0
    return max(mn, min(mx, amount, ctx.stack))


def aggr(ctx: GameContext) -> str:
    """Return the correct aggressive action ('raise' or 'bet')."""
    a = acts(ctx)
    if "raise" in a:
        return "raise"
    if "bet" in a:
        return "bet"
    return "raise"


def pot_odds(ctx: GameContext) -> float:
    return ctx.call_amount / max(1.0, ctx.pot + ctx.call_amount) if ctx.call_amount > 0 else 0.0


def board_hits(ctx: GameContext) -> int:
    """Count how many hole card ranks appear in community cards."""
    if not ctx.community_cards:
        return 0
    hole_ranks = {c[0].upper() for c in ctx.hole_cards}
    board_ranks = {c[0].upper() for c in ctx.community_cards}
    return len(hole_ranks & board_ranks)


def is_ip(ctx: GameContext) -> bool:
    return ctx.is_in_position or ctx.position in ("BTN", "CO", "HJ")


def to_call(ctx: GameContext) -> float:
    return ctx.call_to_amount or ctx.call_amount


def bet_pot(mult: float, ctx: GameContext) -> float:
    return clamp(ctx.pot * mult, ctx)


def open_bb(mult: float, ctx: GameContext) -> float:
    return clamp(ctx.bb_size * mult, ctx)


def raise_to(mult: float, ctx: GameContext) -> float:
    return clamp(max(ctx.call_amount * mult, ctx.bb_size * 2), ctx)


def safe_check_fold(ctx: GameContext) -> Decision:
    if has(ctx, "check"):
        return "check", 0.0, "safe check"
    return "fold", 0.0, "fold"
