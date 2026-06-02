"""agent/strategies/adaptive.py — rule-driven strategy loaded from strategy_rules.json.

The rules file (not this code) is what DeepSeek modifies each cycle.
This code is a pure executor: read rules → apply → return decision.
Never hardcodes thresholds — every number comes from the JSON.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from agent.arbiter import GameContext
from agent.strategies.base import (
    Decision, aggr, bet_pot, board_hits, clamp, has,
    is_ip, open_bb, raise_to, safe_check_fold, to_call,
)

_RULES_PATH = Path(__file__).parent.parent.parent / "strategy_rules.json"
_cached_rules: Optional[dict] = None
_cached_mtime: float = 0.0

_RANK_MAP = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
             "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


def _rank(card: str) -> int:
    return _RANK_MAP.get(card[0].upper(), 0) if card else 0


def _speculative_ehs_bonus(hole_cards: list) -> float:
    """Return EHS bonus/threshold reduction for speculative hands.

    Returns (suited_bonus, is_connector, is_broadway) so caller can apply
    the right see-flop threshold from the rules.
    """
    if len(hole_cards) < 2:
        return 0.0, False, False
    r1, r2 = _rank(hole_cards[0]), _rank(hole_cards[1])
    s1, s2 = hole_cards[0][-1].lower() if hole_cards[0] else "", hole_cards[1][-1].lower() if hole_cards[1] else ""
    suited = s1 == s2 and s1 != ""
    gap = abs(r1 - r2)
    is_connector = 1 <= gap <= 2          # connectors and one-gappers
    is_broadway  = min(r1, r2) >= 10      # both cards T or higher
    suited_bonus = 0.04 if suited else 0.0
    return suited_bonus, is_connector, is_broadway


def _pot_odds_call(ehs: float, call_amount: float, pot: float) -> tuple[bool, str]:
    """Felt ReaderL4 style: equity covers price? -> call is profitable.

    Returns (should_call, reason_string).
    required_equity = call / (call + pot)  — the pot-odds break-even point.
    If EHS >= required_equity * margin, calling is +EV.
    """
    if call_amount <= 0 or (call_amount + pot) <= 0:
        return False, ""
    required = call_amount / (call_amount + pot)
    margin = float(_rules().get("pot_odds", {}).get("call_margin", 1.05))
    covers = ehs >= required * margin
    reason = f"equity {ehs:.0%} {'covers' if covers else 'misses'} price {required:.0%}"
    return covers, reason


def _rules() -> dict:
    """Hot-reload rules when the file changes (picks up mid-session updates)."""
    global _cached_rules, _cached_mtime
    try:
        mtime = _RULES_PATH.stat().st_mtime
        if _cached_rules is None or mtime != _cached_mtime:
            _cached_rules = json.loads(_RULES_PATH.read_text())
            _cached_mtime = mtime
    except Exception:
        pass
    return _cached_rules or {}


def _r(section: str, key: str, default: float = 0.5) -> float:
    return float(_rules().get(section, {}).get(key, default))


def _adj(ehs: float, ip: bool) -> float:
    """Adjust EHS threshold for position."""
    reduction = _r("adjustments", "ip_threshold_reduction", 0.04)
    increase  = _r("adjustments", "oop_threshold_increase", 0.03)
    return ehs - reduction if ip else ehs + increase


# ── preflop ───────────────────────────────────────────────────────────

def _preflop(ctx: GameContext, ehs: float, ip: bool) -> Decision:
    r = _rules().get("preflop", {})
    spec = r.get("speculative_hand_see_flop", {})
    spec_enabled = spec.get("enabled", False)
    suited_bonus, is_connector, is_broadway = _speculative_ehs_bonus(ctx.hole_cards)

    # Effective EHS: suited hands get a bonus (more equity postflop)
    ehs_eff = ehs + (suited_bonus if spec_enabled else 0.0)

    if ctx.is_facing_raise:
        call_min  = r.get("vs_raise_call_ip_min_ehs" if ip else "vs_raise_call_oop_min_ehs", 0.65)
        bet3_min  = r.get("vs_raise_3bet_min_ehs", 0.80)
        bet3_mult = r.get("vs_raise_3bet_sizing_mult", 3.0)

        if ehs_eff >= bet3_min:
            return aggr(ctx), raise_to(bet3_mult, ctx), f"A 3bet value ehs={ehs:.2f}"
        if ehs_eff >= call_min:
            return "call", to_call(ctx), f"A call {'ip' if ip else 'oop'} ehs={ehs:.2f}"

        # Speculative hands defend vs raise to see the flop
        if spec_enabled and ip and (is_connector or is_broadway):
            conn_min = spec.get("connector_see_flop_min_ehs", 0.40)
            bway_min = spec.get("broadway_see_flop_min_ehs", 0.45)
            see_flop_min = bway_min if is_broadway else conn_min
            if ehs_eff >= see_flop_min:
                return "call", to_call(ctx), f"A spec call {'connector' if is_connector else 'broadway'} ehs={ehs:.2f}"

        return "fold", 0.0, f"A fold preflop ehs={ehs:.2f}"

    open_min = r.get("open_ip_min_ehs" if ip else "open_oop_min_ehs", 0.65)
    sizing   = r.get("open_sizing_bb", 3.0)
    if ehs_eff >= open_min:
        return aggr(ctx), open_bb(sizing, ctx), f"A open ehs={ehs:.2f}"

    # BB isolation raise: strong hand in BB, everyone limped/checked → squeeze value
    bb_raise_min = float(r.get("bb_raise_min_ehs", 0.75))
    if (not ip) and ehs_eff >= bb_raise_min and has(ctx, "raise"):
        return aggr(ctx), open_bb(sizing, ctx), f"A BB squeeze ehs={ehs:.2f}"

    # Speculative hands open to see the flop rather than folding
    if spec_enabled and (is_connector or is_broadway):
        conn_min = spec.get("connector_see_flop_min_ehs", 0.40)
        bway_min = spec.get("broadway_see_flop_min_ehs", 0.45)
        see_flop_min = bway_min if is_broadway else conn_min
        if ehs_eff >= see_flop_min and has(ctx, "raise"):
            return aggr(ctx), open_bb(sizing, ctx), f"A spec open {'connector' if is_connector else 'broadway'} ehs={ehs:.2f}"

    return safe_check_fold(ctx)


# ── aggression tier helper ────────────────────────────────────────────

def _aggr_tier(ehs: float, pot: float, ctx: GameContext) -> tuple[float, str] | None:
    """Return (bet_size, reason) for the highest justified aggression tier, or None.

    Tiers read from strategy_rules.json aggression block:
      tier1 (near certain): EHS >= river_allin_min_ehs  → all-in or large overbet
      tier2 (highly likely): EHS >= large_bet_min_ehs   → large bet (1.2x pot)
      tier3 (likely):        EHS >= medium_bet_min_ehs  → medium bet (0.75x pot)
    """
    ag = _rules().get("aggression", {})
    if not ag:
        return None

    allin_min  = float(ag.get("river_allin_min_ehs", 0.90))
    large_min  = float(ag.get("large_bet_min_ehs", 0.78))
    large_mult = float(ag.get("large_bet_pot_mult", 1.2))
    med_min    = float(ag.get("medium_bet_min_ehs", 0.65))
    med_mult   = float(ag.get("medium_bet_pot_mult", 0.75))

    if ehs >= allin_min:
        # All-in or maximum raise — near-certain win
        all_in_size = ctx.stack
        return clamp(all_in_size, ctx), f"A aggr tier1 ALLIN ehs={ehs:.2f}"

    if ehs >= large_min:
        size = clamp(pot * large_mult, ctx)
        return size, f"A aggr tier2 large-bet ehs={ehs:.2f}"

    if ehs >= med_min:
        size = clamp(pot * med_mult, ctx)
        return size, f"A aggr tier3 med-bet ehs={ehs:.2f}"

    return None


# ── flop ──────────────────────────────────────────────────────────────

def _flop(ctx: GameContext, ehs: float, ip: bool) -> Decision:
    r    = _rules().get("flop", {})
    ag   = _rules().get("aggression", {})
    hits = board_hits(ctx)
    bluff = r.get("bluff_enabled", False)

    if ctx.call_amount > 0:
        raise_min  = r.get("facing_bet_raise_min_ehs", 0.83)
        raise_mult = r.get("facing_bet_raise_mult", 3.0)
        call_min   = r.get("facing_bet_call_min_ehs", 0.47)
        fold_max   = r.get("facing_bet_fold_max_ehs", 0.35)

        # Aggressive raise tier: if very strong, raise bigger than standard
        flop_aggr_min = float(ag.get("flop_raise_aggressive_min_ehs", 0.85))
        if ehs >= flop_aggr_min:
            tier = _aggr_tier(ehs, ctx.pot, ctx)
            if tier:
                size, reason = tier
                return aggr(ctx), size, reason
        if ehs >= raise_min:
            return aggr(ctx), raise_to(raise_mult, ctx), f"A flop raise ehs={ehs:.2f}"
        if ehs >= call_min:
            return "call", to_call(ctx), f"A flop call ehs={ehs:.2f}"
        # Pot-odds layer: call even below threshold if equity covers price
        po_call, po_reason = _pot_odds_call(ehs, ctx.call_amount, ctx.pot)
        if po_call and ehs > fold_max:
            return "call", to_call(ctx), f"A flop pot-odds call {po_reason}"
        return "fold", 0.0, f"A flop fold ehs={ehs:.2f} {po_reason}"

    trap_min   = r.get("no_bet_trap_min_ehs", 0.87)
    value_min  = r.get("no_bet_value_min_ehs", 0.62)
    value_pot  = r.get("no_bet_value_sizing_pot", 0.75)
    ob_min     = r.get("no_bet_board_hit_overbet_min_ehs", 0.63)
    ob_pot     = r.get("no_bet_board_hit_overbet_pot", 1.5)
    bluff_max  = r.get("bluff_max_ehs", 0.20)
    bluff_pot  = r.get("bluff_sizing_pot", 0.50)

    adj_trap  = _adj(trap_min, ip)
    adj_value = _adj(value_min, ip)

    if ehs >= adj_trap and has(ctx, "check"):
        return "check", 0.0, f"A flop trap ehs={ehs:.2f}"

    if hits >= 1 and ehs >= ob_min and _rules().get("adjustments", {}).get("board_hit_overbet_enabled", True):
        return aggr(ctx), clamp(ctx.pot * ob_pot, ctx), f"A flop overbet board-hit ehs={ehs:.2f}"

    # Use aggression tier for strong hands — bet bigger when more confident
    if ehs >= adj_value:
        tier = _aggr_tier(ehs, ctx.pot, ctx)
        if tier:
            size, reason = tier
            return aggr(ctx), size, reason
        return aggr(ctx), bet_pot(value_pot, ctx), f"A flop value ehs={ehs:.2f}"

    if bluff and ehs <= bluff_max:
        return aggr(ctx), bet_pot(bluff_pot, ctx), f"A flop bluff ehs={ehs:.2f}"

    return "check", 0.0, f"A flop check ehs={ehs:.2f}"


# ── turn ──────────────────────────────────────────────────────────────

def _turn(ctx: GameContext, ehs: float, ip: bool) -> Decision:
    r    = _rules().get("turn", {})
    bluff = r.get("bluff_enabled", False)

    if ctx.call_amount > 0:
        raise_min  = r.get("facing_bet_raise_min_ehs", 0.86)
        raise_mult = r.get("facing_bet_raise_mult", 3.0)
        call_min   = r.get("facing_bet_call_min_ehs", 0.47)
        fold_max   = r.get("facing_bet_fold_max_ehs", 0.44)

        if ehs >= raise_min:
            tier = _aggr_tier(ehs, ctx.pot, ctx)
            if tier:
                size, reason = tier
                return aggr(ctx), size, reason
            return aggr(ctx), raise_to(raise_mult, ctx), f"A turn raise ehs={ehs:.2f}"
        if ehs >= call_min:
            return "call", to_call(ctx), f"A turn call ehs={ehs:.2f}"
        po_call, po_reason = _pot_odds_call(ehs, ctx.call_amount, ctx.pot)
        if po_call and ehs > fold_max:
            return "call", to_call(ctx), f"A turn pot-odds call {po_reason}"
        return "fold", 0.0, f"A turn fold ehs={ehs:.2f} {po_reason}"

    trap_min  = r.get("no_bet_trap_min_ehs", 0.88)
    value_min = r.get("no_bet_value_min_ehs", 0.67)
    value_pot = r.get("no_bet_value_sizing_pot", 0.80)
    bluff_max = r.get("bluff_max_ehs", 0.20)
    bluff_pot = r.get("bluff_sizing_pot", 0.60)

    if ehs >= _adj(trap_min, ip) and has(ctx, "check"):
        return "check", 0.0, f"A turn trap ehs={ehs:.2f}"

    if ehs >= _adj(value_min, ip):
        tier = _aggr_tier(ehs, ctx.pot, ctx)
        if tier:
            size, reason = tier
            return aggr(ctx), size, reason
        return aggr(ctx), bet_pot(value_pot, ctx), f"A turn value ehs={ehs:.2f}"

    if bluff and ehs <= bluff_max:
        return aggr(ctx), bet_pot(bluff_pot, ctx), f"A turn bluff ehs={ehs:.2f}"
    return "check", 0.0, f"A turn check ehs={ehs:.2f}"


# ── river ──────────────────────────────────────────────────────────────

def _river(ctx: GameContext, ehs: float, ip: bool) -> Decision:
    r    = _rules().get("river", {})
    bluff = r.get("bluff_enabled", False)

    if ctx.call_amount > 0:
        raise_min  = r.get("facing_bet_raise_min_ehs", 0.85)
        raise_mult = r.get("facing_bet_raise_mult", 2.5)
        call_min   = r.get("facing_bet_call_min_ehs", 0.52)
        fold_max   = r.get("facing_bet_fold_max_ehs", 0.50)

        if ehs >= raise_min:
            tier = _aggr_tier(ehs, ctx.pot, ctx)
            if tier:
                size, reason = tier
                return aggr(ctx), size, reason
            return aggr(ctx), raise_to(raise_mult, ctx), f"A river raise ehs={ehs:.2f}"
        if ehs >= call_min:
            return "call", to_call(ctx), f"A river call ehs={ehs:.2f}"
        # River: pot odds is the final arbiter — no more cards after this
        po_call, po_reason = _pot_odds_call(ehs, ctx.call_amount, ctx.pot)
        if po_call and ehs > fold_max:
            return "call", to_call(ctx), f"A river pot-odds call {po_reason}"
        return "fold", 0.0, f"A river fold ehs={ehs:.2f} {po_reason}"

    value_min = r.get("no_bet_value_min_ehs", 0.69)
    value_pot = r.get("no_bet_value_sizing_pot", 0.75)
    bluff_max = r.get("bluff_max_ehs", 0.20)
    bluff_pot = r.get("bluff_sizing_pot", 0.50)

    if ehs >= _adj(value_min, ip):
        # River is the last street — maximise EV with tiered sizing
        tier = _aggr_tier(ehs, ctx.pot, ctx)
        if tier:
            size, reason = tier
            return aggr(ctx), size, reason
        return aggr(ctx), bet_pot(value_pot, ctx), f"A river value ehs={ehs:.2f}"

    if bluff and ehs <= bluff_max:
        return aggr(ctx), bet_pot(bluff_pot, ctx), f"A river bluff ehs={ehs:.2f}"
    return "check", 0.0, f"A river check ehs={ehs:.2f}"


# ── entry ──────────────────────────────────────────────────────────────

def decide(ctx: GameContext, ehs: float, bb_depth: float) -> Decision:
    ip = is_ip(ctx)
    s  = ctx.street
    if s == "preflop": return _preflop(ctx, ehs, ip)
    if s == "flop":    return _flop(ctx, ehs, ip)
    if s == "turn":    return _turn(ctx, ehs, ip)
    return _river(ctx, ehs, ip)
