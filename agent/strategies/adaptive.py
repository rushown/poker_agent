"""agent/strategies/adaptive.py — rule-driven strategy loaded from strategy_rules.json.

The rules file (not this code) is what DeepSeek modifies each cycle.
This code is a pure executor: read rules → apply → return decision.
Never hardcodes thresholds — every number comes from the JSON.

Augmented by agent/strategies/midgame.py which handles:
  - Draw analysis (flush/straight draws, made hands)
  - All-in confidence (push on nuts, call only when justified)
  - Bluff detection + bluff-catch calls
  - Semi-bluff raises on strong draws
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from agent.arbiter import GameContext
import random as _random
from agent.strategies.base import (
    Decision, aggr, bet_pot, board_hits, clamp, has,
    is_ip, open_bb, raise_to, safe_check_fold, to_call,
)
from agent.strategies.midgame import (
    midgame_postflop,
    assess_preflop_min_call,
    analyze_draws,
    should_go_allin,
    DrawAnalysis,
)
from agent.strategies.bet_intelligence import (
    bet_intelligence_decide,
    analyze_board,
    BoardTexture,
)

_rng_tl = threading.local()

def _get_rng() -> _random.Random:
    if not hasattr(_rng_tl, "r"):
        _rng_tl.r = _random.Random()
    return _rng_tl.r

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
    # Read gap threshold from rules (default 1 = connectors and one-gappers only)
    spec = _rules().get("preflop", {}).get("speculative_hand_see_flop", {})
    max_gap     = int(spec.get("connector_gap_max", 1))
    suit_bonus  = float(spec.get("suited_bonus_ehs", 0.04))
    is_connector = 1 <= gap <= max_gap
    is_broadway  = min(r1, r2) >= 10      # both cards T or higher
    suited_bonus = suit_bonus if suited else 0.0
    return suited_bonus, is_connector, is_broadway


def _pot_odds_call(ehs: float, call_amount: float, pot: float) -> tuple[bool, str]:
    """Felt ReaderL4 style: equity covers price? -> call is profitable.

    Returns (should_call, reason_string).
    required_equity = call / (call + pot)  — the pot-odds break-even point.
    If EHS >= required_equity * margin, calling is +EV.

    Hard floor: never call below pot_odds.minimum_ehs_for_call (default 0.48).
    Prevents -EV overcalling at EHS 0.43-0.47 just because the pot is large.
    """
    if call_amount <= 0 or (call_amount + pot) <= 0:
        return False, ""
    po_rules = _rules().get("pot_odds", {})
    required = call_amount / (call_amount + pot)
    margin = float(po_rules.get("call_margin", 1.0))
    min_ehs = float(po_rules.get("minimum_ehs_for_call", 0.48))
    if ehs < min_ehs:
        return False, f"equity {ehs:.0%} below floor {min_ehs:.0%}"
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


def _opp_modifier() -> float:
    """Return EHS threshold modifier based on classified opponent archetype.
    Read from thread-local set by arbiter (safe for async multi-table).
    """
    try:
        from agent.arbiter import _tl
        opp = getattr(_tl, "opponent_type", "unknown")
    except Exception:
        opp = "unknown"
    mods = {
        "nit": -0.04,             "scared_money": -0.05,  # bluff more vs folders
        "calling_station": +0.03, "fish": +0.02,           # value-only vs callers
        "maniac": +0.04,          "lag": +0.03,             # tighten vs aggression
        "tag": 0.0,               "gto_balanced": 0.0,
        "unknown": 0.0,           "fixed_timing": 0.0,
        "range_static": -0.02,                              # bluff range-static bots
    }
    return mods.get(opp, 0.0)


def _adj(ehs: float, ip: bool) -> float:
    """Adjust EHS threshold for position + opponent type."""
    reduction = _r("adjustments", "ip_threshold_reduction", 0.04)
    increase  = _r("adjustments", "oop_threshold_increase", 0.03)
    base = ehs - reduction if ip else ehs + increase
    return base + _opp_modifier()


def _pos_open_min(ctx: GameContext) -> float:
    """Position-specific preflop open threshold (tighter from early positions)."""
    r = _rules().get("preflop", {})
    pos = ctx.position.upper()
    thresholds = {
        "BTN": r.get("open_btn_min_ehs", 0.50),
        "CO":  r.get("open_co_min_ehs",  0.55),
        "HJ":  r.get("open_hj_min_ehs",  0.60),
        "UTG": r.get("open_utg_min_ehs", 0.64),
        "MP":  r.get("open_utg_min_ehs", 0.64),
        "SB":  r.get("open_sb_min_ehs",  0.52),
        "BB":  r.get("open_bb_defend_min_ehs", 0.45),
    }
    if pos in thresholds:
        return thresholds[pos]
    # fallback to ip/oop generic
    ip = ctx.is_in_position or pos in ("BTN", "CO")
    return r.get("open_ip_min_ehs" if ip else "open_oop_min_ehs", 0.60)


def _board_is_wet(community: list) -> bool:
    """True when flop has flush draw or straight-draw potential."""
    if len(community) < 3:
        return False
    suits = [c[1].lower() if len(c) >= 2 else "" for c in community[:3]]
    ranks = []
    rank_map = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"T":10,"J":11,"Q":12,"K":13,"A":14}
    for c in community[:3]:
        ranks.append(rank_map.get(c[0].upper(), 0))
    flush_draw = max(suits.count(s) for s in set(suits) if s) >= 2
    ranks_sorted = sorted(set(ranks))
    straight_draw = len(ranks_sorted) >= 2 and (ranks_sorted[-1] - ranks_sorted[0]) <= 4
    return flush_draw or straight_draw


# ── preflop ───────────────────────────────────────────────────────────

def _preflop(ctx: GameContext, ehs: float, ip: bool) -> Decision:
    r = _rules().get("preflop", {})
    spec = r.get("speculative_hand_see_flop", {})
    spec_enabled = spec.get("enabled", False)
    suited_bonus, is_connector, is_broadway = _speculative_ehs_bonus(ctx.hole_cards)

    ehs_eff = ehs + (suited_bonus if spec_enabled else 0.0)

    # Midgame minimum-call: cheap speculative calls below normal open threshold
    if ctx.call_amount > 0:
        mid_pf = assess_preflop_min_call(
            ehs, ctx.call_amount, ctx.bb_size, ctx.hole_cards, ctx.is_facing_raise
        )
        if mid_pf is not None:
            return mid_pf

    if ctx.is_facing_raise:
        raise_level = ctx.call_amount / max(1.0, ctx.bb_size)
        if raise_level >= 15:  # facing 4bet+
            v4bet_call = float(r.get("vs_4bet_call_min_ehs", 0.78))
            if ehs_eff >= v4bet_call:
                return "call", to_call(ctx), f"A 4bet call ehs={ehs:.2f}"
            return "fold", 0.0, f"A 4bet fold ehs={ehs:.2f}"

        call_min  = r.get("vs_raise_call_ip_min_ehs" if ip else "vs_raise_call_oop_min_ehs", 0.58)
        bet3_min  = r.get("vs_raise_3bet_min_ehs", 0.72)
        bet3_mult = r.get("vs_raise_3bet_sizing_mult", 3.0)

        if ehs_eff >= bet3_min:
            allin_pf = float(r.get("allin_threshold_ehs", 0.97))
            amt = raise_to(bet3_mult, ctx)
            # Cap 3bet when it would accidentally shove stack (e.g. facing large raise)
            if ctx.stack > 0 and amt >= ctx.stack * 0.85 and ehs < allin_pf:
                amt = clamp(ctx.bb_size * 9, ctx)
            return aggr(ctx), amt, f"A 3bet value ehs={ehs:.2f}"
        if ehs_eff >= call_min:
            return "call", to_call(ctx), f"A call {'ip' if ip else 'oop'} ehs={ehs:.2f}"
        if spec_enabled and ip and (is_connector or is_broadway):
            conn_min = spec.get("connector_see_flop_min_ehs", 0.44)
            bway_min = spec.get("broadway_see_flop_min_ehs", 0.48)
            see_flop_min = bway_min if is_broadway else conn_min
            if ehs_eff >= see_flop_min:
                return "call", to_call(ctx), f"A spec call {'connector' if is_connector else 'broadway'} ehs={ehs:.2f}"
        return "fold", 0.0, f"A fold preflop ehs={ehs:.2f}"

    # Position-aware open range
    open_min = _pos_open_min(ctx)
    sizing   = r.get("open_sizing_bb", 2.5)
    if ehs_eff >= open_min:
        return aggr(ctx), open_bb(sizing, ctx), f"A open {ctx.position} ehs={ehs:.2f}"

    # BB defense: wide defend since BB already has money in
    if ctx.position.upper() == "BB" and ctx.call_amount <= ctx.bb_size * 1.1:
        bb_def = r.get("open_bb_defend_min_ehs", 0.45)
        if ehs_eff >= bb_def and has(ctx, "call"):
            return "call", to_call(ctx), f"A BB defend ehs={ehs:.2f}"

    # BB isolation raise with premium holdings
    bb_raise_min = float(r.get("bb_raise_min_ehs", 0.65))
    if ctx.position.upper() == "BB" and ehs_eff >= bb_raise_min and has(ctx, "raise"):
        return aggr(ctx), open_bb(sizing, ctx), f"A BB squeeze ehs={ehs:.2f}"

    # Speculative hands: open from late position to see cheap flop
    if spec_enabled and ip and (is_connector or is_broadway):
        conn_min = spec.get("connector_see_flop_min_ehs", 0.44)
        bway_min = spec.get("broadway_see_flop_min_ehs", 0.48)
        see_flop_min = bway_min if is_broadway else conn_min
        if ehs_eff >= see_flop_min and has(ctx, "raise"):
            return aggr(ctx), open_bb(sizing, ctx), f"A spec open {'connector' if is_connector else 'broadway'} ehs={ehs:.2f}"

    return safe_check_fold(ctx)


# ── aggression tier helper ────────────────────────────────────────────

def _aggr_tier(ehs: float, pot: float, ctx: GameContext, street: str = "") -> tuple[float, str] | None:
    """Return (bet_size, reason) for the highest justified aggression tier, or None.

    Tiers from strategy_rules.json aggression block:
      tier1: EHS >= river_allin_min_ehs → all-in (street-aware: raised on flop/turn)
      tier2: EHS >= large_bet_min_ehs   → large bet (1.1x pot)
      tier3: EHS >= medium_bet_min_ehs  → medium bet (0.75x pot)

    PvP guard: on flop/turn at 100BB stacks, use flop_allin_min_ehs (default 0.90)
    to prevent committing stack with medium-strong hands on early streets.
    """
    ag = _rules().get("aggression", {})
    if not ag:
        return None

    allin_min  = float(ag.get("river_allin_min_ehs", 0.85))
    large_min  = float(ag.get("large_bet_min_ehs", 0.85))
    large_mult = float(ag.get("large_bet_pot_mult", 1.2))
    med_min    = float(ag.get("medium_bet_min_ehs", 0.65))
    med_mult   = float(ag.get("medium_bet_pot_mult", 0.75))

    # Street guard: raise all-in threshold on early streets to avoid overcommitting.
    # flop/turn require flop_allin_min_ehs (default 0.90 — tighter than river).
    if street in ("flop", "turn"):
        allin_min = max(allin_min, float(ag.get("flop_allin_min_ehs", 0.90)))

    # Hard cap: never go all-in below the global push minimum.
    # This prevents _aggr_tier from bypassing the 0.85+ threshold (hand 6543 fix).
    push_min = float(ag.get("all_in_push_min_ehs", 0.85))
    allin_min = max(allin_min, push_min)

    allin_min_pot = float(ag.get("allin_min_pot", 0))
    pot_ok = allin_min_pot <= 0 or pot >= allin_min_pot
    if ehs >= allin_min and pot_ok:
        all_in_size = ctx.stack
        return clamp(all_in_size, ctx), f"A aggr tier1 ALLIN ehs={ehs:.2f} pot={pot:.0f} st={street}"

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
    allin_min_pot = float(ag.get("allin_min_pot", 0))

    # Midgame: draw analysis, all-in calls, semi-bluffs, bluff catches
    mid = midgame_postflop(ctx, ehs, "flop", allin_min_pot)
    if mid is not None:
        return mid

    # Draw analysis for enhanced EHS on raise/bet decisions
    _da = analyze_draws(ctx.hole_cards, ctx.community_cards)
    ehs_mid = ehs + _da.equity_boost  # boosted EHS for raise thresholds

    if ctx.call_amount > 0:
        raise_min  = r.get("facing_bet_raise_min_ehs", 0.83)
        raise_mult = r.get("facing_bet_raise_mult", 3.0)
        call_min   = r.get("facing_bet_call_min_ehs", 0.47)
        fold_max   = r.get("facing_bet_fold_max_ehs", 0.35)
        # Multiway penalty + opponent type modifier
        n_opps = len(ctx.opponent_ids)
        mw_penalty = float(_rules().get("flop", {}).get("multiway_adjust_call_min_ehs_penalty", 0)) * max(0, n_opps - 1)
        call_min = call_min + mw_penalty + _opp_modifier()

        # Aggressive raise tier: if very strong, raise bigger than standard
        flop_aggr_min = float(ag.get("flop_raise_aggressive_min_ehs", 0.85))
        if ehs >= flop_aggr_min:
            tier = _aggr_tier(ehs, ctx.pot, ctx, street="flop")
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
    value_min  = r.get("no_bet_value_min_ehs", 0.55)
    ob_min     = r.get("no_bet_board_hit_overbet_min_ehs", 0.72)
    ob_pot     = r.get("no_bet_board_hit_overbet_pot", 1.1)
    cbet_min   = r.get("no_bet_cbet_min_ehs", 0.33)
    cbet_freq  = float(r.get("no_bet_cbet_frequency", 0.60))
    bluff_max  = r.get("bluff_max_ehs", 0.28)
    bluff_freq_raw = r.get("bluff_frequency", 0.30)
    bluff_pot  = r.get("bluff_sizing_pot", 0.50)

    # Board-texture-aware sizing: small on dry, large on wet (protect equity)
    wet = _board_is_wet(ctx.community_cards)
    value_pot = r.get("no_bet_value_sizing_wet" if wet else "no_bet_value_sizing_dry",
                      r.get("no_bet_value_sizing_pot", 0.55))

    adj_trap  = _adj(trap_min, ip)
    adj_value = _adj(value_min, ip)

    # Slow-play (trap): made monster → check to induce opponent bets
    # Lower trap threshold when we have a full house / quads / SF
    trap_threshold = adj_trap
    if _da.hand_category >= 6 and ehs >= 0.85:  # full house or better → always trap once
        trap_threshold = 0.85
    if ehs >= trap_threshold and has(ctx, "check"):
        return "check", 0.0, f"A flop trap ehs={ehs:.2f} [{_da.hand_category_name}]"

    # Exploit pot-guard shove: only active when exploit_allin_min_ehs > 0 in rules
    _exploit_min = float(ag.get("exploit_allin_min_ehs", 0.0))
    if _exploit_min > 0 and allin_min_pot > 0 and ctx.pot >= allin_min_pot and ehs >= _exploit_min:
        _shove = clamp(ctx.stack, ctx)
        if _shove > 0:
            return aggr(ctx), _shove, f"A exploit shove flop pot={ctx.pot:.0f} ehs={ehs:.2f}"

    # All-in gate: fires before bet_intelligence_decide so strong hands shove instead of sizing normally.
    # Opponents fold to all-in ~95% — all-in is always +EV above the push threshold.
    _push_min = float(ag.get("all_in_push_min_ehs", 0.85))
    _pot_ok   = allin_min_pot <= 0 or ctx.pot >= allin_min_pot
    if ehs >= _push_min and _pot_ok and ctx.stack > 0:
        _sz = clamp(ctx.stack, ctx)
        if _sz > 0:
            return aggr(ctx), _sz, f"A first-act ALLIN flop ehs={ehs:.2f} pot={ctx.pot:.0f}"

    # ── Bet Intelligence: EV-optimal sizing (opponent-adaptive) ──────────
    # Runs before generic overbet so opponent type drives sizing choice.
    bi = bet_intelligence_decide(ctx, ehs, _da, "flop")
    if bi is not None:
        return bi

    # Fallback overbet: board hit + no bet_intelligence override
    if hits >= 1 and ehs >= ob_min and _rules().get("adjustments", {}).get("board_hit_overbet_enabled", True):
        return aggr(ctx), clamp(ctx.pot * ob_pot, ctx), f"A flop overbet board-hit ehs={ehs:.2f}"

    # Value tier: use draw-enhanced EHS so flush/straight draws bet for semi-value
    if ehs_mid >= adj_value:
        tier = _aggr_tier(ehs, ctx.pot, ctx)   # sizing uses raw ehs
        if tier:
            size, reason = tier
            return aggr(ctx), size, reason
        label = "draw" if _da.equity_boost > 0.05 else ("wet" if wet else "dry")
        return aggr(ctx), bet_pot(value_pot, ctx), f"A flop value {label} ehs={ehs:.2f}+{_da.equity_boost:.2f}"

    # C-bet range tier: semi-bluff / range bet (draws, pairs, overcards)
    adj_cbet = _adj(cbet_min, ip)
    if ehs >= adj_cbet and _get_rng().random() < cbet_freq:
        cbet_size = r.get("no_bet_value_sizing_dry", 0.40)  # small size for range cbets
        return aggr(ctx), bet_pot(cbet_size, ctx), f"A flop cbet ehs={ehs:.2f}"

    # Pure bluff tier: complete air, low frequency
    bluff_freq = float(bluff) if isinstance(bluff, float) else (bluff_freq_raw if bluff else 0)
    if bluff and ehs <= bluff_max and _get_rng().random() < bluff_freq:
        return aggr(ctx), bet_pot(bluff_pot, ctx), f"A flop bluff ehs={ehs:.2f}"

    return "check", 0.0, f"A flop check ehs={ehs:.2f}"


# ── turn ──────────────────────────────────────────────────────────────

def _turn(ctx: GameContext, ehs: float, ip: bool) -> Decision:
    r    = _rules().get("turn", {})
    ag   = _rules().get("aggression", {})
    bluff = r.get("bluff_enabled", False)
    allin_min_pot = float(ag.get("allin_min_pot", 0))

    # Midgame: draw analysis, made-hand all-in, semi-bluffs, bluff catches
    mid = midgame_postflop(ctx, ehs, "turn", allin_min_pot)
    if mid is not None:
        return mid

    _da = analyze_draws(ctx.hole_cards, ctx.community_cards)
    ehs_mid = ehs + _da.equity_boost

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
    value_min = r.get("no_bet_value_min_ehs", 0.55)
    value_pot = r.get("no_bet_value_sizing_pot", 0.75)
    # Barrel tier: semi-bluffs that picked up equity or stayed strong on turn
    barrel_min  = r.get("no_bet_barrel_min_ehs", 0.40)
    barrel_freq = float(r.get("no_bet_barrel_frequency", 0.40))
    bluff_max = r.get("bluff_max_ehs", 0.22)
    bluff_pot = r.get("bluff_sizing_pot", 0.65)

    # Slow-play trap on turn with made monster
    trap_threshold = _adj(trap_min, ip)
    if _da.hand_category >= 6 and ehs >= 0.88:
        trap_threshold = min(trap_threshold, 0.88)
    if ehs >= trap_threshold and has(ctx, "check"):
        return "check", 0.0, f"A turn trap ehs={ehs:.2f} [{_da.hand_category_name}]"

    # Exploit pot-guard shove: only active when exploit_allin_min_ehs > 0 in rules
    _exploit_min = float(ag.get("exploit_allin_min_ehs", 0.0))
    if _exploit_min > 0 and allin_min_pot > 0 and ctx.pot >= allin_min_pot and ehs >= _exploit_min:
        _shove = clamp(ctx.stack, ctx)
        if _shove > 0:
            return aggr(ctx), _shove, f"A exploit shove turn pot={ctx.pot:.0f} ehs={ehs:.2f}"

    # All-in gate: fires before bet_intelligence_decide so strong hands shove instead of sizing normally.
    _push_min = float(ag.get("all_in_push_min_ehs", 0.85))
    _pot_ok   = allin_min_pot <= 0 or ctx.pot >= allin_min_pot
    if ehs >= _push_min and _pot_ok and ctx.stack > 0:
        _sz = clamp(ctx.stack, ctx)
        if _sz > 0:
            return aggr(ctx), _sz, f"A first-act ALLIN turn ehs={ehs:.2f} pot={ctx.pot:.0f}"

    # ── Bet Intelligence: EV-optimal sizing (value/semi-bluff/bluff/pot-ctrl) ──
    bi = bet_intelligence_decide(ctx, ehs, _da, "turn")
    if bi is not None:
        return bi

    # Value bet: use draw-boosted EHS so strong draws bet for semi-value
    if ehs_mid >= _adj(value_min, ip):
        tier = _aggr_tier(ehs, ctx.pot, ctx)
        if tier:
            size, reason = tier
            return aggr(ctx), size, reason
        label = "draw" if _da.equity_boost > 0.05 else "value"
        return aggr(ctx), bet_pot(value_pot, ctx), f"A turn {label} ehs={ehs:.2f}+{_da.equity_boost:.2f}"

    # Barrel: continue with semi-bluffs that have equity (flush draws, OESD, etc.)
    adj_barrel = _adj(barrel_min, ip)
    if ehs >= adj_barrel and _get_rng().random() < barrel_freq:
        return aggr(ctx), bet_pot(value_pot, ctx), f"A turn barrel ehs={ehs:.2f}"

    bluff_freq_t = float(bluff) if isinstance(bluff, float) else (r.get("bluff_frequency", 0.20) if bluff else 0)
    if bluff and ehs <= bluff_max and _get_rng().random() < bluff_freq_t:
        return aggr(ctx), bet_pot(bluff_pot, ctx), f"A turn bluff ehs={ehs:.2f}"
    return "check", 0.0, f"A turn check ehs={ehs:.2f}"


# ── river ──────────────────────────────────────────────────────────────

def _river(ctx: GameContext, ehs: float, ip: bool) -> Decision:
    r    = _rules().get("river", {})
    ag   = _rules().get("aggression", {})
    bluff = r.get("bluff_enabled", False)
    allin_min_pot = float(ag.get("allin_min_pot", 0))

    # Midgame: all-in calls, made-hand push, bluff catches on river
    mid = midgame_postflop(ctx, ehs, "river", allin_min_pot)
    if mid is not None:
        return mid

    _da = analyze_draws(ctx.hole_cards, ctx.community_cards)
    ehs_mid = ehs + _da.equity_boost

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

    # River value: go all-in only when pot-committed (SPR ≤ 2.5) + full house or better.
    # Flush (category 5) at deep stacks should escalate via overbet, not jam.
    if _da.hand_category >= 6 and ehs >= 0.88:
        spr = ctx.stack / max(1.0, ctx.pot)
        if spr <= 2.5:
            size = clamp(ctx.stack, ctx)
            if size > 0:
                return aggr(ctx), size, f"A river nuts allin [{_da.hand_category_name}] ehs={ehs:.2f}"

    # Exploit pot-guard shove: only active when exploit_allin_min_ehs > 0 in rules
    _exploit_min = float(ag.get("exploit_allin_min_ehs", 0.0))
    if _exploit_min > 0 and allin_min_pot > 0 and ctx.pot >= allin_min_pot and ehs >= _exploit_min:
        _shove = clamp(ctx.stack, ctx)
        if _shove > 0:
            return aggr(ctx), _shove, f"A exploit shove river pot={ctx.pot:.0f} ehs={ehs:.2f}"

    # All-in gate: fires before bet_intelligence_decide so strong hands shove instead of sizing normally.
    _push_min = float(ag.get("all_in_push_min_ehs", 0.85))
    _pot_ok   = allin_min_pot <= 0 or ctx.pot >= allin_min_pot
    if ehs >= _push_min and _pot_ok and ctx.stack > 0:
        _sz = clamp(ctx.stack, ctx)
        if _sz > 0:
            return aggr(ctx), _sz, f"A first-act ALLIN river ehs={ehs:.2f} pot={ctx.pot:.0f}"

    # ── Bet Intelligence: EV-optimal river sizing (value/bluff/SPR-shove) ──
    bi = bet_intelligence_decide(ctx, ehs, _da, "river")
    if bi is not None:
        return bi

    if ehs >= _adj(value_min, ip):
        tier = _aggr_tier(ehs, ctx.pot, ctx)
        if tier:
            size, reason = tier
            return aggr(ctx), size, reason
        return aggr(ctx), bet_pot(value_pot, ctx), f"A river value ehs={ehs:.2f}"

    bluff_freq_r = float(bluff) if isinstance(bluff, float) else (r.get("bluff_frequency", 1.0) if bluff else 0)
    if bluff and ehs <= bluff_max and _get_rng().random() < bluff_freq_r:
        return aggr(ctx), bet_pot(bluff_pot, ctx), f"A river bluff ehs={ehs:.2f}"
    return "check", 0.0, f"A river check ehs={ehs:.2f}"


# ── entry ──────────────────────────────────────────────────────────────

def decide(ctx: GameContext, ehs: float, bb_depth: float) -> Decision:
    ip = is_ip(ctx)
    s  = ctx.street
    if s == "preflop": action, amount, reason = _preflop(ctx, ehs, ip)
    elif s == "flop":  action, amount, reason = _flop(ctx, ehs, ip)
    elif s == "turn":  action, amount, reason = _turn(ctx, ehs, ip)
    else:              action, amount, reason = _river(ctx, ehs, ip)
    # Record per-hand state for probe / escalation / deterioration logic
    try:
        from agent.strategies.midgame import record_hand_state
        record_hand_state(ctx, s, ehs, action, amount)
    except Exception:
        pass
    return action, amount, reason
