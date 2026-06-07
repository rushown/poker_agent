"""agent/strategies/bet_intelligence.py — Advanced Adaptive Betting Intelligence.

Reasons across five dimensions simultaneously to compute EV-optimal bet sizing:
  1. Hand strength & potential    — EHS + draw analysis
  2. Board texture                — wet/dry/connected/monotone/paired
  3. Opponent modeling            — type-specific fold rates, calling tendencies
  4. Stack-to-pot ratio           — commitment thresholds, leveraged shoving
  5. Strategic deception          — balance value/bluff sizing, multi-street traps

Entry point: bet_intelligence_decide(ctx, ehs, da, street) → Optional[Decision]
Returns None to defer when no superior choice can be computed.

The module thinks in explicit EV:
    EV_bet   = fold_p × pot + call_p × (pot + bet) × win_rate − bet
    EV_check = pot × win_rate   (simplified; real value depends on runout)
    EV_bluff = fold_p × pot − call_p × bet

Optimal sizing maximises EV_bet across a grid of 6 candidate fractions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from engine.hand_eval import card_rank, card_suit
from agent.strategies.midgame import DrawAnalysis

# Re-exported so adaptive.py only needs one import
Decision = Tuple[str, float, str]


# ---------------------------------------------------------------------------
# Opponent type helpers
# ---------------------------------------------------------------------------

def _get_opp_type() -> str:
    try:
        from agent.arbiter import _tl
        return getattr(_tl, "opponent_type", "unknown")
    except Exception:
        return "unknown"


def _get_opp_stats():
    try:
        from agent.arbiter import _tl
        return getattr(_tl, "opponent_stats", None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Board texture
# ---------------------------------------------------------------------------

@dataclass
class BoardTexture:
    wetness: float          # 0.0 (bone dry) → 1.0 (very wet / 3-flush + 3-straight)
    connectivity: float     # 0.0 → 1.0: how connected the ranks are
    flush_possible: bool    # ≥ 3 cards of same suit on board
    monotone: bool          # all board cards same suit
    paired: bool            # board contains a pair
    high_board: bool        # highest rank ≥ J (Broadway board)
    texture_name: str       # "dry" | "semi-wet" | "wet" | "very-wet"

    # Bet-sizing guidance from texture alone
    protection_needed: bool = False   # draws are present; charge them
    value_thin_ok: bool     = False   # dry board: small sizing still extracts value


def analyze_board(community_cards: List[str]) -> BoardTexture:
    """
    Full board texture analysis.

    Example: J♠ 8♦ 3♥ → semi-wet (one straight combination, no flush)
    Example: Q♠ J♠ 2♦ → wet (flush draw possible, Broadway cards)
    Example: 7♥ 2♣ 2♦ → dry (paired, no flush, no straight potential)
    """
    if not community_cards:
        return BoardTexture(0.0, 0.0, False, False, False, False, "dry")

    suits  = [card_suit(c) for c in community_cards if len(c) >= 2]
    ranks  = sorted([card_rank(c) for c in community_cards if c])
    n      = len(community_cards)

    # ── Flush component ───────────────────────────────────────────────
    suit_counts = {s: suits.count(s) for s in set(suits)}
    max_suit    = max(suit_counts.values()) if suit_counts else 0
    flush_possible = max_suit >= 3
    monotone       = max_suit == n and n >= 3

    # ── Connectivity component ────────────────────────────────────────
    unique_ranks = sorted(set(ranks))
    # Count how many ranks fit within a 5-card straight window
    def _in_window(lo):
        return sum(1 for r in unique_ranks if lo <= r <= lo + 4)
    best_window = max((_in_window(r) for r in unique_ranks), default=0)
    connectivity = min(1.0, (best_window - 1) / 4.0) if n >= 2 else 0.0

    # ── Paired board ──────────────────────────────────────────────────
    from collections import Counter
    rank_freq = Counter(ranks)
    paired = any(v >= 2 for v in rank_freq.values())

    # ── High board ────────────────────────────────────────────────────
    high_board = max(ranks) >= 9 if ranks else False   # T or higher

    # ── Wetness score (weighted combo) ────────────────────────────────
    flush_score = (max_suit / max(n, 3))               # 0.33→1.0
    conn_score  = connectivity
    wetness     = (flush_score * 0.50 + conn_score * 0.50)
    if paired:
        wetness *= 0.70   # paired boards are drier (harder to connect)

    if wetness >= 0.70:
        name = "very-wet"
    elif wetness >= 0.45:
        name = "wet"
    elif wetness >= 0.22:
        name = "semi-wet"
    else:
        name = "dry"

    protection = wetness >= 0.45 and not paired
    thin_value = wetness < 0.30

    return BoardTexture(
        wetness=wetness,
        connectivity=connectivity,
        flush_possible=flush_possible,
        monotone=monotone,
        paired=paired,
        high_board=high_board,
        texture_name=name,
        protection_needed=protection,
        value_thin_ok=thin_value,
    )


# ---------------------------------------------------------------------------
# Betting goals
# ---------------------------------------------------------------------------

class BettingGoal(str, Enum):
    VALUE       = "value"        # extract chips from weaker called hands
    PROTECT     = "protect"      # deny equity to draws
    BLUFF       = "bluff"        # win pot without showdown value
    SEMI_BLUFF  = "semi_bluff"   # bet draw + fold equity together
    POT_CONTROL = "pot_control"  # keep pot small; marginal hand
    TRAP        = "trap"         # check strong hand; induce opponent bet


def determine_goal(
    ehs: float,
    da: DrawAnalysis,
    texture: BoardTexture,
    opp_type: str,
    spr: float,
    street: str,
    is_first_to_act: bool,
) -> BettingGoal:
    """
    Determine the primary betting goal using all five intelligence dimensions.

    Examples from spec:
      A♠K♠ on Q♠J♠2♦ (combo draw, EHS~0.55, wet board) → SEMI_BLUFF
      7♠7♦ on 7♥2♣2♦ (full house, dry board, deep) → TRAP
      9♠9♦ on 10♥8♠3♣ (overpair, semi-wet, spr>6) → POT_CONTROL
      A♣K♥ on K♦7♣2♥ (TPTK, dry board) → VALUE
      5♣6♣ on 9♥8♦2♣K♠Q♦ (nothing, river) → BLUFF
    """
    # Trap: made monster (FH/quads/SF) + dry board + deep stacks + first to act.
    # Skip vs calling_station/fish/unknown — they never bluff so slow-play loses value.
    if (
        da.hand_category >= 6
        and texture.wetness < 0.35
        and spr > 4.0
        and is_first_to_act
        and opp_type not in ("calling_station", "fish", "unknown")
    ):
        return BettingGoal.TRAP

    # Trap: flush/straight on dry board early streets.
    # Skip vs calling_station/fish/unknown for same reason.
    if (
        da.hand_category in (4, 5)
        and texture.wetness < 0.25
        and spr > 5.0
        and is_first_to_act
        and opp_type not in ("calling_station", "fish", "unknown")
    ):
        return BettingGoal.TRAP

    # Semi-bluff: strong draw (flush draw or OESD) on early streets
    if (
        (da.flush_draw or da.straight_draw == "oesd")
        and ehs >= 0.38
        and street in ("flop", "turn")
        and not da.hand_category >= 4   # not already made
    ):
        return BettingGoal.SEMI_BLUFF

    # Pot control: marginal hand on wet / semi-wet board at deep stacks
    # Skip vs callers/unknown — they never bluff, so pot control leaks EV.
    # Example: 9♠9♦ on 10♥8♠3♣ vs TAG/NIT only
    if (
        0.45 <= ehs <= 0.63
        and texture.wetness >= 0.35
        and spr > 5.0
        and is_first_to_act
        and street in ("flop", "turn")
        and opp_type not in ("unknown", "calling_station", "fish", "range_static", "fixed_timing")
    ):
        return BettingGoal.POT_CONTROL

    # Protect: vulnerable strong hand + wet board (need to charge draws)
    if (
        ehs >= 0.62
        and texture.protection_needed
        and da.hand_category <= 2   # one/two pair (can be outdrawn)
        and is_first_to_act
    ):
        return BettingGoal.PROTECT

    # Pure bluff: nothing (EHS < 0.35) on appropriate board + opponent profile
    # Must be first to act and board is credible for our range
    bluff_ok_type = opp_type in ("nit", "tag", "scared_money")
    if (
        ehs < 0.36
        and is_first_to_act
        and bluff_ok_type
        and texture.high_board   # high card board hits our rep range
        and street in ("flop", "turn", "river")
    ):
        return BettingGoal.BLUFF

    # Value: EHS >= breakeven vs callers (0.49) or any made hand
    # Arena bots fold only 5% to bets — EV_bet > EV_check at EHS >= 0.49 at any sizing
    if ehs >= 0.49 or da.hand_category >= 3:
        return BettingGoal.VALUE

    # Default: pot control (below breakeven — checking is better)
    return BettingGoal.POT_CONTROL


# ---------------------------------------------------------------------------
# Fold rate model
# ---------------------------------------------------------------------------

# (opponent_type) → (flop_fold, turn_fold, river_fold)
# Represents estimated fold-to-bet probability at a standard ~0.65x pot sizing.
_FOLD_RATES: dict = {
    "calling_station": (0.05, 0.04, 0.03),
    "fish":            (0.18, 0.15, 0.12),
    "tag":             (0.35, 0.30, 0.28),
    "nit":             (0.55, 0.50, 0.45),
    "scared_money":    (0.60, 0.55, 0.50),
    "lag":             (0.18, 0.15, 0.12),
    "maniac":          (0.10, 0.08, 0.06),
    "gto_balanced":    (0.30, 0.28, 0.25),
    "range_static":    (0.05, 0.06, 0.08),
    "fixed_timing":    (0.05, 0.06, 0.08),
    "unknown":         (0.05, 0.06, 0.08),
}

_STREET_IDX = {"flop": 0, "turn": 1, "river": 2, "preflop": 0}


def estimate_fold_rate(
    opp_type: str,
    street: str,
    bet_fraction: float,
    texture: BoardTexture,
) -> float:
    """
    Estimate probability opponent folds to a bet of size `bet_fraction × pot`.

    Adjusted for:
    - Bet sizing: larger bets → higher fold rate (logistic response curve)
    - Board texture: dry boards → higher fold rate (fewer draws to defend)
    - Live opponent stats if available (AF, fold_to_cbet)
    """
    base_rates = _FOLD_RATES.get(opp_type, _FOLD_RATES["unknown"])
    idx  = _STREET_IDX.get(street, 0)
    base = base_rates[idx]

    # Bet size scaling: reference point is 0.65x pot
    # Each 0.5x pot increase adds ~8% fold; each 0.5x decrease subtracts ~8%
    size_delta = (bet_fraction - 0.65) * 0.16
    base += size_delta

    # Dry board: harder to continue without a made hand → fold more
    texture_delta = (0.5 - texture.wetness) * 0.12   # +6% on dry, -6% on wet
    base += texture_delta

    # Live stats override
    opp = _get_opp_stats()
    if opp and opp.hands_seen >= 15:
        # fold_to_cbet (flop continuation bet)
        if opp.fold_to_cbet_flop_opps > 5 and street == "flop":
            observed = opp.fold_to_cbet_flop_count / opp.fold_to_cbet_flop_opps
            base = base * 0.4 + observed * 0.6   # blend toward observed

        # General fold tendency from AF
        total_agg  = opp.bet_count + opp.raise_count
        total_pass = max(1, opp.call_count + opp.check_count)
        af = total_agg / total_pass
        if af < 0.5:    # very passive → folds less to aggression
            base -= 0.08
        elif af > 3.0:  # very aggressive → calls to see what we have
            base -= 0.05

    return max(0.02, min(0.90, base))


# ---------------------------------------------------------------------------
# EV computation
# ---------------------------------------------------------------------------

def ev_bet(pot: float, bet: float, equity: float, fold_prob: float) -> float:
    """
    EV of betting.
      fold_prob × pot   (win pot when they fold)
    + (1-fold_prob) × (pot + bet) × equity   (win big pot × win rate when called)
    − bet × (1-fold_prob) × (1-equity)       (lose bet when called and lose)
    Simplified: EV = fold_p × pot + call_p × (pot + bet) × equity − call_p × bet × (1-equity)
    """
    call_p = 1.0 - fold_prob
    return fold_prob * pot + call_p * (pot + bet) * equity - call_p * bet * (1.0 - equity)


def ev_check(pot: float, equity: float, street: str) -> float:
    """
    EV of checking/not betting (simplified: get equity share of current pot).
    River: pot × equity (no more cards).
    Earlier streets: slightly higher due to more bluff opportunities but we
    ignore the complex tree; approximate with pot × equity.
    """
    return pot * equity


def ev_bluff(pot: float, bet: float, fold_prob: float) -> float:
    """EV of a pure bluff (equity ≈ 0 at showdown)."""
    call_p = 1.0 - fold_prob
    return fold_prob * pot - call_p * bet


def optimal_bet_fraction(
    pot: float,
    stack: float,
    equity: float,
    fold_rates_by_frac: List[Tuple[float, float]],  # [(frac, fold_prob), ...]
    goal: BettingGoal,
) -> float:
    """
    Select the bet fraction that maximises EV from a grid of candidates.
    Returns the winning fraction, or 0.0 to check.
    """
    best_ev   = ev_check(pot, equity, "any")
    best_frac = 0.0

    for frac, fold_p in fold_rates_by_frac:
        bet = min(pot * frac, stack)
        if bet <= 0:
            continue
        if goal == BettingGoal.BLUFF:
            ev = ev_bluff(pot, bet, fold_p)
        else:
            ev = ev_bet(pot, bet, equity, fold_p)

        if ev > best_ev:
            best_ev   = ev
            best_frac = frac

    return best_frac


# ---------------------------------------------------------------------------
# Sizing strategy per opponent type and goal
# ---------------------------------------------------------------------------

# Candidate bet fractions to evaluate (fraction of pot)
_CANDIDATES = [0.25, 0.33, 0.50, 0.65, 0.75, 1.00, 1.25, 1.50]


def _sizing_strategy(
    opp_type: str,
    goal: BettingGoal,
    texture: BoardTexture,
    spr: float,
    street: str,
) -> List[float]:
    """
    Return candidate bet fractions relevant to this situation.
    Narrows the grid based on opponent type and strategic context.

    Key principles:
    - Calling station + value: offer large sizes (0.75-1.5x); they call anything
    - Calling station + bluff: no candidates (return []) — never bluff calling stations
    - TAG + value: balanced 0.33-0.75x; mix value and bluff looks identical
    - Nit + value: small 0.25-0.50x to induce bluff-raises; they fold to big bets
    - Maniac + value: small 0.25-0.40x to keep bluffs in; then raise when re-raised
    - SPR < 3: shove is often the only rational play
    """
    # Cash-game no-rebuy: avoid shoving unless truly committed.
    # SPR threshold raised so we don't accidentally commit entire bankroll.
    if spr < 0.8:
        return [spr]   # unavoidably committed — shove
    if spr < 1.5:
        return [0.75, 1.00]  # large bet but not full shove unless forced

    if goal == BettingGoal.BLUFF:
        if opp_type in ("calling_station", "fish", "maniac", "lag"):
            return []  # never bluff these player types
        # Bluff sizing must match value bet sizing for balance
        if opp_type in ("nit", "scared_money"):
            return [0.65, 0.75, 1.00]   # need credible size to fold them
        return [0.50, 0.65, 0.75]        # balanced with value range

    if goal == BettingGoal.VALUE:
        if opp_type in ("calling_station", "range_static", "fixed_timing"):
            return [0.75, 1.00, 1.25, 1.50]   # confirmed callers: overbet
        if opp_type == "unknown":
            # PvP: unknown opponents may fold to overbets or re-raise them.
            # Analysis: 1.25x/1.5x with EHS 0.52-0.65 vs unknowns cost chips.
            return [0.65, 0.75, 1.00]
        if opp_type in ("fish",):
            return [0.65, 0.75, 1.00]
        if opp_type in ("nit", "scared_money"):
            return [0.25, 0.33, 0.50]          # small to induce; they fold big bets
        if opp_type == "maniac":
            return [0.25, 0.33, 0.50]          # small to keep their bluffs in
        if opp_type in ("tag", "gto_balanced"):
            return [0.33, 0.50, 0.65, 0.75]   # balanced range bets
        return [0.50, 0.65, 0.75]

    if goal == BettingGoal.PROTECT:
        # Need to charge draws; size up on wet boards
        if texture.wetness >= 0.55:
            return [0.75, 1.00, 1.25]
        return [0.65, 0.75, 1.00]

    if goal == BettingGoal.SEMI_BLUFF:
        # Must be large enough to fold out medium-strength hands,
        # but not too large (if called, we still have equity)
        return [0.50, 0.65, 0.75, 1.00]

    if goal == BettingGoal.POT_CONTROL:
        return [0.25, 0.33]   # small to gather information cheaply

    return [0.50, 0.65]   # default


# ---------------------------------------------------------------------------
# Multi-street plan (geometric pot building)
# ---------------------------------------------------------------------------

@dataclass
class StreetPlan:
    flop_frac:  float = 0.65
    turn_frac:  float = 0.75
    river_frac: float = 1.00     # fraction of remaining stack or pot
    approach:   str = "standard"


def _plan_multi_street(
    pot: float,
    stack: float,
    goal: BettingGoal,
    opp_type: str,
    spr: float,
) -> StreetPlan:
    """
    Compute a 3-street geometric betting plan so the stack is naturally
    committed by the river without an awkward shove.

    Principle: if we bet X on each street, we want to end with stack ≈ pot
    on the river for a natural shove.

    Plan types:
      "trap"        — check, small, shove
      "value_build" — 50%, 65%, shove
      "bluff_3x"    — 55%, 65%, 75% (consistent to avoid range tells)
      "pot_control" — 25%, 25%, fold-or-call
      "standard"    — 50%, 65%, 75%
    """
    if goal == BettingGoal.TRAP:
        return StreetPlan(0.0, 0.33, spr, "trap")   # check, small, shove
    if goal == BettingGoal.VALUE:
        if spr < 3:
            return StreetPlan(0.75, 1.0, spr, "fast_commit")
        if opp_type in ("calling_station", "fish"):
            return StreetPlan(0.85, 1.0, spr, "overbet_value")
        return StreetPlan(0.50, 0.70, spr, "value_build")
    if goal == BettingGoal.BLUFF:
        return StreetPlan(0.65, 0.75, 0.90, "bluff_3x")
    if goal == BettingGoal.POT_CONTROL:
        return StreetPlan(0.25, 0.33, 0.0, "pot_control")
    if goal == BettingGoal.SEMI_BLUFF:
        return StreetPlan(0.65, 0.80, spr, "semi_bluff_build")
    return StreetPlan(0.50, 0.65, 0.80, "standard")


# ---------------------------------------------------------------------------
# SPR-aware all-in leverage
# ---------------------------------------------------------------------------

def _spr_decision(
    ctx,   # GameContext
    ehs: float,
    da: DrawAnalysis,
    street: str,
    opp_type: str,
) -> Optional[Decision]:
    """
    When SPR is very low (<= 2.5), evaluate whether to commit stack.

    Spec: "With a stack of 50 BB and a pot of 20 BB, a shove is a 2.5x
    pot overbet. Only use when range is polarized and opponent is capped."
    """
    from agent.strategies.base import clamp, aggr, to_call

    spr = ctx.stack / max(1.0, ctx.pot)
    if spr > 2.5 or ctx.call_amount > 0:
        return None

    # Value: commit with strong hand at low SPR
    if ehs >= 0.65 or da.hand_category >= 3:
        size = clamp(ctx.stack, ctx)
        if size > 0:
            tag = f"[{da.hand_category_name}]" if da.hand_category >= 0 else ""
            return aggr(ctx), size, f"BET-AI: spr={spr:.1f} commit {tag} ehs={ehs:.2f}"

    # Semi-bluff shove at low SPR with big draw
    eff = ehs + da.equity_boost
    if eff >= 0.62 and (da.flush_draw or da.straight_draw == "oesd"):
        size = clamp(ctx.stack, ctx)
        if size > 0:
            return aggr(ctx), size, f"BET-AI: spr={spr:.1f} semi-shove eff={eff:.2f}"

    return None


# ---------------------------------------------------------------------------
# Value bet sizing
# ---------------------------------------------------------------------------

def _value_decision(
    ctx,
    ehs: float,
    da: DrawAnalysis,
    texture: BoardTexture,
    opp_type: str,
    spr: float,
    street: str,
    goal: BettingGoal,
) -> Optional[Decision]:
    """
    Choose optimal value / protection bet size.

    Against calling station: overbet (1.0-1.5x pot).
    Against TAG: balanced mid sizing (0.50-0.65x).
    Against nit: small (0.33x) to induce bluff-raises.
    Against maniac: small (0.33x) to keep their bluffs in; trap with big hand.
    """
    from agent.strategies.base import clamp, aggr, bet_pot

    candidates = _sizing_strategy(opp_type, goal, texture, spr, street)
    if not candidates:
        return None

    fold_rate_pairs = [
        (frac, estimate_fold_rate(opp_type, street, frac, texture))
        for frac in candidates
    ]
    equity = ehs + da.equity_boost * 0.3   # conservative equity estimate for EV calc
    best_frac = optimal_bet_fraction(ctx.pot, ctx.stack, equity, fold_rate_pairs, goal)

    if best_frac <= 0.0:
        return None   # check is best

    size = clamp(ctx.pot * best_frac, ctx)
    if size <= 0:
        return None

    plan = _plan_multi_street(ctx.pot, ctx.stack, goal, opp_type, spr)
    label = "PROTECT" if goal == BettingGoal.PROTECT else "VALUE"
    return (
        aggr(ctx),
        size,
        f"BET-AI:{label} {best_frac:.0%}pot vs {opp_type} [{texture.texture_name}] "
        f"ehs={ehs:.2f} spr={spr:.1f} plan={plan.approach}",
    )


# ---------------------------------------------------------------------------
# Bluff decision
# ---------------------------------------------------------------------------

def _bluff_decision(
    ctx,
    ehs: float,
    texture: BoardTexture,
    opp_type: str,
    spr: float,
    street: str,
) -> Optional[Decision]:
    """
    Pure bluff: no showdown value but board texture + opponent profile
    makes a profitable bluff possible.

    Required fold equity: bet / (pot + bet). Must exceed estimated fold rate.
    Sizing must match value bet sizing for balance.

    Spec: "To make a bluff profitable, I need a fold of about 60% for a pot
    bet. I'll bet 80% pot because it mimics how I'd bet top pair for value."
    """
    from agent.strategies.base import clamp, aggr, bet_pot

    candidates = _sizing_strategy(opp_type, BettingGoal.BLUFF, texture, spr, street)
    if not candidates:
        return None   # player type won't fold; no profitable bluff

    # Find smallest size that achieves positive bluff EV
    best_size = 0.0
    best_ev   = 0.0
    best_frac = 0.0

    for frac in candidates:
        bet      = min(ctx.pot * frac, ctx.stack)
        fold_p   = estimate_fold_rate(opp_type, street, frac, texture)
        req_fold = bet / max(1.0, bet + ctx.pot)   # break-even fold equity
        if fold_p < req_fold:
            continue   # bluff not profitable at this size

        ev = ev_bluff(ctx.pot, bet, fold_p)
        if ev > best_ev:
            best_ev   = ev
            best_size = bet
            best_frac = frac

    if best_size <= 0:
        return None   # no profitable bluff sizing found

    size = clamp(best_size, ctx)
    if size <= 0:
        return None

    req_fold = best_frac / (1 + best_frac)   # required fold rate
    fold_p   = estimate_fold_rate(opp_type, street, best_frac, texture)
    return (
        aggr(ctx),
        size,
        f"BET-AI:BLUFF {best_frac:.0%}pot req_fold={req_fold:.0%} "
        f"est_fold={fold_p:.0%} ev={best_ev:.1f} [{texture.texture_name}]",
    )


# ---------------------------------------------------------------------------
# Semi-bluff decision
# ---------------------------------------------------------------------------

def _semi_bluff_decision(
    ctx,
    ehs: float,
    da: DrawAnalysis,
    texture: BoardTexture,
    opp_type: str,
    spr: float,
    street: str,
) -> Optional[Decision]:
    """
    Semi-bluff: bet draw (flush/OESD) + fold equity.

    Spec: "9 outs to the nuts, 36% equity to improve. Bet 66% pot. This
    denies proper odds to a weaker draw, can fold out middle pairs, and
    even if called, I have ~36% equity to improve by the river."
    """
    from agent.strategies.base import clamp, aggr

    eff_equity = ehs + da.equity_boost    # full draw equity
    candidates  = _sizing_strategy(opp_type, BettingGoal.SEMI_BLUFF, texture, spr, street)
    if not candidates:
        return None

    # Semi-bluff EV uses full equity (we hit and win)
    fold_rate_pairs = [
        (frac, estimate_fold_rate(opp_type, street, frac, texture))
        for frac in candidates
    ]
    best_frac = optimal_bet_fraction(
        ctx.pot, ctx.stack, eff_equity, fold_rate_pairs, BettingGoal.SEMI_BLUFF
    )
    if best_frac <= 0.0:
        return None

    size = clamp(ctx.pot * best_frac, ctx)
    if size <= 0:
        return None

    draw_tag = (
        f"flush {da.flush_outs}outs" if da.flush_draw
        else f"OESD {da.straight_outs}outs"
    )
    return (
        aggr(ctx),
        size,
        f"BET-AI:SEMI {best_frac:.0%}pot [{draw_tag}] eff={eff_equity:.2f} "
        f"vs {opp_type} [{texture.texture_name}]",
    )


# ---------------------------------------------------------------------------
# Pot-control decision
# ---------------------------------------------------------------------------

def _pot_control_decision(
    ctx,
    ehs: float,
    da: DrawAnalysis,
    texture: BoardTexture,
    spr: float,
    street: str,
) -> Optional[Decision]:
    """
    Pot control: marginal/vulnerable hand → check behind or bet very small.

    Spec: "9♠9♦ on 10♥8♠3♣. If I bet big and get called, the turn could
    bring scary cards. I'll check behind for pot control."
    """
    from agent.strategies.base import clamp, aggr

    # On very wet boards or multi-street decision → prefer checking
    if texture.wetness >= 0.50 and spr > 5.0:
        if ctx.call_amount <= 0:
            return "check", 0.0, (
                f"BET-AI:POT-CTRL check [{texture.texture_name}] "
                f"spr={spr:.1f} ehs={ehs:.2f}"
            )

    # Semi-wet / moderate: bet small for thin info
    if texture.wetness < 0.50 and spr > 3.0:
        size = clamp(ctx.pot * 0.30, ctx)
        if size > 0:
            return aggr(ctx), size, (
                f"BET-AI:POT-CTRL small 30%pot spr={spr:.1f} ehs={ehs:.2f}"
            )

    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def bet_intelligence_decide(
    ctx,          # GameContext
    ehs: float,
    da: DrawAnalysis,
    street: str,
) -> Optional[Decision]:
    """
    Compute EV-optimal betting decision using all five intelligence dimensions,
    plus three behavioural systems for unknown rotating opponents:
      - Probe bet (0.28x pot test when medium equity)
      - Bet escalation (1.3x when opponent called last street)
      - EHS deterioration abort (stop committing when equity collapsed)

    Returns Decision override, or None to defer to the adaptive rule engine.
    Called only when ctx.call_amount == 0 (first-to-act / our betting turn).
    """
    from agent.strategies.base import has
    from agent.strategies.midgame import (
        ehs_deterioration_check,
        probe_bet_decide,
        escalate_bet_decide,
    )

    # Only runs when we're first to act (no bet facing us)
    if ctx.call_amount > 0:
        return None
    if not has(ctx, "check") and not has(ctx, "bet") and not has(ctx, "raise"):
        return None

    opp_type = _get_opp_type()
    spr      = ctx.stack / max(1.0, ctx.pot)
    texture  = analyze_board(ctx.community_cards)

    # ── 0a. EHS deterioration abort (highest priority) ───────────────────
    # If our equity collapsed since flop, stop betting; check or fold cheaply.
    det = ehs_deterioration_check(ctx, ehs, street)
    if det is not None:
        return det

    # ── 0b. Graduated nut path / bet escalation ───────────────────────────
    # If we bet a previous street and opponent called, increase this street's
    # bet geometrically (0.5x flop → 0.75x turn → all-in river for nuts).
    esc = escalate_bet_decide(ctx, ehs, street)
    if esc is not None:
        return esc

    # ── 1. SPR leverage check: very low SPR → commit decision ────────────
    spr_dec = _spr_decision(ctx, ehs, da, street, opp_type)
    if spr_dec is not None:
        return spr_dec

    # ── 2. Determine betting goal ─────────────────────────────────────────
    goal = determine_goal(
        ehs, da, texture, opp_type, spr, street,
        is_first_to_act=(ctx.call_amount <= 0),
    )

    # ── 3. Trap: check and let adaptive logic check/trap ──────────────────
    if goal == BettingGoal.TRAP:
        return "check", 0.0, (
            f"BET-AI:TRAP {da.hand_category_name} [{texture.texture_name}] "
            f"spr={spr:.1f} → slow-play"
        )

    # ── 4. Pot control ────────────────────────────────────────────────────
    if goal == BettingGoal.POT_CONTROL:
        # Try probe bet first (cheaper than a 0.33x pot-control bet)
        probe = probe_bet_decide(ctx, ehs, street)
        if probe is not None:
            return probe
        dec = _pot_control_decision(ctx, ehs, da, texture, spr, street)
        if dec is not None:
            return dec
        return None

    # ── 5. Bluff ──────────────────────────────────────────────────────────
    if goal == BettingGoal.BLUFF:
        return _bluff_decision(ctx, ehs, texture, opp_type, spr, street)

    # ── 6. Semi-bluff ─────────────────────────────────────────────────────
    if goal == BettingGoal.SEMI_BLUFF:
        dec = _semi_bluff_decision(ctx, ehs, da, texture, opp_type, spr, street)
        if dec is not None:
            return dec
        # Fall through to value if semi-bluff sizing fails

    # ── 7. Value / Protect ───────────────────────────────────────────────
    if goal in (BettingGoal.VALUE, BettingGoal.PROTECT):
        # For medium-strength value (0.52–0.65 EHS), try probe first
        # on dry boards — tests if opponent folds cheaply
        if 0.52 <= ehs <= 0.65 and texture.texture_name == "dry":
            probe = probe_bet_decide(ctx, ehs, street)
            if probe is not None:
                return probe
        dec = _value_decision(ctx, ehs, da, texture, opp_type, spr, street, goal)
        if dec is not None:
            return dec

    return None   # defer to adaptive rule engine
