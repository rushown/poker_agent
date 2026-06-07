"""agent/strategies/midgame.py — adaptive in-game reasoning module.

Augments the rule-based adaptive.py with draw-aware, opponent-read-enhanced
decisions for all streets. Called from adaptive.py; returns a Decision override
or None (meaning: defer to the existing rule logic).

Key capabilities:
  1. Flush / straight draw detection + outs-based equity boost
  2. Made hand detection (flush, straight, full house, quads, SF)
  3. Opponent bluff detection via sizing tells + tracked stats
  4. All-in confidence: push when made, call only when equity justifies it
  5. Preflop minimum-call assessment for cheap speculative spots
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from engine.hand_eval import best_hand, card_rank, card_suit

# Per-hand semi-bluff raise counter and EHS history.
# Using module-level dicts (not threading.local) so state persists correctly
# across thread-pool workers used by run_in_executor.
_semi_bluff_counts: dict = {}   # (table_id, hand_number) → int
_hand_histories: dict = {}       # (table_id, hand_number) → dict
_state_lock = threading.Lock()


def _hkey(ctx) -> tuple:
    return (getattr(ctx, "table_id", ""), getattr(ctx, "hand_number", None))


def _get_history(ctx) -> dict:
    k = _hkey(ctx)
    with _state_lock:
        if k not in _hand_histories:
            _hand_histories[k] = {
                "flop_ehs":          None,
                "streets_bet":       0,
                "last_bet_fraction": 0.0,
                "last_bet_pot":      0.0,
            }
            if len(_hand_histories) > 40:
                for old in sorted(_hand_histories.keys())[:-40]:
                    del _hand_histories[old]
        return _hand_histories[k]


def record_hand_state(ctx, street: str, ehs: float, action: str, amount: float) -> None:
    """Update within-hand state after each decision. Called from adaptive.py."""
    hist = _get_history(ctx)
    pot  = max(1.0, ctx.pot)
    if street == "flop" and hist["flop_ehs"] is None:
        hist["flop_ehs"] = ehs
    if action in ("bet", "raise") and amount > 0:
        hist["streets_bet"]       += 1
        hist["last_bet_fraction"]  = amount / pot
        hist["last_bet_pot"]       = pot

Decision = Tuple[str, float, str]

_CATEGORY_NAMES = [
    "high_card", "pair", "two_pair", "trips",
    "straight", "flush", "full_house", "quads", "straight_flush",
]


def _get_opp_stats():
    try:
        from agent.arbiter import _tl
        return getattr(_tl, "opponent_stats", None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Draw analysis
# ---------------------------------------------------------------------------

@dataclass
class DrawAnalysis:
    flush_draw: bool = False      # 4 cards to a flush (need 1 more)
    flush_backdoor: bool = False  # 3 cards to a flush (need 2 more)
    flush_complete: bool = False  # made flush (5+ of same suit, holding ≥1)
    flush_outs: int = 0
    flush_suit: str = ""

    straight_draw: str = ""       # "oesd" | "double_gutshot" | "gutshot" | ""
    straight_complete: bool = False
    straight_outs: int = 0

    hand_category: int = -1       # 0-8 from hand_eval score >> 20; -1 = not enough cards
    hand_category_name: str = "none"

    equity_boost: float = 0.0    # add to EHS for raise/bet decisions
    cards_to_come: int = 0       # board cards still to be dealt


def analyze_draws(hole_cards: List[str], community_cards: List[str]) -> DrawAnalysis:
    """
    Analyze hole + community cards for draws and made hands.

    Core insight: if the agent holds 2h 7h and board shows Kh Jh 4c, that is
    4 cards to a flush → 9 outs → equity_boost ≈ +0.18 (flop: 9 * 2 cards * 2%).
    """
    da = DrawAnalysis()
    all_cards = hole_cards + community_cards
    if not hole_cards or len(all_cards) < 2:
        return da

    da.cards_to_come = max(0, 5 - len(community_cards))

    # ── Made hand (need ≥ 5 cards total) ───────────────────────────────────
    if len(all_cards) >= 5:
        try:
            score = best_hand(hole_cards, community_cards)
            cat = min(score >> 20, 8)
            da.hand_category = cat
            da.hand_category_name = _CATEGORY_NAMES[cat]
        except Exception:
            da.hand_category = -1

    # ── Flush analysis ─────────────────────────────────────────────────────
    hole_suits = [card_suit(c) for c in hole_cards if len(c) >= 2]
    all_suits  = [card_suit(c) for c in all_cards  if len(c) >= 2]

    for suit in set(all_suits):
        if hole_suits.count(suit) == 0:
            continue                          # agent doesn't hold this suit
        total = all_suits.count(suit)
        if total >= 5:
            da.flush_complete = True
            da.flush_suit = suit
            da.flush_outs = 0
        elif total == 4 and not da.flush_complete:
            da.flush_draw = True
            da.flush_suit = suit
            da.flush_outs = 13 - total        # 9 outs when holding 2 of suit
        elif total == 3 and len(community_cards) <= 3 and not da.flush_draw:
            da.flush_backdoor = True
            da.flush_outs = max(da.flush_outs, 10)

    # Sync flush_complete with hand category
    if da.hand_category in (5, 8):
        da.flush_complete = True

    # ── Straight analysis ──────────────────────────────────────────────────
    if da.hand_category in (4, 8):
        da.straight_complete = True

    if not da.straight_complete:
        all_rank_ints = sorted(set(card_rank(c) for c in all_cards if c))
        run = _max_consecutive(all_rank_ints)
        if run >= 4:
            da.straight_draw = "oesd"
            da.straight_outs = 8
        elif run >= 3 and _count_gutshots(all_rank_ints) >= 2:
            da.straight_draw = "double_gutshot"
            da.straight_outs = 8
        elif run >= 3:
            da.straight_draw = "gutshot"
            da.straight_outs = 4

    # ── Equity boost: Rule-of-2 / Rule-of-4 ──────────────────────────────
    # Approximation: outs * 4% if 2 cards to come, outs * 2% if 1 card to come.
    # Caps are street-specific: turn (1 card) gets half the flop cap.
    # River (0 cards): missed draws get 0 — no improvement possible.
    mult = 4 if da.cards_to_come >= 2 else 2

    if da.flush_complete or da.straight_complete or da.hand_category >= 5:
        da.equity_boost = 0.18         # made monster: max confidence signal
    elif da.flush_draw and da.straight_draw in ("oesd", "double_gutshot"):
        if da.cards_to_come > 0:
            cap = 0.22 if da.cards_to_come >= 2 else 0.11
            da.equity_boost = min(cap, 15 * mult / 100)
    elif da.flush_draw:
        if da.cards_to_come > 0:
            cap = 0.15 if da.cards_to_come >= 2 else 0.09
            da.equity_boost = min(cap, da.flush_outs * mult / 100)
    elif da.straight_draw == "oesd":
        if da.cards_to_come > 0:
            cap = 0.12 if da.cards_to_come >= 2 else 0.06
            da.equity_boost = min(cap, da.straight_outs * mult / 100)
    elif da.straight_draw in ("gutshot", "double_gutshot"):
        if da.cards_to_come > 0:
            cap = 0.08 if da.cards_to_come >= 2 else 0.04
            da.equity_boost = min(cap, da.straight_outs * mult / 100)
    elif da.flush_backdoor:
        da.equity_boost = 0.03
    elif da.hand_category == 3:   # trips
        da.equity_boost = 0.05
    elif da.hand_category == 2:   # two pair
        da.equity_boost = 0.03

    return da


def _max_consecutive(ranks: List[int]) -> int:
    """Return the length of the longest consecutive run in a sorted unique list."""
    if not ranks:
        return 0
    unique = sorted(set(ranks))
    best = cur = 1
    for i in range(1, len(unique)):
        cur = cur + 1 if unique[i] - unique[i - 1] == 1 else 1
        best = max(best, cur)
    return best


def _count_gutshots(ranks: List[int]) -> int:
    """Count 5-card windows that contain ≥4 of our ranks (each = a gutshot)."""
    unique = sorted(set(ranks))
    count = 0
    for low in unique:
        if sum(1 for r in unique if low <= r <= low + 4) >= 4:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Bluff detection
# ---------------------------------------------------------------------------

@dataclass
class BluffRead:
    bluff_probability: float = 0.0
    is_likely_bluff: bool = False
    is_likely_nuts: bool = False
    call_as_bluff_catch: bool = False
    reason: str = ""


def detect_bluff(
    call_amount: float,
    pot: float,
    ehs: float,
    is_river: bool = False,
) -> BluffRead:
    """
    Estimate probability that opponent's bet is a bluff.

    Uses bet-to-pot sizing (polarization tells) plus opponent stats
    stored in arbiter thread-local (hands_seen, AF, WTSD, river bluff history).
    """
    br = BluffRead()
    if call_amount <= 0 or pot <= 0:
        return br

    btp = call_amount / max(1.0, pot)          # bet-to-pot ratio

    # Sizing heuristic: overbets are polarized; small bets lean value
    if btp >= 1.5:
        base = 0.42
        br.reason = f"overbet {btp:.1f}x; "
    elif btp >= 0.9:
        base = 0.32
        br.reason = f"bigbet {btp:.1f}x; "
    elif btp >= 0.5:
        base = 0.27
        br.reason = f"halfpot {btp:.1f}x; "
    else:
        base = 0.22
        br.reason = f"smallbet {btp:.1f}x; "

    bluff_prob = base
    opp = _get_opp_stats()

    if opp and opp.hands_seen >= 10:
        # Aggression factor: (bets + raises) / calls
        total_agg  = opp.bet_count + opp.raise_count
        total_pass = max(1, opp.call_count + opp.check_count)
        af = total_agg / total_pass
        if af > 2.5:
            bluff_prob += 0.10
            br.reason += f"AF={af:.1f}↑; "
        elif af < 0.7:
            bluff_prob -= 0.08
            br.reason += f"AF={af:.1f}↓; "

        # River bluff-catch history
        if opp.river_bet_count > 3:
            rbluff = opp.river_bluff_caught / opp.river_bet_count
            bluff_prob += (rbluff - 0.20) * 0.6
            if rbluff > 0.30:
                br.reason += f"rbluff={rbluff:.0%}; "

        # WTSD: low = doesn't go to showdown often → may be bluffing
        if opp.wtsd_opps > 5:
            wtsd = opp.wtsd_count / opp.wtsd_opps
            if wtsd < 0.22:
                bluff_prob += 0.06
            elif wtsd > 0.55:
                bluff_prob -= 0.06

    bluff_prob = max(0.05, min(0.90, bluff_prob))
    br.bluff_probability = bluff_prob
    br.is_likely_bluff   = bluff_prob >= 0.42
    br.is_likely_nuts    = bluff_prob <= 0.18

    # Bluff-catch EV: bluff_prob × pot − (1 − bluff_prob) × call > 0?
    required_equity  = call_amount / max(1.0, call_amount + pot)
    bluff_catch_ev   = bluff_prob * pot - (1 - bluff_prob) * call_amount
    br.call_as_bluff_catch = bluff_catch_ev > 0 and ehs > required_equity * 0.75

    return br


# ---------------------------------------------------------------------------
# All-in confidence
# ---------------------------------------------------------------------------

def should_go_allin(
    ehs: float,
    da: DrawAnalysis,
    street: str,
    pot: float,
    allin_min_pot: float = 0.0,
) -> Tuple[bool, str]:
    """
    Should agent push all-in?  Made strong hands lower the EHS threshold.
    Returns (go_allin, reason).
    """
    thresholds = {"preflop": 0.88, "flop": 0.85, "turn": 0.84, "river": 0.82}
    thresh = thresholds.get(street, 0.90)

    if da.hand_category >= 6:      # full house / quads / straight flush
        thresh -= 0.08
    elif da.hand_category == 5:    # flush
        thresh -= 0.05
    elif da.hand_category == 4:    # straight
        thresh -= 0.03
    elif da.flush_complete or da.straight_complete:
        thresh -= 0.03

    if allin_min_pot > 0 and pot < allin_min_pot:
        return False, f"mid:allin blocked pot={pot:.0f}<min={allin_min_pot:.0f}"

    eff = ehs + da.equity_boost
    if eff >= thresh:
        if da.hand_category >= 4:
            tag = f"[{da.hand_category_name}]"
        elif da.equity_boost > 0.05:
            tag = f"[draw+{da.equity_boost:.2f}]"
        else:
            tag = ""
        return True, f"mid:allin eff={eff:.2f}>={thresh:.2f} {tag}".strip()

    return False, f"mid:allin skip eff={eff:.2f}<{thresh:.2f}"


def should_call_allin(
    ehs: float,
    da: DrawAnalysis,
    street: str,
    pot: float,
    call_amount: float,
) -> Tuple[bool, str]:
    """
    Should agent call opponent's all-in?
    Requires effective equity to exceed pot-odds with a confidence margin.
    """
    required = call_amount / max(1.0, call_amount + pot)
    eff      = ehs + da.equity_boost

    # Bots fold to all-in 95% — when they shove, they have real equity.
    # Require a clear edge above pot-odds; 1.08x margin is sufficient.
    min_ehs = max(0.58, required * 1.08)

    if da.hand_category >= 6:      # full house / quads / SF — near-certain winner
        min_ehs = max(0.50, required * 1.02)
    elif da.hand_category == 5:    # flush
        min_ehs = max(0.54, required * 1.04)
    elif da.hand_category == 4:    # straight
        min_ehs = max(0.56, required * 1.05)
    elif da.flush_draw and da.flush_outs >= 9 and da.cards_to_come >= 2:
        min_ehs = max(0.56, required * 1.07)

    if eff >= min_ehs:
        tag = f"[{da.hand_category_name}]" if da.hand_category >= 0 else ""
        return True, f"mid:call_allin eff={eff:.2f}>={min_ehs:.2f} {tag}".strip()

    return False, f"mid:fold_allin eff={eff:.2f}<{min_ehs:.2f} req={required:.2f}"


# ---------------------------------------------------------------------------
# Preflop minimum-call assessment
# ---------------------------------------------------------------------------

def assess_preflop_min_call(
    ehs: float,
    call_amount: float,
    bb_size: float,
    hole_cards: List[str],
    is_facing_raise: bool,
) -> Optional[Decision]:
    """
    Minimum-call / fold assessment for preflop cheap spots.

    Philosophy: with very cheap implied odds, speculative hands (suited
    connectors, low pairs, broadway) are worth calling to see the flop
    even when EHS is below the normal open threshold.

    Returns a Decision only for EHS 0.28–0.44 cheap-call spots.
    Above 0.44, the existing preflop logic is preferred.
    Returns None to defer to existing adaptive logic.
    """
    if not hole_cards or len(hole_cards) < 2:
        return None

    suits = [c[1].lower() for c in hole_cards if len(c) >= 2]
    ranks = [card_rank(c) for c in hole_cards]
    suited      = len(suits) == 2 and suits[0] == suits[1]
    gap         = abs(ranks[0] - ranks[1]) if len(ranks) >= 2 else 99
    is_pair     = gap == 0
    is_connector = 1 <= gap <= 2
    high_rank   = max(ranks) if ranks else 0

    spec_bonus = 0.0
    if suited:        spec_bonus += 0.03
    if is_connector:  spec_bonus += 0.02
    if high_rank >= 10: spec_bonus += 0.02  # broadway card (T/J/Q/K/A)
    if is_pair:       spec_bonus += 0.03

    eff = ehs + spec_bonus

    # Premium / medium: let existing preflop logic handle it
    if eff >= 0.46:
        return None

    cost_bb = call_amount / max(1.0, bb_size)

    if is_facing_raise:
        # Facing aggression: only call with respectable equity
        if eff >= 0.44:
            return None  # borderline — defer
        return "fold", 0.0, f"mid:pf fold vs raise eff={eff:.2f}"

    # Not facing a raise: cheap speculative call
    if cost_bb <= 1.0 and eff >= 0.30:
        return "call", call_amount, f"mid:pf min-call cheap eff={eff:.2f}"
    if cost_bb <= 2.5 and eff >= 0.37:
        return "call", call_amount, f"mid:pf min-call eff={eff:.2f} {cost_bb:.1f}BB"
    if eff < 0.32:
        return "fold", 0.0, f"mid:pf fold weak eff={eff:.2f}"

    return None  # borderline — defer to existing logic


# ---------------------------------------------------------------------------
# Main postflop override
# ---------------------------------------------------------------------------

def _semi_bluff_count(ctx) -> int:
    """Return how many semi-bluff raises we have already made this hand."""
    key = (getattr(ctx, "table_id", ""), getattr(ctx, "hand_number", None))
    with _state_lock:
        return _semi_bluff_counts.get(key, 0)


def _semi_bluff_increment(ctx) -> None:
    """Record that we made a semi-bluff raise this hand."""
    key = (getattr(ctx, "table_id", ""), getattr(ctx, "hand_number", None))
    with _state_lock:
        _semi_bluff_counts[key] = _semi_bluff_counts.get(key, 0) + 1
        if len(_semi_bluff_counts) > 20:
            for old in sorted(_semi_bluff_counts.keys())[:-20]:
                del _semi_bluff_counts[old]


def _semi_bluff_max(street: str) -> int:
    """Max semi-bluff raises allowed per hand from strategy rules."""
    try:
        from agent.strategies.adaptive import _rules
        r = _rules()
        return int(r.get(street, {}).get("semi_bluff_max_raises_per_hand",
               r.get("aggression", {}).get("semi_bluff_max_raises_per_hand", 1)))
    except Exception:
        return 1


def midgame_postflop(
    ctx,               # GameContext (avoid circular type annotation)
    ehs: float,
    street: str,
    allin_min_pot: float = 0.0,
) -> Optional[Decision]:
    """
    Compute draw analysis and return a high-confidence Decision override, or
    None to let the existing adaptive rule logic proceed.

    Override scenarios (in priority order):
      1. Facing near-all-in: call or fold based on confident equity check
      2. Made strong hand (≥straight) + no bet facing: push all-in
      3. Flush draw facing a bet: semi-bluff raise (once per hand max — prevents spiral)
      4. Medium equity + opponent likely bluffing: bluff-catch call
    """
    from agent.strategies.base import clamp, aggr, to_call

    da = analyze_draws(ctx.hole_cards, ctx.community_cards)

    # ── 1. Facing a near-all-in call ────────────────────────────────────
    # Primary guard: ≥40% of remaining stack (was 70% — too loose; 99/265-chip
    # calls at EHS 0.47 were slipping through against value-only bots).
    if ctx.call_amount > 0 and ctx.call_amount >= ctx.stack * 0.40:
        ok, reason = should_call_allin(ehs, da, street, ctx.pot, ctx.call_amount)
        if ok:
            return "call", to_call(ctx), reason
        return "fold", 0.0, reason

    # Secondary guard: medium-large bet (call > 60% of pot) even at shallow stack
    # commitment. Lowered from 80% — analysis showed calls with EHS 0.55-0.72
    # vs 60-80% pot bets cost 1900+ chips (opponents bet non-bluff ranges).
    if ctx.call_amount > 0 and ctx.call_amount >= ctx.pot * 0.60:
        ok, reason = should_call_allin(ehs, da, street, ctx.pot, ctx.call_amount)
        if ok:
            return "call", to_call(ctx), reason
        return "fold", 0.0, reason

    # ── 2. Made strong hand → all-in or defer to trap logic ─────────────
    if ctx.call_amount <= 0 and da.hand_category >= 4 and ehs >= 0.72:
        spr = ctx.stack / max(1.0, ctx.pot)
        # Full house / quads on flop or turn at deep stacks → slow-play (trap):
        # check to induce opponent bets — let adaptive trap logic handle the sizing.
        # Straight flush (cat 8): always all-in immediately — never slow-play the nuts.
        # On the river or at shallow stacks (SPR ≤ 4), push all-in for max value.
        opp_type = getattr(ctx, "opponent_type", "unknown")
        # Against calling stations / unknown bots, skip slow-play — push for max value.
        trap_ok = opp_type not in ("calling_station", "fish", "unknown")
        if da.hand_category in (6, 7) and street in ("flop", "turn") and spr > 4.0 and trap_ok:
            return None  # defer → adaptive trap → check
        ok, reason = should_go_allin(ehs, da, street, ctx.pot, allin_min_pot)
        if ok:
            size = clamp(ctx.stack, ctx)
            if size > 0:
                return aggr(ctx), size, reason

    # ── 3. Flush draw facing a bet → semi-bluff raise ───────────────────
    # Critical guard: only raise ONCE per hand with a draw.
    # Without this, facing a re-raise re-triggers the semi-bluff → geometric spiral
    # (24→72→192→504) that commits the entire stack at ~50% equity. (Hand 6565 fix.)
    already_raised = _semi_bluff_count(ctx)
    max_raises = _semi_bluff_max(street)

    if (
        ctx.call_amount > 0
        and da.flush_draw
        and da.flush_outs >= 9
        and already_raised < max_raises          # ← anti-spiral guard
        and not ctx.call_amount >= ctx.stack * 0.40
    ):
        eff = ehs + da.equity_boost
        semi_thresh = 0.57
        if eff >= semi_thresh:
            size = clamp(ctx.call_amount * 2.8, ctx)
            if size > 0:
                _semi_bluff_increment(ctx)
                return aggr(ctx), size, f"mid:semi-bluff flush draw eff={eff:.2f}"

    # ── 4. OESD facing a bet → semi-bluff raise (less frequent) ─────────
    if (
        ctx.call_amount > 0
        and da.straight_draw == "oesd"
        and not da.flush_draw
        and already_raised < max_raises          # ← anti-spiral guard
        and not ctx.call_amount >= ctx.stack * 0.40
    ):
        eff = ehs + da.equity_boost
        if eff >= 0.60:
            size = clamp(ctx.call_amount * 2.5, ctx)
            if size > 0:
                _semi_bluff_increment(ctx)
                return aggr(ctx), size, f"mid:semi-bluff OESD eff={eff:.2f}"

    # ── 5. Small-bet pot-odds call ────────────────────────────────────────
    # Arena bots never bluff → bluff-catching is always -EV.
    # Only call small bets (≤30% pot) when pure pot-odds are positive.
    if ctx.call_amount > 0 and ctx.call_amount <= ctx.pot * 0.30:
        required = ctx.call_amount / max(1.0, ctx.pot + ctx.call_amount)
        if ehs >= required * 1.05:
            return "call", to_call(ctx), f"mid:pot-odds small-bet ehs={ehs:.2f} req={required:.2f}"

    return None


# ---------------------------------------------------------------------------
# Behavioral systems — optimal static logic for unknown rotating opponents
# ---------------------------------------------------------------------------

def ehs_deterioration_check(ctx, ehs: float, street: str) -> Optional[Decision]:
    """
    System 1 — EHS deterioration abort.

    If our EHS was strong on the flop (≥0.62) but has dropped sharply by the
    turn/river (now <0.48), the board has connected with the opponent or we
    were dominated. Stop betting and check/fold.

    From log analysis: hands 6283 and 6451 showed EHS declines of 0.20+ where
    Plutus kept betting into deteriorating equity, leaking chips.
    """
    hist = _get_history(ctx)
    flop_ehs = hist.get("flop_ehs")
    if flop_ehs is None or street not in ("turn", "river"):
        return None

    drop = flop_ehs - ehs
    # Board significantly hit opponent; equity collapsed.
    # Lowered from (0.18, 0.48) — analysis showed EHS drops of 0.12-0.17
    # (hand 21410: 0.642→0.516, hand 20516: 0.604→partial) went undetected
    # and agent kept betting into trapping opponents losing 3400+ chips.
    if drop >= 0.14 and ehs < 0.52:
        if ctx.call_amount > 0:
            # Facing a bet with deteriorated equity → fold
            if ehs < 0.48:
                return "fold", 0.0, f"mid:deterioration fold flop={flop_ehs:.2f}→{ehs:.2f} drop={drop:.2f}"
            # Marginal: pot odds might justify a call but don't re-raise
            return None
        # Not facing a bet → check (don't bleed chips)
        from agent.strategies.base import has
        if has(ctx, "check"):
            return "check", 0.0, f"mid:deterioration check flop={flop_ehs:.2f}→{ehs:.2f} drop={drop:.2f}"

    return None


def probe_bet_decide(ctx, ehs: float, street: str) -> Optional[Decision]:
    """
    System 2 — Probe bet (test bet).

    When EHS is 0.48–0.64 and we're first to act (no facing bet), use a small
    probe bet (0.28x pot) instead of checking or full value betting.

    Purpose: cheaply collect dead money from folders and gather info.
    Against any unknown opponent — some fold to small bets, giving us free chips.
    If they raise → fold (we've spent 0.28x pot, not 0.75x).
    If they call → we now know they have something; adjust future streets.

    Only fires on flop/turn. Never probes with <0.48 EHS (too weak to probe).
    Never probes after already betting (escalation logic takes over).
    """
    from agent.strategies.base import clamp, aggr, has

    if ctx.call_amount > 0:
        return None   # facing a bet → this is a call/fold decision, not a probe
    if street not in ("flop", "turn"):
        return None
    if not has(ctx, "bet") and not has(ctx, "raise"):
        return None

    hist = _get_history(ctx)
    if hist["streets_bet"] > 0:
        return None   # already bet a previous street; use escalation logic instead

    # Only probe with medium equity — strong hands use full value bet path
    if not (0.48 <= ehs <= 0.64):
        return None

    # Don't probe on very wet boards (opponent likely has draws; they'll call)
    # Board wetness proxy: >2 community cards within 3 ranks = connected
    board = ctx.community_cards
    if board:
        ranks = sorted(set(r for c in board if c for r in [__import__('engine.hand_eval', fromlist=['card_rank']).card_rank(c)]))
        if len(ranks) >= 3 and (ranks[-1] - ranks[0]) <= 4:
            return None   # very connected board → skip probe

    probe_size = clamp(ctx.pot * 0.28, ctx)
    if probe_size <= ctx.bb_size:
        return None   # probe would be too tiny to matter

    return aggr(ctx), probe_size, f"mid:probe 28%pot ehs={ehs:.2f} street={street}"


def escalate_bet_decide(ctx, ehs: float, street: str) -> Optional[Decision]:
    """
    System 3 — Bet escalation when called last street.

    If we bet a previous street and opponent called (pot is growing, we're now
    first to act again on a new street), increase our bet size by ~1.3x.

    Logic:
    - Opponent called → they have equity; they'll pay more next street too
    - Escalating extracts maximum value from calling stations
    - Stops if EHS has deteriorated (see ehs_deterioration_check above)

    Graduated nut path (EHS ≥ 0.82):
      flop 0.50x → turn 0.75x → river all-in
    Value escalation (EHS ≥ 0.62):
      flop 0.50x → turn 0.65x → river 0.85x
    """
    from agent.strategies.base import clamp, aggr, has

    if ctx.call_amount > 0:
        return None   # facing a bet; escalation doesn't apply
    if not has(ctx, "bet") and not has(ctx, "raise"):
        return None
    if street not in ("turn", "river"):
        return None

    hist = _get_history(ctx)
    if hist["streets_bet"] == 0:
        return None   # never bet before; no escalation base

    # Infer opponent called last bet: pot grew after our bet without us acting
    # (pot > last_bet_pot + our_bet = opponent added chips = called)
    last_pot  = hist["last_bet_pot"]
    last_frac = hist["last_bet_fraction"]
    if last_pot <= 0 or last_frac <= 0:
        return None

    expected_pot_if_no_call = last_pot + last_pot * last_frac
    # If current pot is roughly equal to or larger than expected → opponent called
    opp_likely_called = ctx.pot >= expected_pot_if_no_call * 0.85

    if not opp_likely_called:
        return None

    # Nut path: EHS 0.82+ → escalate aggressively toward all-in
    if ehs >= 0.82:
        if street == "turn":
            target = 0.75
        else:  # river
            target = ctx.stack  # all-in on river with near-nut hand
            size = clamp(target, ctx)
            if size > 0:
                return aggr(ctx), size, f"mid:nut-path river ALLIN ehs={ehs:.2f}"
            return None
        size = clamp(ctx.pot * target, ctx)
        if size > 0:
            return aggr(ctx), size, f"mid:nut-path {target:.0%}pot ehs={ehs:.2f} streets_bet={hist['streets_bet']}"
        return None

    # Value escalation: EHS 0.62–0.82
    if ehs >= 0.62:
        # Increase by 1.3x from last bet fraction, capped at 1.0x pot
        new_frac = min(last_frac * 1.3, 1.0)
        # River: don't go all-in here (reserved for nut path above)
        if street == "river":
            new_frac = min(new_frac, 0.85)
        size = clamp(ctx.pot * new_frac, ctx)
        if size > 0:
            return aggr(ctx), size, f"mid:escalate {new_frac:.0%}pot ehs={ehs:.2f} (was {last_frac:.0%})"

    return None
