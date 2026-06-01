"""agent/techniques.py — poker theory helpers with solver-calibrated logic."""
from __future__ import annotations

from typing import Optional

from models.adaptive_memory import AdaptiveMemory, StrategyTuning


def pot_odds_required(call_amount: float, pot: float) -> float:
    if call_amount <= 0:
        return 0.0
    return call_amount / (pot + call_amount)


def minimum_defense_frequency(bet_size: float, pot: float) -> float:
    """MDF = pot / (pot + bet) — fraction of range to defend to break bluffs even."""
    if bet_size <= 0:
        return 1.0
    return pot / (pot + bet_size)


def effective_call_equity(
    call_amount: float,
    pot: float,
    tuning: StrategyTuning,
) -> float:
    """Required equity to call, adjusted by adaptive learning."""
    base = pot_odds_required(call_amount, pot)
    return base + tuning.call_threshold_adj


def should_defend_vs_bet(
    ehs: float,
    call_amount: float,
    pot: float,
    tuning: StrategyTuning,
) -> bool:
    """Defend when equity clears pot odds, with MDF slack for mixed strategy."""
    if call_amount <= 0:
        return True
    req = effective_call_equity(call_amount, pot, tuning)
    mdf = minimum_defense_frequency(call_amount, pot)
    return ehs >= req or (ehs >= req - 0.04 and ehs >= mdf * 0.88)


def value_bet_fraction(ehs: float, tuning: StrategyTuning, board_texture: str = "dry") -> float:
    """Solver-calibrated sizing tiers by equity and board texture.

    Wet/monotone boards warrant larger sizing (more protection needed).
    Dry boards can size down and still get calls from dominated hands.
    """
    if board_texture in ("wet", "monotone"):
        if ehs >= 0.88:
            base = 0.90
        elif ehs >= 0.75:
            base = 0.75
        elif ehs >= 0.62:
            base = 0.60
        else:
            base = 0.45
    elif board_texture == "paired":
        if ehs >= 0.90:
            base = 0.80
        elif ehs >= 0.78:
            base = 0.65
        elif ehs >= 0.62:
            base = 0.50
        else:
            base = 0.38
    else:  # dry
        if ehs >= 0.88:
            base = 0.70
        elif ehs >= 0.75:
            base = 0.55
        elif ehs >= 0.62:
            base = 0.42
        else:
            base = 0.33
    return min(1.2, base * tuning.postflop_value_bias)


def bluff_fraction(
    ehs: float,
    street: str,
    board_texture: str,
    fold_to_cbet: float,
    is_in_position: bool,
    tuning: StrategyTuning,
) -> float:
    """Solver-calibrated bluff sizing by street and texture.

    Larger bluffs on wet boards (more fold equity vs draws, protection).
    Smaller bluffs on dry boards (opponent range is stronger when they call).
    Returns fraction of pot, 0.0 means don't bluff.
    """
    if ehs > 0.42:  # too much equity to bluff — just bet for thin value
        return 0.0

    # Fold equity check — require meaningful fold equity (>50%) to bluff
    # 0.50 threshold ensures we only bluff vs opponents who fold often enough
    if fold_to_cbet < 0.50:
        return 0.0

    if board_texture in ("wet", "monotone"):
        base = 0.66  # larger on wet: more protected narrative
    else:
        base = 0.33  # small on dry: cheap pressure

    # River bluffs need to be larger (no more streets)
    if street == "river":
        base = min(1.0, base * 1.5)

    # OOP gets smaller bluffs (more risk)
    if not is_in_position:
        base *= 0.85

    return min(1.0, base * tuning.bluff_frequency)


def _bluff_breakeven(bet_fraction: float) -> float:
    """Fold equity needed for a bluff of `bet_fraction` pot to break even."""
    return bet_fraction / (1.0 + bet_fraction)


def bluff_allowed(
    ehs: float,
    street: str,
    fold_to_cbet: float,
    tuning: StrategyTuning,
    board_texture: str = "dry",
    is_in_position: bool = True,
) -> bool:
    """Return True when conditions support a bluff."""
    return bluff_fraction(ehs, street, board_texture, fold_to_cbet, is_in_position, tuning) > 0.0


def should_overbet(
    ehs: float,
    street: str,
    board_texture: str,
    is_in_position: bool,
    pot: float,
    stack: float,
) -> bool:
    """Overbet (1.5x-2x pot) when range is polarized and texture favors it.

    Conditions: river/turn with nut advantage on wet/monotone boards, or
    paired boards where we have trips/full house and opponent has showdown value.
    """
    if not is_in_position:
        return False
    spr = stack / max(1, pot)
    if spr < 1.2:  # not enough stack to meaningfully overbet
        return False
    if street == "river":
        return (
            (ehs >= 0.90 and board_texture in ("wet", "monotone", "paired"))
            or (ehs >= 0.95)
        )
    if street == "turn":
        return ehs >= 0.92 and board_texture in ("wet", "monotone")
    return False


def should_check_raise(
    ehs: float,
    street: str,
    board_texture: str,
    aggression_factor: float,
    opponent_cbet_freq: float,
) -> bool:
    """Check-raise when we have equity + opponent cbets too often OOP.

    Check-raise frequency: 15-20% dry, 30-40% wet, 35% monotone (OOP).
    """
    if street not in ("flop", "turn"):
        return False
    if opponent_cbet_freq < 0.55:  # only if they cbet enough to exploit
        return False
    if board_texture in ("wet", "monotone"):
        return ehs >= 0.55 or (ehs < 0.35 and aggression_factor > 2.0)
    return ehs >= 0.65  # dry board: only strong hands


def should_delayed_cbet(
    ehs: float,
    street: str,
    is_in_position: bool,
    board_texture: str,
    checked_flop: bool = False,
) -> bool:
    """Delayed cbet: check flop, bet turn when card improves our range.

    Effective on dry boards where our range has more equity on favorable runouts.
    Size: 50-66% turn.
    """
    if street != "turn" or not is_in_position or not checked_flop:
        return False
    if board_texture == "dry":
        return 0.35 <= ehs <= 0.70
    return 0.45 <= ehs <= 0.75


def should_probe_bet(
    ehs: float,
    street: str,
    is_in_position: bool,
    opponent_checked_back: bool = False,
) -> bool:
    """Probe bet OOP on turn when opponent checked back flop (shows weakness).

    Size: 30-40% pot. Exploits the narrow weak range opponent reveals.
    """
    if street != "turn" or is_in_position or not opponent_checked_back:
        return False
    return ehs >= 0.40


def icm_bubble_threshold(base_bf: float, tuning: StrategyTuning) -> float:
    return base_bf * tuning.icm_tightness


def scale_open_frequency(freq: float, tuning: StrategyTuning, position: str) -> float:
    mult = tuning.preflop_aggression
    if position in ("CO", "BTN", "SB"):
        mult *= tuning.steal_frequency
    return min(1.0, freq * mult)


def exploit_blend_threshold(
    adaptive: Optional[AdaptiveMemory],
    opponent_confidence: float,
) -> float:
    if adaptive is None:
        return 0.45 + opponent_confidence * 0.35
    return adaptive.effective_exploit_blend(opponent_confidence)


def calculate_fold_equity(opponent_fold_to_aggression: float) -> float:
    return max(0.05, min(0.92, opponent_fold_to_aggression))


def induce_bluff(
    ehs: float,
    street: str,
    opponent_af: float,
    can_check: bool,
) -> bool:
    """Check-trap vs hyper-aggressive opponents to induce a bluff."""
    if not can_check or street not in ("flop", "turn"):
        return False
    return ehs < 0.38 and opponent_af >= 3.0


def required_equity_vs_icm(call_amount: float, pot: float, bubble_factor: float) -> float:
    """ICM-adjusted required equity. BF > 1 means we need more equity to call."""
    base = pot_odds_required(call_amount, pot)
    return min(0.95, base * bubble_factor)
