"""agent/techniques.py — established poker theory helpers (MDF, pot odds, blending)."""
from __future__ import annotations

from typing import Optional

from models.adaptive_memory import AdaptiveMemory, StrategyTuning


def pot_odds_required(call_amount: float, pot: float) -> float:
    if call_amount <= 0:
        return 0.0
    return call_amount / (pot + call_amount)


def minimum_defense_frequency(bet_size: float, pot: float) -> float:
    """MDF = pot / (pot + bet) — minimum % of range to defend vs bet."""
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
    """MDF-aware: defend if EHS supports calling or we're near MDF floor."""
    if call_amount <= 0:
        return True
    req = effective_call_equity(call_amount, pot, tuning)
    mdf = minimum_defense_frequency(call_amount, pot)
    # Defend when equity clears threshold OR within MDF slack for mixed strategy
    return ehs >= req or (ehs >= req - 0.04 and ehs >= mdf * 0.85)


def value_bet_fraction(ehs: float, tuning: StrategyTuning) -> float:
    """Solver-inspired sizing tiers scaled by adaptive value bias."""
    if ehs >= 0.88:
        base = 0.80
    elif ehs >= 0.75:
        base = 0.65
    elif ehs >= 0.62:
        base = 0.50
    else:
        base = 0.38
    return min(1.0, base * tuning.postflop_value_bias)


def bluff_allowed(
    ehs: float,
    street: str,
    fold_to_cbet: float,
    tuning: StrategyTuning,
) -> bool:
    """Only bluff with frequency knob + fold equity."""
    if street not in ("flop", "turn", "river"):
        return False
    if ehs > 0.42:
        return False
    threshold = 0.35 - (fold_to_cbet - 0.45) * 0.15
    threshold *= tuning.bluff_frequency
    return ehs < threshold and fold_to_cbet > 0.50


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
    """Estimate fold probability from observed fold-to-aggression stats."""
    return max(0.05, min(0.92, opponent_fold_to_aggression))


def induce_bluff(
    ehs: float,
    street: str,
    opponent_af: float,
    can_check: bool,
) -> bool:
    """Check weak hands vs hyper-aggressive opponents to induce bluffs."""
    if not can_check or street not in ("flop", "turn"):
        return False
    return ehs < 0.38 and opponent_af >= 2.8
