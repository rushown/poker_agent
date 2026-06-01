"""agent/strategy_styles.py — MANIAC/TAG/LAG/NIT style wrappers over GTO baseline."""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from agent.gto_bot import hand_notation, postflop_action, preflop_action, _call_to, _has_check
from models.adaptive_memory import StrategyTuning

_rng = random.Random()


def _tuning_for_style(name: str) -> StrategyTuning:
    t = StrategyTuning()
    if name == "MANIAC":
        t.preflop_aggression = 1.35
        t.bluff_frequency = 1.25
        t.cbet_frequency = 1.2
        t.steal_frequency = 1.3
    elif name == "TAG":
        t.preflop_aggression = 0.85
        t.bluff_frequency = 0.7
        t.cbet_frequency = 1.05
        t.call_threshold_adj = 0.02
    elif name == "LAG":
        t.preflop_aggression = 1.2
        t.bluff_frequency = 1.1
        t.steal_frequency = 1.25
    elif name == "NIT":
        t.preflop_aggression = 0.7
        t.bluff_frequency = 0.45
        t.steal_frequency = 0.75
        t.icm_tightness = 1.15
    t.clamp()
    return t


def play_style(
    style: str,
    *,
    hole: List[str],
    position: str,
    street: str,
    ehs: float,
    pot: float,
    call_amount: float,
    stack: float,
    bb_size: float,
    is_facing_raise: bool,
    facing_raise_size: float,
    is_in_position: bool,
    allowed_actions: List[dict],
    fold_to_3bet: float = 0.5,
    fold_to_cbet: float = 0.45,
) -> Tuple[str, float, str]:
    tuning = _tuning_for_style(style)
    tag = f"[{style}]"

    if style == "MANIAC" and street == "preflop" and not is_facing_raise:
        if _rng.random() < 0.55 and any(a.get("action") == "raise" for a in allowed_actions):
            return "raise", min(stack, bb_size * 3.5), f"{tag} maniac open {hand_notation(hole)}"

    if style == "NIT" and street == "preflop" and ehs < 0.55 and is_facing_raise:
        if _has_check(allowed_actions):
            return "check", 0, f"{tag} nit check"
        return "fold", 0, f"{tag} nit fold"

    if street == "preflop":
        return preflop_action(
            hole=hole,
            position=position,
            is_facing_raise=is_facing_raise,
            facing_raise_size=facing_raise_size,
            bb_size=bb_size,
            stack=stack,
            pot=pot,
            allowed_actions=allowed_actions,
            fold_to_3bet_avg=fold_to_3bet,
            tuning=tuning,
        )

    act, amt, reason = postflop_action(
        ehs=ehs,
        pot=pot,
        call_amount=call_amount,
        stack=stack,
        street=street,
        is_in_position=is_in_position,
        allowed_actions=allowed_actions,
        fold_to_cbet=fold_to_cbet,
        tuning=tuning,
    )
    return act, amt, f"{tag} {reason}"
