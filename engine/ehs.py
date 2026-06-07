"""engine/ehs.py — Effective Hand Strength with caching and street-based samples."""
from __future__ import annotations

import random
from functools import lru_cache
from typing import List, Optional, Tuple

from engine.hand_eval import best_hand, RANKS, SUITS

FULL_DECK = [r + s for r in RANKS for s in SUITS]

# Per-street sample counts (balance speed vs variance)
STREET_SAMPLES = {
    "preflop": 400,   # increased from 200 — reduce 3.5% variance to 2.5% at decision boundaries
    "flop": 600,      # increased from 400
    "turn": 800,      # increased from 600
    "river": 800,     # increased from 600
}

# Tighter villain range when facing aggression (notation buckets by weight)
_FACING_RAISE_WEIGHTS: Tuple[Tuple[str, float], ...] = (
    ("premium", 3.0),   # AA-QQ, AK
    ("strong", 2.0),    # JJ-TT, AQs+
    ("playable", 1.2),
    ("speculative", 0.4),
    ("trash", 0.15),
)


def samples_for_street(street: str) -> int:
    return STREET_SAMPLES.get(street.lower(), 400)


def clear_ehs_cache() -> None:
    _ehs_cached.cache_clear()


def _remaining_deck(known: List[str]) -> List[str]:
    known_set = set(c.upper()[0] + c[1].lower() for c in known)
    return [c for c in FULL_DECK if c not in known_set]


def _hole_bucket(hole: List[str]) -> str:
    """Rough preflop bucket for weighted opponent sampling."""
    from engine.hand_eval import card_rank

    if len(hole) < 2:
        return "trash"
    r1, r2 = card_rank(hole[0]), card_rank(hole[1])
    high, low = max(r1, r2), min(r1, r2)
    suited = hole[0][1].lower() == hole[1][1].lower()
    if r1 == r2 and r1 >= 10:
        return "premium"
    if r1 == r2 and r1 >= 8:
        return "strong"
    if high >= 12 and low >= 10 and suited:
        return "premium"
    if high >= 12 and low >= 9:
        return "strong"
    if high >= 11:
        return "playable"
    if suited and high - low <= 4:
        return "speculative"
    return "trash"


def _weighted_opponent_hole(deck: List[str], facing_raise: bool, rng: random.Random) -> List[str]:
    if not facing_raise or len(deck) < 2:
        return rng.sample(deck, 2)
    # Rejection sample: prefer strong combos when villain raised
    for _ in range(30):
        opp = rng.sample(deck, 2)
        bucket = _hole_bucket(opp)
        weight = next((w for b, w in _FACING_RAISE_WEIGHTS if b == bucket), 0.5)
        if rng.random() < weight / 3.0:
            return opp
    return rng.sample(deck, 2)


def _ehs_key(hole: Tuple[str, ...], community: Tuple[str, ...], samples: int, facing: bool) -> Tuple:
    return (hole, community, samples, facing)


@lru_cache(maxsize=512)
def _ehs_cached(
    hole: Tuple[str, ...],
    community: Tuple[str, ...],
    samples: int,
    facing_raise: bool,
    seed: int,
) -> float:
    rng = random.Random(seed)
    hole_l = list(hole)
    comm_l = list(community)
    known = hole_l + comm_l
    deck = _remaining_deck(known)
    cards_to_deal = 5 - len(comm_l)
    opponent_hole_size = 2

    ahead = behind = tied = 0
    ppot_ahead_behind = [0, 0]
    ppot_counts = [0, 0]

    for _ in range(samples):
        remaining = list(deck)
        rng.shuffle(remaining)
        opp_hole = _weighted_opponent_hole(remaining, facing_raise, rng)
        used = set(opp_hole)
        rest = [c for c in remaining if c not in used]
        need = cards_to_deal
        if len(rest) < need:
            continue
        runout = rest[:need]
        full_board = comm_l + runout

        my_score = best_hand(hole_l, full_board)
        opp_score = best_hand(opp_hole, full_board)

        if my_score > opp_score:
            ahead += 1
        elif my_score < opp_score:
            behind += 1
        else:
            tied += 1

        my_now = best_hand(hole_l, comm_l) if comm_l else 0
        opp_now = best_hand(opp_hole, comm_l) if comm_l else 0

        if my_now > opp_now:
            ppot_counts[0] += 1
            if my_score < opp_score:
                ppot_ahead_behind[0] += 1
        elif my_now < opp_now:
            ppot_counts[1] += 1
            if my_score > opp_score:
                ppot_ahead_behind[1] += 1

    total = ahead + behind + tied
    if total == 0:
        return 0.5

    hs = (ahead + 0.5 * tied) / total
    npot = ppot_ahead_behind[0] / ppot_counts[0] if ppot_counts[0] > 0 else 0.0
    ppot = ppot_ahead_behind[1] / ppot_counts[1] if ppot_counts[1] > 0 else 0.0
    ehs = hs * (1.0 - npot) + (1.0 - hs) * ppot
    return max(0.0, min(1.0, ehs))


def calculate_ehs(
    hole: List[str],
    community: List[str],
    samples: int = 400,
    seed: Optional[int] = None,
    street: str = "flop",
    facing_raise: bool = False,
) -> float:
    """Return EHS ∈ [0, 1] with LRU cache."""
    if samples <= 0:
        samples = samples_for_street(street)
    s = seed if seed is not None else random.randint(0, 2**31 - 1)
    return _ehs_cached(
        tuple(sorted(hole)),
        tuple(sorted(community)),
        samples,
        facing_raise,
        s,
    )


def ehs_to_bucket(ehs: float) -> str:
    if ehs >= 0.85:
        return "monster"
    if ehs >= 0.65:
        return "strong"
    if ehs >= 0.45:
        return "medium"
    if ehs >= 0.25:
        return "weak"
    return "trash"


def pot_odds(call_amount: float, pot_size: float) -> float:
    if call_amount <= 0:
        return 0.0
    total = pot_size + call_amount
    return call_amount / total
