"""engine/hand_eval.py — fast 7-card hand evaluator.

Returns an integer rank: higher is better.
No external dependencies — pure Python.

Rank encoding (higher = stronger hand):
  8 = straight flush, 7 = quads, 6 = full house, 5 = flush,
  4 = straight, 3 = trips, 2 = two pair, 1 = pair, 0 = high card
  Full value = (category << 20) | tiebreaker_bits
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_MAP = {r: i for i, r in enumerate(RANKS)}

# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

def card_rank(c: str) -> int:
    return RANK_MAP[c[0].upper()]

def card_suit(c: str) -> str:
    return c[1].lower()

def parse_card(c: str) -> Tuple[int, str]:
    """'Ah' -> (12, 'h')"""
    return card_rank(c), card_suit(c)


def hand_notation(hole: List[str]) -> str:
    """['Ah','Kd'] -> 'AKo',  ['Qh','Qc'] -> 'QQ'"""
    if len(hole) < 2:
        return "??"
    r1, r2 = card_rank(hole[0]), card_rank(hole[1])
    s1, s2 = hole[0][1].lower(), hole[1][1].lower()
    high, low = RANKS[max(r1, r2)], RANKS[min(r1, r2)]
    if r1 == r2:
        return high + high
    return high + low + ("s" if s1 == s2 else "o")


# ---------------------------------------------------------------------------
# 5-card evaluator
# ---------------------------------------------------------------------------

def _eval5(cards: List[str]) -> int:
    """Evaluate exactly 5 cards; return integer score (higher = better)."""
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    suits = [card_suit(c) for c in cards]

    is_flush = len(set(suits)) == 1
    is_straight = (
        (ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5) or
        # Wheel: A-2-3-4-5
        ranks == [12, 3, 2, 1, 0]
    )
    if ranks == [12, 3, 2, 1, 0]:
        straight_high = 3  # 5-high straight
    else:
        straight_high = ranks[0]

    from collections import Counter
    cnt = Counter(ranks)
    freq = sorted(cnt.values(), reverse=True)
    groups = sorted(cnt.keys(), key=lambda r: (cnt[r], r), reverse=True)

    if is_straight and is_flush:
        return (8 << 20) | straight_high
    if freq[0] == 4:
        return (7 << 20) | (groups[0] << 4) | groups[1]
    if freq[0] == 3 and freq[1] == 2:
        return (6 << 20) | (groups[0] << 4) | groups[1]
    if is_flush:
        tiebreak = sum(r << (4 * (4 - i)) for i, r in enumerate(ranks))
        return (5 << 20) | tiebreak
    if is_straight:
        return (4 << 20) | straight_high
    if freq[0] == 3:
        kickers = sorted([r for r in ranks if r != groups[0]], reverse=True)
        return (3 << 20) | (groups[0] << 8) | (kickers[0] << 4) | kickers[1]
    if freq[0] == 2 and freq[1] == 2:
        pair1, pair2 = groups[0], groups[1]
        kicker = [r for r in ranks if r not in (pair1, pair2)][0]
        return (2 << 20) | (max(pair1, pair2) << 8) | (min(pair1, pair2) << 4) | kicker
    if freq[0] == 2:
        kickers = sorted([r for r in ranks if r != groups[0]], reverse=True)
        return (1 << 20) | (groups[0] << 12) | (kickers[0] << 8) | (kickers[1] << 4) | kickers[2]
    tiebreak = sum(r << (4 * (4 - i)) for i, r in enumerate(ranks))
    return tiebreak


def best_hand(hole: List[str], community: List[str]) -> int:
    """Return best 5-card score from hole + community cards (2+3/4/5)."""
    all_cards = hole + community
    if len(all_cards) < 5:
        # Pre-flop or partial board: evaluate what we have
        return _eval5(all_cards) if len(all_cards) == 5 else _eval_partial(all_cards)
    return max(_eval5(list(combo)) for combo in combinations(all_cards, 5))


def _eval_partial(cards: List[str]) -> int:
    """Score fewer than 5 cards (pre-flop / flop only). Good enough for relative comparisons."""
    from collections import Counter
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    suits = [card_suit(c) for c in cards]
    cnt = Counter(ranks)
    freq = sorted(cnt.values(), reverse=True)
    groups = sorted(cnt.keys(), key=lambda r: (cnt[r], r), reverse=True)
    if freq[0] == 4:
        return (7 << 20) | groups[0]
    if freq[0] == 3:
        return (3 << 20) | groups[0]
    if freq[0] == 2 and len(freq) > 1 and freq[1] == 2:
        return (2 << 20) | (groups[0] << 4) | groups[1]
    if freq[0] == 2:
        return (1 << 20) | groups[0]
    return sum(r << (4 * (len(ranks) - 1 - i)) for i, r in enumerate(ranks))


def hand_name(score: int) -> str:
    cat = score >> 20
    names = ["High Card", "Pair", "Two Pair", "Trips", "Straight",
             "Flush", "Full House", "Quads", "Straight Flush"]
    return names[cat] if 0 <= cat <= 8 else "Unknown"
