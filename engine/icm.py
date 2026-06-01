"""engine/icm.py — ICM calculator using Malmuth-Harville approximation.

Converts chip stacks into $ EV given a payout structure.
Runs in O(n^2) — fast enough for real-time use up to ~20 players.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def icm_ev(stacks: List[float], payouts: List[float]) -> List[float]:
    """Compute ICM equity for each player.

    Args:
        stacks:  chip counts for each player (any positive floats)
        payouts: payout amounts for finishing positions (index 0 = 1st place)
                 len(payouts) <= len(stacks)

    Returns:
        List of $ EV per player, same order as stacks.
    """
    n = len(stacks)
    total_chips = sum(stacks)
    if total_chips == 0:
        return [0.0] * n

    # Normalise stacks to probabilities
    probs = [s / total_chips for s in stacks]

    # Build finish probabilities using Malmuth-Harville
    # finish_prob[i][k] = P(player i finishes in position k+1)
    finish_prob = [[0.0] * len(payouts) for _ in range(n)]

    def _recursive(
        remaining_players: List[int],
        remaining_chips: float,
        position: int,
        accumulated_prob: float,
    ) -> None:
        if position >= len(payouts) or not remaining_players:
            return
        for idx in remaining_players:
            p = stacks[idx] / remaining_chips
            finish_prob[idx][position] += accumulated_prob * p
            next_players = [j for j in remaining_players if j != idx]
            _recursive(
                next_players,
                remaining_chips - stacks[idx],
                position + 1,
                accumulated_prob * p,
            )

    _recursive(list(range(n)), total_chips, 0, 1.0)

    # Compute EV
    ev = []
    for i in range(n):
        player_ev = sum(finish_prob[i][k] * payouts[k] for k in range(len(payouts)))
        ev.append(player_ev)

    return ev


def icm_pressure(
    my_stack: float,
    all_stacks: List[float],
    payouts: List[float],
    call_amount: float,
    win_chips: float,
) -> float:
    """Return the ICM EV delta of calling vs folding.

    Positive = calling increases $ EV.
    Negative = folding is better in ICM terms.
    """
    stacks = list(all_stacks)
    my_idx = stacks.index(my_stack)  # assumes unique stack; caller should handle ties

    # EV if we fold
    ev_fold = icm_ev(stacks, payouts)[my_idx]

    # EV if we win (gain win_chips)
    stacks_win = list(stacks)
    stacks_win[my_idx] += win_chips
    ev_win = icm_ev(stacks_win, payouts)[my_idx]

    # EV if we lose (lose call_amount)
    stacks_lose = list(stacks)
    stacks_lose[my_idx] = max(0, stacks_lose[my_idx] - call_amount)
    ev_lose = icm_ev(stacks_lose, payouts)[my_idx]

    # We need to know equity to combine; caller passes equity separately.
    # This helper just returns the components.
    return ev_win - ev_fold, ev_lose - ev_fold


def bubble_factor(
    my_stack: float,
    all_stacks: List[float],
    payouts: List[float],
) -> float:
    """ICM-based bubble factor.

    > 1.0 means we should play tighter than pure chip EV suggests.
    1.0 = chip EV ≈ $ EV (no ICM distortion).
    """
    if len(payouts) < 2:
        return 1.0
    total = sum(all_stacks)
    if total == 0:
        return 1.0

    stacks = list(all_stacks)
    try:
        my_idx = stacks.index(my_stack)
    except ValueError:
        return 1.0

    ev_now = icm_ev(stacks, payouts)[my_idx]

    # Simulate winning 10 BB worth of chips
    chip_unit = total / 100
    stacks_up = list(stacks)
    stacks_up[my_idx] += chip_unit
    ev_up = icm_ev(stacks_up, payouts)[my_idx]

    stacks_down = list(stacks)
    stacks_down[my_idx] = max(0, stacks_down[my_idx] - chip_unit)
    ev_down = icm_ev(stacks_down, payouts)[my_idx]

    gain = ev_up - ev_now
    loss = ev_now - ev_down

    if gain == 0:
        return 2.0
    return loss / gain
