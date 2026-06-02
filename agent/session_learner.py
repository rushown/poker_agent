"""agent/session_learner.py — post-hand review hook + rich hand outcome logging."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from loguru import logger
from models.adaptive_memory import AdaptiveMemory


def on_hand_complete(
    adaptive: AdaptiveMemory,
    table_id: str,
    hand_number: Any,
    prev_table: Dict,
    current_table: Dict,
    my_agent_id: str,
    meta_learner: Any = None,
    meta_strategy: str = "EXPLOIT",
    dashboard: Any = None,
    strategy_mode: str = "gto",
    brutal_check: Any = None,
    action_ev: float = 0.0,
    baseline_ev: float = 0.0,
) -> None:
    """Finalize learning for a completed hand using stack + winner info."""
    seats_prev = prev_table.get("seats") or prev_table.get("players") or []
    seats_curr = current_table.get("seats") or current_table.get("players") or []
    bb = float(
        current_table.get("bigBlind")
        or prev_table.get("bigBlind")
        or 100
    )

    stack_end = _my_stack(seats_curr, my_agent_id)
    stack_start = _my_stack(seats_prev, my_agent_id)
    if stack_end is None:
        stack_end = stack_start or 0.0

    won = _did_we_win(
        current_table.get("winners") or prev_table.get("winners") or [],
        my_agent_id,
    )

    delta = (stack_end - stack_start) if stack_start is not None else 0.0
    adaptive.finish_hand(table_id, hand_number, stack_end, won=won, bb_size=bb)

    if meta_learner is not None:
        meta_learner.record_outcome(
            meta_strategy, delta, won is True, bb_size=bb
        )

    if brutal_check is not None:
        brutal_check.record_hand(
            strategy_mode=strategy_mode,
            meta_strategy=meta_strategy,
            chip_delta=delta,
            bb_size=bb,
            action_ev=action_ev,
            baseline_ev=baseline_ev,
            won=won,
        )

    if dashboard is not None:
        dashboard.record_hand(
            strategy_mode=strategy_mode,
            meta_strategy=meta_strategy,
            stack_delta=delta,
            bb_size=bb,
            vpip=False,
            pfr=False,
            won=won,
        )

    # Rich hand outcome event — parsed by cycle_engine for opponent modeling
    _log_hand_outcome(
        table_id=table_id,
        hand_number=hand_number,
        my_agent_id=my_agent_id,
        prev_table=prev_table,
        current_table=current_table,
        delta=delta,
        won=won,
        bb=bb,
    )


def _log_hand_outcome(
    table_id: str,
    hand_number: Any,
    my_agent_id: str,
    prev_table: Dict,
    current_table: Dict,
    delta: float,
    won: Optional[bool],
    bb: float,
) -> None:
    """Emit a structured hand_outcome event for cycle_engine analysis."""
    seats = prev_table.get("seats") or prev_table.get("players") or []
    my_seat = next((s for s in seats if (s.get("agentId") or s.get("id","")) == my_agent_id), {})
    hole = my_seat.get("holeCards") or my_seat.get("cards") or []
    board = (current_table.get("communityCards") or current_table.get("boardCards")
             or prev_table.get("communityCards") or prev_table.get("boardCards") or [])

    # Opponent actions this hand (from prev_table action history if available)
    opp_actions = []
    for s in seats:
        aid = s.get("agentId") or s.get("id", "")
        if aid == my_agent_id:
            continue
        opp_actions.append({
            "id": aid,
            "stack": float(s.get("stack") or s.get("stackChips") or 0),
            "status": s.get("status", ""),
        })

    winners = current_table.get("winners") or prev_table.get("winners") or []
    winner_ids = [w.get("agentId") or w.get("winnerId","") for w in winners]
    pot = float(prev_table.get("pot") or prev_table.get("potChips") or 0)

    payload = {
        "event": "hand_outcome",
        "ts": time.time(),
        "table_id": table_id,
        "hand_number": hand_number,
        "hole_cards": hole,
        "board_cards": board,
        "pot": pot,
        "chip_delta": round(delta, 2),
        "bb_size": bb,
        "bb_delta": round(delta / max(1, bb), 2),
        "won": won,
        "winner_ids": winner_ids,
        "opponents": opp_actions,
    }
    logger.info(json.dumps(payload))


def _my_stack(seats: List[Dict], agent_id: str) -> Optional[float]:
    for s in seats:
        sid = s.get("agentId") or s.get("id", "")
        if sid == agent_id:
            return float(s.get("stack") or s.get("stackChips") or 0)
    return None


def _did_we_win(winners: List, agent_id: str) -> Optional[bool]:
    if not winners:
        return None
    for w in winners:
        wid = w.get("agentId") or w.get("id") or w.get("winnerId", "")
        if wid == agent_id:
            return True
    return False
