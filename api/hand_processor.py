"""api/hand_processor.py — replay buffered hand actions into OpponentTracker."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from loguru import logger
from models.opponent_tracker import OpponentTracker

_STREETS = ("preflop", "flop", "turn", "river")
_BLIND_ACTIONS = frozenset({"sb", "bb", "smallblind", "bigblind", "ante", "post"})


def process_full_hand(
    actions: List[Dict[str, Any]],
    seats: List[Dict[str, Any]],
    my_agent_id: str,
    tracker: OpponentTracker,
    winners: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Walk actions in order and update opponent stats."""
    if not actions and not seats:
        return

    try:
        by_street = _group_by_street(actions)
        opponent_ids = _opponent_ids(seats, my_agent_id)

        for opp_id in opponent_ids:
            tracker.record_hand_seen(opp_id)
            _process_preflop_hand(opp_id, by_street.get("preflop", []), tracker, seats)
            _process_postflop_hand(opp_id, by_street, tracker)

            if _saw_street(opp_id, by_street.get("river", [])):
                won = _did_win(opp_id, winners or [])
                tracker.record_wtsd(opp_id, went=True, won=won)

            for seat in seats:
                sid = seat.get("agentId") or seat.get("id", "")
                if sid == opp_id:
                    chat = seat.get("chatMessage") or seat.get("message")
                    if chat:
                        tracker.record_chat(opp_id, str(chat))

        tracker.save()
    except Exception as e:
        logger.debug(f"process_full_hand error: {e}")


def process_completed_hand(
    hand: Dict[str, Any],
    my_agent_id: str,
    tracker: OpponentTracker,
) -> None:
    """Legacy entry: use snapshot if it already has full history."""
    seats = hand.get("seats") or hand.get("players", [])
    actions_by_street = hand.get("actionsByStreet") or {}
    all_actions: List[Dict] = []
    if actions_by_street:
        for street in _STREETS:
            for a in actions_by_street.get(street, []):
                rec = dict(a)
                rec.setdefault("street", street)
                all_actions.append(rec)
    else:
        all_actions = list(hand.get("actions") or hand.get("actionHistory") or [])

    winners = hand.get("winners") or (hand.get("finalState") or {}).get("winners")
    process_full_hand(all_actions, seats, my_agent_id, tracker, winners)


def _group_by_street(actions: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {s: [] for s in _STREETS}
    for a in actions:
        street = str(a.get("street") or a.get("round") or "preflop").lower()
        if street not in out:
            street = "preflop"
        out[street].append(a)
    return out


def _opponent_ids(seats: List[Dict], my_agent_id: str) -> List[str]:
    ids: List[str] = []
    for s in seats:
        sid = s.get("agentId") or s.get("id") or s.get("playerId", "")
        if sid and sid != my_agent_id:
            ids.append(sid)
    return ids


def _actor(action: Dict) -> str:
    return action.get("agentId") or action.get("playerId") or action.get("actor", "")


def _act_name(action: Dict) -> str:
    return str(action.get("action") or action.get("type") or "").lower()


def _is_aggressive(act: str) -> bool:
    return act in ("bet", "raise", "all-in", "allin")


def _process_preflop_hand(
    opp_id: str,
    preflop: List[Dict],
    tracker: OpponentTracker,
    seats: List[Dict],
) -> None:
    raises_before_opp = 0
    opp_acted = False
    opp_was_pfr = False
    first_voluntary = True

    for a in preflop:
        actor = _actor(a)
        act = _act_name(a)

        if act in _BLIND_ACTIONS:
            continue

        if actor == opp_id:
            opp_acted = True
            if first_voluntary and act in ("call", "raise", "bet", "all-in", "allin"):
                tracker.record_vpip(opp_id)
                first_voluntary = False
            if act in ("raise", "bet", "all-in", "allin"):
                tracker.record_pfr(opp_id)
                if raises_before_opp >= 1:
                    tracker.record_3bet_opportunity(opp_id, True)
                opp_was_pfr = True
            elif raises_before_opp >= 2 and act == "fold":
                tracker.record_fold_to_3bet(opp_id, True)
            elif raises_before_opp >= 2 and act in ("call", "raise", "all-in", "allin"):
                tracker.record_fold_to_3bet(opp_id, False)

            if raises_before_opp == 1 and act in ("raise", "all-in", "allin"):
                tracker.record_3bet_opportunity(opp_id, True)

            # Steal defence: faced open raise, folded
            if raises_before_opp == 1 and act == "fold":
                tracker.record_fold_to_steal(opp_id, True)
            elif raises_before_opp == 1 and act in ("call", "raise", "all-in", "allin"):
                tracker.record_fold_to_steal(opp_id, False)

            tracker.record_action(opp_id, act)
        else:
            if _is_aggressive(act):
                raises_before_opp += 1
                if opp_acted and actor != opp_id:
                    tracker.record_3bet_opportunity(opp_id, False)

    # Open-raise (RFI) opportunity: no raise before opp's first voluntary action
    if not opp_was_pfr and opp_acted:
        had_raise_before = any(
            _is_aggressive(_act_name(a))
            for a in preflop
            if _actor(a) != opp_id and _act_name(a) not in _BLIND_ACTIONS
        )
        if not had_raise_before:
            tracker.record_raise_first_in(
                opp_id,
                any(
                    _is_aggressive(_act_name(a))
                    for a in preflop
                    if _actor(a) == opp_id
                ),
            )


def _process_postflop_hand(
    opp_id: str,
    by_street: Dict[str, List[Dict]],
    tracker: OpponentTracker,
) -> None:
    preflop = by_street.get("preflop", [])
    opp_was_pfr = _was_preflop_aggressor(opp_id, preflop)
    saw_flop = bool(by_street.get("flop"))
    opp_cbet_opp_recorded = False

    for street in ("flop", "turn", "river"):
        street_actions = by_street.get(street, [])
        if not street_actions:
            continue

        first_aggressor: Optional[str] = None
        facing_bet = False

        for a in street_actions:
            actor = _actor(a)
            act = _act_name(a)

            if act in _BLIND_ACTIONS:
                continue

            if _is_aggressive(act) and first_aggressor is None:
                first_aggressor = actor

            if actor == opp_id:
                tracker.record_action(opp_id, act)

                if (
                    street == "flop"
                    and opp_was_pfr
                    and saw_flop
                    and not opp_cbet_opp_recorded
                    and first_aggressor is None
                ):
                    tracker.record_cbet_opportunity(opp_id, _is_aggressive(act))
                    opp_cbet_opp_recorded = True

                if facing_bet and first_aggressor and first_aggressor != opp_id:
                    if street == "flop":
                        tracker.record_fold_to_cbet(opp_id, act == "fold", street="flop")
                    elif street == "turn":
                        tracker.record_fold_to_cbet(opp_id, act == "fold", street="turn")

                if street == "flop" and act == "raise" and facing_bet:
                    tracker.record_check_raise_flop(opp_id, True)

                if street == "river" and _is_aggressive(act):
                    tracker.record_river_aggression(opp_id)

            if actor != opp_id and _is_aggressive(act):
                facing_bet = True
                if actor != opp_id and street == "flop" and opp_was_pfr and not opp_cbet_opp_recorded:
                    # Opponent was PFR but someone else bet first — still mark cbet opp
                    opp_cbet_opp_recorded = True


def _was_preflop_aggressor(opp_id: str, preflop: List[Dict]) -> bool:
    for a in preflop:
        if _actor(a) == opp_id and _is_aggressive(_act_name(a)):
            return True
    return False


def _saw_street(agent_id: str, actions: List[Dict]) -> bool:
    return any(_actor(a) == agent_id for a in actions)


def _did_win(opp_id: str, winners: List[Dict]) -> bool:
    for w in winners:
        wid = w.get("agentId") or w.get("id") or w.get("winnerId", "")
        if wid == opp_id:
            return True
    return False
