"""api/state_parser.py — translates raw dev.fun table JSON into GameContext."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agent.arbiter import GameContext
from loguru import logger


def parse_table(
    table: Dict[str, Any],
    my_agent_id: str,
    payouts: Optional[List[float]] = None,
) -> Optional[GameContext]:
    try:
        table = _normalize_table_snapshot(table)
        table_id = table.get("tableId") or table.get("id", "")
        seats = table.get("seats") or table.get("players", [])

        my_seat = None
        for s in seats:
            sid = s.get("agentId") or s.get("id") or s.get("playerId", "")
            if sid == my_agent_id:
                my_seat = s
                break

        if my_seat is None:
            logger.warning(f"Could not find our seat in table {table_id}")
            return None

        community = _normalise_cards(
            table.get("communityCards")
            or table.get("boardCards")
            or table.get("board", [])
        )
        hole = _normalise_cards(my_seat.get("holeCards") or my_seat.get("cards", []))
        if len(hole) != 2:
            logger.debug(f"Invalid hole cards: {my_seat.get('holeCards')!r}")
            return None

        my_stack = float(
            my_seat.get("stack")
            or my_seat.get("stackChips")
            or 0
        )
        pot = float(table.get("pot") or table.get("potChips") or 0)
        bb_size = float(
            table.get("bigBlind")
            or table.get("bb")
            or table.get("bigBlindChips")
            or 2
        )
        street = _parse_street(table, community)
        position = _parse_position(my_seat, seats, table)
        is_ip = _is_in_position(my_seat, seats, table)
        allowed = _normalize_allowed_actions(
            table.get("allowedActions") or my_seat.get("allowedActions")
        )

        call_amount = _extract_call_amount(allowed, table, my_seat, pot)
        call_to = _extract_call_to_amount(allowed, table, my_seat, call_amount)
        committed = float(
            my_seat.get("currentBetChips")
            or my_seat.get("currentBet")
            or my_seat.get("betThisStreet")
            or 0
        )
        facing_raise = call_amount > bb_size * 1.05 and street == "preflop"
        facing_raise_size = call_amount if facing_raise else 0.0

        opponent_ids = [
            s.get("agentId") or s.get("id", "")
            for s in seats
            if (s.get("agentId") or s.get("id", "")) != my_agent_id
            and not _seat_folded(s)
        ]

        if payouts is None:
            payouts = _parse_payouts(table)

        deadline = table.get("actionDeadline") or table.get("deadline")
        if not deadline and table.get("actionDeadlineAt"):
            # API uses epoch ms
            deadline = float(table["actionDeadlineAt"]) / 1000.0

        ctx = GameContext(
            hole_cards=hole,
            community_cards=community,
            pot=pot,
            call_amount=call_amount,
            call_to_amount=call_to,
            committed_chips=committed,
            stack=my_stack,
            bb_size=bb_size,
            street=street,
            position=position,
            is_in_position=is_ip,
            all_stacks=[
                float(s.get("stack") or s.get("stackChips") or 0)
                for s in seats
                if not _seat_folded(s)
            ],
            payouts=payouts,
            allowed_actions=allowed,
            opponent_ids=opponent_ids,
            is_facing_raise=facing_raise,
            facing_raise_size=facing_raise_size,
            table_id=table_id,
            hand_number=table.get("handNumber") or table.get("handId") or table.get("tableNumber"),
        )
        return ctx

    except Exception as e:
        logger.exception(f"Error parsing table state: {e}")
        return None


def _normalize_table_snapshot(table: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap benchmark/start response { table: {...} } if needed."""
    if "table" in table and isinstance(table["table"], dict):
        inner = dict(table["table"])
        if "match" in table:
            inner["_match"] = table["match"]
        return inner
    return table


def prepare_table_for_runner(table: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize snapshot fields used by the polling loop (deadlines, ids)."""
    out = _normalize_table_snapshot(dict(table))
    if not out.get("actionDeadline") and not out.get("deadline"):
        raw = out.get("actionDeadlineAt")
        if raw is not None:
            ts = float(raw)
            out["actionDeadline"] = ts / 1000.0 if ts > 1e12 else ts
    if not out.get("tableId") and out.get("id"):
        out["tableId"] = out["id"]
    return out


def hero_is_to_act(table: Dict[str, Any], agent_id: str) -> bool:
    """True when benchmark/pending snapshot shows it is our decision."""
    acting = table.get("actingSeatNumber")
    if acting is None:
        acting = table.get("currentSeatNumber")
    self_seat = table.get("selfSeatNumber")
    if acting is not None and self_seat is not None:
        if int(acting) != int(self_seat):
            return False
    else:
        for seat in table.get("seats") or table.get("players") or []:
            sid = seat.get("agentId") or seat.get("id") or ""
            if sid == agent_id and str(seat.get("status", "")).lower() in (
                "folded",
                "out",
                "eliminated",
            ):
                return False

    aa = table.get("allowedActions")
    if isinstance(aa, dict):
        avail = aa.get("availableActions") or []
        return bool(avail)
    if isinstance(aa, list) and aa:
        return True
    return acting is not None and self_seat is not None


def normalize_allowed_actions(raw: Any) -> List[Dict]:
    """Public wrapper for allowedActions dict → list."""
    return _normalize_allowed_actions(raw)


def _normalize_allowed_actions(raw: Any) -> List[Dict]:
    """Convert API allowedActions object or list to arbiter-friendly list."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, dict):
        return []

    actions: List[Dict] = []
    available = [str(a).lower() for a in (raw.get("availableActions") or [])]

    for name in available:
        act: Dict[str, Any] = {"action": name.replace("_", "-")}
        if name == "call":
            inc = raw.get("callAmount") or raw.get("callChips")
            to_amt = raw.get("callToAmount") or raw.get("callTo")
            if inc is not None:
                act["amount"] = float(inc)
            if to_amt is not None:
                act["toAmount"] = float(to_amt)
            elif inc is not None:
                act["toAmount"] = float(inc)
        elif name in ("raise", "bet"):
            rr = raw.get("raiseRange") or raw.get("betRange") or {}
            act["minAmount"] = float(
                rr.get("min") or raw.get("minRaiseTo") or raw.get("minBet") or 0
            )
            act["maxAmount"] = float(
                rr.get("max")
                or raw.get("maxCommit")
                or raw.get("allInToAmount")
                or 999999
            )
        elif name in ("all-in", "allin"):
            act["minAmount"] = float(raw.get("minRaiseTo") or raw.get("callToAmount") or 0)
            act["maxAmount"] = float(
                raw.get("allInToAmount") or raw.get("maxCommit") or 999999
            )
        actions.append(act)

    if raw.get("canCheck") and not any(a.get("action") == "check" for a in actions):
        actions.append({"action": "check", "amount": 0})

    return actions


def _seat_folded(seat: Dict) -> bool:
    if seat.get("isFolded"):
        return True
    return str(seat.get("status", "")).lower() in ("folded", "out", "eliminated")


def extract_table_id(table: Dict) -> str:
    return table.get("tableId") or table.get("id", "")


def extract_opponent_actions(table: Dict, my_agent_id: str) -> List[Dict]:
    actions = []
    for action in (
        table.get("actionHistory")
        or table.get("actions")
        or table.get("recentEvents")
        or []
    ):
        summary = action.get("summary") or action
        agent = (
            action.get("agentId")
            or summary.get("agentId")
            or action.get("playerId")
            or ""
        )
        act_name = action.get("action") or summary.get("action") or action.get("type", "")
        if agent and agent != my_agent_id and act_name:
            actions.append(
                {
                    "agentId": agent,
                    "action": act_name,
                    "amount": action.get("amount") or summary.get("amount"),
                    "street": (action.get("street") or "").lower(),
                }
            )
    return actions


_CARD_RANK_MAP = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "7", "8": "8", "9": "9", "T": "T", "10": "T",
    "J": "J", "Q": "Q", "K": "K", "A": "A",
}
_SUIT_MAP = {
    "c": "c", "d": "d", "h": "h", "s": "s",
    "C": "c", "D": "d", "H": "h", "S": "s",
    "♣": "c", "♦": "d", "♥": "h", "♠": "s",
}


def _normalise_card(c: Any) -> Optional[str]:
    if not c:
        return None
    if isinstance(c, dict):
        rank = str(c.get("rank") or c.get("value") or "")
        suit = str(c.get("suit") or "")
        c = rank + suit
    c = str(c).strip()
    if len(c) < 2:
        return None
    # "7c", "Th", "As"
    if len(c) == 2:
        rank_raw, suit_raw = c[0].upper(), c[1]
    else:
        rank_raw = c[:-1].upper()
        suit_raw = c[-1]
    rank = _CARD_RANK_MAP.get(rank_raw)
    suit = _SUIT_MAP.get(suit_raw)
    if rank and suit:
        return rank + suit.lower() if len(suit) == 1 else rank + suit
    return None


def _normalise_cards(cards: Any) -> List[str]:
    if not cards:
        return []
    if isinstance(cards, str):
        parts = cards.split()
        return [c for c in (_normalise_card(p) for p in parts) if c]
    return [c for c in (_normalise_card(x) for x in cards) if c]


def _parse_street(table: Dict, community: List[str]) -> str:
    street = (table.get("street") or table.get("round") or "").lower()
    if street in ("preflop", "flop", "turn", "river", "showdown", "predeal"):
        if street in ("showdown", "predeal"):
            return "river" if community and len(community) >= 5 else "preflop"
        return street
    n = len(community)
    if n == 0:
        return "preflop"
    if n == 3:
        return "flop"
    if n == 4:
        return "turn"
    return "river"


_POSITION_MAP = {0: "UTG", 1: "MP", 2: "CO", 3: "BTN", 4: "SB", 5: "BB"}


def _parse_position(my_seat: Dict, seats: List[Dict], table: Dict) -> str:
    pos = my_seat.get("position") or my_seat.get("seatPosition") or ""
    if isinstance(pos, str):
        pu = pos.upper()
        if pu in _POSITION_MAP.values():
            return pu
    if isinstance(pos, int):
        return _POSITION_MAP.get(pos, "MP")

    # Infer from seat number in 6-max (approximate)
    sn = my_seat.get("seatNumber") or my_seat.get("seat")
    if sn is not None and len(seats) <= 6:
        idx = int(sn) - 1
        return _POSITION_MAP.get(idx % 6, "MP")

    btn_seat = table.get("buttonSeat") or table.get("dealerSeat")
    if btn_seat is not None:
        active = [s for s in seats if not _seat_folded(s) and not s.get("isSittingOut")]
        for i, s in enumerate(active):
            if (s.get("seat") or s.get("seatIndex") or s.get("seatNumber")) == btn_seat:
                my_idx = active.index(my_seat) if my_seat in active else 0
                offset = (my_idx - i) % len(active)
                return _POSITION_MAP.get(offset, "MP")
    return "MP"


def _is_in_position(my_seat: Dict, seats: List[Dict], table: Dict) -> bool:
    pos = _parse_position(my_seat, seats, table)
    return pos in ("BTN", "CO")


def _extract_call_amount(
    allowed_actions: List[Dict],
    table: Dict,
    my_seat: Dict,
    pot: float,
) -> float:
    for a in allowed_actions:
        if a.get("action") == "call":
            amt = a.get("amount")
            if amt is not None:
                return float(amt)
            if a.get("callAmount") is not None:
                return float(a["callAmount"])

    aa = table.get("allowedActions")
    if isinstance(aa, dict):
        for key in ("callAmount", "callChips", "callToAmount"):
            if aa.get(key) is not None:
                return float(aa[key])

    to_call = my_seat.get("amountToCall") or my_seat.get("callAmount")
    if to_call is not None:
        return float(to_call)

    current_bet = float(
        my_seat.get("currentBet")
        or my_seat.get("currentBetChips")
        or my_seat.get("betThisStreet")
        or 0
    )
    max_bet = max(
        (
            float(s.get("currentBet") or s.get("currentBetChips") or s.get("betThisStreet") or 0)
            for s in (table.get("seats") or table.get("players") or [])
        ),
        default=0,
    )
    inferred = max(0.0, max_bet - current_bet)
    if inferred > 0:
        return inferred

    if table.get("currentBet") is not None and current_bet == 0:
        return max(0.0, float(table["currentBet"]) - float(my_seat.get("totalCommittedChips") or 0))

    has_raise = any(a.get("action") in ("raise", "bet") for a in allowed_actions)
    has_fold = any(a.get("action") == "fold" for a in allowed_actions)
    if has_raise and has_fold and not any(a.get("action") == "check" for a in allowed_actions):
        for a in allowed_actions:
            if a.get("action") in ("raise", "bet"):
                mn = float(a.get("minAmount", 0))
                if mn > 0:
                    return float(table.get("currentBet") or mn)
    return 0.0


def _extract_call_to_amount(
    allowed_actions: List[Dict],
    table: Dict,
    my_seat: Dict,
    call_amount: float,
) -> float:
    for a in allowed_actions:
        if a.get("action") == "call":
            if a.get("toAmount") is not None:
                return float(a["toAmount"])
    aa = table.get("allowedActions")
    if isinstance(aa, dict):
        for key in ("callToAmount", "callTo"):
            if aa.get(key) is not None:
                return float(aa[key])
    committed = float(
        my_seat.get("currentBetChips")
        or my_seat.get("currentBet")
        or 0
    )
    return committed + call_amount


def _parse_payouts(table: Dict) -> Optional[List[float]]:
    comp = table.get("competition") or {}
    raw = table.get("payouts") or comp.get("payouts") or table.get("prizeStructure")
    if not raw:
        return None
    if isinstance(raw, list):
        out = []
        for p in raw:
            if isinstance(p, (int, float)):
                out.append(float(p))
            elif isinstance(p, dict):
                out.append(float(p.get("amount") or p.get("prize") or 0))
        return out if out else None
    return None
