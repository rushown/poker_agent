"""api/action_amount.py — convert strategy amounts to Arena toAmount semantics."""
from __future__ import annotations

from typing import Any, Dict, Optional


def _my_seat(table: Dict[str, Any], agent_id: str) -> Optional[Dict]:
    for s in table.get("seats") or table.get("players") or []:
        sid = s.get("agentId") or s.get("id") or s.get("playerId", "")
        if sid == agent_id:
            return s
    return None


def committed_chips(seat: Optional[Dict]) -> float:
    if not seat:
        return 0.0
    return float(
        seat.get("currentBetChips")
        or seat.get("currentBet")
        or seat.get("betThisStreet")
        or 0
    )


def raw_allowed(table: Dict[str, Any]) -> Dict[str, Any]:
    aa = table.get("allowedActions")
    return aa if isinstance(aa, dict) else {}


def call_to_amount(table: Dict[str, Any], agent_id: str) -> float:
    """Total chips committed on this street after a call."""
    aa = raw_allowed(table)
    for key in ("callToAmount", "callTo"):
        if aa.get(key) is not None:
            return float(aa[key])
    seat = _my_seat(table, agent_id)
    committed = committed_chips(seat)
    inc = float(aa.get("callAmount") or aa.get("callChips") or 0)
    return committed + inc


def min_raise_to(table: Dict[str, Any]) -> float:
    aa = raw_allowed(table)
    mn = aa.get("minRaiseTo")
    if mn is not None:
        return float(mn)
    rr = aa.get("raiseRange") or aa.get("betRange") or {}
    return float(rr.get("min") or 0)


def max_commit(table: Dict[str, Any], agent_id: str) -> float:
    aa = raw_allowed(table)
    for key in ("allInToAmount", "maxCommit"):
        if aa.get(key) is not None:
            return float(aa[key])
    seat = _my_seat(table, agent_id)
    if seat:
        return float(seat.get("stackChips") or seat.get("stack") or 0)
    return 999999.0


def format_submission_amount(
    action: str,
    amount: float,
    table: Dict[str, Any],
    agent_id: str,
) -> float:
    """
    Arena expects `amount` as total chips committed on this street after acting.
    Strategy code may use incremental call sizes or raise-to targets; normalize here.
    """
    act = (action or "").lower().replace("_", "-")
    if act in ("fold", "check"):
        return 0.0

    seat = _my_seat(table, agent_id)
    committed = committed_chips(seat)
    aa = raw_allowed(table)
    call_inc = float(aa.get("callAmount") or aa.get("callChips") or 0)
    target_call = call_to_amount(table, agent_id)
    mn_raise = min_raise_to(table)
    mx = max_commit(table, agent_id)

    if act == "call":
        if amount <= 0:
            return target_call
        # Incremental call amount (e.g. 11) vs toAmount (13)
        if abs(amount - call_inc) < 0.01 or amount < committed:
            return target_call
        if amount < target_call and amount > committed:
            return target_call
        return min(mx, amount)

    if act in ("all-in", "allin"):
        return mx

    if act in ("raise", "bet"):
        if amount <= 0:
            return mn_raise if mn_raise > 0 else mx
        # Already a valid toAmount
        if mn_raise > 0 and amount >= mn_raise - 0.01:
            return min(mx, amount)
        # Open size given as total bet from 0 (unopened)
        if committed <= 0 and amount >= (aa.get("minBet") or aa.get("bigBlindChips") or 2):
            return min(mx, max(amount, mn_raise or amount))
        # Incremental raise mistake: add committed
        if amount < committed + call_inc + 0.01:
            return min(mx, max(mn_raise, committed + amount))
        return min(mx, max(mn_raise, amount))

    return amount


def build_action_payload(
    table_id: str,
    action: str,
    amount: float,
    table: Dict[str, Any],
    agent_id: str,
    message: str = "",
) -> Dict[str, Any]:
    """Arena POST /texas/action body with toAmount-normalized amount."""
    body: Dict[str, Any] = {
        "tableId": table_id,
        "action": action.lower().replace("_", "-"),
    }
    amt = format_submission_amount(action, amount, table, agent_id)
    if amt > 0:
        body["amount"] = round(amt, 2)
    if message:
        body["message"] = message[:200]
    return body


def self_test() -> list:
    errors: list = []
    table = {
        "tableId": "t1",
        "seats": [{"agentId": "me", "currentBetChips": 2, "stackChips": 200}],
        "allowedActions": {
            "callAmount": 11,
            "callToAmount": 13,
            "minRaiseTo": 22,
            "allInToAmount": 200,
            "raiseRange": {"min": 22, "max": 200},
        },
    }
    payload = build_action_payload("t1", "call", 11, table, "me")
    if payload.get("amount") != 13:
        errors.append(f"call payload amount expected 13 got {payload.get('amount')}")
    payload_r = build_action_payload("t1", "raise", 22, table, "me")
    if payload_r.get("amount") != 22:
        errors.append(f"raise payload expected 22 got {payload_r.get('amount')}")
    payload_ai = build_action_payload("t1", "all-in", 0, table, "me")
    if payload_ai.get("amount") != 200:
        errors.append(f"all-in expected 200 got {payload_ai.get('amount')}")
    return errors
