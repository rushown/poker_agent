"""agent/decision_log.py — structured JSON decision lines for plutus.log."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from loguru import logger


def log_decision(
    *,
    table_id: str,
    hand_number: Any,
    strategy_mode: str,
    meta_strategy: str,
    action: str,
    amount: float,
    ehs: float,
    decision_time_ms: float,
    opponent_ids: Optional[list] = None,
    error: str = "",
    hand_result: str = "",
) -> None:
    payload: Dict[str, Any] = {
        "event": "decision",
        "ts": time.time(),
        "table_id": table_id,
        "hand_number": hand_number,
        "strategy_mode": strategy_mode,
        "meta_strategy": meta_strategy,
        "action_taken": action,
        "amount": amount,
        "ehs": round(ehs, 4),
        "decision_time_ms": round(decision_time_ms, 1),
        "opponent_ids": opponent_ids or [],
        "hand_result": hand_result,
    }
    if error:
        payload["error"] = error
    logger.bind(structured=True).info(json.dumps(payload))
