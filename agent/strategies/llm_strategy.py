"""agent/strategies/llm_strategy.py — L2 LLM-driven strategy for Juliana.

Decision flow each hand:
  1. Call DeepSeek chat with compressed game state + Juliana persona
  2. Parse JSON response → action + amount
  3. Fall back to ADAPTIVE rule-based on any failure or timeout
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import httpx

from agent.arbiter import GameContext
from agent.strategies.adaptive import decide as _fallback

_API_URL = "https://api.deepseek.com/chat/completions"

_SYSTEM = """\
You are Juliana — a deceptive no-limit Texas Hold'em player who loves to mislead.

Strategy:
- Strong hands (EHS ≥ 0.75): slow-play. Check/call flop to trap, raise turn or river.
- Medium hands (EHS 0.45–0.74): bet ~0.65× pot. Looks like a bluff, often isn't.
- Weak hands (EHS < 0.45): bluff when in position on dry boards. Fold out of position.
- Preflop: open wide. 3-bet polarized. Call 3-bets with suited connectors.
- Sizing is always polarized: 0.5× pot (probe) or 1× pot (value/big bluff). Never in between.
- Never min-raise. Never limp. Bet medium pairs as bluffs — never check them.

Respond with ONLY valid compact JSON, no markdown:
{"action":"fold|check|call|raise|bet","amount":<int or 0>,"reason":"<10 words max>"}
"action" must be exactly one of the provided allowed_actions."""


def _build_prompt(ctx: GameContext, ehs: float) -> str:
    allowed = []
    for a in ctx.allowed_actions:
        if isinstance(a, dict):
            name = a.get("action", "")
            mn, mx = a.get("min", 0), a.get("max", 0)
            allowed.append(f"{name}(min={mn},max={mx})" if (mn or mx) else name)
        else:
            allowed.append(str(a))
    board = " ".join(ctx.community_cards) if ctx.community_cards else "none"
    return (
        f"hole={ctx.hole_cards} board={board} street={ctx.street} "
        f"pos={ctx.position} ip={'yes' if ctx.is_in_position else 'no'} "
        f"pot={ctx.pot:.0f} stack={ctx.stack:.0f} call={ctx.call_amount:.0f} "
        f"ehs={ehs:.3f} facing_raise={'yes' if ctx.is_facing_raise else 'no'} "
        f"actions=[{','.join(allowed)}]"
    )


def _allowed_names(ctx: GameContext) -> set[str]:
    names = set()
    for a in ctx.allowed_actions:
        n = (a.get("action", "") if isinstance(a, dict) else str(a)).lower()
        names.add(n)
    return names


def decide(ctx: GameContext, ehs: float, bb_depth: float) -> Tuple[str, float, str]:
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return _fallback(ctx, ehs, bb_depth)

    try:
        with httpx.Client(timeout=2.5) as client:
            r = client.post(
                _API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": _build_prompt(ctx, ehs)},
                    ],
                    "max_tokens": 80,
                    "temperature": 0.75,
                    "response_format": {"type": "json_object"},
                },
            )
            r.raise_for_status()
            data = json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception:
        return _fallback(ctx, ehs, bb_depth)

    action = str(data.get("action", "")).lower().strip()
    amount = float(data.get("amount") or 0)
    reason = str(data.get("reason", "Juliana"))[:80]

    allowed = _allowed_names(ctx)

    # Normalize all-in if model outputs it
    if action in ("allin", "all-in", "all_in"):
        action = "raise" if "raise" in allowed else "bet" if "bet" in allowed else "call"
        amount = ctx.stack

    if action not in allowed:
        return _fallback(ctx, ehs, bb_depth)

    return action, amount, f"[Juliana] {reason}"
