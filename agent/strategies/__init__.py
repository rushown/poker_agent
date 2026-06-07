"""agent/strategies — single adaptive strategy driven by strategy_rules.json.
DeepSeek modifies strategy_rules.json each cycle. This code never changes.
Set ARENA_STRATEGY=ADAPTIVE to use it (or leave unset to use the full arbiter).
"""
from __future__ import annotations
from typing import Callable, Dict, Tuple
from agent.arbiter import GameContext
Decision = Tuple[str, float, str]
StrategyFn = Callable[[GameContext, float, float], Decision]
from agent.strategies.adaptive import decide as _adaptive
REGISTRY: Dict[str, StrategyFn] = {
    "ADAPTIVE": _adaptive,
}
NUMBERED_STRATEGIES = frozenset(REGISTRY.keys())
NAMES: Dict[str, str] = {
    "ADAPTIVE": "Adaptive — rule-driven from strategy_rules.json, refined each cycle by DeepSeek",
}
def route(strategy_id: str, ctx: GameContext, ehs: float, bb_depth: float) -> Decision:
    fn = REGISTRY.get(strategy_id.upper())
    if fn is None:
        raise ValueError(f"Unknown strategy '{strategy_id}'")
    return fn(ctx, ehs, bb_depth)
def list_strategies() -> str:
    lines = ["Available ARENA_STRATEGY values:", ""]
    for sid, desc in NAMES.items():
        lines.append(f"  {sid:10s} {desc}")
    return "\n".join(lines)
