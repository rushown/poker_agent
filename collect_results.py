#!/usr/bin/env python3
"""collect_results.py — fetch benchmark results for all 9 strategy agents and save comparison."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from api.arena_client import ArenaClient, ArenaAPIError

AGENTS_DIR = Path("/home/ocean/vscode/plutus-agents")
RESULTS_FILE = Path("/home/ocean/vscode/poker_agent/strategy_results.json")

STRATEGY_DESCRIPTIONS = {
    "S1": "AggroValue — large pot-sized bets when ahead",
    "S2": "AllInSpecialist — shove EHS>0.80, fold otherwise",
    "S3": "BoardTexture — 1.5x pot overbet when board pairs hole",
    "S4": "PurePotOdds — call/fold on pure equity math",
    "S5": "PositionExploiter — wide IP, ultra-tight OOP",
    "S6": "Polarized3Bet — nuts or air only",
    "S7": "SlowPlayTrap — check-raise monsters",
    "S8": "ManiacPressure — 80% open, constant pressure",
    "S9": "TightNit — top 12% hands, zero bluffs",
}


def load_agent_env(agent_dir: Path) -> dict:
    env = {}
    env_file = agent_dir / ".env"
    if not env_file.exists():
        return env
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def fetch_benchmark(api_key: str, competition_id: str, base_url: str) -> dict:
    client = ArenaClient(
        api_key=api_key,
        base_url=base_url,
    )
    client._api_key = api_key
    try:
        status = client.get_benchmark_status(competition_id)
        return status.get("match") or status
    except ArenaAPIError as e:
        return {"error": str(e), "status": e.status}


def main():
    results = {}
    print(f"\n{'Strategy':<8} {'Agent Name':<22} {'Hands':>6} {'BB/100':>8} {'ChipΔ':>8} {'Status'}")
    print("─" * 70)

    for sid in [f"S{i}" for i in range(1, 10)]:
        agent_dir = AGENTS_DIR / sid
        if not agent_dir.exists():
            print(f"{sid:<8} {'(no dir)':<22}")
            continue

        env = load_agent_env(agent_dir)
        api_key = env.get("ARENA_API_KEY", "")
        agent_name = env.get("AGENT_NAME", sid)
        comp_id = env.get("ARENA_COMPETITION_ID", "seed_poker_eval_s1")
        base_url = env.get("ARENA_BASE_URL", "https://arena.dev.fun")

        if not api_key or not api_key.startswith("arena_sk_"):
            print(f"{sid:<8} {agent_name:<22} {'not registered':>24}")
            results[sid] = {"strategy": sid, "name": agent_name, "error": "not registered"}
            continue

        match = fetch_benchmark(api_key, comp_id, base_url)
        time.sleep(0.5)

        bb100 = match.get("rawBbPer100") or match.get("adjustedBbPer100") or 0
        hands = match.get("completedHands") or 0
        target = match.get("targetHands") or 500
        chip_delta = match.get("rawChipDelta") or 0
        status = match.get("status") or match.get("phase") or "unknown"
        agent_id = match.get("agentId", env.get("ARENA_AGENT_ID", ""))[:12]
        error = match.get("error")

        print(f"{sid:<8} {agent_name:<22} {hands:>5}/{target:<4} {bb100:>+7.1f} {chip_delta:>+8.0f}  {status}")

        results[sid] = {
            "strategy": sid,
            "description": STRATEGY_DESCRIPTIONS.get(sid, ""),
            "name": agent_name,
            "agent_id": agent_id,
            "competition_id": comp_id,
            "hands_played": hands,
            "target_hands": target,
            "raw_bb_per_100": bb100,
            "chip_delta": chip_delta,
            "status": status,
            "error": error,
        }

    print("─" * 70)

    # Rank completed agents by BB/100
    ranked = [
        v for v in results.values()
        if isinstance(v.get("raw_bb_per_100"), (int, float)) and v.get("hands_played", 0) > 0
    ]
    ranked.sort(key=lambda x: x["raw_bb_per_100"], reverse=True)

    if ranked:
        print("\n🏆 Ranking (by BB/100):")
        for rank, r in enumerate(ranked, 1):
            bar = "█" * max(0, int((r["raw_bb_per_100"] + 50) / 5))
            print(f"  #{rank} {r['strategy']} {r['name']:<22} {r['raw_bb_per_100']:>+7.1f}  {bar}")

    # Save results
    output = {
        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agents": results,
        "ranking": [r["strategy"] for r in ranked],
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
