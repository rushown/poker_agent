"""agent/owner_messages.py — owner-facing messages per arena.md formatting."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from loguru import logger


def derive_handle(name: str) -> str:
    """arena.md: handle from name, lowercase, alnum+underscore, max 30."""
    h = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower().replace(" ", "_"))
    while "__" in h:
        h = h.replace("__", "_")
    return h.strip("_")[:30] or "agent"


def write_owner_message(path: str, message: str, *, log_summary: bool = True) -> None:
    """Append owner message to file. Logs a one-line summary (not full secrets)."""
    block = f"\n━━━ Plutus · {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} ━━━\n{message.strip()}\n"
    if log_summary:
        first_line = message.strip().split("\n", 1)[0][:120]
        logger.info(f"[OWNER] {first_line}")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)
    except OSError as e:
        logger.warning(f"Could not write owner message file: {e}")


def format_registration_success(
    agent_id: str,
    api_key: str,
    claim_url: str = "",
) -> str:
    return (
        "Registered. Save the API key — it's the only copy.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Agent ID: `{agent_id}`\n"
        f"API Key: `{api_key}`\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Ready to compete. Set ARENA_API_KEY in .env or keep `.arena-credentials`.\n\n"
        "───────────────────────────────────\n"
        "💡 Claim your agent for prizes and the leaderboard:\n"
        f"{claim_url or '(run claim-status after register)'}"
    )


def format_competition_picker(competitions: List[Dict[str, Any]]) -> str:
    lines = [
        "🏟 Multiple live arenas are available. Pick one:\n",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    for i, c in enumerate(competitions[:10], 1):
        skill = c.get("skillFile") or ""
        fee = c.get("entryFee") or c.get("entry_fee") or ""
        from api.arena_onboarding import competition_mode_label

        mode = competition_mode_label(c)
        lines.append(
            f"\n**{i}.** `{c.get('name', '?')}`\n"
            f"gameType `{c.get('gameType')}` · season `{c.get('seasonNumber', '?')}`\n"
            f"competitionId `{c.get('id')}` · mode: **{mode}**\n"
            + (f"skill `{skill}`\n" if skill else "")
            + (f"entry `{fee}`" if fee else "")
        )
    lines.append(
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Set **ARENA_COMPETITION_ID** in `.env` to your choice, then restart.\n"
        "Or run: `python -m agent.runner --list-competitions`"
    )
    return "\n".join(lines)


def format_invitation_offer(inv: Dict[str, Any]) -> str:
    return (
        f"🎁 You have {inv.get('rewardAmount', '?')} reserved from "
        f"{inv.get('partnerName', 'a partner')} for "
        f"{inv.get('templateName', 'arena entry')}. "
        "I'll claim it automatically on startup."
    )


def format_entry_fee_help(
    competition_name: str,
    amount: str,
    currency: str,
    chain: str,
    wallet_address: str,
) -> str:
    return (
        f"To enter **{competition_name}** I need **{amount} {currency}** on **{chain}**.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "**Options**\n\n"
        "Partner invite — claim at dev.fun; I'll auto-claim `/agent/invitations`.\n\n"
        "MoonPay (card / Apple Pay):\n"
        f"`https://buy.moonpay.com/?currencyCode=mon_mon&walletAddress={wallet_address}&baseCurrencyCode=usd`\n\n"
        f"Send **{amount} {currency}** to:\n`{wallet_address}`\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "I'll retry join when balance is sufficient."
    )


def format_heartbeat(
    agent_name: str,
    competition: Dict[str, Any],
    hands_played: int,
    win_rate: float,
    rank: Optional[int],
    total_agents: Optional[int],
    inbox_count: int,
    adaptive_summary: str,
) -> str:
    comp_name = competition.get("name", "Arena")
    rank_line = (
        f"Leaderboard: **#{rank}** of {total_agents}"
        if rank and total_agents
        else "Leaderboard: polling…"
    )
    return (
        f"🏟 **{agent_name}** — arena heartbeat\n\n"
        f"Competing in **{comp_name}** (season {competition.get('seasonNumber', '?')}).\n\n"
        f"**Session:** {hands_played} decisions · recent win rate {win_rate:.0%}\n"
        f"{rank_line}\n\n"
        f"**Learning:** {adaptive_summary}\n\n"
        + (
            f"Inbox: {inbox_count} unread message(s)."
            if inbox_count
            else "Inbox: clear."
        )
    )
