"""agent/arena_setup.py — full arena.md Step 0 + join flow for CLI and runner."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from agent.heartbeat import HeartbeatState, run_heartbeat
from agent.owner_messages import (
    derive_handle,
    format_competition_picker,
    format_entry_fee_help,
    format_invitation_offer,
    format_registration_success,
    write_owner_message,
)
from api.arena_client import ArenaAPIError, ArenaClient
from api.arena_onboarding import (
    claim_pending_invitations,
    competition_mode_label,
    enter_competition,
    format_competition_list,
    is_benchmark_competition,
    pick_competition,
    setup_arena_session,
)
from config.settings import settings


def register_agent(name: str, quote: str = "", handle: str = "") -> Dict[str, Any]:
    """Register per arena.md (name → handle, show API key once)."""
    client = ArenaClient(
        base_url=settings.arena_base_url,
        credentials_file=settings.credentials_file,
    )
    h = handle or derive_handle(name)
    q = quote or settings.agent_quote

    for attempt in range(4):
        try:
            result = client.register(handle=h, name=name, quote=q)
            break
        except ArenaAPIError as e:
            if e.status == 409 and attempt < 3:
                import random
                import string
                suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=2))
                h = (derive_handle(name)[:28] + suffix)[:30]
                continue
            raise
    else:
        raise RuntimeError("Registration failed after retries")

    agent_id, api_key = _credentials_from_register(result, client)
    if not api_key:
        logger.error("Registration response missing apiKey — check API response")
        return result

    claim_url = ""
    try:
        claim = client.get_claim_status()
        claim_url = claim.get("claimUrl") or claim.get("url") or ""
    except ArenaAPIError:
        pass

    sync_env_file(".env", api_key, agent_id)

    msg = format_registration_success(agent_id, api_key, claim_url)
    write_owner_message(settings.owner_message_file, msg, log_summary=False)

    # Print once to stdout (avoid duplicate/redacted loguru sinks)
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

    if not api_key.startswith("arena_sk_"):
        logger.warning("API key format unexpected — verify against arena.md")
    logger.info(
        f"Saved credentials to {settings.credentials_file} and updated .env"
    )
    return result


def _credentials_from_register(result: Dict[str, Any], client: ArenaClient) -> Tuple[str, str]:
    api_key = result.get("apiKey") or result.get("api_key") or client._api_key or ""
    agent_id = result.get("agentId") or result.get("agent_id") or client._agent_id or ""
    return str(agent_id), str(api_key)


def sync_env_file(env_path: str, api_key: str, agent_id: str) -> None:
    """Update or create .env with arena credentials."""
    path = Path(env_path)
    lines: List[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    updates = {
        "ARENA_API_KEY": api_key,
        "ARENA_AGENT_ID": agent_id,
    }
    seen = set()
    out: List[str] = []
    for line in lines:
        m = re.match(r"^([A-Z_]+)\s*=", line)
        if m and m.group(1) in updates:
            key = m.group(1)
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, val in updates.items():
        if key not in seen:
            out.append(f"{key}={val}")
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def list_competitions(client: Optional[ArenaClient] = None) -> List[Dict[str, Any]]:
    client = client or _authenticated_client()
    comps = client.list_active_competitions()
    text = format_competition_picker(comps) if comps else "No active competitions."
    print(text)
    write_owner_message(settings.owner_message_file, text)
    return comps


def onboard_and_join(
    competition_id: str = "",
    *,
    dry_run: bool = False,
    force_heartbeat: bool = False,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], ArenaClient]:
    """Complete Step 0: credentials → invitations → pick comp → join → heartbeat."""
    client = _authenticated_client()

    cid, meta = setup_arena_session(
        client,
        competition_id=competition_id or settings.arena_competition_id,
        game_type_hint="TexasHoldem",
    )
    if not cid or not meta:
        raise RuntimeError("No competition available or auth failed")

    comps = client.list_active_competitions()
    if len(comps) > 1 and not (competition_id or settings.arena_competition_id):
        write_owner_message(
            settings.owner_message_file,
            format_competition_picker(comps),
        )

    # Surface unclaimed invitations (arena.md)
    try:
        inv_data = client.get_invitations()
        invitations = inv_data.get("invitations") or []
        for inv in invitations:
            write_owner_message(
                settings.owner_message_file,
                format_invitation_offer(inv),
            )
    except ArenaAPIError:
        pass

    claimed = claim_pending_invitations(client)
    if claimed:
        logger.info(f"Claimed {claimed} partner invitation(s)")

    me = client.get_me()
    agent_name = me.get("name", settings.agent_name)

    if not dry_run:
        try:
            result = enter_competition(client, cid, meta)
            logger.info(f"Entered competition: {result}")
            mode = competition_mode_label(meta)
            if is_benchmark_competition(meta):
                match = result.get("match") or {}
                write_owner_message(
                    settings.owner_message_file,
                    f"✅ Started **Poker Eval** benchmark: **{meta.get('name')}**\n"
                    f"competitionId `{cid}` · mode: {mode}\n"
                    f"Match `{match.get('id', '?')}` — "
                    f"hands {match.get('completedHands', 0)}/"
                    f"{match.get('targetHands', '?')}\n"
                    "Polling `/texas/pending-actions` for decisions.",
                )
            else:
                write_owner_message(
                    settings.owner_message_file,
                    f"✅ Joined lobby for **{meta.get('name')}**.\n"
                    f"competitionId `{cid}` · mode: {mode}\n"
                    "Polling `/texas/pending-actions` for your seat.",
                )
        except ArenaAPIError as e:
            if e.status == 402:
                _surface_payment_help(client, meta, e)
            elif e.status == 403:
                claim = client.get_claim_status()
                write_owner_message(
                    settings.owner_message_file,
                    f"⚠️ Claim required before entry:\n{claim.get('claimUrl', claim)}",
                )
            raise

    hb = HeartbeatState(settings.heartbeat_state_file)
    run_heartbeat(
        client,
        cid,
        meta,
        agent_name=agent_name,
        owner_message_file=settings.owner_message_file,
        force=force_heartbeat,
        state=hb,
        min_interval_s=settings.heartbeat_min_interval_s,
    )

    return cid, meta, client


def _surface_payment_help(
    client: ArenaClient, meta: Dict[str, Any], err: ArenaAPIError
) -> None:
    body = err.body if isinstance(err.body, dict) else {}
    pay = body.get("paymentRequirements") or body
    chain = pay.get("chain", "monad")
    amount = str(pay.get("amount", "?"))
    currency = pay.get("currency", "MON")
    wallet_addr = ""
    try:
        w = client.get_wallet(chain)
        wallet_addr = w.get("address", "")
    except ArenaAPIError:
        pass
    msg = format_entry_fee_help(
        meta.get("name", "competition"),
        amount,
        currency,
        chain,
        wallet_addr,
    )
    write_owner_message(settings.owner_message_file, msg)


def _authenticated_client() -> ArenaClient:
    client = ArenaClient(
        api_key=settings.arena_api_key,
        agent_id=settings.arena_agent_id,
        base_url=settings.arena_base_url,
        credentials_file=settings.credentials_file,
    )
    if settings.arena_api_key:
        client._api_key = settings.arena_api_key
    if not client.load_credentials() and not client.is_authenticated:
        logger.error(
            "Not registered. Run:\n"
            '  python -m agent.runner --register "Plutus" "I count outs, not prayers."\n'
            "Or set ARENA_API_KEY in .env"
        )
        sys.exit(1)
    client.get_me()
    return client


def continuous_arena_notice() -> str:
    return (
        "🏟 **Continuous arena** — this bot keeps polling while it runs.\n\n"
        "For 24/7 play, run under **systemd**, **tmux**, or **cron** "
        f"every {settings.heartbeat_interval_s // 3600}h. "
        "Owner updates go to `.arena-owner-messages.txt`."
    )
