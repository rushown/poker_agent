"""api/arena_onboarding.py — dev.fun Arena onboarding per /skills/arena.md."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from api.arena_client import ArenaAPIError, ArenaClient
from config.settings import settings

POKER_GAME_TYPES = frozenset({"texasholdem", "pokereval"})


def pick_competition(
    competitions: List[Dict[str, Any]],
    competition_id: str = "",
    game_type_hint: str = "",
) -> Optional[Dict[str, Any]]:
    """Select competition per arena.md fallback order."""
    if not competitions:
        return None

    if competition_id:
        for c in competitions:
            if str(c.get("id", "")) == competition_id:
                return c
        return None

    pool = competitions
    if game_type_hint:
        gt = game_type_hint.lower().replace("_", "").replace(" ", "")
        filtered = [
            c for c in competitions
            if gt in str(c.get("gameType", "")).lower().replace("_", "").replace(" ", "")
        ]
        if filtered:
            pool = filtered
    else:
        poker = [
            c for c in competitions
            if _normalize_game_type(c.get("gameType", "")) in POKER_GAME_TYPES
            or "holdem" in str(c.get("gameType", "")).lower()
            or "poker" in str(c.get("gameType", "")).lower()
        ]
        if poker:
            pool = poker

    def sort_key(c: Dict) -> Tuple:
        start = c.get("startAt") or c.get("launchAt") or 0
        season = c.get("seasonNumber") or 0
        return (start, season)

    return max(pool, key=sort_key)


def _normalize_game_type(gt: str) -> str:
    return str(gt).lower().replace("_", "").replace(" ", "")


def format_competition_list(competitions: List[Dict[str, Any]]) -> str:
    lines = ["Multiple live arenas are available:", ""]
    for i, c in enumerate(competitions[:10], 1):
        skill = c.get("skillFile") or _default_skill_file(c.get("gameType", ""))
        mode = competition_mode_label(c)
        lines.append(
            f"{i}. `{c.get('name', '?')}` — `{c.get('gameType')}`, "
            f"season `{c.get('seasonNumber', '?')}`, id `{c.get('id')}`\n"
            f"   mode: {mode}"
            + (f", skill `{skill}`" if skill else "")
        )
    lines.append("\nSet ARENA_COMPETITION_ID to choose one.")
    return "\n".join(lines)


def _default_skill_file(game_type: str) -> str:
    gt = _normalize_game_type(game_type)
    if gt == "pokereval":
        return "/skills/poker-eval.md"
    if gt in POKER_GAME_TYPES or "holdem" in gt:
        return "/skills/texas-holdem.md"
    return ""


def is_benchmark_competition(comp: Dict[str, Any]) -> bool:
    """Poker Eval / PVE uses /texas/benchmark/start, not /texas/join."""
    gt = _normalize_game_type(comp.get("gameType", ""))
    if gt == "pokereval":
        return True
    skill = str(comp.get("skillFile") or "").lower()
    if "poker-eval" in skill or "pokereval" in skill:
        return True
    cid = str(comp.get("id") or "").lower()
    name = str(comp.get("name") or "").lower()
    if "poker_eval" in cid or "pokereval" in cid or ("eval" in name and "poker" in name):
        return True
    mode = str(comp.get("mode") or comp.get("competitionMode") or "").lower()
    if mode in ("benchmark", "pve", "eval"):
        return True
    return False


def competition_mode_label(comp: Dict[str, Any]) -> str:
    return "benchmark (PVE)" if is_benchmark_competition(comp) else "matchmaking lobby"


def claim_pending_invitations(client: ArenaClient) -> int:
    """Claim all pending partner invitations. Returns count claimed."""
    try:
        data = client.get_invitations()
    except ArenaAPIError as e:
        logger.debug(f"Invitations check failed: {e}")
        return 0

    invitations = data.get("invitations") or data if isinstance(data, list) else []
    if isinstance(data, dict) and not invitations:
        invitations = []

    claimed = 0
    for inv in invitations:
        rid = inv.get("id") or inv.get("redemptionId")
        if not rid:
            continue
        try:
            client.claim_invitation(str(rid))
            partner = inv.get("partnerName", "partner")
            amount = inv.get("rewardAmount", "?")
            logger.info(f"Claimed invitation from {partner}: {amount}")
            claimed += 1
        except ArenaAPIError as e:
            logger.warning(f"Invitation claim failed: {e}")
    return claimed


def start_benchmark_match(client: ArenaClient, competition_id: str) -> Dict[str, Any]:
    """Start or resume Poker Eval benchmark; then poll pending-actions."""
    claim_pending_invitations(client)
    try:
        result = client.start_benchmark(competition_id)
    except ArenaAPIError as e:
        if e.status == 409:
            logger.info("Benchmark match already exists — resuming via status endpoint")
            result = client.get_benchmark_status(competition_id)
        elif e.status == 403:
            claim = client.get_claim_status()
            logger.error(f"Claim required: {claim.get('claimUrl', claim)}")
            raise
        else:
            raise

    match = result.get("match") or {}
    logger.info(
        f"Benchmark match: status={match.get('status')} "
        f"phase={match.get('phase')} "
        f"hands={match.get('completedHands', 0)}/{match.get('targetHands', '?')}"
    )
    return result


def join_competition(
    client: ArenaClient,
    competition_id: str,
    competition_meta: Optional[Dict[str, Any]] = None,
    *,
    max_payment_wait_s: float = 120.0,
) -> Dict[str, Any]:
    """Enter competition: benchmark start OR matchmaking lobby join."""
    if competition_meta and is_benchmark_competition(competition_meta):
        return start_benchmark_match(client, competition_id)

    claim_pending_invitations(client)

    try:
        return client.join_table(competition_id)
    except ArenaAPIError as e:
        if e.status == 409:
            err_text = str(e.body).lower()
            if "benchmark" in err_text:
                logger.info("Routing to /texas/benchmark/start (eval competition)")
                return start_benchmark_match(client, competition_id)
        if e.status == 403:
            claim = client.get_claim_status()
            logger.error(
                "Agent must be claimed by X-verified owner. "
                f"Claim URL: {claim.get('claimUrl', claim)}"
            )
            raise
        if e.status != 402:
            raise

    body = e.body if isinstance(e.body, dict) else {}
    if isinstance(body, str):
        try:
            import json
            body = json.loads(body)
        except Exception:
            body = {}
    payment = body.get("paymentRequirements") or body
    if not payment:
        raise ArenaAPIError(402, body)

    chain = payment.get("chain", "monad")
    to_addr = payment.get("to", "")
    amount = str(payment.get("amount", "0"))
    currency = payment.get("currency", "MON")

    logger.info(
        f"Entry fee: {amount} {currency} on {chain} → {to_addr[:10]}..."
    )

    tx_hash = _pay_entry_fee(client, chain, to_addr, amount, max_payment_wait_s)
    return client.join_table(competition_id, tx_hash=tx_hash)


def enter_competition(
    client: ArenaClient,
    competition_id: str,
    competition_meta: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Unified entry: picks benchmark vs lobby automatically."""
    return join_competition(
        client, competition_id, competition_meta, **kwargs
    )


def _pay_entry_fee(
    client: ArenaClient,
    chain: str,
    to_addr: str,
    amount: str,
    max_wait_s: float,
) -> str:
    claim_pending_invitations(client)

    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        wallet = client.get_wallet(chain)
        balance = float(wallet.get("nativeBalance") or wallet.get("balance") or 0)
        need = float(amount)

        if balance >= need and to_addr:
            result = client.transfer_native(chain, to_addr, amount)
            tx_hash = (
                result.get("txHash")
                or result.get("hash")
                or result.get("transactionHash")
                or ""
            )
            if tx_hash:
                logger.info(f"Entry fee paid tx={tx_hash}")
                return str(tx_hash)
            logger.warning(f"Transfer response missing txHash: {result}")

        claim_pending_invitations(client)
        time.sleep(10)

    raise ArenaAPIError(
        402,
        {
            "error": "Insufficient wallet balance for entry fee",
            "needed": amount,
            "chain": chain,
            "hint": "Claim /agent/invitations, fund wallet, or set ARENA_INVITE_CODE for faucet",
        },
    )


def setup_arena_session(
    client: ArenaClient,
    competition_id: str = "",
    game_type_hint: str = "",
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Step 0 returning-player flow: verify, pick competition, return metadata."""
    if not client.load_credentials() and not client.is_authenticated:
        return None, None

    try:
        me = client.get_me()
        agent_id = (
            me.get("agentId")
            or me.get("id")
            or client.agent_id
            or settings.arena_agent_id
        )
        if agent_id and not client._agent_id:
            client._agent_id = str(agent_id)
        logger.info(f"Arena agent: {me.get('name')} ({agent_id})")
    except ArenaAPIError:
        return None, None

    try:
        client.introspect()
    except ArenaAPIError as e:
        logger.warning(f"Introspection failed: {e}")

    claim_pending_invitations(client)

    comps = client.list_active_competitions()
    if len(comps) > 1 and not competition_id:
        logger.info(format_competition_list(comps))

    chosen = pick_competition(comps, competition_id, game_type_hint)
    if not chosen:
        logger.warning("No active competition found")
        return None, None

    cid = str(chosen.get("id", ""))
    skill = chosen.get("skillFile") or _default_skill_file(chosen.get("gameType", ""))
    if skill:
        logger.info(f"Competition skill file (read for rules): {skill}")

    return cid, chosen
