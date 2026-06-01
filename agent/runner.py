"""agent/runner.py — async polling loop with arena.md onboarding + heartbeat."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from asyncio import Lock
from typing import Any, Dict, List, Optional

from loguru import logger

from agent.arbiter import StrategyArbiter
from agent.arena_setup import (
    continuous_arena_notice,
    list_competitions,
    onboard_and_join,
    register_agent,
)
from agent.heartbeat import HeartbeatState, run_heartbeat
from agent.health import start_health_server
from api.arena_client import ArenaClient, ArenaAPIError
from api.hand_buffer import HandHistoryBuffer
from api.hand_processor import process_full_hand
from api.action_amount import format_submission_amount
from api.state_parser import parse_table, extract_table_id, normalize_allowed_actions
from agent.session_learner import on_hand_complete
from config.logging_setup import configure_logging
from config.settings import settings
from models.adaptive_memory import AdaptiveMemory
from agent.ab_test import ABTestRunner
from agent.brutal_check import BrutalSelfCheck
from agent.decision_log import log_decision
from agent.meta_learner import MetaLearner
from agent.performance_dashboard import PerformanceDashboard
from agent.play_stats import PlayStatsLogger
from agent.self_test_runner import run_startup_self_tests
from api.action_amount import build_action_payload
from models.opponent_tracker import OpponentTracker

STRATEGY_VERSION = "v4-brutal"


def safe_fallback(allowed_actions) -> tuple[str, float]:
    acts = _action_names(allowed_actions)
    if "check" in acts:
        return "check", 0
    return "fold", 0


def _action_names(allowed_actions) -> set[str]:
    if isinstance(allowed_actions, dict):
        return {str(a).lower() for a in (allowed_actions.get("availableActions") or [])}
    if isinstance(allowed_actions, list):
        return {a.get("action", "").lower() for a in allowed_actions if isinstance(a, dict)}
    return set()


class PokerRunner:
    def __init__(
        self,
        competition_id: str = "",
        dry_run: bool = False,
        max_hands: Optional[int] = None,
    ):
        self.competition_id = competition_id or settings.arena_competition_id
        self.dry_run = dry_run
        self.max_hands = max_hands
        self.hands_played = 0
        self._running = True
        self.payouts: Optional[List[float]] = None
        self.competition_meta: Dict[str, Any] = {}
        self._prev_hand: Dict[str, Any] = {}
        self._hand_buffer = HandHistoryBuffer()
        self._decision_times: List[float] = []
        self._heartbeat_state = HeartbeatState(settings.heartbeat_state_file)
        self._last_heartbeat_loop = time.time()
        self._last_poll = 0.0
        self._active_tables: set = set()
        self._shutting_down = False
        self._pending_actions = 0
        self._table_locks: Dict[str, Lock] = {}
        self._action_tx_log: List[Dict[str, Any]] = []
        self._started_at = time.time()

        self.tracker = OpponentTracker(settings.state_file)
        self.adaptive = AdaptiveMemory(settings.adaptive_state_file)
        self.brutal = BrutalSelfCheck()
        self.meta = MetaLearner()
        self.ab_test = ABTestRunner()
        self.arbiter = StrategyArbiter(
            self.tracker, self.adaptive, self.meta, self.brutal
        )
        self.play_stats = PlayStatsLogger(settings.decisions_log_file)
        self.dashboard = PerformanceDashboard()
        self.client = ArenaClient(
            api_key=settings.arena_api_key,
            agent_id=settings.arena_agent_id,
            base_url=settings.arena_base_url,
            credentials_file=settings.credentials_file,
        )

        self._health = start_health_server(
            settings.health_port, self._health_status, self._metrics_status
        )

    def _health_status(self) -> dict:
        avg_ms = (
            sum(self._decision_times) / len(self._decision_times)
            if self._decision_times
            else 0
        )
        idle = time.time() - self.brutal.last_action_at
        degraded = (
            self.brutal.api_errors_400 > 0
            or idle > settings.watchdog_idle_s
            or self.brutal.should_rollback()
        )
        return {
            "status": "degraded" if degraded else "ok",
            "uptime_s": round(time.time() - self._started_at, 1),
            "strategy_version": STRATEGY_VERSION,
            "hands_played": self.hands_played,
            "win_rate": round(self.adaptive.recent_win_rate(), 3),
            "active_tables": list(self._active_tables),
            "current_strategy": self.arbiter.current_mode,
            "meta_strategy": self.meta.active,
            "avg_decision_ms": round(avg_ms, 1),
            "competition_id": self.competition_id,
            "adaptive": self.adaptive.summary(),
            "last_heartbeat": self._heartbeat_state.last_heartbeat_at,
            **self.brutal.health_dict(),
        }

    def _metrics_status(self) -> dict:
        return {
            "dashboard": self.dashboard.snapshot(),
            "meta_scoreboard": self.meta.scoreboard(),
            "bot_validation": self.arbiter.detector.validate_classifications(
                {
                    oid: self.tracker.get(oid)
                    for oid in self.tracker.all_archetypes()
                }
            ),
            "brutal": self.brutal.metrics_dict(),
            "ab_tests": {
                k: {"hands": v.hands, "bb_per_hand": v.bb_per_hand}
                for k, v in self.ab_test._variants.items()
            },
        }

    def _shutdown(self, *_):
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("Shutdown — finishing pending actions then saving state")
        deadline = time.time() + 8.0
        while self._pending_actions > 0 and time.time() < deadline:
            time.sleep(0.1)
        self.tracker.save()
        self.adaptive.save()
        self.meta.save()
        self.adaptive.export_strategy_report()
        self.brutal.save()
        self.meta.save()
        self.ab_test.save()
        self._save_action_tx_log()
        self._heartbeat_state.save()
        self._running = False

    def _save_action_tx_log(self) -> None:
        path = settings.action_tx_log
        if not self._action_tx_log:
            return
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._action_tx_log[-500:], f, indent=2)
        os.replace(tmp, path)

    def _table_lock(self, table_id: str) -> Lock:
        if table_id not in self._table_locks:
            self._table_locks[table_id] = Lock()
        return self._table_locks[table_id]

    def _watchdog_check(self) -> None:
        idle = time.time() - self.brutal.last_action_at
        if idle > settings.watchdog_idle_s and self._running:
            logger.warning(f"Watchdog: no action for {idle:.0f}s")
            self.brutal._alert(f"watchdog idle {idle:.0f}s")

    def setup(self) -> bool:
        try:
            cid, meta, client = onboard_and_join(
                competition_id=self.competition_id,
                dry_run=self.dry_run,
                force_heartbeat=True,
            )
            self.client = client
            self.competition_id = cid or ""
            self.competition_meta = meta or {}
        except ArenaAPIError as e:
            logger.error(f"Arena setup failed: {e}")
            return False
        except RuntimeError as e:
            logger.error(str(e))
            return False

        from agent.owner_messages import write_owner_message
        write_owner_message(settings.owner_message_file, continuous_arena_notice())

        self._load_payouts()
        self.meta.reset_match()
        return bool(self.competition_id)

    def _rate_limit_poll(self) -> None:
        elapsed = time.time() - self._last_poll
        wait = max(0.0, settings.min_poll_interval_s - elapsed)
        if wait > 0:
            time.sleep(wait)
        self._last_poll = time.time()

    async def _async_rate_limit_poll(self) -> None:
        elapsed = time.time() - self._last_poll
        wait = max(0.0, settings.min_poll_interval_s - elapsed)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_poll = time.time()

    def _refresh_match_context(self) -> None:
        if not self.competition_id:
            return
        try:
            st = self.client.get_benchmark_status(self.competition_id)
            match = st.get("match") or st
            hands = int(match.get("completedHands") or self.hands_played)
            target = int(match.get("targetHands") or 500)
            seats_alive = len(
                [
                    s
                    for s in (st.get("table") or {}).get("seats", [])
                    if str(s.get("status", "")).lower() not in ("eliminated", "out")
                ]
            )
            self.arbiter.set_match_context(
                hands,
                target,
                seats_alive or self.arbiter.players_remaining,
            )
            self.match_hands_played = hands
        except Exception:
            pass

    def _load_payouts(self) -> None:
        if not self.competition_id:
            return
        try:
            comp = self.client.get_competition(self.competition_id)
            self.payouts = self.client.parse_payouts(comp)
            if self.payouts:
                logger.info(f"ICM payouts loaded: {self.payouts}")
        except Exception as e:
            logger.warning(f"Could not load payouts: {e}")

    def _maybe_heartbeat(self, force: bool = False) -> None:
        if not self.competition_id:
            return
        run_heartbeat(
            self.client,
            self.competition_id,
            self.competition_meta,
            agent_name=settings.agent_name,
            hands_played=self.hands_played,
            win_rate=self.adaptive.recent_win_rate(),
            adaptive_summary=self.adaptive.summary(),
            owner_message_file=settings.owner_message_file,
            force=force,
            state=self._heartbeat_state,
            min_interval_s=settings.heartbeat_min_interval_s,
        )

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        if settings.use_async:
            asyncio.run(self._async_run())
        else:
            self._sync_run()

    def _sync_run(self) -> None:
        prev_states: dict = {}
        self._maybe_heartbeat(force=False)
        while self._running:
            if self.max_hands and self.hands_played >= self.max_hands:
                break
            if time.time() - self._last_heartbeat_loop >= settings.heartbeat_interval_s:
                self._maybe_heartbeat()
                self._last_heartbeat_loop = time.time()
            self._refresh_match_context()
            self._watchdog_check()
            self._rate_limit_poll()
            try:
                tables = self.client.get_pending_actions(self.competition_id)
            except ArenaAPIError as e:
                logger.warning(f"Poll error {e.status}")
                time.sleep(self.client._breaker.backoff_sleep())
                continue
            if not tables:
                time.sleep(settings.poll_interval_s)
                continue
            tables = sorted(
                tables,
                key=lambda t: t.get("actionDeadline") or t.get("deadline") or 9e18,
            )
            for table in tables:
                self._handle_table_sync(table, prev_states)
            time.sleep(0.2)

    async def _async_run(self) -> None:
        prev_states: dict = {}
        self._maybe_heartbeat(force=False)
        while self._running:
            if self.max_hands and self.hands_played >= self.max_hands:
                break
            if time.time() - self._last_heartbeat_loop >= settings.heartbeat_interval_s:
                self._maybe_heartbeat()
                self._last_heartbeat_loop = time.time()
            self._refresh_match_context()
            self._watchdog_check()
            await self._async_rate_limit_poll()
            try:
                tables = await self.client.async_get_pending_actions(
                    self.competition_id
                )
            except ArenaAPIError as e:
                logger.warning(f"Poll error {e.status}")
                await asyncio.sleep(self.client._breaker.backoff_sleep())
                continue
            except Exception as e:
                logger.warning(f"Poll exception: {e}")
                await asyncio.sleep(settings.poll_interval_s)
                continue

            if not tables:
                await asyncio.sleep(settings.poll_interval_s)
                continue

            tables = sorted(
                tables,
                key=lambda t: t.get("actionDeadline") or t.get("deadline") or 9e18,
            )
            await asyncio.gather(
                *[self._handle_table_async(t, prev_states) for t in tables]
            )
            await asyncio.sleep(0.15)

        await self.client.aclose()

    def _buffer_and_finalize(self, table: dict, table_id: str, prev_states: dict) -> None:
        self._hand_buffer.ingest(table, table_id)
        prev = prev_states.get(table_id, {})
        prev_hand = self._prev_hand.get(table_id) or prev.get("handNumber") or prev.get("handId")
        curr_hand = table.get("handNumber") or table.get("handId")

        if prev_hand and curr_hand and prev_hand != curr_hand:
            _, actions = self._hand_buffer.pop_completed(table_id, prev_hand)
            seats = prev.get("seats") or prev.get("players", [])
            winners = table.get("winners") or prev.get("winners")
            if actions:
                process_full_hand(
                    actions, seats, self.client.agent_id, self.tracker, winners
                )
            else:
                from api.hand_processor import process_completed_hand
                process_completed_hand(prev, self.client.agent_id, self.tracker)

            on_hand_complete(
                self.adaptive,
                table_id,
                prev_hand,
                prev,
                table,
                self.client.agent_id,
                meta_learner=self.meta,
                meta_strategy=self.meta.active,
                dashboard=self.dashboard,
                strategy_mode=self.arbiter.current_mode,
                brutal_check=self.brutal,
                baseline_ev=float(prev.get("pot") or prev.get("potChips") or 0) * 0.1,
            )
            self.match_hands_played = max(self.match_hands_played, self.hands_played)

        if curr_hand:
            self._prev_hand[table_id] = curr_hand
        self._hand_buffer.prune_stale()
        prev_states[table_id] = table

    def _handle_table_sync(self, table: dict, prev_states: dict) -> None:
        table_id = extract_table_id(table)
        with self._table_lock(table_id):
            self._buffer_and_finalize(table, table_id, prev_states)
            self._decide_and_submit(table, table_id)

    async def _handle_table_async(self, table: dict, prev_states: dict) -> None:
        table_id = extract_table_id(table)
        async with self._table_lock(table_id):
            self._buffer_and_finalize(table, table_id, prev_states)
            deadline = table.get("actionDeadline") or table.get("deadline") or 0
        budget = settings.decision_budget_s
        if deadline:
            budget = min(budget, max(0.2, deadline - time.time() - 0.2))
            try:
                await asyncio.wait_for(
                    self._decide_and_submit_async(table, table_id, deadline),
                    timeout=budget,
                )
            except asyncio.TimeoutError:
                allowed = table.get("allowedActions") or []
                action, amount = safe_fallback(allowed)
                await self._submit_async(
                    table_id, action, amount, table, "Hard timeout — safe fallback"
                )

    def _decide_and_submit(self, table: dict, table_id: str) -> None:
        if self._shutting_down:
            return
        self._active_tables.add(table_id)
        self._pending_actions += 1
        try:
            allowed = table.get("allowedActions") or []
            ctx = parse_table(table, self.client.agent_id, payouts=self.payouts)
            if ctx is None:
                action, amount = safe_fallback(allowed)
                self._submit_sync(table_id, action, amount, table, "Parse fallback")
                return

            if ctx.hand_number:
                self.arbiter.on_new_hand(table_id, ctx.hand_number, ctx.stack)

            deadline = table.get("actionDeadline") or table.get("deadline") or 0
            if deadline and (deadline - time.time()) < 0.25:
                action, amount = safe_fallback(allowed)
                self._submit_sync(table_id, action, amount, table, "Clock emergency")
                return

            t0 = time.monotonic()
            try:
                action, amount, chat = self.arbiter.decide(ctx, deadline=deadline or 0)
            except Exception as e:
                logger.exception(f"Arbiter error: {e}")
                action, amount, chat = safe_fallback(allowed)[0], 0, "Error fallback"

            self._decision_times.append((time.monotonic() - t0) * 1000)
            if len(self._decision_times) > 500:
                self._decision_times = self._decision_times[-200:]

            action, amount = self._validate_action(
                action, amount, ctx.allowed_actions, ctx.stack
            )
            self.play_stats.record_decision(
                table_id=table_id,
                hand_number=ctx.hand_number,
                position=ctx.position,
                hole=ctx.hole_cards,
                street=ctx.street,
                action=action,
                amount=amount,
                is_facing_raise=ctx.is_facing_raise,
            )
            log_decision(
                table_id=table_id,
                hand_number=ctx.hand_number,
                strategy_mode=self.arbiter.current_mode,
                meta_strategy=self.meta.active,
                action=action,
                amount=amount,
                ehs=self.arbiter._last_ehs,
                decision_time_ms=(time.monotonic() - t0) * 1000,
                opponent_ids=ctx.opponent_ids,
            )
            self._submit_sync(table_id, action, amount, table, chat)
        finally:
            self._pending_actions = max(0, self._pending_actions - 1)
            self._active_tables.discard(table_id)

    async def _decide_and_submit_async(
        self, table: dict, table_id: str, deadline: float
    ) -> None:
        if self._shutting_down:
            return
        self._active_tables.add(table_id)
        self._pending_actions += 1
        try:
            allowed = table.get("allowedActions") or []
            ctx = parse_table(table, self.client.agent_id, payouts=self.payouts)
            if ctx is None:
                action, amount = safe_fallback(allowed)
                await self._submit_async(table_id, action, amount, table, "Parse fallback")
                return

            if ctx.hand_number:
                self.arbiter.on_new_hand(table_id, ctx.hand_number, ctx.stack)

            if deadline and (deadline - time.time()) < 0.25:
                action, amount = safe_fallback(allowed)
                await self._submit_async(table_id, action, amount, table, "Clock emergency")
                return

            loop = asyncio.get_event_loop()
            t0 = time.monotonic()

            def _decide():
                return self.arbiter.decide(ctx, deadline=deadline or 0)

            try:
                action, amount, chat = await loop.run_in_executor(None, _decide)
            except Exception as e:
                logger.exception(f"Arbiter error: {e}")
                action, amount, chat = safe_fallback(allowed)[0], 0, "Error fallback"

            self._decision_times.append((time.monotonic() - t0) * 1000)
            if len(self._decision_times) > 500:
                self._decision_times = self._decision_times[-200:]

            action, amount = self._validate_action(
                action, amount, ctx.allowed_actions, ctx.stack
            )
            self.play_stats.record_decision(
                table_id=table_id,
                hand_number=ctx.hand_number,
                position=ctx.position,
                hole=ctx.hole_cards,
                street=ctx.street,
                action=action,
                amount=amount,
                is_facing_raise=ctx.is_facing_raise,
            )
            log_decision(
                table_id=table_id,
                hand_number=ctx.hand_number,
                strategy_mode=self.arbiter.current_mode,
                meta_strategy=self.meta.active,
                action=action,
                amount=amount,
                ehs=self.arbiter._last_ehs,
                decision_time_ms=(time.monotonic() - t0) * 1000,
                opponent_ids=ctx.opponent_ids,
            )
            await self._submit_async(table_id, action, amount, table, chat)
        finally:
            self._pending_actions = max(0, self._pending_actions - 1)
            self._active_tables.discard(table_id)

    def _validate_action(
        self,
        action: str,
        amount: float,
        allowed: Any,
        stack: float,
    ) -> tuple[str, float]:
        allowed_list = (
            allowed
            if isinstance(allowed, list)
            else normalize_allowed_actions(allowed)
        )
        allowed_names = _action_names(allowed_list)
        if action.lower() in allowed_names:
            return action, amount
        if action == "raise" and "bet" in allowed_names:
            return "bet", amount
        if action == "bet" and "raise" in allowed_names:
            return "raise", amount
        return safe_fallback(allowed_list)

    def _submit_sync(
        self, table_id: str, action: str, amount: float, table: dict, chat: str
    ) -> None:
        amount = format_submission_amount(
            action, amount, table, self.client.agent_id
        )
        if self.dry_run:
            logger.info(f"[DRY-RUN] {action} {amount:.1f} | {chat[:80]}")
            return
        payload = build_action_payload(
            table_id, action, amount, table, self.client.agent_id, chat
        )
        tx_id = f"{table_id}:{time.time():.3f}"
        self._action_tx_log.append({"id": tx_id, "payload": payload, "ts": time.time()})
        try:
            self.client.submit_action_safe(table_id, action, amount, chat)
            self.brutal.record_api_call(True, 200)
            logger.info(f"✓ {action.upper()} {amount:.1f} | {chat[:80]}")
            self.hands_played += 1
        except ArenaAPIError as e:
            self.brutal.record_api_call(False, e.status)
            logger.error(f"Submit failed {e.status}: {e.body}")

    async def _submit_async(
        self, table_id: str, action: str, amount: float, table: dict, chat: str
    ) -> None:
        amount = format_submission_amount(
            action, amount, table, self.client.agent_id
        )
        if self.dry_run:
            logger.info(f"[DRY-RUN] {action} {amount:.1f} | {chat[:80]}")
            return
        payload = build_action_payload(
            table_id, action, amount, table, self.client.agent_id, chat
        )
        self._action_tx_log.append(
            {"id": f"{table_id}:{time.time():.3f}", "payload": payload, "ts": time.time()}
        )
        try:
            await self.client.async_submit_action_safe(table_id, action, amount, chat)
            self.brutal.record_api_call(True, 200)
            logger.info(f"✓ {action.upper()} {amount:.1f} | {chat[:80]}")
            self.hands_played += 1
        except ArenaAPIError as e:
            self.brutal.record_api_call(False, e.status)
            logger.error(f"Submit failed {e.status}: {e.body}")


def main():
    parser = argparse.ArgumentParser(description="Plutus — dev.fun Arena Poker Agent")
    parser.add_argument("--competition-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-hands", type=int, default=None)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument(
        "--register",
        nargs="+",
        metavar=("NAME", "QUOTE"),
        help='Register: --register "Plutus" "optional quote"',
    )
    parser.add_argument(
        "--list-competitions",
        action="store_true",
        help="List live competitions (arena.md picker)",
    )
    parser.add_argument(
        "--onboard",
        action="store_true",
        help="Run Step 0 + join lobby without playing",
    )
    parser.add_argument(
        "--heartbeat",
        action="store_true",
        help="Send one owner heartbeat now",
    )
    args = parser.parse_args()

    configure_logging(
        settings.log_level,
        json_logs=True,
        log_file=settings.log_file,
    )

    if not args.register and not args.list_competitions and not args.onboard and not args.heartbeat:
        run_startup_self_tests()

    if args.register:
        name = args.register[0]
        quote = args.register[1] if len(args.register) > 1 else settings.agent_quote
        register_agent(name, quote)
        sys.exit(0)

    if args.list_competitions:
        list_competitions()
        sys.exit(0)

    if args.onboard:
        try:
            onboard_and_join(
                competition_id=args.competition_id,
                dry_run=args.dry_run,
                force_heartbeat=True,
            )
        except ArenaAPIError as e:
            logger.error(f"Onboard failed: {e}")
            sys.exit(1)
        sys.exit(0)

    if args.heartbeat:
        client = ArenaClient(
            api_key=settings.arena_api_key,
            base_url=settings.arena_base_url,
            credentials_file=settings.credentials_file,
        )
        if settings.arena_api_key:
            client._api_key = settings.arena_api_key
        client.load_credentials()
        cid = args.competition_id or settings.arena_competition_id
        meta = client.get_competition(cid) if cid else {}
        run_heartbeat(
            client,
            cid,
            meta,
            force=True,
            owner_message_file=settings.owner_message_file,
        )
        sys.exit(0)

    if args.sync:
        settings.use_async = False

    runner = PokerRunner(
        competition_id=args.competition_id,
        dry_run=args.dry_run,
        max_hands=args.max_hands,
    )
    if not runner.setup():
        sys.exit(1)
    runner.run()
    logger.info("Agent stopped.")


if __name__ == "__main__":
    main()
