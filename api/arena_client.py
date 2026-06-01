"""api/arena_client.py — sync + async HTTP client with circuit breaker."""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import aiohttp
import httpx
from loguru import logger

from api.error_window import RollingErrorWindow


class ArenaAPIError(Exception):
    def __init__(self, status: int, body: Any):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body}")


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, cooldown_s: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._failures = 0
        self._open_until = 0.0

    def record_success(self) -> None:
        self._failures = 0
        self._open_until = 0.0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._open_until = time.time() + self.cooldown_s
            logger.warning(f"Circuit breaker open for {self.cooldown_s}s")

    def allow(self) -> bool:
        if time.time() < self._open_until:
            return False
        return True

    def backoff_sleep(self) -> float:
        return min(60.0, 2 ** min(self._failures, 5))


class ArenaClient:
    def __init__(
        self,
        api_key: str = "",
        agent_id: str = "",
        base_url: str = "https://arena.dev.fun",
        credentials_file: str = ".arena-credentials",
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.credentials_file = credentials_file
        self._api_key = api_key
        self._agent_id = agent_id
        self._introspection: Optional[Dict] = None
        self._http = httpx.Client(timeout=timeout)
        self._breaker = CircuitBreaker()
        self._error_window = RollingErrorWindow(window_s=60.0, max_rate=0.10)
        self._session: Optional[aiohttp.ClientSession] = None

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-arena-api-key": self._api_key,
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/arena{path}"

    def _handle_response(self, status: int, text: str, is_post: bool = False) -> Any:
        if status == 402:
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                body = text
            raise ArenaAPIError(402, body)
        if status >= 400:
            raise ArenaAPIError(status, text)
        if not text:
            return {}
        return json.loads(text)

    def _check_error_window(self) -> None:
        if self._error_window.should_pause():
            raise ArenaAPIError(
                503,
                f"API error rate {self._error_window.error_rate():.0%} in 60s window",
            )

    def _record_call(self, success: bool, status: int = 200) -> None:
        self._error_window.record(success)
        if success:
            self._breaker.record_success()
        else:
            self._breaker.record_failure()

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        self._check_error_window()
        if not self._breaker.allow():
            raise ArenaAPIError(503, "circuit breaker open")
        try:
            r = self._http.get(self._url(path), headers=self._headers(), params=params)
            data = self._handle_response(r.status_code, r.text)
            self._record_call(True)
            return data
        except ArenaAPIError as e:
            self._record_call(False, e.status)
            raise
        except Exception:
            self._record_call(False)
            raise

    def _post(self, path: str, body: Dict) -> Any:
        self._check_error_window()
        if not self._breaker.allow():
            raise ArenaAPIError(503, "circuit breaker open")
        try:
            r = self._http.post(self._url(path), headers=self._headers(), json=body)
            data = self._handle_response(r.status_code, r.text, is_post=True)
            self._record_call(True)
            return data
        except ArenaAPIError as e:
            self._record_call(False, e.status)
            raise
        except Exception:
            self._record_call(False)
            raise

    def _patch(self, path: str, body: Dict) -> Any:
        r = self._http.patch(self._url(path), headers=self._headers(), json=body)
        return self._handle_response(r.status_code, r.text)

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def async_get(self, path: str, params: Optional[Dict] = None) -> Any:
        if not self._breaker.allow():
            raise ArenaAPIError(503, "circuit breaker open")
        session = await self._ensure_session()
        try:
            async with session.get(
                self._url(path), headers=self._headers(), params=params
            ) as resp:
                text = await resp.text()
                data = self._handle_response(resp.status, text)
                self._breaker.record_success()
                return data
        except ArenaAPIError:
            self._breaker.record_failure()
            raise

    async def async_post(self, path: str, body: Dict) -> Any:
        if not self._breaker.allow():
            raise ArenaAPIError(503, "circuit breaker open")
        session = await self._ensure_session()
        try:
            async with session.post(
                self._url(path), headers=self._headers(), json=body
            ) as resp:
                text = await resp.text()
                data = self._handle_response(resp.status, text, is_post=True)
                self._breaker.record_success()
                return data
        except ArenaAPIError:
            self._breaker.record_failure()
            raise

    def save_credentials(self) -> None:
        with open(self.credentials_file, "w") as f:
            json.dump({"api_key": self._api_key, "agent_id": self._agent_id}, f)
        os.chmod(self.credentials_file, 0o600)

    def load_credentials(self) -> bool:
        if not os.path.exists(self.credentials_file):
            return False
        try:
            raw = open(self.credentials_file).read().strip()
            if raw.startswith("{"):
                data = json.loads(raw)
                self._api_key = data.get("api_key") or data.get("apiKey", "")
                self._agent_id = data.get("agent_id") or data.get("agentId", "")
            else:
                for line in raw.splitlines():
                    line = line.strip()
                    if "=" not in line or line.startswith("#"):
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"')
                    if k in ("api_key", "apiKey"):
                        self._api_key = v
                    elif k in ("agent_id", "agentId"):
                        self._agent_id = v
            return bool(self._api_key)
        except Exception:
            return False

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def is_authenticated(self) -> bool:
        return bool(self._api_key)

    def introspect(self) -> Dict:
        if self._introspection is None:
            self._introspection = self._get("/__introspection")
        return self._introspection

    def list_active_competitions(self) -> List[Dict]:
        return self._get("/competition/list-active")

    def get_competition(self, competition_id: str) -> Dict:
        return self._get("/competition", params={"competitionId": competition_id})

    def parse_payouts(self, comp: Dict) -> Optional[List[float]]:
        raw = comp.get("payouts") or comp.get("prizePool") or comp.get("prizes")
        if not raw:
            return None
        if isinstance(raw, list):
            out: List[float] = []
            for p in raw:
                if isinstance(p, (int, float)):
                    out.append(float(p))
                elif isinstance(p, dict):
                    out.append(float(p.get("amount") or p.get("prize") or 0))
            return out if out and sum(out) > 0 else None
        return None

    def register(self, handle: str, name: str, quote: str = "") -> Dict:
        result = self._post("/auth/register", {"handle": handle, "name": name, "quote": quote})
        self._api_key = result.get("apiKey", "")
        self._agent_id = result.get("agentId", "")
        self.save_credentials()
        return result

    def get_me(self) -> Dict:
        return self._get("/agent/me")

    def get_leaderboard(self, competition_id: str) -> Any:
        return self._get(
            "/competition/leaderboard", params={"competitionId": competition_id}
        )

    def get_claim_status(self) -> Dict:
        return self._get("/auth/claim/status")

    def get_invitations(self) -> Dict:
        return self._get("/agent/invitations")

    def claim_invitation(self, redemption_id: str) -> Dict:
        return self._post(f"/agent/invitations/{redemption_id}/claim", {})

    def get_wallet(self, chain: str = "monad") -> Dict:
        return self._get("/agent/wallet", params={"chain": chain})

    def transfer_native(self, chain: str, to: str, amount: str) -> Dict:
        return self._post(
            "/agent/wallet/transfer/native",
            {"chain": chain, "to": to, "amount": str(amount)},
        )

    def join_table(self, competition_id: str = "", tx_hash: str = "") -> Dict:
        body: Dict = {"competitionId": competition_id}
        if not competition_id:
            body = {}
        if tx_hash:
            body["txHash"] = tx_hash
        return self._post("/texas/join", body)

    def start_benchmark(self, competition_id: str) -> Dict:
        """POST /texas/benchmark/start — Poker Eval PVE (not matchmaking lobby)."""
        return self._post(
            "/texas/benchmark/start",
            {"competitionId": competition_id},
        )

    def get_benchmark_status(self, competition_id: str = "") -> Dict:
        """GET /texas/benchmark/status — match progress (bb/100, hands, phase)."""
        params = {}
        if competition_id:
            params["competitionId"] = competition_id
        return self._get("/texas/benchmark/status", params=params or None)

    def get_pending_actions(self, competition_id: str = "") -> List[Dict]:
        params = {"competitionId": competition_id} if competition_id else None
        if not params:
            raise ArenaAPIError(
                400,
                "competitionId required for GET /texas/pending-actions",
            )
        result = self._get("/texas/pending-actions", params=params)
        if isinstance(result, dict):
            return result.get("tables", [])
        return result or []

    async def async_get_pending_actions(self, competition_id: str = "") -> List[Dict]:
        params = {"competitionId": competition_id} if competition_id else None
        if not params:
            raise ArenaAPIError(
                400,
                "competitionId required for GET /texas/pending-actions",
            )
        result = await self.async_get("/texas/pending-actions", params=params)
        if isinstance(result, dict):
            return result.get("tables", [])
        return result or []

    def submit_action(
        self, table_id: str, action: str, amount: float = 0, message: str = ""
    ) -> Dict:
        body: Dict = {"tableId": table_id, "action": action}
        if amount > 0:
            body["amount"] = round(amount, 2)
        if message:
            body["message"] = message[:200]
        return self._post("/texas/action", body)

    async def async_submit_action(
        self, table_id: str, action: str, amount: float = 0, message: str = ""
    ) -> Dict:
        body: Dict = {"tableId": table_id, "action": action}
        if amount > 0:
            body["amount"] = round(amount, 2)
        if message:
            body["message"] = message[:200]
        return await self.async_post("/texas/action", body)

    def get_hand_history(self, hand_id: str) -> Dict:
        """Fetch completed hand if API exposes it."""
        for path in (
            f"/texas/hand/{hand_id}",
            "/texas/hand",
        ):
            try:
                if path.endswith(hand_id):
                    return self._get(path)
                return self._get(path, params={"handId": hand_id})
            except ArenaAPIError as e:
                if e.status == 404:
                    continue
                raise
        raise ArenaAPIError(404, "hand history not found")

    def get_inbox(self) -> Any:
        return self._get("/agent/messages/inbox")

    def redeem_faucet(self, invite_code: str) -> Dict:
        return self._post("/agent/wallet/faucet", {"inviteCode": invite_code})

    def _retry_sleep(self, attempt: int) -> None:
        base = min(2.0, 0.15 * (2**attempt))
        time.sleep(base + random.uniform(0, 0.25))

    async def _async_retry_sleep(self, attempt: int) -> None:
        base = min(2.0, 0.15 * (2**attempt))
        await asyncio.sleep(base + random.uniform(0, 0.25))

    def submit_action_safe(
        self,
        table_id: str,
        action: str,
        amount: float = 0,
        message: str = "",
        retries: int = 2,
    ) -> Optional[Dict]:
        for attempt in range(retries + 1):
            try:
                return self.submit_action(table_id, action, amount, message)
            except ArenaAPIError as e:
                if e.status in (409, 400) and attempt < retries:
                    self._retry_sleep(attempt)
                    continue
                raise
        return None

    async def async_submit_action_safe(
        self,
        table_id: str,
        action: str,
        amount: float = 0,
        message: str = "",
        retries: int = 2,
    ) -> Optional[Dict]:
        for attempt in range(retries + 1):
            try:
                return await self.async_submit_action(table_id, action, amount, message)
            except ArenaAPIError as e:
                if e.status in (409, 400) and attempt < retries:
                    await self._async_retry_sleep(attempt)
                    continue
                raise
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._http.close()
