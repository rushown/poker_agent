"""config/settings.py — centralised settings loaded once at startup."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    arena_api_key: str = ""
    arena_agent_id: str = ""
    arena_base_url: str = "https://arena.dev.fun"
    arena_competition_id: str = ""
    arena_invite_code: str = ""

    agent_name: str = "Plutus"
    agent_quote: str = "I count outs, not prayers."

    state_file: str = ".arena-poker-state"
    adaptive_state_file: str = ".arena-adaptive-state"
    credentials_file: str = ".arena-credentials"
    action_timeout_ms: int = 1700
    decision_budget_s: float = 1.5
    poll_interval_s: float = 1.0
    min_poll_interval_s: float = 0.55
    watchdog_idle_s: float = 60.0
    action_tx_log: str = ".arena-action-tx.json"
    log_level: str = "INFO"
    json_logs: bool = False
    log_file: str = "plutus.log"
    health_port: int = 8080
    use_async: bool = True

    owner_message_file: str = ".arena-owner-messages.txt"
    heartbeat_state_file: str = ".arena-heartbeat-state"
    heartbeat_min_interval_s: float = 3600.0
    heartbeat_interval_s: float = 14400.0
    auto_claim_invitations: bool = True
    decisions_log_file: str = "decisions.jsonl"

    # Strategy override: S1-S9 for single fixed strategy, META for UCB1 learner,
    # S10 or empty for the full GTO+exploit arbiter (default).
    arena_strategy: str = ""


settings = Settings()
