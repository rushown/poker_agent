"""Launch a second Plutus instance with isolated env, credentials, and state files."""
import json, os, sys
from pathlib import Path

os.environ["ENV_FILE"] = "plutus2/.env"

def _load_root_env():
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        if k and k not in os.environ:
            os.environ[k] = v.strip()

_load_root_env()

_is_registering = "--register" in sys.argv

_creds_path = Path("plutus2/.arena-credentials")
if _creds_path.exists():
    _c = json.loads(_creds_path.read_text())
    _api_key  = os.environ.get("PLUTUS2_ARENA_API_KEY",  _c.get("apiKey", ""))
    _agent_id = os.environ.get("PLUTUS2_ARENA_AGENT_ID", _c.get("agentId", ""))
else:
    _api_key  = os.environ.get("PLUTUS2_ARENA_API_KEY", "")
    _agent_id = os.environ.get("PLUTUS2_ARENA_AGENT_ID", "")
    if not _api_key or not _agent_id:
        if not _is_registering:
            sys.exit("ERROR: plutus2/.arena-credentials missing and PLUTUS2_ARENA_API_KEY/PLUTUS2_ARENA_AGENT_ID not set")

_ENV = {
    "ARENA_API_KEY":         _api_key,
    "ARENA_AGENT_ID":        _agent_id,
    "ARENA_COMPETITION_ID":  os.environ.get("PLUTUS2_COMPETITION_ID", "cmq57o53r0bhw18g23qkydb08"),
    "AGENT_NAME":            "Plutus",
    "AGENT_QUOTE":           "I count outs, not prayers.",
    "HEALTH_PORT":           "8087",
    "ARENA_STRATEGY":        "ADAPTIVE",
    "USE_ASYNC":             "true",
    "DECISION_BUDGET_S":     "1.5",
    "POLL_INTERVAL_S":       "1.5",
    "CREDENTIALS_FILE":      "plutus2/.arena-credentials",
    "STATE_FILE":            "plutus2/.arena-poker-state",
    "ADAPTIVE_STATE_FILE":   "plutus2/.arena-adaptive-state",
    "ACTION_TX_LOG":         "plutus2/.arena-action-tx.json",
    "DECISIONS_LOG_FILE":    "plutus2/decisions.jsonl",
    "HEARTBEAT_STATE_FILE":  "plutus2/.arena-heartbeat-state",
    "OWNER_MESSAGE_FILE":    "plutus2/.arena-owner-messages.txt",
    "LOG_FILE":              "plutus2/plutus2.log",
    "JSON_LOGS":             "false",
    "LOG_LEVEL":             "INFO",
}
for k, v in _ENV.items():
    os.environ[k] = v

import config.settings as _settings_mod
from pydantic_settings import BaseSettings, SettingsConfigDict

class Plutus2Settings(_settings_mod.Settings):
    model_config = SettingsConfigDict(env_file="plutus2/.env", extra="ignore")

_settings_mod.settings = Plutus2Settings()

s = _settings_mod.settings
if not _is_registering:
    assert s.arena_agent_id == _agent_id, f"Wrong agent loaded: {s.arena_agent_id}"

import agent.arena_setup as _setup_mod
_orig_sync = _setup_mod.sync_env_file

def _plutus2_sync_env(path: str, api_key: str, agent_id: str) -> None:
    _orig_sync("plutus2/.env", api_key, agent_id)

_setup_mod.sync_env_file = _plutus2_sync_env

import agent.strategies.adaptive as _adp
_adp._RULES_PATH = Path("strategy_rules.json")
_adp._cached_rules = None

from agent.runner import main
sys.argv = ["run_plutus2.py"] + sys.argv[1:]
main()
