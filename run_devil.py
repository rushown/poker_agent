"""Launch devil with isolated env, credentials, state files, and dominate strategy."""
import json, os, sys
from pathlib import Path

os.environ["ENV_FILE"] = "devil/.env"

# Load root .env first so DEVIL_ARENA_* vars are available (no-op if already set)
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

# Load credentials: env vars (from root .env or shell) take priority over .arena-credentials
_creds_path = Path("devil/.arena-credentials")
if _creds_path.exists():
    _c = json.loads(_creds_path.read_text())
    _devil_api_key  = os.environ.get("DEVIL_ARENA_API_KEY",  _c.get("apiKey", ""))
    _devil_agent_id = os.environ.get("DEVIL_ARENA_AGENT_ID", _c.get("agentId", ""))
else:
    _devil_api_key  = os.environ.get("DEVIL_ARENA_API_KEY", "")
    _devil_agent_id = os.environ.get("DEVIL_ARENA_AGENT_ID", "")
    if not _devil_api_key or not _devil_agent_id:
        sys.exit("ERROR: devil/.arena-credentials not found and DEVIL_ARENA_API_KEY/DEVIL_ARENA_AGENT_ID not set")

_DEVIL_ENV = {
    "ARENA_API_KEY":         _devil_api_key,
    "ARENA_AGENT_ID":        _devil_agent_id,
    "ARENA_COMPETITION_ID":  "cmpy2qy65002ud9ej6b7jjq0l",
    "AGENT_NAME":            "devil",
    "AGENT_QUOTE":           "Your chips belong to me.",
    "HEALTH_PORT":           "8082",
    "ARENA_STRATEGY":        "ADAPTIVE",
    "USE_ASYNC":             "true",
    "DECISION_BUDGET_S":     "1.5",
    "POLL_INTERVAL_S":       "1.5",
    "CREDENTIALS_FILE":      "devil/.arena-credentials",
    "STATE_FILE":            "devil/.arena-poker-state",
    "ADAPTIVE_STATE_FILE":   "devil/.arena-adaptive-state",
    "ACTION_TX_LOG":         "devil/.arena-action-tx.json",
    "DECISIONS_LOG_FILE":    "devil/decisions.jsonl",
    "HEARTBEAT_STATE_FILE":  "devil/.arena-heartbeat-state",
    "OWNER_MESSAGE_FILE":    "devil/.arena-owner-messages.txt",
    "LOG_FILE":              "devil/devil.log",
    "JSON_LOGS":             "false",
    "LOG_LEVEL":             "INFO",
}
for k, v in _DEVIL_ENV.items():
    os.environ[k] = v

import config.settings as _settings_mod
from pydantic_settings import BaseSettings, SettingsConfigDict

class DevilSettings(_settings_mod.Settings):
    model_config = SettingsConfigDict(env_file="devil/.env", extra="ignore")

_settings_mod.settings = DevilSettings()

s = _settings_mod.settings
assert s.arena_agent_id == _devil_agent_id, f"Wrong agent loaded: {s.arena_agent_id}"

# Redirect --register writes to devil/.env (prevents overwriting main .env)
import agent.arena_setup as _setup_mod
_orig_sync = _setup_mod.sync_env_file
def _devil_sync_env(path: str, api_key: str, agent_id: str) -> None:
    _orig_sync("devil/.env", api_key, agent_id)
_setup_mod.sync_env_file = _devil_sync_env

import agent.strategies.adaptive as _adp
_adp._RULES_PATH = Path("devil/strategy_rules.json")
_adp._cached_rules = None

import agent.strategies.midgame as _mid

# Dominate synthesis: fixed thresholds 0.85 all streets, no made-hand reductions
def _devil_go_allin(ehs, da, street, pot, allin_min_pot=0.0):
    thresh = 0.85
    if allin_min_pot > 0 and pot < allin_min_pot:
        return False, f"devil:allin pot<min({allin_min_pot})"
    eff = ehs + da.equity_boost
    if eff >= thresh:
        tag = f"[{da.hand_category_name}]" if da.hand_category >= 4 else ""
        return True, f"devil:allin eff={eff:.2f}>={thresh:.2f} {tag}".strip()
    return False, f"devil:allin skip eff={eff:.2f}"

def _devil_call_allin(ehs, da, street, pot, call_amount):
    required = call_amount / max(1.0, call_amount + pot)
    eff = ehs + da.equity_boost
    min_ehs = max(0.55, required * 1.08)
    if   da.hand_category >= 5: min_ehs = max(0.48, required * 1.03)
    elif da.hand_category == 4: min_ehs = max(0.50, required * 1.05)
    elif da.flush_draw and da.flush_outs >= 9 and da.cards_to_come >= 2:
        min_ehs = max(0.45, required * 1.02)
    if eff >= min_ehs:
        tag = f"[{da.hand_category_name}]" if da.hand_category >= 0 else ""
        return True, f"devil:call_allin eff={eff:.2f}>={min_ehs:.2f} {tag}".strip()
    return False, f"devil:fold_allin eff={eff:.2f}<{min_ehs:.2f}"

_mid.should_go_allin   = _devil_go_allin
_mid.should_call_allin = _devil_call_allin

from agent.runner import main
sys.argv = ["run_devil.py"]
main()
