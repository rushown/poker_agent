"""Launch Plutus_Aggro with isolated env, credentials, state files, and strategy."""
import json, os, sys
from pathlib import Path

# Must set ENV_FILE before any settings import
os.environ["ENV_FILE"] = "plutus_aggro/.env"

# Load root .env first so AGGRO_ARENA_* vars are available (no-op if already set)
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
_creds_path = Path("plutus_aggro/.arena-credentials")
if _creds_path.exists():
    _c = json.loads(_creds_path.read_text())
    _aggro_api_key  = os.environ.get("AGGRO_ARENA_API_KEY",  _c.get("apiKey", ""))
    _aggro_agent_id = os.environ.get("AGGRO_ARENA_AGENT_ID", _c.get("agentId", ""))
else:
    _aggro_api_key  = os.environ.get("AGGRO_ARENA_API_KEY", "")
    _aggro_agent_id = os.environ.get("AGGRO_ARENA_AGENT_ID", "")
    if not _aggro_api_key or not _aggro_agent_id:
        sys.exit("ERROR: plutus_aggro/.arena-credentials not found and AGGRO_ARENA_API_KEY/AGGRO_ARENA_AGENT_ID not set")

# Pre-set all values so pydantic picks them up regardless of which .env it reads
_AGGRO_ENV = {
    "ARENA_API_KEY":         _aggro_api_key,
    "ARENA_AGENT_ID":        _aggro_agent_id,
    "ARENA_COMPETITION_ID":  "cmpy2qy65002ud9ej6b7jjq0l",
    "AGENT_NAME":            "Plutus_Aggro",
    "AGENT_QUOTE":           "I raise first, ask questions never.",
    "HEALTH_PORT":           "8081",
    "ARENA_STRATEGY":        "ADAPTIVE",
    "USE_ASYNC":             "true",
    "DECISION_BUDGET_S":     "1.5",
    "POLL_INTERVAL_S":       "1.5",
    "CREDENTIALS_FILE":      "plutus_aggro/.arena-credentials",
    "STATE_FILE":            "plutus_aggro/.arena-poker-state",
    "ADAPTIVE_STATE_FILE":   "plutus_aggro/.arena-adaptive-state",
    "ACTION_TX_LOG":         "plutus_aggro/.arena-action-tx.json",
    "DECISIONS_LOG_FILE":    "plutus_aggro/decisions.jsonl",
    "HEARTBEAT_STATE_FILE":  "plutus_aggro/.arena-heartbeat-state",
    "OWNER_MESSAGE_FILE":    "plutus_aggro/.arena-owner-messages.txt",
    "LOG_FILE":              "plutus_aggro/plutus_aggro.log",
    "JSON_LOGS":             "false",
    "LOG_LEVEL":             "INFO",
}
for k, v in _AGGRO_ENV.items():
    os.environ[k] = v

# Patch pydantic Settings before it initialises from .env
import config.settings as _settings_mod
from pydantic_settings import BaseSettings, SettingsConfigDict

class AggroSettings(_settings_mod.Settings):
    model_config = SettingsConfigDict(env_file="plutus_aggro/.env", extra="ignore")

_settings_mod.settings = AggroSettings()

# Verify correct agent loaded
s = _settings_mod.settings
assert "aggro" in s.agent_name.lower() or s.arena_agent_id == _aggro_agent_id, \
    f"Wrong agent loaded: {s.arena_agent_id}"

# Redirect --register writes to plutus_aggro/.env (prevents overwriting main .env)
import agent.arena_setup as _setup_mod
_orig_sync = _setup_mod.sync_env_file
def _aggro_sync_env(path: str, api_key: str, agent_id: str) -> None:
    _orig_sync("plutus_aggro/.env", api_key, agent_id)
_setup_mod.sync_env_file = _aggro_sync_env

# Redirect strategy rules to aggro version
import agent.strategies.adaptive as _adp
_adp._RULES_PATH = Path("plutus_aggro/strategy_rules_aggro.json")
_adp._cached_rules = None   # force reload

# Patch midgame thresholds: aggro (less conservative than main Plutus)
import agent.strategies.midgame as _mid

def _aggro_go_allin(ehs, da, street, pot, allin_min_pot=0.0):
    thresholds = {"preflop": 0.86, "flop": 0.88, "turn": 0.86, "river": 0.84}
    thresh = thresholds.get(street, 0.87)
    if   da.hand_category >= 6: thresh -= 0.10
    elif da.hand_category == 5: thresh -= 0.07
    elif da.hand_category == 4: thresh -= 0.05
    elif da.flush_complete or da.straight_complete: thresh -= 0.05
    if allin_min_pot > 0 and pot < allin_min_pot:
        return False, f"aggro:allin pot<min"
    eff = ehs + da.equity_boost
    if eff >= thresh:
        tag = f"[{da.hand_category_name}]" if da.hand_category >= 4 else ""
        return True, f"aggro:allin eff={eff:.2f}>={thresh:.2f} {tag}".strip()
    return False, f"aggro:allin skip eff={eff:.2f}"

def _aggro_call_allin(ehs, da, street, pot, call_amount):
    required = call_amount / max(1.0, call_amount + pot)
    eff = ehs + da.equity_boost
    min_ehs = max(0.55, required * 1.08)
    if   da.hand_category >= 5: min_ehs = max(0.48, required * 1.03)
    elif da.hand_category == 4: min_ehs = max(0.50, required * 1.05)
    elif da.flush_draw and da.flush_outs >= 9 and da.cards_to_come >= 2:
        min_ehs = max(0.45, required * 1.02)
    if eff >= min_ehs:
        tag = f"[{da.hand_category_name}]" if da.hand_category >= 0 else ""
        return True, f"aggro:call_allin eff={eff:.2f}>={min_ehs:.2f} {tag}".strip()
    return False, f"aggro:fold_allin eff={eff:.2f}<{min_ehs:.2f}"

_mid.should_go_allin   = _aggro_go_allin
_mid.should_call_allin = _aggro_call_allin

# Run
from agent.runner import main
sys.argv = ["run_aggro.py"]
main()
