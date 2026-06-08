"""Parameterized launcher for any bluff_27 instance.

Usage:
    python3 run_bluff27_instance.py <suffix>              # run instance
    python3 run_bluff27_instance.py <suffix> --register   # register new agent
    python3 run_bluff27_instance.py <suffix> --list-competitions

Examples:
    python3 run_bluff27_instance.py 1          # runs bluff_271 instance
    python3 run_bluff27_instance.py 2          # runs bluff_272 instance
    python3 run_bluff27_instance.py 1 --register "bluff_271" "I bluff with the worst hand."

The folder is always  bluff_27{suffix}/
The health port is    8083 + int(suffix)   (so instance 1=8084, 2=8085, ...)
"""
import json, os, sys
from pathlib import Path

if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
    print(__doc__)
    sys.exit(1)

_SUFFIX    = sys.argv[1]
_FOLDER    = f"bluff_27{_SUFFIX}"
_PORT      = str(8083 + int(_SUFFIX))
_NAME      = f"bluff_27{_SUFFIX}"
_PASS_ARGS = sys.argv[2:]          # everything after the suffix

# Remove suffix from argv so runner sees normal args
sys.argv = [f"run_bluff27_instance.py"] + _PASS_ARGS

os.environ["ENV_FILE"] = f"{_FOLDER}/.env"


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

_creds_path = Path(f"{_FOLDER}/.arena-credentials")
if _creds_path.exists():
    _c = json.loads(_creds_path.read_text())
    _api_key  = os.environ.get(f"BLUFF27{_SUFFIX}_ARENA_API_KEY",  _c.get("apiKey", ""))
    _agent_id = os.environ.get(f"BLUFF27{_SUFFIX}_ARENA_AGENT_ID", _c.get("agentId", ""))
else:
    _api_key  = os.environ.get(f"BLUFF27{_SUFFIX}_ARENA_API_KEY", "")
    _agent_id = os.environ.get(f"BLUFF27{_SUFFIX}_ARENA_AGENT_ID", "")
    if not _api_key or not _agent_id:
        if not _is_registering:
            sys.exit(f"ERROR: {_FOLDER}/.arena-credentials missing. Run: python3 run_bluff27_instance.py {_SUFFIX} --register \"{_NAME}\"")

_ENV = {
    "ARENA_API_KEY":         _api_key,
    "ARENA_AGENT_ID":        _agent_id,
    "ARENA_COMPETITION_ID":  os.environ.get("BLUFF27_COMPETITION_ID", "cmpy2qy65002ud9ej6b7jjq0l"),
    "AGENT_NAME":            _NAME,
    "AGENT_QUOTE":           "I bluff with the worst hand. And I win.",
    "HEALTH_PORT":           _PORT,
    "ARENA_STRATEGY":        "ADAPTIVE",
    "USE_ASYNC":             "true",
    "DECISION_BUDGET_S":     "1.5",
    "POLL_INTERVAL_S":       "1.5",
    "CREDENTIALS_FILE":      f"{_FOLDER}/.arena-credentials",
    "STATE_FILE":            f"{_FOLDER}/.arena-poker-state",
    "ADAPTIVE_STATE_FILE":   f"{_FOLDER}/.arena-adaptive-state",
    "ACTION_TX_LOG":         f"{_FOLDER}/.arena-action-tx.json",
    "DECISIONS_LOG_FILE":    f"{_FOLDER}/decisions.jsonl",
    "HEARTBEAT_STATE_FILE":  f"{_FOLDER}/.arena-heartbeat-state",
    "OWNER_MESSAGE_FILE":    f"{_FOLDER}/.arena-owner-messages.txt",
    "LOG_FILE":              f"{_FOLDER}/{_NAME}.log",
    "JSON_LOGS":             "false",
    "LOG_LEVEL":             "INFO",
}
for k, v in _ENV.items():
    os.environ[k] = v

import config.settings as _settings_mod
from pydantic_settings import BaseSettings, SettingsConfigDict


class _InstanceSettings(_settings_mod.Settings):
    model_config = SettingsConfigDict(env_file=f"{_FOLDER}/.env", extra="ignore")


_settings_mod.settings = _InstanceSettings()

s = _settings_mod.settings
if not _is_registering:
    assert s.arena_agent_id == _agent_id, f"Wrong agent loaded: {s.arena_agent_id}"

import agent.arena_setup as _setup_mod
_orig_sync = _setup_mod.sync_env_file


def _instance_sync_env(path: str, api_key: str, agent_id: str) -> None:
    _orig_sync(f"{_FOLDER}/.env", api_key, agent_id)


_setup_mod.sync_env_file = _instance_sync_env

import agent.strategies.adaptive as _adp
_adp._RULES_PATH = Path("bluff_27/strategy_rules_bluff27.json")  # all instances share same rules
_adp._cached_rules = None

import agent.strategies.midgame as _mid


# ── 7-2 offsuit detection (shared logic) ────────────────────────────────────

def _is_72_offsuit(hole_cards: list) -> bool:
    if len(hole_cards) != 2:
        return False
    ranks = set(c[0].upper() for c in hole_cards if c)
    suits = [c[1].lower() for c in hole_cards if len(c) >= 2]
    return '7' in ranks and '2' in ranks and len(set(suits)) == 2


def _has_board_pair(hole_cards: list, community: list) -> bool:
    hole_ranks = {c[0].upper() for c in hole_cards if c}
    board_ranks = {c[0].upper() for c in community if c}
    return bool(hole_ranks & board_ranks)


_ORIG_ADAPTIVE_DECIDE = None


def _bluff27_decide(ctx, ehs: float, bb_depth: float):
    from agent.strategies.base import clamp, aggr, open_bb, to_call

    if _is_72_offsuit(ctx.hole_cards):
        if _has_board_pair(ctx.hole_cards, ctx.community_cards):
            return _ORIG_ADAPTIVE_DECIDE(ctx, ehs, bb_depth)

        street = ctx.street

        if street == "preflop":
            if ctx.call_amount >= ctx.stack * 0.80:
                return "fold", 0.0, f"{_NAME}:fold_vs_allin 7-2"
            if ctx.call_amount > 0 and ctx.call_amount < ctx.stack * 0.30:
                shove = clamp(ctx.stack, ctx)
                return aggr(ctx), shove, f"{_NAME}:3bet_allin 7-2"
            amt = open_bb(3.0, ctx)
            return aggr(ctx), amt, f"{_NAME}:open_3bb 7-2"

        if street == "flop":
            if ctx.call_amount >= ctx.stack * 0.40:
                return "fold", 0.0, f"{_NAME}:fold_vs_allin_flop 7-2"
            shove = clamp(ctx.stack, ctx)
            if shove > 0:
                return aggr(ctx), shove, f"{_NAME}:allin_flop 7-2 ev=+4.97"
            return "check", 0.0, f"{_NAME}:check_flop 7-2"

        if street in ("turn", "river"):
            if ctx.call_amount > 0:
                return "fold", 0.0, f"{_NAME}:give_up_{street} 7-2"
            return "check", 0.0, f"{_NAME}:check_{street} 7-2"

    return _ORIG_ADAPTIVE_DECIDE(ctx, ehs, bb_depth)


import agent.strategies.adaptive as _adp_module
_ORIG_ADAPTIVE_DECIDE = _adp_module.decide
_adp_module.decide = _bluff27_decide


def _go_allin(ehs, da, street, pot, allin_min_pot=0.0):
    thresholds = {"preflop": 0.82, "flop": 0.82, "turn": 0.82, "river": 0.82}
    thresh = thresholds.get(street, 0.82)
    if da.hand_category >= 6:  thresh -= 0.08
    elif da.hand_category == 5: thresh -= 0.05
    elif da.hand_category == 4: thresh -= 0.03
    if allin_min_pot > 0 and pot < allin_min_pot:
        return False, f"{_NAME}:allin pot<min"
    eff = ehs + da.equity_boost
    if eff >= thresh:
        tag = f"[{da.hand_category_name}]" if da.hand_category >= 4 else ""
        return True, f"{_NAME}:allin eff={eff:.2f}>={thresh:.2f} {tag}".strip()
    return False, f"{_NAME}:allin skip eff={eff:.2f}"


def _call_allin(ehs, da, street, pot, call_amount):
    required = call_amount / max(1.0, call_amount + pot)
    eff = ehs + da.equity_boost
    min_ehs = max(0.55, required * 1.08)
    if da.hand_category >= 5:  min_ehs = max(0.48, required * 1.03)
    elif da.hand_category == 4: min_ehs = max(0.50, required * 1.05)
    elif da.flush_draw and da.flush_outs >= 9 and da.cards_to_come >= 2:
        min_ehs = max(0.45, required * 1.02)
    if eff >= min_ehs:
        tag = f"[{da.hand_category_name}]" if da.hand_category >= 0 else ""
        return True, f"{_NAME}:call_allin eff={eff:.2f}>={min_ehs:.2f} {tag}".strip()
    return False, f"{_NAME}:fold_allin eff={eff:.2f}<{min_ehs:.2f}"


_mid.should_go_allin   = _go_allin
_mid.should_call_allin = _call_allin

from agent.runner import main
main()
