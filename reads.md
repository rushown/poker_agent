ocean@fedora:~/vscode/poker_agent$ cat run_bluff27.py

# Parameterized instance launcher (for instances 1-N)
cat run_bluff27_instance.py

# Spawn + register helper
cat spawn_bluff27.py

# Strategy rules (bluff override + normal hand rules)
cat bluff_27/strategy_rules_bluff27.json

# Strategy engine that bluff logic patches into
cat agent/strategies/adaptive.py

# Midgame module (draw analysis, all-in logic)
cat agent/strategies/midgame.py

# Base helpers (clamp, open_bb, bet_pot, aggr, etc.)
cat agent/strategies/base.py

# Runner entry point
cat agent/runner.py
"""Launch bluff_27 agent for S2 2-7 Offsuit Bluff ($2,500) tournament.

Strategy: detect 7-2 offsuit -> open 3BB -> flop all-in (97.4% fold probability).
Normal hands: plutus_optimal_v7 rules unchanged.
DO NOT shove preflop with 7-2 (EV=-28.35). Open small -> flop shove (EV=+1.48).
"""
import json, os, sys
from pathlib import Path

os.environ["ENV_FILE"] = "bluff_27/.env"


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

_creds_path = Path("bluff_27/.arena-credentials")
if _creds_path.exists():
    _c = json.loads(_creds_path.read_text())
    _b27_api_key  = os.environ.get("BLUFF27_ARENA_API_KEY",  _c.get("apiKey", ""))
    _b27_agent_id = os.environ.get("BLUFF27_ARENA_AGENT_ID", _c.get("agentId", ""))
else:
    _b27_api_key  = os.environ.get("BLUFF27_ARENA_API_KEY", "")
    _b27_agent_id = os.environ.get("BLUFF27_ARENA_AGENT_ID", "")
    if not _b27_api_key or not _b27_agent_id:
        if not _is_registering:
            sys.exit("ERROR: bluff_27/.arena-credentials not found and BLUFF27_ARENA_API_KEY/BLUFF27_ARENA_AGENT_ID not set")

_BLUFF27_ENV = {
    "ARENA_API_KEY":         _b27_api_key,
    "ARENA_AGENT_ID":        _b27_agent_id,
    "ARENA_COMPETITION_ID":  os.environ.get("BLUFF27_COMPETITION_ID", ""),
    "AGENT_NAME":            "bluff_27",
    "AGENT_QUOTE":           "I bluff with the worst hand. And I win.",
    "HEALTH_PORT":           "8083",
    "ARENA_STRATEGY":        "ADAPTIVE",
    "USE_ASYNC":             "true",
    "DECISION_BUDGET_S":     "1.5",
    "POLL_INTERVAL_S":       "1.5",
    "CREDENTIALS_FILE":      "bluff_27/.arena-credentials",
    "STATE_FILE":            "bluff_27/.arena-poker-state",
    "ADAPTIVE_STATE_FILE":   "bluff_27/.arena-adaptive-state",
    "ACTION_TX_LOG":         "bluff_27/.arena-action-tx.json",
    "DECISIONS_LOG_FILE":    "bluff_27/decisions.jsonl",
    "HEARTBEAT_STATE_FILE":  "bluff_27/.arena-heartbeat-state",
    "OWNER_MESSAGE_FILE":    "bluff_27/.arena-owner-messages.txt",
    "LOG_FILE":              "bluff_27/bluff27.log",
    "JSON_LOGS":             "false",
    "LOG_LEVEL":             "INFO",
}
for k, v in _BLUFF27_ENV.items():
    os.environ[k] = v

import config.settings as _settings_mod
from pydantic_settings import BaseSettings, SettingsConfigDict


class Bluff27Settings(_settings_mod.Settings):
    model_config = SettingsConfigDict(env_file="bluff_27/.env", extra="ignore")


_settings_mod.settings = Bluff27Settings()

s = _settings_mod.settings
if not _is_registering:
    assert s.arena_agent_id == _b27_agent_id, f"Wrong agent loaded: {s.arena_agent_id}"

import agent.arena_setup as _setup_mod
_orig_sync = _setup_mod.sync_env_file


def _bluff27_sync_env(path: str, api_key: str, agent_id: str) -> None:
    _orig_sync("bluff_27/.env", api_key, agent_id)


_setup_mod.sync_env_file = _bluff27_sync_env

# Point adaptive strategy at bluff_27 rules
import agent.strategies.adaptive as _adp
_adp._RULES_PATH = Path("bluff_27/strategy_rules_bluff27.json")
_adp._cached_rules = None

import agent.strategies.midgame as _mid


# ── 7-2 offsuit detection ────────────────────────────────────────────────────

def _is_72_offsuit(hole_cards: list) -> bool:
    """Return True if hole cards are exactly 7-2 offsuit (worst hand in Hold'em)."""
    if len(hole_cards) != 2:
        return False
    ranks = set(c[0].upper() for c in hole_cards if c)
    suits = [c[1].lower() for c in hole_cards if len(c) >= 2]
    return '7' in ranks and '2' in ranks and len(set(suits)) == 2


def _has_board_pair(hole_cards: list, community: list) -> bool:
    """Return True if the board pairs one of our hole cards (7 or 2 on board)."""
    hole_ranks = {c[0].upper() for c in hole_cards if c}
    board_ranks = {c[0].upper() for c in community if c}
    return bool(hole_ranks & board_ranks)


# ── 7-2 bluff override ───────────────────────────────────────────────────────
# Injected into midgame/adaptive via monkey-patching the decide function.
#
# EV MATH (from 8-agent DeepSeek analysis 2026-06-08):
#   EV_preflop_allin  = -28.35 chips → NEVER do this
#   EV_open_3bb       = +0.48 chips (preflop fold 48%)
#   EV_flop_allin     = +4.97 chips (95% fold after called preflop)
#   Total EV sequence = +1.48 chips per 7-2 hand, fold prob = 97.4%
#   EV_cbet_only      = -0.31 chips → useless against 5% c-bet fold bots

_ORIG_ADAPTIVE_DECIDE = None


def _bluff27_decide(ctx, ehs: float, bb_depth: float):
    """Override adaptive.decide: inject 7-2 bluff logic before normal strategy."""
    from agent.strategies.base import clamp, aggr, open_bb, to_call, safe_check_fold

    if _is_72_offsuit(ctx.hole_cards):
        # Made hand check: if 7 or 2 is on board we have a pair — use normal strategy
        if _has_board_pair(ctx.hole_cards, ctx.community_cards):
            return _ORIG_ADAPTIVE_DECIDE(ctx, ehs, bb_depth)

        street = ctx.street

        # ── Preflop bluff sequence ─────────────────────────────────────────
        if street == "preflop":
            # Facing opponent all-in: fold (EV=-28.35 is too costly)
            stack_in = ctx.call_amount
            if stack_in >= ctx.stack * 0.80:
                return "fold", 0.0, "bluff27:fold_vs_allin 7-2"

            # Facing a raise: 3-bet all-in (pot still small, 95% fold)
            if ctx.call_amount > 0 and ctx.call_amount < ctx.stack * 0.30:
                shove = clamp(ctx.stack, ctx)
                return aggr(ctx), shove, f"bluff27:3bet_allin preflop 7-2"

            # No bet facing or standard raise: open 3BB
            amt = open_bb(3.0, ctx)
            return aggr(ctx), amt, "bluff27:open_3bb 7-2"

        # ── Postflop bluff sequence ────────────────────────────────────────
        # Flop: push all-in (EV=+4.97 vs EV_cbet=-0.31)
        if street == "flop":
            # Facing opponent all-in: fold (we have 31% equity at best)
            if ctx.call_amount >= ctx.stack * 0.40:
                return "fold", 0.0, "bluff27:fold_vs_allin_flop 7-2"
            # Go all-in regardless of whether facing a bet or acting first
            shove = clamp(ctx.stack, ctx)
            if shove > 0:
                return aggr(ctx), shove, f"bluff27:allin_flop 7-2 ev=+4.97"
            return "check", 0.0, "bluff27:check_flop 7-2 (stack=0)"

        # Turn/River: if we're still in after flop call, give up — opponent has a hand
        if street in ("turn", "river"):
            if ctx.call_amount > 0:
                return "fold", 0.0, f"bluff27:give_up_{street} 7-2 (called flop)"
            return "check", 0.0, f"bluff27:check_{street} 7-2"

    # Not 7-2 offsuit: use normal optimal strategy
    return _ORIG_ADAPTIVE_DECIDE(ctx, ehs, bb_depth)


# Patch adaptive.decide
import agent.strategies.adaptive as _adp_module
_ORIG_ADAPTIVE_DECIDE = _adp_module.decide
_adp_module.decide = _bluff27_decide


# ── All-in thresholds (same as plutus_optimal_v7) ────────────────────────────

def _bluff27_go_allin(ehs, da, street, pot, allin_min_pot=0.0):
    """Normal all-in threshold: 0.82 across all streets (proven optimal)."""
    thresholds = {"preflop": 0.82, "flop": 0.82, "turn": 0.82, "river": 0.82}
    thresh = thresholds.get(street, 0.82)
    if da.hand_category >= 6:
        thresh -= 0.08
    elif da.hand_category == 5:
        thresh -= 0.05
    elif da.hand_category == 4:
        thresh -= 0.03
    if allin_min_pot > 0 and pot < allin_min_pot:
        return False, f"b27:allin pot<min({allin_min_pot})"
    eff = ehs + da.equity_boost
    if eff >= thresh:
        tag = f"[{da.hand_category_name}]" if da.hand_category >= 4 else ""
        return True, f"b27:allin eff={eff:.2f}>={thresh:.2f} {tag}".strip()
    return False, f"b27:allin skip eff={eff:.2f}"


def _bluff27_call_allin(ehs, da, street, pot, call_amount):
    """Normal call-all-in: require justified equity (opponents never bluff)."""
    required = call_amount / max(1.0, call_amount + pot)
    eff = ehs + da.equity_boost
    min_ehs = max(0.55, required * 1.08)
    if da.hand_category >= 5:
        min_ehs = max(0.48, required * 1.03)
    elif da.hand_category == 4:
        min_ehs = max(0.50, required * 1.05)
    elif da.flush_draw and da.flush_outs >= 9 and da.cards_to_come >= 2:
        min_ehs = max(0.45, required * 1.02)
    if eff >= min_ehs:
        tag = f"[{da.hand_category_name}]" if da.hand_category >= 0 else ""
        return True, f"b27:call_allin eff={eff:.2f}>={min_ehs:.2f} {tag}".strip()
    return False, f"b27:fold_allin eff={eff:.2f}<{min_ehs:.2f}"


_mid.should_go_allin   = _bluff27_go_allin
_mid.should_call_allin = _bluff27_call_allin

from agent.runner import main
# Preserve CLI args (--register, --list-competitions, etc.) — only strip script name
sys.argv = ["run_bluff27.py"] + sys.argv[1:]
main()
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
#!/usr/bin/env python3
"""Spawn and register a new bluff_27 instance.

Usage:
    python3 spawn_bluff27.py <suffix> [competition_id]

Creates  bluff_27{suffix}/  folder, copies strategy rules, registers a new
arena agent, and saves credentials. After this, launch with:
    python3 run_bluff27_instance.py <suffix>

Examples:
    python3 spawn_bluff27.py 1                          # bluff_271, Playground S1
    python3 spawn_bluff27.py 2 cmpy2qy65002ud9ej6b7jjq0l
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

SUFFIX       = sys.argv[1]
FOLDER       = f"bluff_27{SUFFIX}"
NAME         = f"bluff_27{SUFFIX}"
PORT         = str(8083 + int(SUFFIX))
COMP_ID      = sys.argv[2] if len(sys.argv) > 2 else "cmpy2qy65002ud9ej6b7jjq0l"
RULES_SRC    = "bluff_27/strategy_rules_bluff27.json"

print(f"Spawning {FOLDER} (port {PORT}, competition {COMP_ID})")

# ── Create folder ────────────────────────────────────────────────────────────
p = Path(FOLDER)
p.mkdir(exist_ok=True)

# State files
for fname in (".arena-heartbeat-state", ".arena-owner-messages.txt"):
    (p / fname).touch()
for fname in (".arena-poker-state", ".arena-adaptive-state", ".arena-action-tx.json"):
    if not (p / fname).exists():
        (p / fname).write_text("{}")

# Credentials (blank — will be filled by --register)
creds = p / ".arena-credentials"
if not creds.exists() or json.loads(creds.read_text()).get("apiKey", "") == "":
    creds.write_text('{"apiKey":"","agentId":""}')

# .env
env_path = p / ".env"
env_path.write_text(f"""\
ARENA_COMPETITION_ID={COMP_ID}
AGENT_NAME={NAME}
AGENT_QUOTE=I bluff with the worst hand. And I win.
ARENA_INVITE_CODE=
ARENA_BASE_URL=https://arena.dev.fun

ARENA_STRATEGY=ADAPTIVE

LOG_LEVEL=INFO
JSON_LOGS=false
HEALTH_PORT={PORT}
USE_ASYNC=true
DECISION_BUDGET_S=1.5
POLL_INTERVAL_S=1.5
HEARTBEAT_MIN_INTERVAL_S=3600
HEARTBEAT_INTERVAL_S=14400
OWNER_MESSAGE_FILE={FOLDER}/.arena-owner-messages.txt
ARENA_API_KEY=
ARENA_AGENT_ID=
""")

print(f"  Created {FOLDER}/")

# ── Register ─────────────────────────────────────────────────────────────────
print(f"  Registering agent '{NAME}' ...")
result = subprocess.run(
    [sys.executable, "run_bluff27_instance.py", SUFFIX,
     "--register", NAME, "I bluff with the worst hand. And I win."],
    capture_output=False,
    text=True,
)
if result.returncode != 0:
    print(f"  Registration failed (exit {result.returncode})")
    sys.exit(1)

# Verify credentials were saved
time.sleep(1)
try:
    c = json.loads(creds.read_text())
    aid = c.get("agentId", "")
    key = c.get("apiKey", "")
    if aid and key:
        print(f"\n  Agent ID : {aid}")
        print(f"  API Key  : {key[:30]}...")
        print(f"\n  Launch with: python3 run_bluff27_instance.py {SUFFIX}")
    else:
        print("  WARNING: credentials file looks empty after registration")
except Exception as e:
    print(f"  WARNING: could not read credentials: {e}")

print(f"\nDone. {FOLDER} is ready.")
{
  "version": "bluff27_v1",
  "cycle": "bluff27_v1",
  "core_principle": "S2 2-7 Offsuit Bluff ($2500 prize): detect 7-2 offsuit -> open 3BB -> flop all-in (97.4% fold prob). Normal hands: plutus_optimal_v7 EV strategy unchanged. NEVER preflop all-in with 7-2 (EV=-28.35). Always open small -> flop shove (EV=+1.48).",
  "preflop": {
    "open_ip_min_ehs": 0.36,
    "open_oop_min_ehs": 0.46,
    "open_sizing_bb": 3.0,
    "open_bb_defend_min_ehs": 0.32,
    "vs_raise_call_ip_min_ehs": 0.46,
    "vs_raise_call_oop_min_ehs": 0.54,
    "bb_raise_min_ehs": 0.6,
    "vs_raise_3bet_min_ehs": 0.76,
    "speculative_hand_see_flop": {
      "enabled": true,
      "connector_gap_max": 2,
      "suited_bonus_ehs": 0.06,
      "broadway_min_rank": 9,
      "broadway_see_flop_min_ehs": 0.44,
      "connector_see_flop_min_ehs": 0.4
    },
    "steal_attempts": {
      "ip_steal_min_ehs": 0.34,
      "sb_steal_min_ehs": 0.36,
      "steal_sizing_bb": 3.0
    },
    "vs_raise_3bet_sizing_mult": 2.5,
    "vs_4bet_call_min_ehs": 0.78,
    "vs_4bet_fold_max_ehs": 0.76,
    "limp_strategy": "never_limp",
    "min_raise_bb": 2.0,
    "max_raise_bb": 14.0,
    "allin_threshold_ehs": 0.82
  },
  "flop": {
    "no_bet_value_min_ehs": 0.49,
    "no_bet_value_sizing_pot": 1.0,
    "no_bet_value_sizing_dry": 0.85,
    "no_bet_value_sizing_wet": 1.25,
    "c_bet_frequency": 0.8,
    "c_bet_range_advantage_enabled": true,
    "facing_bet_call_min_ehs": 0.46,
    "facing_bet_raise_min_ehs": 0.9,
    "bluff_enabled": false,
    "bluff_frequency": 0,
    "bluff_sizing_pot": 0.6,
    "bluff_max_ehs": 0.3,
    "multiway_adjust_call_min_ehs_penalty": 0.04,
    "no_bet_trap_min_ehs": 1.0,
    "facing_bet_fold_max_ehs": 0.34,
    "no_bet_cbet_min_ehs": 0.49,
    "no_bet_cbet_frequency": 0.0,
    "no_bet_board_hit_overbet_min_ehs": 0.68,
    "no_bet_board_hit_overbet_pot": 1.3,
    "facing_bet_raise_mult": 3.0,
    "allin_threshold_ehs": 0.82,
    "check_raise_allow": true,
    "check_raise_min_ehs": 0.88,
    "check_raise_sizing_pot": 3.0,
    "flop_allin_min_ehs": 0.82,
    "flop_raise_aggressive_min_ehs": 0.88,
    "semi_bluff_max_raises_per_hand": 1
  },
  "turn": {
    "facing_bet_call_min_ehs": 0.52,
    "facing_bet_raise_min_ehs": 0.88,
    "bluff_enabled": false,
    "bluff_frequency": 0,
    "bluff_sizing_pot": 0.7,
    "bluff_max_ehs": 0.24,
    "multiway_adjust_call_min_ehs_penalty": 0.04,
    "no_bet_value_min_ehs": 0.54,
    "no_bet_value_sizing_pot": 1.3,
    "no_bet_trap_min_ehs": 1.0,
    "facing_bet_fold_max_ehs": 0.38,
    "barrel_min_ehs": 0.48,
    "barrel_sizing_pot": 0.85,
    "no_bet_barrel_min_ehs": 0.48,
    "no_bet_barrel_frequency": 0.55,
    "facing_bet_raise_mult": 2.8,
    "allin_threshold_ehs": 0.82,
    "overbet_enabled": true,
    "overbet_min_ehs": 0.58,
    "overbet_sizing_pot": 1.5,
    "semi_bluff_max_raises_per_hand": 1
  },
  "river": {
    "facing_bet_call_min_ehs": 0.58,
    "facing_bet_raise_min_ehs": 0.82,
    "bluff_enabled": false,
    "bluff_frequency": 0,
    "bluff_sizing_pot": 0.8,
    "bluff_max_ehs": 0.22,
    "multiway_adjust_call_min_ehs_penalty": 0.03,
    "no_bet_value_min_ehs": 0.49,
    "no_bet_value_sizing_pot": 1.75,
    "facing_bet_fold_max_ehs": 0.42,
    "bet_frequency": 1.0,
    "bet_min_ehs": 0.49,
    "bet_sizing_pot": 1.75,
    "allin_threshold_ehs": 0.82,
    "overbet_enabled": true,
    "overbet_min_ehs": 0.58,
    "overbet_sizing_pot": 2.0
  },
  "aggression": {
    "river_allin_min_ehs": 0.82,
    "all_in_push_min_ehs": 0.82,
    "all_in_call_min_ehs": 0.82,
    "pot_commit_max_ehs_trap": 0.65,
    "aggressive_opponent_trap_calls_enabled": true,
    "large_bet_min_ehs": 0.68,
    "large_bet_pot_mult": 1.5,
    "medium_bet_min_ehs": 0.54,
    "medium_bet_pot_mult": 1.0,
    "flop_allin_min_ehs": 0.82,
    "flop_raise_aggressive_min_ehs": 0.88,
    "semi_bluff_max_raises_per_hand": 1,
    "allin_min_pot": 6
  },
  "adjustments": {
    "ip_threshold_reduction": 0.05,
    "oop_threshold_increase": 0.03,
    "board_hit_overbet_enabled": true
  },
  "cycle_notes": "bluff27_v1: 8-agent DeepSeek analysis 2026-06-08. 7-2 bluff: open 3BB->flop allin. EV_flop_allin=+4.97 vs EV_cbet=-0.31. Fold prob 97.4%. Normal hands: plutus_optimal_v7 unchanged.",
  "pot_odds": {
    "call_margin": 1.0,
    "minimum_ehs_for_call": 0.44,
    "note": "EHS floor prevents bad calls below breakeven. 15 sub-0.38 calls found in v6 session \u2014 investigate code bypass path."
  },
  "bluff_27_override": {
    "enabled": true,
    "detection": "hole cards contain exactly 7 and 2, different suits",
    "preflop_raise_bb": 3.0,
    "preflop_shove_if_facing_raise": true,
    "preflop_fold_to_opponent_allin": true,
    "flop_action": "all_in",
    "flop_ev": 4.97,
    "turn_action_if_called_flop": "fold",
    "river_action": "fold",
    "fold_to_opponent_allin": true,
    "made_pair_ehs_threshold": 0.52,
    "ev_math": "open 3BB (48% fold = +1.44) then flop allin (95% fold, ev=+4.97). Total fold prob=97.4%, EV=+1.48 chips. EV_preflop_allin=-28.35 NEVER shove pre.",
    "notes": "If 7 or 2 on board (EHS>=0.52), skip override and let normal strategy handle it."
  },
  "short_stack": {
    "threshold_bb": 20,
    "shove_min_ehs": 0.62,
    "note": "Stack < 20BB: push all-in at EHS >= 0.62"
  }
"""agent/strategies/adaptive.py — rule-driven strategy loaded from strategy_rules.json.

The rules file (not this code) is what DeepSeek modifies each cycle.
This code is a pure executor: read rules → apply → return decision.
Never hardcodes thresholds — every number comes from the JSON.

Augmented by agent/strategies/midgame.py which handles:
  - Draw analysis (flush/straight draws, made hands)
  - All-in confidence (push on nuts, call only when justified)
  - Bluff detection + bluff-catch calls
  - Semi-bluff raises on strong draws
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from agent.arbiter import GameContext
import random as _random
from agent.strategies.base import (
    Decision, aggr, bet_pot, board_hits, clamp, has,
    is_ip, open_bb, raise_to, safe_check_fold, to_call,
)
from agent.strategies.midgame import (
    midgame_postflop,
    assess_preflop_min_call,
    analyze_draws,
    should_go_allin,
    DrawAnalysis,
)
from agent.strategies.bet_intelligence import (
    bet_intelligence_decide,
    analyze_board,
    BoardTexture,
)

_rng_tl = threading.local()

def _get_rng() -> _random.Random:
    if not hasattr(_rng_tl, "r"):
        _rng_tl.r = _random.Random()
    return _rng_tl.r

_RULES_PATH = Path(__file__).parent.parent.parent / "strategy_rules.json"
_cached_rules: Optional[dict] = None
_cached_mtime: float = 0.0

_RANK_MAP = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
             "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


def _rank(card: str) -> int:
    return _RANK_MAP.get(card[0].upper(), 0) if card else 0


def _speculative_ehs_bonus(hole_cards: list) -> float:
    """Return EHS bonus/threshold reduction for speculative hands.

    Returns (suited_bonus, is_connector, is_broadway) so caller can apply
    the right see-flop threshold from the rules.
    """
    if len(hole_cards) < 2:
        return 0.0, False, False
    r1, r2 = _rank(hole_cards[0]), _rank(hole_cards[1])
    s1, s2 = hole_cards[0][-1].lower() if hole_cards[0] else "", hole_cards[1][-1].lower() if hole_cards[1] else ""
    suited = s1 == s2 and s1 != ""
    gap = abs(r1 - r2)
    # Read gap threshold from rules (default 1 = connectors and one-gappers only)
    spec = _rules().get("preflop", {}).get("speculative_hand_see_flop", {})
    max_gap     = int(spec.get("connector_gap_max", 1))
    suit_bonus  = float(spec.get("suited_bonus_ehs", 0.04))
    is_connector = 1 <= gap <= max_gap
    is_broadway  = min(r1, r2) >= 10      # both cards T or higher
    suited_bonus = suit_bonus if suited else 0.0
    return suited_bonus, is_connector, is_broadway


def _pot_odds_call(ehs: float, call_amount: float, pot: float) -> tuple[bool, str]:
    """Felt ReaderL4 style: equity covers price? -> call is profitable.

    Returns (should_call, reason_string).
    required_equity = call / (call + pot)  — the pot-odds break-even point.
    If EHS >= required_equity * margin, calling is +EV.

    Hard floor: never call below pot_odds.minimum_ehs_for_call (default 0.48).
    Prevents -EV overcalling at EHS 0.43-0.47 just because the pot is large.
    """
    if call_amount <= 0 or (call_amount + pot) <= 0:
        return False, ""
    po_rules = _rules().get("pot_odds", {})
    required = call_amount / (call_amount + pot)
    margin = float(po_rules.get("call_margin", 1.0))
    min_ehs = float(po_rules.get("minimum_ehs_for_call", 0.48))
    if ehs < min_ehs:
        return False, f"equity {ehs:.0%} below floor {min_ehs:.0%}"
    covers = ehs >= required * margin
    reason = f"equity {ehs:.0%} {'covers' if covers else 'misses'} price {required:.0%}"
    return covers, reason


def _rules() -> dict:
    """Hot-reload rules when the file changes (picks up mid-session updates)."""
    global _cached_rules, _cached_mtime
    try:
        mtime = _RULES_PATH.stat().st_mtime
        if _cached_rules is None or mtime != _cached_mtime:
            _cached_rules = json.loads(_RULES_PATH.read_text())
            _cached_mtime = mtime
    except Exception:
        pass
    return _cached_rules or {}


def _r(section: str, key: str, default: float = 0.5) -> float:
    return float(_rules().get(section, {}).get(key, default))


def _opp_modifier() -> float:
    """Return EHS threshold modifier based on classified opponent archetype.
    Read from thread-local set by arbiter (safe for async multi-table).
    """
    try:
        from agent.arbiter import _tl
        opp = getattr(_tl, "opponent_type", "unknown")
    except Exception:
        opp = "unknown"
    mods = {
        "nit": -0.04,             "scared_money": -0.05,  # bluff more vs folders
        "calling_station": +0.03, "fish": +0.02,           # value-only vs callers
        "maniac": +0.04,          "lag": +0.03,             # tighten vs aggression
        "tag": 0.0,               "gto_balanced": 0.0,
        "unknown": 0.0,           "fixed_timing": 0.0,
        "range_static": -0.02,                              # bluff range-static bots
    }
    return mods.get(opp, 0.0)


def _adj(ehs: float, ip: bool) -> float:
    """Adjust EHS threshold for position + opponent type."""
    reduction = _r("adjustments", "ip_threshold_reduction", 0.04)
    increase  = _r("adjustments", "oop_threshold_increase", 0.03)
    base = ehs - reduction if ip else ehs + increase
    return base + _opp_modifier()


def _pos_open_min(ctx: GameContext) -> float:
    """Position-specific preflop open threshold (tighter from early positions)."""
    r = _rules().get("preflop", {})
    pos = ctx.position.upper()
    thresholds = {
        "BTN": r.get("open_btn_min_ehs", 0.50),
        "CO":  r.get("open_co_min_ehs",  0.55),
        "HJ":  r.get("open_hj_min_ehs",  0.60),
        "UTG": r.get("open_utg_min_ehs", 0.64),
        "MP":  r.get("open_utg_min_ehs", 0.64),
        "SB":  r.get("open_sb_min_ehs",  0.52),
        "BB":  r.get("open_bb_defend_min_ehs", 0.45),
    }
    if pos in thresholds:
        return thresholds[pos]
    # fallback to ip/oop generic
    ip = ctx.is_in_position or pos in ("BTN", "CO")
    return r.get("open_ip_min_ehs" if ip else "open_oop_min_ehs", 0.60)


def _board_is_wet(community: list) -> bool:
    """True when flop has flush draw or straight-draw potential."""
    if len(community) < 3:
        return False
    suits = [c[1].lower() if len(c) >= 2 else "" for c in community[:3]]
    ranks = []
    rank_map = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"T":10,"J":11,"Q":12,"K":13,"A":14}
    for c in community[:3]:
        ranks.append(rank_map.get(c[0].upper(), 0))
    flush_draw = max(suits.count(s) for s in set(suits) if s) >= 2
    ranks_sorted = sorted(set(ranks))
    straight_draw = len(ranks_sorted) >= 2 and (ranks_sorted[-1] - ranks_sorted[0]) <= 4
    return flush_draw or straight_draw


# ── preflop ───────────────────────────────────────────────────────────

def _preflop(ctx: GameContext, ehs: float, ip: bool) -> Decision:
    r = _rules().get("preflop", {})
    spec = r.get("speculative_hand_see_flop", {})
    spec_enabled = spec.get("enabled", False)
    suited_bonus, is_connector, is_broadway = _speculative_ehs_bonus(ctx.hole_cards)

    ehs_eff = ehs + (suited_bonus if spec_enabled else 0.0)

    # Midgame minimum-call: cheap speculative calls below normal open threshold
    if ctx.call_amount > 0:
        mid_pf = assess_preflop_min_call(
            ehs, ctx.call_amount, ctx.bb_size, ctx.hole_cards, ctx.is_facing_raise
        )
        if mid_pf is not None:
            return mid_pf

    if ctx.is_facing_raise:
        raise_level = ctx.call_amount / max(1.0, ctx.bb_size)
        if raise_level >= 15:  # facing 4bet+
            v4bet_call = float(r.get("vs_4bet_call_min_ehs", 0.78))
            if ehs_eff >= v4bet_call:
                return "call", to_call(ctx), f"A 4bet call ehs={ehs:.2f}"
            return "fold", 0.0, f"A 4bet fold ehs={ehs:.2f}"

        call_min  = r.get("vs_raise_call_ip_min_ehs" if ip else "vs_raise_call_oop_min_ehs", 0.58)
        bet3_min  = r.get("vs_raise_3bet_min_ehs", 0.72)
        bet3_mult = r.get("vs_raise_3bet_sizing_mult", 3.0)

        if ehs_eff >= bet3_min:
            allin_pf = float(r.get("allin_threshold_ehs", 0.97))
            amt = raise_to(bet3_mult, ctx)
            # Cap 3bet when it would accidentally shove stack (e.g. facing large raise)
            if ctx.stack > 0 and amt >= ctx.stack * 0.85 and ehs < allin_pf:
                amt = clamp(ctx.bb_size * 9, ctx)
            return aggr(ctx), amt, f"A 3bet value ehs={ehs:.2f}"
        if ehs_eff >= call_min:
            return "call", to_call(ctx), f"A call {'ip' if ip else 'oop'} ehs={ehs:.2f}"
        if spec_enabled and ip and (is_connector or is_broadway):
            conn_min = spec.get("connector_see_flop_min_ehs", 0.44)
            bway_min = spec.get("broadway_see_flop_min_ehs", 0.48)
            see_flop_min = bway_min if is_broadway else conn_min
            if ehs_eff >= see_flop_min:
                return "call", to_call(ctx), f"A spec call {'connector' if is_connector else 'broadway'} ehs={ehs:.2f}"
        return "fold", 0.0, f"A fold preflop ehs={ehs:.2f}"

    # Position-aware open range
    open_min = _pos_open_min(ctx)
    sizing   = r.get("open_sizing_bb", 2.5)
    if ehs_eff >= open_min:
        return aggr(ctx), open_bb(sizing, ctx), f"A open {ctx.position} ehs={ehs:.2f}"

    # BB defense: wide defend since BB already has money in
    if ctx.position.upper() == "BB" and ctx.call_amount <= ctx.bb_size * 1.1:
        bb_def = r.get("open_bb_defend_min_ehs", 0.45)
        if ehs_eff >= bb_def and has(ctx, "call"):
            return "call", to_call(ctx), f"A BB defend ehs={ehs:.2f}"

    # BB isolation raise with premium holdings
    bb_raise_min = float(r.get("bb_raise_min_ehs", 0.65))
    if ctx.position.upper() == "BB" and ehs_eff >= bb_raise_min and has(ctx, "raise"):
        return aggr(ctx), open_bb(sizing, ctx), f"A BB squeeze ehs={ehs:.2f}"

    # Speculative hands: open from late position to see cheap flop
    if spec_enabled and ip and (is_connector or is_broadway):
        conn_min = spec.get("connector_see_flop_min_ehs", 0.44)
        bway_min = spec.get("broadway_see_flop_min_ehs", 0.48)
        see_flop_min = bway_min if is_broadway else conn_min
        if ehs_eff >= see_flop_min and has(ctx, "raise"):
            return aggr(ctx), open_bb(sizing, ctx), f"A spec open {'connector' if is_connector else 'broadway'} ehs={ehs:.2f}"

    return safe_check_fold(ctx)


# ── aggression tier helper ────────────────────────────────────────────

def _aggr_tier(ehs: float, pot: float, ctx: GameContext, street: str = "") -> tuple[float, str] | None:
    """Return (bet_size, reason) for the highest justified aggression tier, or None.

    Tiers from strategy_rules.json aggression block:
      tier1: EHS >= river_allin_min_ehs → all-in (street-aware: raised on flop/turn)
      tier2: EHS >= large_bet_min_ehs   → large bet (1.1x pot)
      tier3: EHS >= medium_bet_min_ehs  → medium bet (0.75x pot)

    PvP guard: on flop/turn at 100BB stacks, use flop_allin_min_ehs (default 0.90)
    to prevent committing stack with medium-strong hands on early streets.
    """
    ag = _rules().get("aggression", {})
    if not ag:
        return None

    allin_min  = float(ag.get("river_allin_min_ehs", 0.85))
    large_min  = float(ag.get("large_bet_min_ehs", 0.85))
    large_mult = float(ag.get("large_bet_pot_mult", 1.2))
    med_min    = float(ag.get("medium_bet_min_ehs", 0.65))
    med_mult   = float(ag.get("medium_bet_pot_mult", 0.75))

    # Street guard: raise all-in threshold on early streets to avoid overcommitting.
    # flop/turn require flop_allin_min_ehs (default 0.90 — tighter than river).
    if street in ("flop", "turn"):
        allin_min = max(allin_min, float(ag.get("flop_allin_min_ehs", 0.90)))

    # Hard cap: never go all-in below the global push minimum.
    # This prevents _aggr_tier from bypassing the 0.85+ threshold (hand 6543 fix).
    push_min = float(ag.get("all_in_push_min_ehs", 0.85))
    allin_min = max(allin_min, push_min)

    allin_min_pot = float(ag.get("allin_min_pot", 0))
    pot_ok = allin_min_pot <= 0 or pot >= allin_min_pot
    if ehs >= allin_min and pot_ok:
        all_in_size = ctx.stack
        return clamp(all_in_size, ctx), f"A aggr tier1 ALLIN ehs={ehs:.2f} pot={pot:.0f} st={street}"

    if ehs >= large_min:
        size = clamp(pot * large_mult, ctx)
        return size, f"A aggr tier2 large-bet ehs={ehs:.2f}"

    if ehs >= med_min:
        size = clamp(pot * med_mult, ctx)
        return size, f"A aggr tier3 med-bet ehs={ehs:.2f}"

    return None


# ── flop ──────────────────────────────────────────────────────────────

def _flop(ctx: GameContext, ehs: float, ip: bool) -> Decision:
    r    = _rules().get("flop", {})
    ag   = _rules().get("aggression", {})
    hits = board_hits(ctx)
    bluff = r.get("bluff_enabled", False)
    allin_min_pot = float(ag.get("allin_min_pot", 0))

    # Midgame: draw analysis, all-in calls, semi-bluffs, bluff catches
    mid = midgame_postflop(ctx, ehs, "flop", allin_min_pot)
    if mid is not None:
        return mid

    # Draw analysis for enhanced EHS on raise/bet decisions
    _da = analyze_draws(ctx.hole_cards, ctx.community_cards)
    ehs_mid = ehs + _da.equity_boost  # boosted EHS for raise thresholds

    if ctx.call_amount > 0:
        raise_min  = r.get("facing_bet_raise_min_ehs", 0.83)
        raise_mult = r.get("facing_bet_raise_mult", 3.0)
        call_min   = r.get("facing_bet_call_min_ehs", 0.47)
        fold_max   = r.get("facing_bet_fold_max_ehs", 0.35)
        # Multiway penalty + opponent type modifier
        n_opps = len(ctx.opponent_ids)
        mw_penalty = float(_rules().get("flop", {}).get("multiway_adjust_call_min_ehs_penalty", 0)) * max(0, n_opps - 1)
        call_min = call_min + mw_penalty + _opp_modifier()

        # Aggressive raise tier: if very strong, raise bigger than standard
        flop_aggr_min = float(ag.get("flop_raise_aggressive_min_ehs", 0.85))
        if ehs >= flop_aggr_min:
            tier = _aggr_tier(ehs, ctx.pot, ctx, street="flop")
            if tier:
                size, reason = tier
                return aggr(ctx), size, reason
        if ehs >= raise_min:
            return aggr(ctx), raise_to(raise_mult, ctx), f"A flop raise ehs={ehs:.2f}"
        if ehs >= call_min:
            return "call", to_call(ctx), f"A flop call ehs={ehs:.2f}"
        # Pot-odds layer: call even below threshold if equity covers price
        po_call, po_reason = _pot_odds_call(ehs, ctx.call_amount, ctx.pot)
        if po_call and ehs > fold_max:
            return "call", to_call(ctx), f"A flop pot-odds call {po_reason}"
        return "fold", 0.0, f"A flop fold ehs={ehs:.2f} {po_reason}"

    trap_min   = r.get("no_bet_trap_min_ehs", 0.87)
    value_min  = r.get("no_bet_value_min_ehs", 0.55)
    ob_min     = r.get("no_bet_board_hit_overbet_min_ehs", 0.72)
    ob_pot     = r.get("no_bet_board_hit_overbet_pot", 1.1)
    cbet_min   = r.get("no_bet_cbet_min_ehs", 0.33)
    cbet_freq  = float(r.get("no_bet_cbet_frequency", 0.60))
    bluff_max  = r.get("bluff_max_ehs", 0.28)
    bluff_freq_raw = r.get("bluff_frequency", 0.30)
    bluff_pot  = r.get("bluff_sizing_pot", 0.50)

    # Board-texture-aware sizing: small on dry, large on wet (protect equity)
    wet = _board_is_wet(ctx.community_cards)
    value_pot = r.get("no_bet_value_sizing_wet" if wet else "no_bet_value_sizing_dry",
                      r.get("no_bet_value_sizing_pot", 0.55))

    adj_trap  = _adj(trap_min, ip)
    adj_value = _adj(value_min, ip)

    # Slow-play (trap): made monster → check to induce opponent bets
    # Lower trap threshold when we have a full house / quads / SF
    trap_threshold = adj_trap
    if _da.hand_category >= 6 and ehs >= 0.85:  # full house or better → always trap once
        trap_threshold = 0.85
    if ehs >= trap_threshold and has(ctx, "check"):
        return "check", 0.0, f"A flop trap ehs={ehs:.2f} [{_da.hand_category_name}]"

    # Exploit pot-guard shove: only active when exploit_allin_min_ehs > 0 in rules
    _exploit_min = float(ag.get("exploit_allin_min_ehs", 0.0))
    if _exploit_min > 0 and allin_min_pot > 0 and ctx.pot >= allin_min_pot and ehs >= _exploit_min:
        _shove = clamp(ctx.stack, ctx)
        if _shove > 0:
            return aggr(ctx), _shove, f"A exploit shove flop pot={ctx.pot:.0f} ehs={ehs:.2f}"

    # All-in gate: fires before bet_intelligence_decide so strong hands shove instead of sizing normally.
    # Opponents fold to all-in ~95% — all-in is always +EV above the push threshold.
    _push_min = float(ag.get("all_in_push_min_ehs", 0.85))
    _pot_ok   = allin_min_pot <= 0 or ctx.pot >= allin_min_pot
    if ehs >= _push_min and _pot_ok and ctx.stack > 0:
        _sz = clamp(ctx.stack, ctx)
        if _sz > 0:
            return aggr(ctx), _sz, f"A first-act ALLIN flop ehs={ehs:.2f} pot={ctx.pot:.0f}"

    # ── Bet Intelligence: EV-optimal sizing (opponent-adaptive) ──────────
    # Runs before generic overbet so opponent type drives sizing choice.
    bi = bet_intelligence_decide(ctx, ehs, _da, "flop")
    if bi is not None:
        return bi

    # Fallback overbet: board hit + no bet_intelligence override
    if hits >= 1 and ehs >= ob_min and _rules().get("adjustments", {}).get("board_hit_overbet_enabled", True):
        return aggr(ctx), clamp(ctx.pot * ob_pot, ctx), f"A flop overbet board-hit ehs={ehs:.2f}"

    # Value tier: use draw-enhanced EHS so flush/straight draws bet for semi-value
    if ehs_mid >= adj_value:
        tier = _aggr_tier(ehs, ctx.pot, ctx)   # sizing uses raw ehs
        if tier:
            size, reason = tier
            return aggr(ctx), size, reason
        label = "draw" if _da.equity_boost > 0.05 else ("wet" if wet else "dry")
        return aggr(ctx), bet_pot(value_pot, ctx), f"A flop value {label} ehs={ehs:.2f}+{_da.equity_boost:.2f}"

    # C-bet range tier: semi-bluff / range bet (draws, pairs, overcards)
    adj_cbet = _adj(cbet_min, ip)
    if ehs >= adj_cbet and _get_rng().random() < cbet_freq:
        cbet_size = r.get("no_bet_value_sizing_dry", 0.40)  # small size for range cbets
        return aggr(ctx), bet_pot(cbet_size, ctx), f"A flop cbet ehs={ehs:.2f}"

    # Pure bluff tier: complete air, low frequency
    bluff_freq = float(bluff) if isinstance(bluff, float) else (bluff_freq_raw if bluff else 0)
    if bluff and ehs <= bluff_max and _get_rng().random() < bluff_freq:
        return aggr(ctx), bet_pot(bluff_pot, ctx), f"A flop bluff ehs={ehs:.2f}"

    return "check", 0.0, f"A flop check ehs={ehs:.2f}"


# ── turn ──────────────────────────────────────────────────────────────

def _turn(ctx: GameContext, ehs: float, ip: bool) -> Decision:
    r    = _rules().get("turn", {})
    ag   = _rules().get("aggression", {})
    bluff = r.get("bluff_enabled", False)
    allin_min_pot = float(ag.get("allin_min_pot", 0))

    # Midgame: draw analysis, made-hand all-in, semi-bluffs, bluff catches
    mid = midgame_postflop(ctx, ehs, "turn", allin_min_pot)
    if mid is not None:
        return mid

    _da = analyze_draws(ctx.hole_cards, ctx.community_cards)
    ehs_mid = ehs + _da.equity_boost

    if ctx.call_amount > 0:
        raise_min  = r.get("facing_bet_raise_min_ehs", 0.86)
        raise_mult = r.get("facing_bet_raise_mult", 3.0)
        call_min   = r.get("facing_bet_call_min_ehs", 0.47)
        fold_max   = r.get("facing_bet_fold_max_ehs", 0.44)

        if ehs >= raise_min:
            tier = _aggr_tier(ehs, ctx.pot, ctx)
            if tier:
                size, reason = tier
                return aggr(ctx), size, reason
            return aggr(ctx), raise_to(raise_mult, ctx), f"A turn raise ehs={ehs:.2f}"
        if ehs >= call_min:
            return "call", to_call(ctx), f"A turn call ehs={ehs:.2f}"
        po_call, po_reason = _pot_odds_call(ehs, ctx.call_amount, ctx.pot)
        if po_call and ehs > fold_max:
            return "call", to_call(ctx), f"A turn pot-odds call {po_reason}"
        return "fold", 0.0, f"A turn fold ehs={ehs:.2f} {po_reason}"

    trap_min  = r.get("no_bet_trap_min_ehs", 0.88)
    value_min = r.get("no_bet_value_min_ehs", 0.55)
    value_pot = r.get("no_bet_value_sizing_pot", 0.75)
    # Barrel tier: semi-bluffs that picked up equity or stayed strong on turn
    barrel_min  = r.get("no_bet_barrel_min_ehs", 0.40)
    barrel_freq = float(r.get("no_bet_barrel_frequency", 0.40))
    bluff_max = r.get("bluff_max_ehs", 0.22)
    bluff_pot = r.get("bluff_sizing_pot", 0.65)

    # Slow-play trap on turn with made monster
    trap_threshold = _adj(trap_min, ip)
    if _da.hand_category >= 6 and ehs >= 0.88:
        trap_threshold = min(trap_threshold, 0.88)
    if ehs >= trap_threshold and has(ctx, "check"):
        return "check", 0.0, f"A turn trap ehs={ehs:.2f} [{_da.hand_category_name}]"

    # Exploit pot-guard shove: only active when exploit_allin_min_ehs > 0 in rules
    _exploit_min = float(ag.get("exploit_allin_min_ehs", 0.0))
    if _exploit_min > 0 and allin_min_pot > 0 and ctx.pot >= allin_min_pot and ehs >= _exploit_min:
        _shove = clamp(ctx.stack, ctx)
        if _shove > 0:
            return aggr(ctx), _shove, f"A exploit shove turn pot={ctx.pot:.0f} ehs={ehs:.2f}"

    # All-in gate: fires before bet_intelligence_decide so strong hands shove instead of sizing normally.
    _push_min = float(ag.get("all_in_push_min_ehs", 0.85))
    _pot_ok   = allin_min_pot <= 0 or ctx.pot >= allin_min_pot
    if ehs >= _push_min and _pot_ok and ctx.stack > 0:
        _sz = clamp(ctx.stack, ctx)
        if _sz > 0:
            return aggr(ctx), _sz, f"A first-act ALLIN turn ehs={ehs:.2f} pot={ctx.pot:.0f}"

    # ── Bet Intelligence: EV-optimal sizing (value/semi-bluff/bluff/pot-ctrl) ──
    bi = bet_intelligence_decide(ctx, ehs, _da, "turn")
    if bi is not None:
        return bi

    # Value bet: use draw-boosted EHS so strong draws bet for semi-value
    if ehs_mid >= _adj(value_min, ip):
        tier = _aggr_tier(ehs, ctx.pot, ctx)
        if tier:
            size, reason = tier
            return aggr(ctx), size, reason
        label = "draw" if _da.equity_boost > 0.05 else "value"
        return aggr(ctx), bet_pot(value_pot, ctx), f"A turn {label} ehs={ehs:.2f}+{_da.equity_boost:.2f}"

    # Barrel: continue with semi-bluffs that have equity (flush draws, OESD, etc.)
    adj_barrel = _adj(barrel_min, ip)
    if ehs >= adj_barrel and _get_rng().random() < barrel_freq:
        return aggr(ctx), bet_pot(value_pot, ctx), f"A turn barrel ehs={ehs:.2f}"

    bluff_freq_t = float(bluff) if isinstance(bluff, float) else (r.get("bluff_frequency", 0.20) if bluff else 0)
    if bluff and ehs <= bluff_max and _get_rng().random() < bluff_freq_t:
        return aggr(ctx), bet_pot(bluff_pot, ctx), f"A turn bluff ehs={ehs:.2f}"
    return "check", 0.0, f"A turn check ehs={ehs:.2f}"


# ── river ──────────────────────────────────────────────────────────────

def _river(ctx: GameContext, ehs: float, ip: bool) -> Decision:
    r    = _rules().get("river", {})
    ag   = _rules().get("aggression", {})
    bluff = r.get("bluff_enabled", False)
    allin_min_pot = float(ag.get("allin_min_pot", 0))

    # Midgame: all-in calls, made-hand push, bluff catches on river
    mid = midgame_postflop(ctx, ehs, "river", allin_min_pot)
    if mid is not None:
        return mid

    _da = analyze_draws(ctx.hole_cards, ctx.community_cards)
    ehs_mid = ehs + _da.equity_boost

    if ctx.call_amount > 0:
        raise_min  = r.get("facing_bet_raise_min_ehs", 0.85)
        raise_mult = r.get("facing_bet_raise_mult", 2.5)
        call_min   = r.get("facing_bet_call_min_ehs", 0.52)
        fold_max   = r.get("facing_bet_fold_max_ehs", 0.50)

        if ehs >= raise_min:
            tier = _aggr_tier(ehs, ctx.pot, ctx)
            if tier:
                size, reason = tier
                return aggr(ctx), size, reason
            return aggr(ctx), raise_to(raise_mult, ctx), f"A river raise ehs={ehs:.2f}"
        if ehs >= call_min:
            return "call", to_call(ctx), f"A river call ehs={ehs:.2f}"
        # River: pot odds is the final arbiter — no more cards after this
        po_call, po_reason = _pot_odds_call(ehs, ctx.call_amount, ctx.pot)
        if po_call and ehs > fold_max:
            return "call", to_call(ctx), f"A river pot-odds call {po_reason}"
        return "fold", 0.0, f"A river fold ehs={ehs:.2f} {po_reason}"

    value_min = r.get("no_bet_value_min_ehs", 0.69)
    value_pot = r.get("no_bet_value_sizing_pot", 0.75)
    bluff_max = r.get("bluff_max_ehs", 0.20)
    bluff_pot = r.get("bluff_sizing_pot", 0.50)

    # River value: go all-in only when pot-committed (SPR ≤ 2.5) + full house or better.
    # Flush (category 5) at deep stacks should escalate via overbet, not jam.
    if _da.hand_category >= 6 and ehs >= 0.88:
        spr = ctx.stack / max(1.0, ctx.pot)
        if spr <= 2.5:
            size = clamp(ctx.stack, ctx)
            if size > 0:
                return aggr(ctx), size, f"A river nuts allin [{_da.hand_category_name}] ehs={ehs:.2f}"

    # Exploit pot-guard shove: only active when exploit_allin_min_ehs > 0 in rules
    _exploit_min = float(ag.get("exploit_allin_min_ehs", 0.0))
    if _exploit_min > 0 and allin_min_pot > 0 and ctx.pot >= allin_min_pot and ehs >= _exploit_min:
        _shove = clamp(ctx.stack, ctx)
        if _shove > 0:
            return aggr(ctx), _shove, f"A exploit shove river pot={ctx.pot:.0f} ehs={ehs:.2f}"

    # All-in gate: fires before bet_intelligence_decide so strong hands shove instead of sizing normally.
    _push_min = float(ag.get("all_in_push_min_ehs", 0.85))
    _pot_ok   = allin_min_pot <= 0 or ctx.pot >= allin_min_pot
    if ehs >= _push_min and _pot_ok and ctx.stack > 0:
        _sz = clamp(ctx.stack, ctx)
        if _sz > 0:
            return aggr(ctx), _sz, f"A first-act ALLIN river ehs={ehs:.2f} pot={ctx.pot:.0f}"

    # ── Bet Intelligence: EV-optimal river sizing (value/bluff/SPR-shove) ──
    bi = bet_intelligence_decide(ctx, ehs, _da, "river")
    if bi is not None:
        return bi

    if ehs >= _adj(value_min, ip):
        tier = _aggr_tier(ehs, ctx.pot, ctx)
        if tier:
            size, reason = tier
            return aggr(ctx), size, reason
        return aggr(ctx), bet_pot(value_pot, ctx), f"A river value ehs={ehs:.2f}"

    bluff_freq_r = float(bluff) if isinstance(bluff, float) else (r.get("bluff_frequency", 1.0) if bluff else 0)
    if bluff and ehs <= bluff_max and _get_rng().random() < bluff_freq_r:
        return aggr(ctx), bet_pot(bluff_pot, ctx), f"A river bluff ehs={ehs:.2f}"
    return "check", 0.0, f"A river check ehs={ehs:.2f}"


# ── entry ──────────────────────────────────────────────────────────────

def decide(ctx: GameContext, ehs: float, bb_depth: float) -> Decision:
    ip = is_ip(ctx)
    s  = ctx.street
    if s == "preflop": action, amount, reason = _preflop(ctx, ehs, ip)
    elif s == "flop":  action, amount, reason = _flop(ctx, ehs, ip)
    elif s == "turn":  action, amount, reason = _turn(ctx, ehs, ip)
    else:              action, amount, reason = _river(ctx, ehs, ip)
    # Record per-hand state for probe / escalation / deterioration logic
    try:
        from agent.strategies.midgame import record_hand_state
        record_hand_state(ctx, s, ehs, action, amount)
    except Exception:
        pass
    return action, amount, reason
"""agent/strategies/midgame.py — adaptive in-game reasoning module.

Augments the rule-based adaptive.py with draw-aware, opponent-read-enhanced
decisions for all streets. Called from adaptive.py; returns a Decision override
or None (meaning: defer to the existing rule logic).

Key capabilities:
  1. Flush / straight draw detection + outs-based equity boost
  2. Made hand detection (flush, straight, full house, quads, SF)
  3. Opponent bluff detection via sizing tells + tracked stats
  4. All-in confidence: push when made, call only when equity justifies it
  5. Preflop minimum-call assessment for cheap speculative spots
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from engine.hand_eval import best_hand, card_rank, card_suit

# Per-hand semi-bluff raise counter and EHS history.
# Using module-level dicts (not threading.local) so state persists correctly
# across thread-pool workers used by run_in_executor.
_semi_bluff_counts: dict = {}   # (table_id, hand_number) → int
_hand_histories: dict = {}       # (table_id, hand_number) → dict
_state_lock = threading.Lock()


def _hkey(ctx) -> tuple:
    return (getattr(ctx, "table_id", ""), getattr(ctx, "hand_number", None))


def _get_history(ctx) -> dict:
    k = _hkey(ctx)
    with _state_lock:
        if k not in _hand_histories:
            _hand_histories[k] = {
                "flop_ehs":          None,
                "streets_bet":       0,
                "last_bet_fraction": 0.0,
                "last_bet_pot":      0.0,
            }
            if len(_hand_histories) > 40:
                for old in sorted(_hand_histories.keys())[:-40]:
                    del _hand_histories[old]
        return _hand_histories[k]


def record_hand_state(ctx, street: str, ehs: float, action: str, amount: float) -> None:
    """Update within-hand state after each decision. Called from adaptive.py."""
    hist = _get_history(ctx)
    pot  = max(1.0, ctx.pot)
    if street == "flop" and hist["flop_ehs"] is None:
        hist["flop_ehs"] = ehs
    if action in ("bet", "raise") and amount > 0:
        hist["streets_bet"]       += 1
        hist["last_bet_fraction"]  = amount / pot
        hist["last_bet_pot"]       = pot

Decision = Tuple[str, float, str]

_CATEGORY_NAMES = [
    "high_card", "pair", "two_pair", "trips",
    "straight", "flush", "full_house", "quads", "straight_flush",
]


def _get_opp_stats():
    try:
        from agent.arbiter import _tl
        return getattr(_tl, "opponent_stats", None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Draw analysis
# ---------------------------------------------------------------------------

@dataclass
class DrawAnalysis:
    flush_draw: bool = False      # 4 cards to a flush (need 1 more)
    flush_backdoor: bool = False  # 3 cards to a flush (need 2 more)
    flush_complete: bool = False  # made flush (5+ of same suit, holding ≥1)
    flush_outs: int = 0
    flush_suit: str = ""

    straight_draw: str = ""       # "oesd" | "double_gutshot" | "gutshot" | ""
    straight_complete: bool = False
    straight_outs: int = 0

    hand_category: int = -1       # 0-8 from hand_eval score >> 20; -1 = not enough cards
    hand_category_name: str = "none"

    equity_boost: float = 0.0    # add to EHS for raise/bet decisions
    cards_to_come: int = 0       # board cards still to be dealt


def analyze_draws(hole_cards: List[str], community_cards: List[str]) -> DrawAnalysis:
    """
    Analyze hole + community cards for draws and made hands.

    Core insight: if the agent holds 2h 7h and board shows Kh Jh 4c, that is
    4 cards to a flush → 9 outs → equity_boost ≈ +0.18 (flop: 9 * 2 cards * 2%).
    """
    da = DrawAnalysis()
    all_cards = hole_cards + community_cards
    if not hole_cards or len(all_cards) < 2:
        return da

    da.cards_to_come = max(0, 5 - len(community_cards))

    # ── Made hand (need ≥ 5 cards total) ───────────────────────────────────
    if len(all_cards) >= 5:
        try:
            score = best_hand(hole_cards, community_cards)
            cat = min(score >> 20, 8)
            da.hand_category = cat
            da.hand_category_name = _CATEGORY_NAMES[cat]
        except Exception:
            da.hand_category = -1

    # ── Flush analysis ─────────────────────────────────────────────────────
    hole_suits = [card_suit(c) for c in hole_cards if len(c) >= 2]
    all_suits  = [card_suit(c) for c in all_cards  if len(c) >= 2]

    for suit in set(all_suits):
        if hole_suits.count(suit) == 0:
            continue                          # agent doesn't hold this suit
        total = all_suits.count(suit)
        if total >= 5:
            da.flush_complete = True
            da.flush_suit = suit
            da.flush_outs = 0
        elif total == 4 and not da.flush_complete:
            da.flush_draw = True
            da.flush_suit = suit
            da.flush_outs = 13 - total        # 9 outs when holding 2 of suit
        elif total == 3 and len(community_cards) <= 3 and not da.flush_draw:
            da.flush_backdoor = True
            da.flush_outs = max(da.flush_outs, 10)

    # Sync flush_complete with hand category
    if da.hand_category in (5, 8):
        da.flush_complete = True

    # ── Straight analysis ──────────────────────────────────────────────────
    if da.hand_category in (4, 8):
        da.straight_complete = True

    if not da.straight_complete:
        all_rank_ints = sorted(set(card_rank(c) for c in all_cards if c))
        run = _max_consecutive(all_rank_ints)
        if run >= 4:
            da.straight_draw = "oesd"
            da.straight_outs = 8
        elif run >= 3 and _count_gutshots(all_rank_ints) >= 2:
            da.straight_draw = "double_gutshot"
            da.straight_outs = 8
        elif run >= 3:
            da.straight_draw = "gutshot"
            da.straight_outs = 4

    # ── Equity boost: Rule-of-2 / Rule-of-4 ──────────────────────────────
    # Approximation: outs * 4% if 2 cards to come, outs * 2% if 1 card to come.
    # Caps are street-specific: turn (1 card) gets half the flop cap.
    # River (0 cards): missed draws get 0 — no improvement possible.
    mult = 4 if da.cards_to_come >= 2 else 2

    if da.flush_complete or da.straight_complete or da.hand_category >= 5:
        da.equity_boost = 0.18         # made monster: max confidence signal
    elif da.flush_draw and da.straight_draw in ("oesd", "double_gutshot"):
        if da.cards_to_come > 0:
            cap = 0.22 if da.cards_to_come >= 2 else 0.11
            da.equity_boost = min(cap, 15 * mult / 100)
    elif da.flush_draw:
        if da.cards_to_come > 0:
            cap = 0.15 if da.cards_to_come >= 2 else 0.09
            da.equity_boost = min(cap, da.flush_outs * mult / 100)
    elif da.straight_draw == "oesd":
        if da.cards_to_come > 0:
            cap = 0.12 if da.cards_to_come >= 2 else 0.06
            da.equity_boost = min(cap, da.straight_outs * mult / 100)
    elif da.straight_draw in ("gutshot", "double_gutshot"):
        if da.cards_to_come > 0:
            cap = 0.08 if da.cards_to_come >= 2 else 0.04
            da.equity_boost = min(cap, da.straight_outs * mult / 100)
    elif da.flush_backdoor:
        da.equity_boost = 0.03
    elif da.hand_category == 3:   # trips
        da.equity_boost = 0.05
    elif da.hand_category == 2:   # two pair
        da.equity_boost = 0.03

    return da


def _max_consecutive(ranks: List[int]) -> int:
    """Return the length of the longest consecutive run in a sorted unique list."""
    if not ranks:
        return 0
    unique = sorted(set(ranks))
    best = cur = 1
    for i in range(1, len(unique)):
        cur = cur + 1 if unique[i] - unique[i - 1] == 1 else 1
        best = max(best, cur)
    return best


def _count_gutshots(ranks: List[int]) -> int:
    """Count 5-card windows that contain ≥4 of our ranks (each = a gutshot)."""
    unique = sorted(set(ranks))
    count = 0
    for low in unique:
        if sum(1 for r in unique if low <= r <= low + 4) >= 4:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Bluff detection
# ---------------------------------------------------------------------------

@dataclass
class BluffRead:
    bluff_probability: float = 0.0
    is_likely_bluff: bool = False
    is_likely_nuts: bool = False
    call_as_bluff_catch: bool = False
    reason: str = ""


def detect_bluff(
    call_amount: float,
    pot: float,
    ehs: float,
    is_river: bool = False,
) -> BluffRead:
    """
    Estimate probability that opponent's bet is a bluff.

    Uses bet-to-pot sizing (polarization tells) plus opponent stats
    stored in arbiter thread-local (hands_seen, AF, WTSD, river bluff history).
    """
    br = BluffRead()
    if call_amount <= 0 or pot <= 0:
        return br

    btp = call_amount / max(1.0, pot)          # bet-to-pot ratio

    # Sizing heuristic: overbets are polarized; small bets lean value
    if btp >= 1.5:
        base = 0.42
        br.reason = f"overbet {btp:.1f}x; "
    elif btp >= 0.9:
        base = 0.32
        br.reason = f"bigbet {btp:.1f}x; "
    elif btp >= 0.5:
        base = 0.27
        br.reason = f"halfpot {btp:.1f}x; "
    else:
        base = 0.22
        br.reason = f"smallbet {btp:.1f}x; "

    bluff_prob = base
    opp = _get_opp_stats()

    if opp and opp.hands_seen >= 10:
        # Aggression factor: (bets + raises) / calls
        total_agg  = opp.bet_count + opp.raise_count
        total_pass = max(1, opp.call_count + opp.check_count)
        af = total_agg / total_pass
        if af > 2.5:
            bluff_prob += 0.10
            br.reason += f"AF={af:.1f}↑; "
        elif af < 0.7:
            bluff_prob -= 0.08
            br.reason += f"AF={af:.1f}↓; "

        # River bluff-catch history
        if opp.river_bet_count > 3:
            rbluff = opp.river_bluff_caught / opp.river_bet_count
            bluff_prob += (rbluff - 0.20) * 0.6
            if rbluff > 0.30:
                br.reason += f"rbluff={rbluff:.0%}; "

        # WTSD: low = doesn't go to showdown often → may be bluffing
        if opp.wtsd_opps > 5:
            wtsd = opp.wtsd_count / opp.wtsd_opps
            if wtsd < 0.22:
                bluff_prob += 0.06
            elif wtsd > 0.55:
                bluff_prob -= 0.06

    bluff_prob = max(0.05, min(0.90, bluff_prob))
    br.bluff_probability = bluff_prob
    br.is_likely_bluff   = bluff_prob >= 0.42
    br.is_likely_nuts    = bluff_prob <= 0.18

    # Bluff-catch EV: bluff_prob × pot − (1 − bluff_prob) × call > 0?
    required_equity  = call_amount / max(1.0, call_amount + pot)
    bluff_catch_ev   = bluff_prob * pot - (1 - bluff_prob) * call_amount
    br.call_as_bluff_catch = bluff_catch_ev > 0 and ehs > required_equity * 0.75

    return br


# ---------------------------------------------------------------------------
# All-in confidence
# ---------------------------------------------------------------------------

def should_go_allin(
    ehs: float,
    da: DrawAnalysis,
    street: str,
    pot: float,
    allin_min_pot: float = 0.0,
) -> Tuple[bool, str]:
    """
    Should agent push all-in?  Made strong hands lower the EHS threshold.
    Returns (go_allin, reason).
    """
    thresholds = {"preflop": 0.88, "flop": 0.85, "turn": 0.84, "river": 0.82}
    thresh = thresholds.get(street, 0.90)

    if da.hand_category >= 6:      # full house / quads / straight flush
        thresh -= 0.08
    elif da.hand_category == 5:    # flush
        thresh -= 0.05
    elif da.hand_category == 4:    # straight
        thresh -= 0.03
    elif da.flush_complete or da.straight_complete:
        thresh -= 0.03

    if allin_min_pot > 0 and pot < allin_min_pot:
        return False, f"mid:allin blocked pot={pot:.0f}<min={allin_min_pot:.0f}"

    eff = ehs + da.equity_boost
    if eff >= thresh:
        if da.hand_category >= 4:
            tag = f"[{da.hand_category_name}]"
        elif da.equity_boost > 0.05:
            tag = f"[draw+{da.equity_boost:.2f}]"
        else:
            tag = ""
        return True, f"mid:allin eff={eff:.2f}>={thresh:.2f} {tag}".strip()

    return False, f"mid:allin skip eff={eff:.2f}<{thresh:.2f}"


def should_call_allin(
    ehs: float,
    da: DrawAnalysis,
    street: str,
    pot: float,
    call_amount: float,
) -> Tuple[bool, str]:
    """
    Should agent call opponent's all-in?
    Requires effective equity to exceed pot-odds with a confidence margin.
    """
    required = call_amount / max(1.0, call_amount + pot)
    eff      = ehs + da.equity_boost

    # Bots fold to all-in 95% — when they shove, they have real equity.
    # Require a clear edge above pot-odds; 1.08x margin is sufficient.
    min_ehs = max(0.58, required * 1.08)

    if da.hand_category >= 6:      # full house / quads / SF — near-certain winner
        min_ehs = max(0.50, required * 1.02)
    elif da.hand_category == 5:    # flush
        min_ehs = max(0.54, required * 1.04)
    elif da.hand_category == 4:    # straight
        min_ehs = max(0.56, required * 1.05)
    elif da.flush_draw and da.flush_outs >= 9 and da.cards_to_come >= 2:
        min_ehs = max(0.56, required * 1.07)

    if eff >= min_ehs:
        tag = f"[{da.hand_category_name}]" if da.hand_category >= 0 else ""
        return True, f"mid:call_allin eff={eff:.2f}>={min_ehs:.2f} {tag}".strip()

    return False, f"mid:fold_allin eff={eff:.2f}<{min_ehs:.2f} req={required:.2f}"


# ---------------------------------------------------------------------------
# Preflop minimum-call assessment
# ---------------------------------------------------------------------------

def assess_preflop_min_call(
    ehs: float,
    call_amount: float,
    bb_size: float,
    hole_cards: List[str],
    is_facing_raise: bool,
) -> Optional[Decision]:
    """
    Minimum-call / fold assessment for preflop cheap spots.

    Philosophy: with very cheap implied odds, speculative hands (suited
    connectors, low pairs, broadway) are worth calling to see the flop
    even when EHS is below the normal open threshold.

    Returns a Decision only for EHS 0.28–0.44 cheap-call spots.
    Above 0.44, the existing preflop logic is preferred.
    Returns None to defer to existing adaptive logic.
    """
    if not hole_cards or len(hole_cards) < 2:
        return None

    suits = [c[1].lower() for c in hole_cards if len(c) >= 2]
    ranks = [card_rank(c) for c in hole_cards]
    suited      = len(suits) == 2 and suits[0] == suits[1]
    gap         = abs(ranks[0] - ranks[1]) if len(ranks) >= 2 else 99
    is_pair     = gap == 0
    is_connector = 1 <= gap <= 2
    high_rank   = max(ranks) if ranks else 0

    spec_bonus = 0.0
    if suited:        spec_bonus += 0.03
    if is_connector:  spec_bonus += 0.02
    if high_rank >= 10: spec_bonus += 0.02  # broadway card (T/J/Q/K/A)
    if is_pair:       spec_bonus += 0.03

    eff = ehs + spec_bonus

    # Premium / medium: let existing preflop logic handle it
    if eff >= 0.46:
        return None

    cost_bb = call_amount / max(1.0, bb_size)

    if is_facing_raise:
        # Facing aggression: only call with respectable equity
        if eff >= 0.44:
            return None  # borderline — defer
        return "fold", 0.0, f"mid:pf fold vs raise eff={eff:.2f}"

    # Not facing a raise: cheap speculative call
    if cost_bb <= 1.0 and eff >= 0.30:
        return "call", call_amount, f"mid:pf min-call cheap eff={eff:.2f}"
    if cost_bb <= 2.5 and eff >= 0.37:
        return "call", call_amount, f"mid:pf min-call eff={eff:.2f} {cost_bb:.1f}BB"
    if eff < 0.32:
        return "fold", 0.0, f"mid:pf fold weak eff={eff:.2f}"

    return None  # borderline — defer to existing logic


# ---------------------------------------------------------------------------
# Main postflop override
# ---------------------------------------------------------------------------

def _semi_bluff_count(ctx) -> int:
    """Return how many semi-bluff raises we have already made this hand."""
    key = (getattr(ctx, "table_id", ""), getattr(ctx, "hand_number", None))
    with _state_lock:
        return _semi_bluff_counts.get(key, 0)


def _semi_bluff_increment(ctx) -> None:
    """Record that we made a semi-bluff raise this hand."""
    key = (getattr(ctx, "table_id", ""), getattr(ctx, "hand_number", None))
    with _state_lock:
        _semi_bluff_counts[key] = _semi_bluff_counts.get(key, 0) + 1
        if len(_semi_bluff_counts) > 20:
            for old in sorted(_semi_bluff_counts.keys())[:-20]:
                del _semi_bluff_counts[old]


def _semi_bluff_max(street: str) -> int:
    """Max semi-bluff raises allowed per hand from strategy rules."""
    try:
        from agent.strategies.adaptive import _rules
        r = _rules()
        return int(r.get(street, {}).get("semi_bluff_max_raises_per_hand",
               r.get("aggression", {}).get("semi_bluff_max_raises_per_hand", 1)))
    except Exception:
        return 1


def midgame_postflop(
    ctx,               # GameContext (avoid circular type annotation)
    ehs: float,
    street: str,
    allin_min_pot: float = 0.0,
) -> Optional[Decision]:
    """
    Compute draw analysis and return a high-confidence Decision override, or
    None to let the existing adaptive rule logic proceed.

    Override scenarios (in priority order):
      1. Facing near-all-in: call or fold based on confident equity check
      2. Made strong hand (≥straight) + no bet facing: push all-in
      3. Flush draw facing a bet: semi-bluff raise (once per hand max — prevents spiral)
      4. Medium equity + opponent likely bluffing: bluff-catch call
    """
    from agent.strategies.base import clamp, aggr, to_call

    da = analyze_draws(ctx.hole_cards, ctx.community_cards)

    # ── 1. Facing a near-all-in call ────────────────────────────────────
    # Primary guard: ≥40% of remaining stack (was 70% — too loose; 99/265-chip
    # calls at EHS 0.47 were slipping through against value-only bots).
    if ctx.call_amount > 0 and ctx.call_amount >= ctx.stack * 0.40:
        ok, reason = should_call_allin(ehs, da, street, ctx.pot, ctx.call_amount)
        if ok:
            return "call", to_call(ctx), reason
        return "fold", 0.0, reason

    # Secondary guard: medium-large bet (call > 60% of pot) even at shallow stack
    # commitment. Lowered from 80% — analysis showed calls with EHS 0.55-0.72
    # vs 60-80% pot bets cost 1900+ chips (opponents bet non-bluff ranges).
    if ctx.call_amount > 0 and ctx.call_amount >= ctx.pot * 0.60:
        ok, reason = should_call_allin(ehs, da, street, ctx.pot, ctx.call_amount)
        if ok:
            return "call", to_call(ctx), reason
        return "fold", 0.0, reason

    # ── 2. Made strong hand → all-in or defer to trap logic ─────────────
    if ctx.call_amount <= 0 and da.hand_category >= 4 and ehs >= 0.72:
        spr = ctx.stack / max(1.0, ctx.pot)
        # Full house / quads on flop or turn at deep stacks → slow-play (trap):
        # check to induce opponent bets — let adaptive trap logic handle the sizing.
        # Straight flush (cat 8): always all-in immediately — never slow-play the nuts.
        # On the river or at shallow stacks (SPR ≤ 4), push all-in for max value.
        opp_type = getattr(ctx, "opponent_type", "unknown")
        # Against calling stations / unknown bots, skip slow-play — push for max value.
        trap_ok = opp_type not in ("calling_station", "fish", "unknown")
        if da.hand_category in (6, 7) and street in ("flop", "turn") and spr > 4.0 and trap_ok:
            return None  # defer → adaptive trap → check
        ok, reason = should_go_allin(ehs, da, street, ctx.pot, allin_min_pot)
        if ok:
            size = clamp(ctx.stack, ctx)
            if size > 0:
                return aggr(ctx), size, reason

    # ── 3. Flush draw facing a bet → semi-bluff raise ───────────────────
    # Critical guard: only raise ONCE per hand with a draw.
    # Without this, facing a re-raise re-triggers the semi-bluff → geometric spiral
    # (24→72→192→504) that commits the entire stack at ~50% equity. (Hand 6565 fix.)
    already_raised = _semi_bluff_count(ctx)
    max_raises = _semi_bluff_max(street)

    if (
        ctx.call_amount > 0
        and da.flush_draw
        and da.flush_outs >= 9
        and already_raised < max_raises          # ← anti-spiral guard
        and not ctx.call_amount >= ctx.stack * 0.40
    ):
        eff = ehs + da.equity_boost
        semi_thresh = 0.57
        if eff >= semi_thresh:
            size = clamp(ctx.call_amount * 2.8, ctx)
            if size > 0:
                _semi_bluff_increment(ctx)
                return aggr(ctx), size, f"mid:semi-bluff flush draw eff={eff:.2f}"

    # ── 4. OESD facing a bet → semi-bluff raise (less frequent) ─────────
    if (
        ctx.call_amount > 0
        and da.straight_draw == "oesd"
        and not da.flush_draw
        and already_raised < max_raises          # ← anti-spiral guard
        and not ctx.call_amount >= ctx.stack * 0.40
    ):
        eff = ehs + da.equity_boost
        if eff >= 0.60:
            size = clamp(ctx.call_amount * 2.5, ctx)
            if size > 0:
                _semi_bluff_increment(ctx)
                return aggr(ctx), size, f"mid:semi-bluff OESD eff={eff:.2f}"

    # ── 5. Small-bet pot-odds call ────────────────────────────────────────
    # Arena bots never bluff → bluff-catching is always -EV.
    # Only call small bets (≤30% pot) when pure pot-odds are positive.
    if ctx.call_amount > 0 and ctx.call_amount <= ctx.pot * 0.30:
        required = ctx.call_amount / max(1.0, ctx.pot + ctx.call_amount)
        if ehs >= required * 1.05:
            return "call", to_call(ctx), f"mid:pot-odds small-bet ehs={ehs:.2f} req={required:.2f}"

    return None


# ---------------------------------------------------------------------------
# Behavioral systems — optimal static logic for unknown rotating opponents
# ---------------------------------------------------------------------------

def ehs_deterioration_check(ctx, ehs: float, street: str) -> Optional[Decision]:
    """
    System 1 — EHS deterioration abort.

    If our EHS was strong on the flop (≥0.62) but has dropped sharply by the
    turn/river (now <0.48), the board has connected with the opponent or we
    were dominated. Stop betting and check/fold.

    From log analysis: hands 6283 and 6451 showed EHS declines of 0.20+ where
    Plutus kept betting into deteriorating equity, leaking chips.
    """
    hist = _get_history(ctx)
    flop_ehs = hist.get("flop_ehs")
    if flop_ehs is None or street not in ("turn", "river"):
        return None

    drop = flop_ehs - ehs
    # Board significantly hit opponent; equity collapsed.
    # Lowered from (0.18, 0.48) — analysis showed EHS drops of 0.12-0.17
    # (hand 21410: 0.642→0.516, hand 20516: 0.604→partial) went undetected
    # and agent kept betting into trapping opponents losing 3400+ chips.
    if drop >= 0.14 and ehs < 0.52:
        if ctx.call_amount > 0:
            # Facing a bet with deteriorated equity → fold
            if ehs < 0.48:
                return "fold", 0.0, f"mid:deterioration fold flop={flop_ehs:.2f}→{ehs:.2f} drop={drop:.2f}"
            # Marginal: pot odds might justify a call but don't re-raise
            return None
        # Not facing a bet → check (don't bleed chips)
        from agent.strategies.base import has
        if has(ctx, "check"):
            return "check", 0.0, f"mid:deterioration check flop={flop_ehs:.2f}→{ehs:.2f} drop={drop:.2f}"

    return None


def probe_bet_decide(ctx, ehs: float, street: str) -> Optional[Decision]:
    """
    System 2 — Probe bet (test bet).

    When EHS is 0.48–0.64 and we're first to act (no facing bet), use a small
    probe bet (0.28x pot) instead of checking or full value betting.

    Purpose: cheaply collect dead money from folders and gather info.
    Against any unknown opponent — some fold to small bets, giving us free chips.
    If they raise → fold (we've spent 0.28x pot, not 0.75x).
    If they call → we now know they have something; adjust future streets.

    Only fires on flop/turn. Never probes with <0.48 EHS (too weak to probe).
    Never probes after already betting (escalation logic takes over).
    """
    from agent.strategies.base import clamp, aggr, has

    if ctx.call_amount > 0:
        return None   # facing a bet → this is a call/fold decision, not a probe
    if street not in ("flop", "turn"):
        return None
    if not has(ctx, "bet") and not has(ctx, "raise"):
        return None

    hist = _get_history(ctx)
    if hist["streets_bet"] > 0:
        return None   # already bet a previous street; use escalation logic instead

    # Only probe with medium equity — strong hands use full value bet path
    if not (0.48 <= ehs <= 0.64):
        return None

    # Don't probe on very wet boards (opponent likely has draws; they'll call)
    # Board wetness proxy: >2 community cards within 3 ranks = connected
    board = ctx.community_cards
    if board:
        ranks = sorted(set(r for c in board if c for r in [__import__('engine.hand_eval', fromlist=['card_rank']).card_rank(c)]))
        if len(ranks) >= 3 and (ranks[-1] - ranks[0]) <= 4:
            return None   # very connected board → skip probe

    probe_size = clamp(ctx.pot * 0.28, ctx)
    if probe_size <= ctx.bb_size:
        return None   # probe would be too tiny to matter

    return aggr(ctx), probe_size, f"mid:probe 28%pot ehs={ehs:.2f} street={street}"


def escalate_bet_decide(ctx, ehs: float, street: str) -> Optional[Decision]:
    """
    System 3 — Bet escalation when called last street.

    If we bet a previous street and opponent called (pot is growing, we're now
    first to act again on a new street), increase our bet size by ~1.3x.

    Logic:
    - Opponent called → they have equity; they'll pay more next street too
    - Escalating extracts maximum value from calling stations
    - Stops if EHS has deteriorated (see ehs_deterioration_check above)

    Graduated nut path (EHS ≥ 0.82):
      flop 0.50x → turn 0.75x → river all-in
    Value escalation (EHS ≥ 0.62):
      flop 0.50x → turn 0.65x → river 0.85x
    """
    from agent.strategies.base import clamp, aggr, has

    if ctx.call_amount > 0:
        return None   # facing a bet; escalation doesn't apply
    if not has(ctx, "bet") and not has(ctx, "raise"):
        return None
    if street not in ("turn", "river"):
        return None

    hist = _get_history(ctx)
    if hist["streets_bet"] == 0:
        return None   # never bet before; no escalation base

    # Infer opponent called last bet: pot grew after our bet without us acting
    # (pot > last_bet_pot + our_bet = opponent added chips = called)
    last_pot  = hist["last_bet_pot"]
    last_frac = hist["last_bet_fraction"]
    if last_pot <= 0 or last_frac <= 0:
        return None

    expected_pot_if_no_call = last_pot + last_pot * last_frac
    # If current pot is roughly equal to or larger than expected → opponent called
    opp_likely_called = ctx.pot >= expected_pot_if_no_call * 0.85

    if not opp_likely_called:
        return None

    # Nut path: EHS 0.82+ → escalate aggressively toward all-in
    if ehs >= 0.82:
        if street == "turn":
            target = 0.75
        else:  # river
            target = ctx.stack  # all-in on river with near-nut hand
            size = clamp(target, ctx)
            if size > 0:
                return aggr(ctx), size, f"mid:nut-path river ALLIN ehs={ehs:.2f}"
            return None
        size = clamp(ctx.pot * target, ctx)
        if size > 0:
            return aggr(ctx), size, f"mid:nut-path {target:.0%}pot ehs={ehs:.2f} streets_bet={hist['streets_bet']}"
        return None

    # Value escalation: EHS 0.62–0.82
    if ehs >= 0.62:
        # Increase by 1.3x from last bet fraction, capped at 1.0x pot
        new_frac = min(last_frac * 1.3, 1.0)
        # River: don't go all-in here (reserved for nut path above)
        if street == "river":
            new_frac = min(new_frac, 0.85)
        size = clamp(ctx.pot * new_frac, ctx)
        if size > 0:
            return aggr(ctx), size, f"mid:escalate {new_frac:.0%}pot ehs={ehs:.2f} (was {last_frac:.0%})"

    return None
"""agent/strategies/base.py — shared helpers for S1-S9 strategy modules."""
from __future__ import annotations

from typing import Tuple

from agent.arbiter import GameContext

Decision = Tuple[str, float, str]


def acts(ctx: GameContext) -> set:
    return {a.get("action", "").lower() for a in ctx.allowed_actions}


def has(ctx: GameContext, action: str) -> bool:
    return action in acts(ctx)


def bet_bounds(ctx: GameContext) -> tuple:
    for a in ctx.allowed_actions:
        if a.get("action") in ("bet", "raise", "all-in"):
            return float(a.get("minAmount", 0)), float(a.get("maxAmount", ctx.stack))
    return 0.0, ctx.stack


def clamp(amount: float, ctx: GameContext) -> float:
    if amount <= 0:
        return 0.0
    mn, mx = bet_bounds(ctx)
    if mn == 0.0 and mx == ctx.stack and not any(
        a.get("action") in ("bet", "raise", "all-in") for a in ctx.allowed_actions
    ):
        return 0.0
    return max(mn, min(mx, amount, ctx.stack))


def aggr(ctx: GameContext) -> str:
    """Return the correct aggressive action ('raise' or 'bet')."""
    a = acts(ctx)
    if "raise" in a:
        return "raise"
    if "bet" in a:
        return "bet"
    return "raise"


def pot_odds(ctx: GameContext) -> float:
    return ctx.call_amount / max(1.0, ctx.pot + ctx.call_amount) if ctx.call_amount > 0 else 0.0


def board_hits(ctx: GameContext) -> int:
    """Count how many hole card ranks appear in community cards."""
    if not ctx.community_cards:
        return 0
    hole_ranks = {c[0].upper() for c in ctx.hole_cards}
    board_ranks = {c[0].upper() for c in ctx.community_cards}
    return len(hole_ranks & board_ranks)


def is_ip(ctx: GameContext) -> bool:
    return ctx.is_in_position or ctx.position in ("BTN", "CO", "HJ")


def to_call(ctx: GameContext) -> float:
    return ctx.call_to_amount or ctx.call_amount


def bet_pot(mult: float, ctx: GameContext) -> float:
    return clamp(ctx.pot * mult, ctx)


def open_bb(mult: float, ctx: GameContext) -> float:
    return clamp(ctx.bb_size * mult, ctx)


def raise_to(mult: float, ctx: GameContext) -> float:
    return clamp(max(ctx.call_amount * mult, ctx.bb_size * 2), ctx)


def safe_check_fold(ctx: GameContext) -> Decision:
    if has(ctx, "check"):
        return "check", 0.0, "safe check"
    return "fold", 0.0, "fold"
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

from agent.brutal_check import BrutalSelfCheck
from agent.decision_log import log_decision
from agent.meta_learner import MetaLearner
from agent.performance_dashboard import PerformanceDashboard
from agent.play_stats import PlayStatsLogger
from agent.self_test_runner import run_startup_self_tests
from api.action_amount import build_action_payload
from models.opponent_tracker import OpponentTracker

STRATEGY_VERSION = "v3-universal"


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
        self.hands_played = 0       # action submissions (not used for stop condition)
        self.match_hands_played = 0 # arena completedHands (used for stop condition)
        self._running = True
        self.payouts: Optional[List[float]] = None
        self.competition_meta: Dict[str, Any] = {}
        self._prev_hand: Dict[str, Any] = {}
        self._hand_buffer = HandHistoryBuffer()
        self._decision_times: List[float] = []
        self._heartbeat_state = HeartbeatState(settings.heartbeat_state_file)
        self._last_heartbeat_loop = time.time()
        self._last_poll = 0.0
        self._last_match_refresh = 0.0
        self._active_tables: set = set()
        self._shutting_down = False
        self._pending_actions = 0
        self._table_locks: Dict[str, Lock] = {}
        self._action_tx_log: List[Dict[str, Any]] = []
        self._started_at = time.time()
        self._expired_tables: Dict[str, float] = {}
        self._bootstrap_table: Optional[Dict[str, Any]] = None
        self._match_phase: str = ""
        self._bankroll_chips: int = -1
        self._total_chips: int = -1
        self._last_rebuy_attempt: float = 0.0

        self.tracker = OpponentTracker(settings.state_file)
        self.adaptive = AdaptiveMemory(settings.adaptive_state_file)
        self.brutal = BrutalSelfCheck()
        self.meta = MetaLearner()
        _strat = os.getenv("ARENA_STRATEGY", "").strip().upper() or "ADAPTIVE"
        logger.info(f"Strategy: {_strat} | agent={settings.agent_name}")
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
            or self.brutal.should_rollback()
        )
        phase = (self._match_phase or "").lower()
        if (
            idle > settings.watchdog_idle_s
            and phase not in ("waiting_user", "waiting", "")
            and self._active_tables
        ):
            degraded = True
        brutal_h = self.brutal.health_dict()
        return {
            "status": brutal_h.get("status", "degraded" if degraded else "ok"),
            "errors_last_100": brutal_h.get("errors_last_100", self.brutal.api_errors_400),
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
        phase = (self._match_phase or "").lower()
        if phase in ("waiting_user", "waiting"):
            return
        if idle > settings.watchdog_idle_s and self._running:
            logger.warning(f"Watchdog: no action for {idle:.0f}s")
            self.brutal._alert(f"watchdog idle {idle:.0f}s")

    def _poll_action_tables(self) -> List[Dict]:
        return self.client.fetch_action_tables(self.competition_id)

    async def _async_poll_action_tables(self) -> List[Dict]:
        return await self.client.async_fetch_action_tables(self.competition_id)

    def _consume_bootstrap_table(self, prev_states: dict) -> bool:
        if not self._bootstrap_table:
            return False
        table = self._bootstrap_table
        self._bootstrap_table = None
        self._handle_table_sync(table, prev_states)
        return True

    async def _async_consume_bootstrap_table(self, prev_states: dict) -> bool:
        if not self._bootstrap_table:
            return False
        table = self._bootstrap_table
        self._bootstrap_table = None
        await self._handle_table_async(table, prev_states)
        return True

    def setup(self) -> bool:
        import httpx as _httpx
        for _attempt in range(5):
            try:
                cid, meta, client = onboard_and_join(
                    competition_id=self.competition_id,
                    dry_run=self.dry_run,
                    force_heartbeat=True,
                )
                self.client = client
                self.competition_id = cid or ""
                self.competition_meta = meta or {}
                break
            except (_httpx.ReadTimeout, _httpx.ConnectTimeout):
                _wait = 20 * (_attempt + 1)
                logger.warning(f"Arena timeout (attempt {_attempt+1}/5), retry in {_wait}s")
                time.sleep(_wait)
                if _attempt == 4:
                    logger.error("Arena setup failed after 5 retries (timeout)")
                    return False
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
        self._seed_bootstrap_table()
        return bool(self.competition_id)

    def _seed_bootstrap_table(self) -> None:
        """If benchmark already has our decision, act on first poll cycle."""
        if not self.competition_id:
            return
        try:
            from api.state_parser import hero_is_to_act, prepare_table_for_runner

            st = self.client.get_benchmark_status(self.competition_id)
            match = st.get("match") or {}
            self._match_phase = str(match.get("phase") or "")
            tbl = st.get("table")
            if tbl and hero_is_to_act(tbl, self.client.agent_id):
                self._bootstrap_table = prepare_table_for_runner(tbl)
                logger.info(
                    f"Bootstrap: hero to act on table {self._bootstrap_table.get('tableId')}"
                )
        except Exception as e:
            logger.debug(f"Bootstrap table check skipped: {e!r}")

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
            self._match_phase = str(match.get("phase") or "")
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
            participant = st.get("participant") or {}
            if participant:
                self._bankroll_chips = int(participant.get("bankrollChips", -1))
                self._total_chips = int(participant.get("totalChips", -1))
        except Exception:
            pass

    def _maybe_rebuy(self) -> None:
        if self.dry_run or not self.competition_id:
            return
        # Only rebuy when completely bust (totalChips == 0); bankrollChips alone
        # can be 0 while chips remain on the table, triggering a premature 403.
        if self._total_chips != 0:
            return
        cooldown = 60.0
        if time.time() - self._last_rebuy_attempt < cooldown:
            return
        self._last_rebuy_attempt = time.time()
        try:
            result = self.client.rebuy(self.competition_id)
            participant = result.get("participant") or {}
            new_chips = participant.get("totalChips", "?")
            rebuy_count = participant.get("rebuyCount", "?")
            self._bankroll_chips = int(participant.get("bankrollChips", 0))
            self._total_chips = int(participant.get("totalChips", 0))
            logger.info(
                f"Rebuy credited — totalChips={new_chips} rebuyCount={rebuy_count}"
            )
        except ArenaAPIError as e:
            if e.status == 402:
                logger.warning("Rebuy requires on-chain payment — skipping")
            elif e.status == 400:
                logger.warning(f"Rebuy not available: {e.body}")
            else:
                logger.warning(f"Rebuy failed {e.status}: {e.body}")

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
        _503_streak = 0
        self._maybe_heartbeat(force=False)
        while self._running:
            if self.max_hands and self.match_hands_played >= self.max_hands:
                break
            if time.time() - self._last_heartbeat_loop >= settings.heartbeat_interval_s:
                self._maybe_heartbeat()
                self._last_heartbeat_loop = time.time()
            if time.time() - self._last_match_refresh >= 30.0:
                self._refresh_match_context()
                self._last_match_refresh = time.time()
            self._watchdog_check()
            self._rate_limit_poll()
            if self._consume_bootstrap_table(prev_states):
                continue
            try:
                tables = self._poll_action_tables()
                _503_streak = 0
            except ArenaAPIError as e:
                if e.status == 503:
                    _503_streak += 1
                    backoff = min(60.0, 2.0 ** min(_503_streak, 6))
                    logger.warning(f"Poll 503 (streak {_503_streak}), backoff {backoff:.0f}s: {e.body}")
                    time.sleep(backoff)
                else:
                    logger.warning(f"Poll error {e.status}: {e.body}")
                    time.sleep(self.client._breaker.backoff_sleep())
                continue
            except Exception as e:
                logger.warning(f"Poll exception: {type(e).__name__}: {e!r}")
                time.sleep(settings.poll_interval_s)
                continue
            if not tables:
                self.brutal.record_action()
                self._maybe_rebuy()
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
        _503_streak = 0
        self._maybe_heartbeat(force=False)
        while self._running:
            if self.max_hands and self.match_hands_played >= self.max_hands:
                break
            if time.time() - self._last_heartbeat_loop >= settings.heartbeat_interval_s:
                self._maybe_heartbeat()
                self._last_heartbeat_loop = time.time()
            if time.time() - self._last_match_refresh >= 30.0:
                self._refresh_match_context()
                self._last_match_refresh = time.time()
            self._watchdog_check()
            await self._async_rate_limit_poll()
            if await self._async_consume_bootstrap_table(prev_states):
                continue
            try:
                tables = await self._async_poll_action_tables()
                _503_streak = 0
            except ArenaAPIError as e:
                if e.status == 503:
                    _503_streak += 1
                    backoff = min(60.0, 2.0 ** min(_503_streak, 6))
                    logger.warning(f"Poll 503 (streak {_503_streak}), backoff {backoff:.0f}s: {e.body}")
                    await asyncio.sleep(backoff)
                else:
                    logger.warning(f"Poll error {e.status}: {e.body}")
                    await asyncio.sleep(self.client._breaker.backoff_sleep())
                continue
            except Exception as e:
                logger.warning(f"Poll exception: {type(e).__name__}: {e!r}")
                await asyncio.sleep(settings.poll_interval_s)
                continue

            if not tables:
                self.brutal.record_action()
                self._maybe_rebuy()
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

        if prev_hand is not None and curr_hand is not None and prev_hand != curr_hand:
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
        if self._expired_tables.get(table_id, 0) > time.time():
            return
        with self._table_lock(table_id):
            self._buffer_and_finalize(table, table_id, prev_states)
            self._decide_and_submit(table, table_id)

    async def _handle_table_async(self, table: dict, prev_states: dict) -> None:
        table_id = extract_table_id(table)
        if self._expired_tables.get(table_id, 0) > time.time():
            return
        async with self._table_lock(table_id):
            self._buffer_and_finalize(table, table_id, prev_states)
            deadline = (
                table.get("actionDeadline")
                or table.get("deadline")
                or 0
            )
            if not deadline and table.get("actionDeadlineAt"):
                raw = float(table["actionDeadlineAt"])
                deadline = raw / 1000.0 if raw > 1e12 else raw
        budget = settings.decision_budget_s
        if deadline:
            budget = min(budget, max(0.2, deadline - time.time() - 0.2))
        submitted: list[bool] = [False]
        try:
            await asyncio.wait_for(
                self._decide_and_submit_async(table, table_id, deadline, submitted),
                timeout=budget,
            )
        except asyncio.TimeoutError:
            if not submitted[0]:
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
                action, amount = safe_fallback(allowed)
                chat = "Error fallback"

            self._decision_times.append((time.monotonic() - t0) * 1000)
            if len(self._decision_times) > 500:
                self._decision_times = self._decision_times[-200:]

            action, amount = self._validate_action(
                action, amount, ctx.allowed_actions, ctx.stack
            )
            self.play_stats.log_decision(
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
        self, table: dict, table_id: str, deadline: float, submitted: list[bool] | None = None
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
                if submitted is not None:
                    submitted[0] = True
                await self._submit_async(table_id, action, amount, table, "Parse fallback")
                return

            if ctx.hand_number:
                self.arbiter.on_new_hand(table_id, ctx.hand_number, ctx.stack)

            if deadline and (deadline - time.time()) < 0.25:
                action, amount = safe_fallback(allowed)
                if submitted is not None:
                    submitted[0] = True
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
                action, amount = safe_fallback(allowed)
                chat = "Error fallback"

            self._decision_times.append((time.monotonic() - t0) * 1000)
            if len(self._decision_times) > 500:
                self._decision_times = self._decision_times[-200:]

            action, amount = self._validate_action(
                action, amount, ctx.allowed_actions, ctx.stack
            )
            self.play_stats.log_decision(
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
            if submitted is not None:
                submitted[0] = True
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
        if self.dry_run:
            amt = format_submission_amount(
                action, amount, table, self.client.agent_id
            )
            logger.info(f"[DRY-RUN] {action} {amt:.1f} | {chat[:80]}")
            return
        reasoning = (chat or "").strip() or (
            f"{self.arbiter.current_mode} | {self.meta.active} | {action}"
        )
        payload = build_action_payload(
            table_id,
            action,
            amount,
            table,
            self.client.agent_id,
            reasoning=reasoning,
            message=chat,
        )
        tx_id = f"{table_id}:{time.time():.3f}"
        self._action_tx_log.append({"id": tx_id, "payload": payload, "ts": time.time()})
        try:
            self._post_action_payload(payload, table_id, table)
        except ArenaAPIError as e:
            self.brutal.record_api_call(False, e.status)
            if e.status == 400:
                self._submit_safe_fallback(table_id, table, chat, original_error=e)
            elif e.status == 409:
                self._expired_tables[table_id] = time.time() + 30.0
                logger.warning(f"Table {table_id} no longer active (409), suppressing for 30s")
            elif e.status == 504:
                logger.warning(f"Table {table_id} action timed out, skipping")
            else:
                logger.error(f"Submit failed {e.status}: {e.body}")
        except Exception as e:
            logger.warning(f"Table {table_id} submit error ({type(e).__name__}), skipping")

    def _post_action_payload(
        self, payload: Dict[str, Any], table_id: str, table: dict
    ) -> None:
        result = self.client.submit_action_payload_safe(payload)
        if result is None:
            raise ArenaAPIError(409, "table no longer active")
        self.brutal.record_api_call(True, 200)
        self.brutal.record_action()
        act = str(payload.get("action", "fold")).upper()
        amt = payload.get("amount", 0)
        logger.info(f"✓ {act} {amt} | {str(payload.get('reasoning', ''))[:80]}")
        self.hands_played += 1

    def _submit_safe_fallback(
        self,
        table_id: str,
        table: dict,
        chat: str,
        *,
        original_error: Optional[ArenaAPIError] = None,
    ) -> None:
        """Recover from 400 by re-reading table state and playing a legal action."""
        fresh = table
        try:
            for t in self.client.fetch_action_tables(self.competition_id):
                if extract_table_id(t) == table_id:
                    fresh = t
                    break
        except Exception:
            pass
        allowed = fresh.get("allowedActions") or []
        action, amount = safe_fallback(allowed)
        reasoning = (chat or "").strip() or "400 recovery safe action"
        payload = build_action_payload(
            table_id,
            action,
            amount,
            fresh,
            self.client.agent_id,
            reasoning=reasoning,
            message=reasoning,
        )
        try:
            self._post_action_payload(payload, table_id, fresh)
            logger.warning(
                f"Recovered from 400 via {action} "
                f"(was: {original_error.body if original_error else '?'})"
            )
        except ArenaAPIError as e2:
            logger.error(f"Submit failed {e2.status} after 400 recovery: {e2.body}")

    async def _submit_async(
        self, table_id: str, action: str, amount: float, table: dict, chat: str
    ) -> None:
        if self.dry_run:
            amt = format_submission_amount(
                action, amount, table, self.client.agent_id
            )
            logger.info(f"[DRY-RUN] {action} {amt:.1f} | {chat[:80]}")
            return
        reasoning = (chat or "").strip() or (
            f"{self.arbiter.current_mode} | {self.meta.active} | {action}"
        )
        payload = build_action_payload(
            table_id,
            action,
            amount,
            table,
            self.client.agent_id,
            reasoning=reasoning,
            message=chat,
        )
        self._action_tx_log.append(
            {"id": f"{table_id}:{time.time():.3f}", "payload": payload, "ts": time.time()}
        )
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, lambda: self._post_action_payload(payload, table_id, table)
            )
        except ArenaAPIError as e:
            self.brutal.record_api_call(False, e.status)
            if e.status == 400:
                await loop.run_in_executor(
                    None,
                    lambda: self._submit_safe_fallback(
                        table_id, table, chat, original_error=e
                    ),
                )
            elif e.status == 409:
                self._expired_tables[table_id] = time.time() + 30.0
                logger.warning(f"Table {table_id} no longer active (409), suppressing for 30s")
            elif e.status == 504:
                logger.warning(f"Table {table_id} action timed out, skipping")
            else:
                logger.error(f"Submit failed {e.status}: {e.body}")
        except Exception as e:
            logger.warning(f"Table {table_id} submit error ({type(e).__name__}), skipping")


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
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="Print all available ARENA_STRATEGY values and exit",
    )
    args = parser.parse_args()

    configure_logging(
        settings.log_level,
        json_logs=True,
        log_file=settings.log_file,
    )

    if args.list_strategies:
        from agent.strategies import list_strategies
        print(list_strategies())
        sys.exit(0)

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
ocean@fedora:~/vscode/poker_agent$ 
