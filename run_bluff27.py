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
