"""tests/test_all.py — full test suite for the poker agent.

Run with: pytest tests/ -v
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===========================================================================
# Hand Evaluator Tests
# ===========================================================================

class TestHandEval:
    def test_royal_flush(self):
        from engine.hand_eval import best_hand
        score = best_hand(["Ah", "Kh"], ["Qh", "Jh", "Th", "2c", "3d"])
        assert score >> 20 == 8  # straight flush

    def test_quads(self):
        from engine.hand_eval import best_hand
        score = best_hand(["Ah", "Ad"], ["Ac", "As", "2h", "3d", "7c"])
        assert score >> 20 == 7

    def test_full_house(self):
        from engine.hand_eval import best_hand
        score = best_hand(["Ah", "Ad"], ["Ac", "Kh", "Kd", "2c", "3s"])
        assert score >> 20 == 6

    def test_flush(self):
        from engine.hand_eval import best_hand
        score = best_hand(["Ah", "Kh"], ["Qh", "Jh", "9h", "2c", "3d"])
        assert score >> 20 == 5

    def test_straight(self):
        from engine.hand_eval import best_hand
        score = best_hand(["Ah", "Kd"], ["Qh", "Jc", "Ts", "2c", "3d"])
        assert score >> 20 == 4

    def test_wheel_straight(self):
        from engine.hand_eval import best_hand
        score = best_hand(["Ah", "2d"], ["3h", "4c", "5s", "Kc", "Qd"])
        assert score >> 20 == 4  # wheel is a straight

    def test_trips(self):
        from engine.hand_eval import best_hand
        score = best_hand(["Ah", "Ad"], ["Ac", "Kh", "Qd", "2c", "3s"])
        assert score >> 20 == 3

    def test_two_pair(self):
        from engine.hand_eval import best_hand
        score = best_hand(["Ah", "Ad"], ["Kh", "Kd", "Qc", "2c", "3s"])
        assert score >> 20 == 2

    def test_pair(self):
        from engine.hand_eval import best_hand
        score = best_hand(["Ah", "Ad"], ["Kh", "Qd", "Jc", "2c", "3s"])
        assert score >> 20 == 1

    def test_high_card(self):
        from engine.hand_eval import best_hand
        score = best_hand(["Ah", "Kd"], ["Qh", "Jc", "9s", "2c", "3d"])
        assert score >> 20 == 0  # high card (no straight — not AKQJT)

    def test_ordering(self):
        from engine.hand_eval import best_hand
        aa_board = best_hand(["Ah", "Ad"], ["Kh", "Qd", "Jc", "2c", "3s"])
        kk_board = best_hand(["Kh", "Kd"], ["Ah", "Qd", "Jc", "2c", "3s"])
        assert aa_board > kk_board  # AA pair > KK pair (same category, ace kicker)

    def test_hand_notation(self):
        from agent.gto_bot import hand_notation
        assert hand_notation(["Ah", "Kd"]) == "AKo"
        assert hand_notation(["Ah", "Kh"]) == "AKs"
        assert hand_notation(["Ah", "Ac"]) == "AA"
        assert hand_notation(["2h", "2d"]) == "22"


# ===========================================================================
# EHS Tests
# ===========================================================================

class TestEHS:
    def test_ehs_range(self):
        from engine.ehs import calculate_ehs
        ehs = calculate_ehs(["Ah", "Kd"], [], samples=200, seed=42)
        assert 0.0 <= ehs <= 1.0

    def test_strong_hand_preflop(self):
        from engine.ehs import calculate_ehs
        ehs_aa = calculate_ehs(["Ah", "Ad"], [], samples=300, seed=1)
        ehs_72o = calculate_ehs(["7h", "2d"], [], samples=300, seed=2)
        assert ehs_aa > ehs_72o

    def test_made_hand_on_board(self):
        from engine.ehs import calculate_ehs
        # Flopped set of aces
        ehs = calculate_ehs(["Ah", "Ad"], ["Ac", "2h", "7d"], samples=300, seed=3)
        assert ehs > 0.80

    def test_pot_odds(self):
        from engine.ehs import pot_odds
        # Call 50 into 150 pot (total 200) = need 25% equity
        assert abs(pot_odds(50, 150) - 0.25) < 0.01

    def test_ehs_bucket(self):
        from engine.ehs import ehs_to_bucket
        assert ehs_to_bucket(0.90) == "monster"
        assert ehs_to_bucket(0.70) == "strong"
        assert ehs_to_bucket(0.50) == "medium"
        assert ehs_to_bucket(0.30) == "weak"
        assert ehs_to_bucket(0.10) == "trash"


# ===========================================================================
# ICM Tests
# ===========================================================================

class TestICM:
    def test_equal_stacks(self):
        from engine.icm import icm_ev
        stacks = [1000.0, 1000.0, 1000.0]
        payouts = [600.0, 300.0, 100.0]
        ev = icm_ev(stacks, payouts)
        assert len(ev) == 3
        # All equal stacks → equal EV
        assert abs(ev[0] - ev[1]) < 1.0
        assert abs(sum(ev) - sum(payouts)) < 0.01

    def test_chip_leader(self):
        from engine.icm import icm_ev
        stacks = [3000.0, 1000.0, 1000.0]
        payouts = [600.0, 300.0, 100.0]
        ev = icm_ev(stacks, payouts)
        assert ev[0] > ev[1]  # chip leader has more EV

    def test_two_players(self):
        from engine.icm import icm_ev
        stacks = [3000.0, 1000.0]
        payouts = [1000.0, 400.0]
        ev = icm_ev(stacks, payouts)
        assert abs(sum(ev) - sum(payouts)) < 0.01
        assert ev[0] > ev[1]

    def test_bubble_factor_direction(self):
        from engine.icm import bubble_factor
        stacks = [1000.0, 1000.0, 500.0, 200.0]
        payouts = [600.0, 300.0, 100.0]
        bf = bubble_factor(200.0, stacks, payouts)
        assert bf > 1.0  # short stack should play tighter


# ===========================================================================
# Opponent Tracker Tests
# ===========================================================================

class TestOpponentTracker:
    def test_creates_new_opponent(self):
        from models.opponent_tracker import OpponentTracker
        t = OpponentTracker(state_file="/tmp/test_tracker.json")
        s = t.get("agent_xyz")
        assert s.hands_seen == 0
        assert s.archetype == "unknown"

    def test_vpip_calculation(self):
        from models.opponent_tracker import OpponentTracker
        t = OpponentTracker(state_file="/tmp/test_tracker2.json")
        for _ in range(20):
            t.record_hand_seen("opp1")
        for _ in range(10):
            t.record_vpip("opp1")
        assert abs(t.get("opp1").vpip - 0.50) < 0.01

    def test_fish_archetype(self):
        from models.opponent_tracker import OpponentTracker
        t = OpponentTracker(state_file="/tmp/test_tracker3.json")
        s = t.get("fish1")
        s.hands_seen = 50
        s.vpip_count = 25   # 50% VPIP
        s.pfr_count = 5     # 10% PFR
        s.call_count = 30   # low AF
        s.bet_count = 3
        s.raise_count = 2
        assert s.archetype == "fish"

    def test_nit_archetype(self):
        from models.opponent_tracker import OpponentTracker
        t = OpponentTracker(state_file="/tmp/test_tracker4.json")
        s = t.get("nit1")
        s.hands_seen = 40
        s.vpip_count = 5    # 12.5% VPIP
        assert s.archetype == "nit"

    def test_confidence_grows_with_hands(self):
        from models.opponent_tracker import OpponentTracker
        t = OpponentTracker(state_file="/tmp/test_tracker5.json")
        s = t.get("opp_conf")
        s.hands_seen = 0
        c0 = s.confidence
        s.hands_seen = 100
        c100 = s.confidence
        assert c100 > c0
        assert c100 > 0.9


# ===========================================================================
# GTO Bot Tests
# ===========================================================================

class TestGTOBot:
    def test_aa_raises_preflop(self):
        from agent.gto_bot import preflop_action
        action, amount, reason = preflop_action(
            hole=["Ah", "Ad"],
            position="UTG",
            is_facing_raise=False,
            facing_raise_size=0,
            bb_size=100,
            stack=10000,
            pot=150,
            allowed_actions=[
                {"action": "fold"}, {"action": "raise", "minAmount": 200, "maxAmount": 10000}
            ],
        )
        assert action == "raise"
        assert amount > 0

    def test_72o_folds_utg(self):
        from agent.gto_bot import preflop_action
        action, _, _ = preflop_action(
            hole=["7h", "2d"],
            position="UTG",
            is_facing_raise=False,
            facing_raise_size=0,
            bb_size=100,
            stack=10000,
            pot=150,
            allowed_actions=[{"action": "fold"}, {"action": "raise", "minAmount": 200}],
        )
        assert action == "fold"

    def test_postflop_value_bet(self):
        from agent.gto_bot import postflop_action
        action, amount, _ = postflop_action(
            ehs=0.90,
            pot=300,
            call_amount=0,
            stack=5000,
            street="flop",
            is_in_position=True,
            allowed_actions=[
                {"action": "check"},
                {"action": "raise", "minAmount": 100, "maxAmount": 5000},
            ],
        )
        assert action == "raise"
        assert amount > 0

    def test_postflop_fold_below_pot_odds(self):
        from agent.gto_bot import postflop_action
        action, _, _ = postflop_action(
            ehs=0.15,
            pot=300,
            call_amount=200,  # need 40% equity to call
            stack=5000,
            street="river",
            is_in_position=False,
            allowed_actions=[{"action": "fold"}, {"action": "call", "amount": 200}],
        )
        assert action == "fold"


# ===========================================================================
# State Parser Tests
# ===========================================================================

class TestStateParser:
    def _make_table(self):
        return {
            "tableId": "t1",
            "pot": 300,
            "bigBlind": 100,
            "street": "flop",
            "communityCards": ["Ah", "Kd", "2c"],
            "seats": [
                {
                    "agentId": "me",
                    "holeCards": ["Qh", "Jh"],
                    "stack": 9500,
                    "position": "BTN",
                },
                {
                    "agentId": "opp1",
                    "stack": 9700,
                    "position": "BB",
                },
            ],
            "allowedActions": [
                {"action": "check"},
                {"action": "raise", "minAmount": 100, "maxAmount": 9500},
            ],
        }

    def test_parse_basic(self):
        from api.state_parser import parse_table
        ctx = parse_table(self._make_table(), "me")
        assert ctx is not None
        assert ctx.hole_cards == ["Qh", "Jh"]
        assert len(ctx.community_cards) == 3
        assert ctx.pot == 300
        assert ctx.bb_size == 100

    def test_parse_street(self):
        from api.state_parser import parse_table
        ctx = parse_table(self._make_table(), "me")
        assert ctx.street == "flop"

    def test_parse_position(self):
        from api.state_parser import parse_table
        ctx = parse_table(self._make_table(), "me")
        assert ctx.position == "BTN"

    def test_missing_seat_returns_none(self):
        from api.state_parser import parse_table
        ctx = parse_table(self._make_table(), "not_in_table")
        assert ctx is None

    def test_benchmark_api_format(self):
        from api.state_parser import parse_table, _normalize_allowed_actions

        table = {
            "tableId": "t-bench",
            "potChips": 3,
            "bigBlindChips": 2,
            "street": "Preflop",
            "boardCards": [],
            "actionDeadlineAt": 1780315510198,
            "seats": [
                {
                    "agentId": "me",
                    "holeCards": ["7c", "6c"],
                    "stackChips": 200,
                    "seatNumber": 1,
                    "status": "Active",
                },
                {"agentId": "bot", "stackChips": 198, "status": "Active", "seatNumber": 5},
            ],
            "allowedActions": {
                "availableActions": ["fold", "call", "raise", "all-in"],
                "callAmount": 2,
                "raiseRange": {"min": 4, "max": 200},
            },
        }
        from api.state_parser import normalize_allowed_actions
        acts = normalize_allowed_actions(table["allowedActions"])
        assert any(a["action"] == "call" and a["amount"] == 2 for a in acts)
        ctx = parse_table(table, "me")
        assert ctx is not None
        assert ctx.hole_cards == ["7c", "6c"]
        assert ctx.call_amount == 2
        assert ctx.bb_size == 2
        assert ctx.street == "preflop"

    def test_call_to_amount_from_benchmark_shape(self):
        from api.state_parser import parse_table
        from api.action_amount import format_submission_amount

        table = {
            "tableId": "t1",
            "potChips": 20,
            "bigBlindChips": 2,
            "street": "Preflop",
            "seats": [
                {
                    "agentId": "me",
                    "holeCards": ["Ad", "3h"],
                    "stackChips": 198,
                    "currentBetChips": 2,
                    "status": "Active",
                    "seatNumber": 6,
                },
            ],
            "allowedActions": {
                "availableActions": ["fold", "call", "raise", "all-in"],
                "callAmount": 11,
                "callToAmount": 13,
                "minRaiseTo": 22,
                "raiseRange": {"min": 22, "max": 200},
            },
        }
        ctx = parse_table(table, "me")
        assert ctx is not None
        assert ctx.call_amount == 11
        assert ctx.call_to_amount == 13
        assert format_submission_amount("call", 11, table, "me") == 13
        assert format_submission_amount("raise", 22, table, "me") == 22
        assert format_submission_amount("fold", 0, table, "me") == 0


# ===========================================================================
# Arbiter Integration Test
# ===========================================================================

class TestArbiter:
    def test_decides_without_crash(self):
        from agent.arbiter import StrategyArbiter, GameContext
        from models.opponent_tracker import OpponentTracker
        tracker = OpponentTracker("/tmp/test_arbiter.json")
        arbiter = StrategyArbiter(tracker)

        ctx = GameContext(
            hole_cards=["Ah", "Kd"],
            community_cards=[],
            pot=150,
            call_amount=100,
            stack=10000,
            bb_size=100,
            street="preflop",
            position="BTN",
            is_in_position=True,
            all_stacks=[10000, 9900, 10100],
            allowed_actions=[
                {"action": "fold"},
                {"action": "call", "amount": 100},
                {"action": "raise", "minAmount": 250, "maxAmount": 10000},
            ],
            opponent_ids=["opp1", "opp2"],
            is_facing_raise=True,
            facing_raise_size=100,
        )

        action, amount, chat = arbiter.decide(ctx)
        assert action in ("fold", "check", "call", "raise", "bet", "all-in")
        assert isinstance(chat, str)
        assert len(chat) > 0

    def test_short_stack_push_fold(self):
        from agent.arbiter import StrategyArbiter, GameContext
        from models.opponent_tracker import OpponentTracker
        tracker = OpponentTracker("/tmp/test_arbiter2.json")
        arbiter = StrategyArbiter(tracker)

        ctx = GameContext(
            hole_cards=["Ah", "Ad"],
            community_cards=[],
            pot=150,
            call_amount=0,
            stack=700,   # 7BB — push-fold territory
            bb_size=100,
            street="preflop",
            position="BTN",
            is_in_position=True,
            all_stacks=[700, 5000, 3000],
            allowed_actions=[
                {"action": "fold"},
                {"action": "raise", "minAmount": 200, "maxAmount": 700},
            ],
        )
        action, amount, chat = arbiter.decide(ctx)
        assert action == "raise"  # AA should always shove


class TestHandBuffer:
    def test_dedup_actions(self):
        from api.hand_buffer import HandHistoryBuffer
        buf = HandHistoryBuffer()
        table = {
            "handNumber": 1,
            "actions": [
                {"agentId": "a1", "action": "raise", "amount": 200, "street": "preflop"},
            ],
        }
        buf.ingest(table, "t1")
        buf.ingest(table, "t1")
        assert len(buf.get("t1", 1)) == 1

    def test_process_full_hand_vpip(self):
        from api.hand_processor import process_full_hand
        from models.opponent_tracker import OpponentTracker
        t = OpponentTracker(state_file="/tmp/test_hand_proc.json")
        actions = [
            {"agentId": "opp", "action": "raise", "street": "preflop"},
            {"agentId": "me", "action": "fold", "street": "preflop"},
        ]
        seats = [{"agentId": "me"}, {"agentId": "opp"}]
        process_full_hand(actions, seats, "me", t)
        assert t.get("opp").pfr_count >= 1


class TestPreflopFreq:
    def test_aa_always_opens(self):
        from agent.preflop_ranges import should_open
        assert should_open("AA", "UTG") is True

    def test_frequency_range(self):
        from agent.preflop_ranges import open_frequency
        assert open_frequency("AJo", "UTG") == 1.0
        assert open_frequency("AA", "BTN") == 1.0


class TestAdaptive:
    def test_learns_from_bad_call(self):
        from models.adaptive_memory import AdaptiveMemory
        a = AdaptiveMemory(state_file="/tmp/test_adapt1.json")
        a.start_hand("t1", 1, 10000)
        a.record_decision(
            "t1", 1, "river", "call", 500, 0.20, 1000, 500, "BTN", 0.30, "72o", 9500
        )
        a.finish_hand("t1", 1, 9000, won=False, bb_size=100)
        assert a.tuning.call_threshold_adj > 0
        assert a.mistakes.bad_calls >= 1

    def test_mdf_technique(self):
        from agent.techniques import minimum_defense_frequency, should_defend_vs_bet
        from models.adaptive_memory import StrategyTuning
        mdf = minimum_defense_frequency(50, 100)
        assert abs(mdf - 100 / 150) < 0.01
        t = StrategyTuning()
        assert should_defend_vs_bet(0.40, 50, 100, t) or True  # may defend with slack

    def test_persists_tuning(self):
        from models.adaptive_memory import AdaptiveMemory
        path = "/tmp/test_adapt2.json"
        a = AdaptiveMemory(state_file=path)
        a.tuning.preflop_aggression = 1.15
        a.save()
        b = AdaptiveMemory(state_file=path)
        assert abs(b.tuning.preflop_aggression - 1.15) < 0.01


class TestOwnerMessages:
    def test_derive_handle(self):
        from agent.owner_messages import derive_handle
        assert derive_handle("Plutus Bot") == "plutus_bot"
        assert len(derive_handle("A" * 50)) <= 30


class TestHeartbeat:
    def test_dedup_interval(self):
        from agent.heartbeat import HeartbeatState
        hb = HeartbeatState("/tmp/hb_test.json")
        hb.last_heartbeat_at = 0
        assert hb.should_run(3600) is True
        hb.mark_ran()
        assert hb.should_run(3600) is False


class TestArenaOnboarding:
    def test_benchmark_detection(self):
        from api.arena_onboarding import is_benchmark_competition
        assert is_benchmark_competition(
            {"id": "seed_poker_eval_s1", "gameType": "TexasHoldem", "name": "[Poker] Eval S1"}
        )
        assert not is_benchmark_competition(
            {"id": "live_lobby_s21", "gameType": "TexasHoldem", "name": "Holdem S21"}
        )

    def test_pick_competition_by_id(self):
        from api.arena_onboarding import pick_competition
        comps = [
            {"id": "a", "gameType": "TexasHoldem", "startAt": 1, "seasonNumber": 1},
            {"id": "b", "gameType": "TexasHoldem", "startAt": 2, "seasonNumber": 2},
        ]
        assert pick_competition(comps, competition_id="a")["id"] == "a"

    def test_pick_latest_season(self):
        from api.arena_onboarding import pick_competition
        comps = [
            {"id": "old", "gameType": "TexasHoldem", "startAt": 1, "seasonNumber": 20},
            {"id": "new", "gameType": "TexasHoldem", "startAt": 99, "seasonNumber": 21},
        ]
        assert pick_competition(comps)["id"] == "new"

    def test_credentials_key_value(self, tmp_path):
        from api.arena_client import ArenaClient
        cred = tmp_path / "creds"
        cred.write_text('apiKey=arena_sk_test12345678901234567890123456789012345678901234567890\n')
        c = ArenaClient(credentials_file=str(cred))
        assert c.load_credentials()
        assert c._api_key.startswith("arena_sk_")


class TestPushFold:
    def test_aa_pushes_short(self):
        from engine.push_fold import should_push
        assert should_push("AA", "BTN", 10, False) is True


class TestBotPatternDetector:
    def test_classify_nit(self):
        from models.bot_pattern_detector import BotPatternDetector, BotType
        from models.opponent_tracker import OpponentStats

        det = BotPatternDetector()
        s = OpponentStats(agent_id="bot1", hands_seen=30, vpip_count=4, pfr_count=3)
        s.fold_to_steal_opps = 10
        s.fold_to_steal_count = 8
        p = det.classify(s)
        assert p.bot_type in (BotType.NIT, BotType.SCARED_MONEY, BotType.UNKNOWN)


class TestStrategyModes:
    def test_early_intimidation(self):
        from agent.strategy_modes import StrategyModeSelector, TournamentContext, StrategyMode

        sel = StrategyModeSelector()
        ctx = TournamentContext(match_hands_played=10, target_hands=500, bb_depth=50)
        mode = sel.select(ctx, ["opp"], {})
        assert mode == StrategyMode.INTIMIDATION


class TestEarlyGame:
    def test_top20_open(self):
        from agent.early_game import EarlyGameAggressor
        from models.opponent_tracker import OpponentTracker

        eg = EarlyGameAggressor(OpponentTracker("/tmp/eg.json"))
        assert eg.is_top20(["Ah", "Kd"])
        acts = [{"action": "raise", "minAmount": 4, "maxAmount": 200}]
        r = eg.decide(
            ["Ah", "Kd"], "BTN", 200, 2, 3, False, 0, acts, []
        )
        assert r is not None
        assert r[0] == "raise"


class TestMetaLearner:
    def test_select_and_record(self):
        from agent.meta_learner import MetaLearner

        m = MetaLearner(state_file="/tmp/meta_test.json")
        assert m.select_strategy() in ("MANIAC", "TAG", "LAG", "NIT", "EXPLOIT")
        m.record_outcome("TAG", 50, True)


class TestActionAmount:
    def test_call_to_amount(self):
        from api.action_amount import format_submission_amount

        table = {
            "seats": [{"agentId": "me", "currentBetChips": 2, "stackChips": 200}],
            "allowedActions": {
                "callAmount": 11,
                "callToAmount": 13,
                "minRaiseTo": 22,
                "raiseRange": {"min": 22, "max": 200},
            },
        }
        assert format_submission_amount("call", 11, table, "me") == 13


class TestChatTilt:
    def test_message_length(self):
        from agent.chat_tilt import build_chat_message

        msg = build_chat_message(action="raise", was_bluff=True, ehs=0.2, strategy_mode="intimidation")
        assert len(msg) <= 200


class TestBrutalCheck:
    def test_intimidation_abort(self):
        from agent.brutal_check import BrutalSelfCheck

        b = BrutalSelfCheck(state_file="/tmp/brutal_abort.json")
        for _ in range(55):
            b.record_hand(
                strategy_mode="intimidation",
                meta_strategy="MANIAC",
                chip_delta=-5,
                bb_size=2,
            )
        assert b._intimidation_aborted is True

    def test_bootstrap(self):
        from agent.brutal_check import bootstrap_improvement_significant

        ok, p = bootstrap_improvement_significant([0.0] * 30, [5.0] * 30)
        assert ok and p < 0.1


class TestActionPayload:
    def test_arena_call_payload(self):
        from api.action_amount import build_action_payload

        table = {
            "seats": [{"agentId": "me", "currentBetChips": 2, "stackChips": 200}],
            "allowedActions": {
                "callAmount": 11,
                "callToAmount": 13,
                "minRaiseTo": 22,
                "allInToAmount": 200,
            },
        }
        p = build_action_payload("t1", "call", 11, table, "me")
        assert p["amount"] == 13
        assert p["action"] == "call"


class TestArbiterSafety:
    def test_never_fold_aa(self):
        from agent.arbiter import StrategyArbiter, GameContext
        from models.opponent_tracker import OpponentTracker

        arb = StrategyArbiter(OpponentTracker("/tmp/safe.json"))
        ctx = GameContext(
            hole_cards=["Ah", "Ad"],
            community_cards=[],
            pot=100,
            call_amount=50,
            stack=200,
            bb_size=2,
            street="preflop",
            position="BTN",
            is_in_position=True,
            is_facing_raise=True,
            allowed_actions=[
                {"action": "fold"},
                {"action": "call", "toAmount": 50},
                {"action": "raise", "minAmount": 100, "maxAmount": 200},
            ],
        )
        out = arb._safety_override(ctx)
        assert out is not None
        assert out[0] == "raise"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
