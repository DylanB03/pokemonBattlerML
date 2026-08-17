from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from pokemon_battler.interaction_features import validate_interaction_observation
from pokemon_battler.live_eval import DEFAULT_TEAM, build_parser
from pokemon_battler.live_policy import (
    InteractionPlayer,
    parse_showdown_rating_update,
)
from pokemon_battler.live_state import (
    LiveBattleTracker,
    battle_to_metamon_state,
    exact_live_legal_action_ids,
    live_action_to_order,
)
from pokemon_battler.training_data import InteractionInferenceCollator


class Named:
    def __init__(self, name: str) -> None:
        self.name = name


def fake_move(name: str, *, power: int = 80):
    return SimpleNamespace(
        id=name,
        type=Named("psychic"),
        category=Named("special" if power else "status"),
        base_power=power,
        accuracy=1.0,
        priority=0,
        current_pp=16,
        max_pp=16,
    )


def fake_pokemon(
    species: str,
    *,
    moves: list[object] | None = None,
    active: bool = False,
    fainted: bool = False,
    hp: float = 1.0,
    tera_type: str | None = "water",
    request_tera_type: str | None = None,
):
    move_list = moves or [fake_move("tackle", power=40)]
    return SimpleNamespace(
        species=species,
        base_species=species,
        name=species,
        current_hp_fraction=hp,
        types=[Named("psychic")],
        tera_type=Named(tera_type) if tera_type else None,
        _last_request=(
            {"teraType": request_tera_type} if request_tera_type else None
        ),
        item="leftovers",
        ability="pressure",
        level=100,
        status=Named("fnt") if fainted else None,
        effects={},
        moves={move.id: move for move in move_list},
        boosts={
            "atk": 0,
            "spa": 0,
            "def": 0,
            "spd": 0,
            "spe": 0,
            "accuracy": 0,
            "evasion": 0,
        },
        base_stats={
            "hp": 100,
            "atk": 100,
            "def": 100,
            "spa": 100,
            "spd": 100,
            "spe": 100,
        },
        active=active,
        fainted=fainted,
        last_move=None,
        _selected_in_teampreview=False,
    )


def fake_battle():
    moves = [
        fake_move("thunderbolt", power=90),
        fake_move("calmmind", power=0),
        fake_move("moonblast", power=95),
        fake_move("psyshock", power=80),
    ]
    active = fake_pokemon("ironvaliant", moves=moves, active=True)
    corviknight = fake_pokemon("corviknight")
    gholdengo = fake_pokemon("gholdengo")
    opponent = fake_pokemon("clefable", active=True)
    return SimpleNamespace(
        battle_tag="battle-gen9ou-test",
        format="gen9ou",
        turn=1,
        last_request={"rqid": 1},
        active_pokemon=active,
        opponent_active_pokemon=opponent,
        team={"active": active, "gholdengo": gholdengo, "corviknight": corviknight},
        opponent_team={"opponent": opponent},
        teampreview_opponent_team=[opponent],
        available_moves=[moves[0], moves[2]],
        available_switches=[gholdengo],
        can_tera=True,
        force_switch=False,
        reviving=False,
        side_conditions={},
        opponent_side_conditions={},
        fields={},
        weather={},
        won=None,
        lost=None,
    )


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    @staticmethod
    def encode(text: str, add_special_tokens: bool = True) -> list[int]:
        prefix = [2] if add_special_tokens else []
        return prefix + [3 + (ord(character) % 101) for character in text]


class LivePolicyTests(unittest.TestCase):
    def test_finished_battle_callback_includes_terminal_progress_fields(self) -> None:
        events = []
        player = object.__new__(InteractionPlayer)
        player.decision_callback = events.append
        player.trace_writer = None
        battle = SimpleNamespace(
            battle_tag="battle-gen9ou-finished",
            won=True,
            lost=False,
            opponent_username="HumanOpponent",
            turn=19,
            rating=1100,
            opponent_rating=1080,
        )

        player._battle_finished_callback(battle)

        self.assertEqual(
            events,
            [
                {
                    "event": "battle_finished",
                    "battle_id": "battle-gen9ou-finished",
                    "won": True,
                    "lost": False,
                    "opponent": "HumanOpponent",
                    "turns": 19,
                    "rating": 1100,
                    "opponent_rating": 1080,
                    "rating_update": None,
                }
            ],
        )

    def test_showdown_rating_update_parser_reads_exact_ladder_transition(self) -> None:
        update = parse_showdown_rating_update(
            "ATSskipper5's rating: 1074 &rarr; "
            "<strong>1100</strong><br />(+26 for winning)",
            username="ats skipper 5",
        )
        self.assertEqual(
            update,
            {
                "before": 1074,
                "after": 1100,
                "change": 26,
                "result": "winning",
            },
        )
        self.assertIsNone(
            parse_showdown_rating_update(
                "Opponent's rating: 1100 &rarr; <strong>1074</strong>"
                "<br />(-26 for losing)",
                username="ATSskipper5",
            )
        )

    def test_rating_update_is_attached_to_finished_battle_event(self) -> None:
        events = []
        player = object.__new__(InteractionPlayer)
        player.ps_client = SimpleNamespace(username="ATSskipper5")
        player.rating_updates = {}
        player.decision_callback = events.append
        player.trace_writer = None
        raw_messages = [
            [">battle-gen9ou-123"],
            ["", "win", "ATSskipper5"],
            [
                "",
                "raw",
                "ATSskipper5's rating: 1090 &rarr; "
                "<strong>1112</strong><br />(+22 for winning)",
            ],
        ]
        player._capture_rating_updates(raw_messages)
        battle = SimpleNamespace(
            battle_tag="battle-gen9ou-123",
            won=True,
            lost=False,
            opponent_username="Opponent",
            turn=12,
            rating=1090,
            opponent_rating=1100,
        )

        player._battle_finished_callback(battle)

        self.assertEqual(events[0]["rating_update"]["after"], 1112)
        self.assertEqual(events[0]["rating_update"]["change"], 22)

    def test_showdown_room_rename_is_filtered_and_aliased(self) -> None:
        player = object.__new__(InteractionPlayer)
        player.ps_client = SimpleNamespace(username="ATSskipper5")
        player.rating_updates = {}
        battle = SimpleNamespace(battle_tag="battle-gen9ou-123")
        player._battles = {battle.battle_tag: battle}
        messages = [
            [">battle-gen9ou-123"],
            [
                "",
                "noinit",
                "rename",
                "battle-gen9ou-123-private",
                "Opponent vs. ATSskipper5",
            ],
        ]

        with patch(
            "poke_env.player.player.Player._handle_battle_message",
            new_callable=AsyncMock,
        ) as base_handler:
            asyncio.run(player._handle_battle_message(messages))

        base_handler.assert_awaited_once_with([[">battle-gen9ou-123"]])
        self.assertIs(player._battles["battle-gen9ou-123-private"], battle)

    def test_normal_battle_messages_are_forwarded_unchanged(self) -> None:
        player = object.__new__(InteractionPlayer)
        player.ps_client = SimpleNamespace(username="ATSskipper5")
        player.rating_updates = {}
        player._battles = {}
        messages = [
            [">battle-gen9ou-123"],
            ["", "turn", "14"],
            ["", "win", "ATSskipper5"],
        ]

        with patch(
            "poke_env.player.player.Player._handle_battle_message",
            new_callable=AsyncMock,
        ) as base_handler:
            asyncio.run(player._handle_battle_message(messages))

        base_handler.assert_awaited_once_with(messages)

    def test_public_preview_policy_can_randomize_the_lead(self) -> None:
        player = object.__new__(InteractionPlayer)
        player.team_preview_policy = "random"
        player.random_teampreview = lambda _battle: "/team 321"
        self.assertEqual(player.teampreview(fake_battle()), "/team 321")

    def test_missing_learned_preview_warning_is_emitted_once(self) -> None:
        player = object.__new__(InteractionPlayer)
        player.team_preview_policy = "learned"
        player.runtime = SimpleNamespace(preview_head=None)
        player._missing_preview_warning_emitted = False
        logger = Mock()

        with patch.object(
            InteractionPlayer, "logger", new_callable=PropertyMock, return_value=logger
        ):
            self.assertEqual(player.teampreview(fake_battle()), "/team 123")
            self.assertEqual(player.teampreview(fake_battle()), "/team 123")
        logger.warning.assert_called_once()

    def test_exact_mask_preserves_full_alphabetical_action_slots(self) -> None:
        battle = fake_battle()
        # Full order: calmmind, moonblast, psyshock, thunderbolt. Only Moonblast
        # and Thunderbolt are available. Full switch order is Corviknight, Gholdengo,
        # but only Gholdengo is currently legal.
        self.assertEqual(
            exact_live_legal_action_ids(battle),
            [1, 3, 5, 10, 12],
        )
        order = live_action_to_order(battle, 10)
        self.assertIsNotNone(order)
        self.assertEqual(order.order.id, "moonblast")
        self.assertTrue(order.terastallize)
        self.assertIsNone(live_action_to_order(battle, 4))

    def test_forced_switch_removes_moves_and_tera(self) -> None:
        battle = fake_battle()
        battle.force_switch = True
        battle.available_moves = []
        self.assertEqual(exact_live_legal_action_ids(battle), [5])

    def test_tracker_builds_unlabelled_rows_and_deduplicates_requests(self) -> None:
        battle = fake_battle()
        tracker = LiveBattleTracker(battle.battle_tag)
        first = tracker.observe(battle)
        validate_interaction_observation(first)
        self.assertNotIn("action_id", first)
        self.assertEqual(first["turn_index"], 0)
        self.assertEqual(first["history_events"], [])

        battle.turn = 2
        battle.last_request = {"rqid": 2}
        battle.opponent_active_pokemon.current_hp_fraction = 0.5
        battle.active_pokemon.last_move = battle.active_pokemon.moves["moonblast"]
        second = tracker.observe(battle)
        self.assertEqual(second["turn_index"], 1)
        self.assertEqual(len(second["history_events"]), 1)
        self.assertAlmostEqual(second["history_events"][0]["opponent_hp_delta"], -0.5)

        duplicate = tracker.observe(battle)
        self.assertEqual(duplicate["turn_index"], 1)
        self.assertEqual(len(duplicate["history_events"]), 1)
        self.assertEqual(tracker.decision_count, 2)

    def test_live_state_retains_all_conditions_effects_and_request_tera_type(self) -> None:
        battle = fake_battle()
        battle.active_pokemon.tera_type = None
        battle.active_pokemon._last_request = {"teraType": "Water"}
        battle.active_pokemon.effects = {
            Named("substitute"): 0,
            Named("leech_seed"): 0,
        }
        battle.side_conditions = {
            Named("stealth_rock"): 1,
            Named("spikes"): 3,
            Named("reflect"): 1,
        }

        state = battle_to_metamon_state(battle)

        self.assertEqual(state["player_active_pokemon"]["tera_type"], "water")
        self.assertEqual(
            state["player_active_pokemon"]["effect"],
            "leechseed substitute",
        )
        self.assertEqual(
            state["player_conditions"],
            "reflect spikes3 stealthrock",
        )

    def test_inference_collator_does_not_require_targets(self) -> None:
        row = LiveBattleTracker("battle-gen9ou-test").observe(fake_battle())
        batch = InteractionInferenceCollator(
            FakeTokenizer(),
            max_length=20_000,
            prompt_format="mechanics-v2",
        )([row])
        self.assertNotIn("action_ids", batch)
        self.assertNotIn("value_targets", batch)
        self.assertEqual(tuple(batch["legal_action_mask"].shape), (1, 13))
        self.assertEqual(int(batch["legal_action_mask"].sum()), 5)

    def test_bundled_team_and_cli_defaults_are_usable(self) -> None:
        team = ConstantTeambuilder(DEFAULT_TEAM.read_text(encoding="utf-8"))
        self.assertEqual(len(team.team), 6)
        args = build_parser().parse_args([])
        self.assertEqual(args.checkpoint, Path("outputs/interaction-v1-1epoch/policy/final"))
        self.assertEqual(args.team_file, DEFAULT_TEAM)
        self.assertEqual(args.opponent, "heuristic")


if __name__ == "__main__":
    unittest.main()
