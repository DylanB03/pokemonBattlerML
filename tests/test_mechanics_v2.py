from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pokemon_battler.actions import ACTION_COUNT
from pokemon_battler.mechanics_v2 import (
    MECHANICS_FEATURE_COUNT,
    MECHANICS_FEATURE_NAMES,
    MECHANICS_IDENTITY_COUNT,
    MECHANICS_IDENTITY_NAMES,
    candidate_feature_matrix,
    candidate_identity_matrix,
)
from pokemon_battler.mechanics_cache import build_feature_cache, cache_is_current
from pokemon_battler.prompting import render_prompt
from pokemon_battler.training_data import JsonlOffsetDataset, MechanicsCacheDataset
from tests.helpers import move, state


def feature(matrix: list[list[float]], action_id: int, name: str) -> float:
    return matrix[action_id][MECHANICS_FEATURE_NAMES.index(name)]


def identity(matrix: list[list[int]], action_id: int, name: str) -> int:
    return matrix[action_id][MECHANICS_IDENTITY_NAMES.index(name)]


class MechanicsV2Tests(unittest.TestCase):
    def test_schema_has_numeric_and_categorical_candidate_inputs(self) -> None:
        battle_state = state(forced_switch=True)
        features = candidate_feature_matrix(battle_state)
        identities = candidate_identity_matrix(battle_state)

        self.assertEqual(MECHANICS_FEATURE_COUNT, 207)
        self.assertEqual(MECHANICS_IDENTITY_COUNT, 32)
        self.assertEqual(len(features), ACTION_COUNT)
        self.assertTrue(all(len(row) == MECHANICS_FEATURE_COUNT for row in features))
        self.assertTrue(all(len(row) == MECHANICS_IDENTITY_COUNT for row in identities))
        self.assertTrue(all(value == 0 for value in identities[0]))
        self.assertNotEqual(
            identity(identities, 4, "actor_species"),
            identity(identities, 5, "actor_species"),
        )

    def test_compact_prompt_retains_names_without_move_stat_prose(self) -> None:
        prompt = render_prompt(state(), "mechanics-v2").lower()

        self.assertIn("zoroark", prompt)
        self.assertIn("a0:ironhead", prompt)
        self.assertIn("saltcure", prompt)
        self.assertNotIn("base_power", prompt)
        self.assertNotIn("current_pp", prompt)

    def test_special_status_moves_are_not_collapsed(self) -> None:
        battle_state = state(can_tera=False)
        reflect = move("reflect") | {"category": "status", "base_power": 0}
        light_screen = move("lightscreen") | {"category": "status", "base_power": 0}
        battle_state["player_active_pokemon"]["moves"] = [reflect, light_screen]
        features = candidate_feature_matrix(battle_state)
        identities = candidate_identity_matrix(battle_state)

        # Alphabetical order is Light Screen, Reflect.
        self.assertEqual(feature(features, 0, "sets_light_screen"), 1.0)
        self.assertEqual(feature(features, 1, "sets_reflect"), 1.0)
        self.assertNotEqual(
            identity(identities, 0, "candidate_move"),
            identity(identities, 1, "candidate_move"),
        )

    def test_distinct_abilities_have_distinct_exact_ids(self) -> None:
        first = state(can_tera=False)
        second = state(can_tera=False)
        first["player_active_pokemon"]["ability"] = "longreach"
        second["player_active_pokemon"]["ability"] = "technician"

        first_id = identity(candidate_identity_matrix(first), 0, "actor_ability")
        second_id = identity(candidate_identity_matrix(second), 0, "actor_ability")

        self.assertNotEqual(first_id, second_id)

    def test_side_conditions_and_original_tera_stab_remain_explicit(self) -> None:
        battle_state = state()
        battle_state["player_conditions"] = "stealthrock spikes3 reflect"
        battle_state["opponent_conditions"] = "toxicspikes2 lightscreen"
        battle_state["player_active_pokemon"]["moves"] = [move("tackle")]
        battle_state["player_active_pokemon"]["tera_type"] = "fire"
        features = candidate_feature_matrix(battle_state)

        self.assertEqual(feature(features, 0, "player_side_stealth_rock"), 1.0)
        self.assertEqual(feature(features, 0, "player_side_spikes_layers"), 1.0)
        self.assertEqual(feature(features, 0, "player_side_reflect"), 1.0)
        self.assertEqual(feature(features, 0, "opponent_side_toxic_spikes_layers"), 1.0)
        self.assertEqual(feature(features, 0, "opponent_side_light_screen"), 1.0)
        self.assertEqual(feature(features, 0, "stab_scaled"), 0.75)
        self.assertEqual(feature(features, 9, "stab_scaled"), 0.75)

    def test_v2_cache_is_isolated_from_v1(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = Path(temp_dir) / "tiny.jsonl"
            data_file.write_text(
                json.dumps({"state": state(forced_switch=True), "action_id": 4}) + "\n",
                encoding="utf-8",
            )
            metadata = build_feature_cache(data_file, progress_every=0)
            cache = Path(metadata["cache_file"])
            dataset = MechanicsCacheDataset(JsonlOffsetDataset(data_file), cache)

            self.assertTrue(cache.name.endswith(".mechanics-v2.npy"))
            self.assertTrue(cache_is_current(data_file, cache))
            self.assertEqual(metadata["schema"], "mechanics-v2")
            self.assertEqual(
                dataset[0]["_mechanics_features"].shape,
                (ACTION_COUNT, MECHANICS_FEATURE_COUNT),
            )


if __name__ == "__main__":
    unittest.main()
