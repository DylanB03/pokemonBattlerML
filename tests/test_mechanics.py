from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from pokemon_battler.core.actions import ACTION_COUNT
from pokemon_battler.core.mechanics import (
    MECHANICS_FEATURE_COUNT,
    MECHANICS_FEATURE_NAMES,
    candidate_feature_matrix,
)
from pokemon_battler.data.mechanics_cache import build_feature_cache, cache_is_current
from pokemon_battler.core.prompting import render_prompt
from pokemon_battler.data.training_data import JsonlOffsetDataset, MechanicsCacheDataset
from tests.helpers import move, state


def feature(matrix: list[list[float]], action_id: int, name: str) -> float:
    return matrix[action_id][MECHANICS_FEATURE_NAMES.index(name)]


class MechanicsTests(unittest.TestCase):
    def test_matrix_is_dense_mask_compatible_and_name_free(self) -> None:
        battle_state = state(forced_switch=True)
        matrix = candidate_feature_matrix(battle_state)
        prompt = render_prompt(battle_state, "mechanics-v1").lower()

        self.assertEqual(len(matrix), ACTION_COUNT)
        self.assertTrue(all(len(row) == MECHANICS_FEATURE_COUNT for row in matrix))
        self.assertTrue(all(value == 0 for value in matrix[0]))
        self.assertEqual(feature(matrix, 4, "action_switch"), 1.0)
        self.assertNotIn("thunderbolt", prompt)
        self.assertNotIn("saltcure", prompt)

    def test_type_immunity_and_status_boosts_are_explicit(self) -> None:
        immune_state = state(can_tera=False)
        immune_state["opponent_active_pokemon"]["types"] = "ground notype"
        matrix = candidate_feature_matrix(immune_state)
        # Alphabetical order is ironhead, protect, quickattack, thunderbolt.
        self.assertEqual(feature(matrix, 3, "effectiveness_immune"), 1.0)
        self.assertEqual(feature(matrix, 3, "damage_max_target_hp"), 0.0)

        boost_state = state(can_tera=False)
        setup = move("swordsdance")
        setup.update({"category": "status", "base_power": 0})
        boost_state["player_active_pokemon"]["moves"] = [setup]
        matrix = candidate_feature_matrix(boost_state)
        self.assertAlmostEqual(feature(matrix, 0, "self_atk_delta"), 2 / 6)
        self.assertEqual(feature(matrix, 0, "move_status"), 1.0)

    def test_switch_hazard_cost_respects_heavy_duty_boots(self) -> None:
        battle_state = state(forced_switch=True)
        without_boots = candidate_feature_matrix(battle_state)
        self.assertGreater(feature(without_boots, 4, "switch_entry_damage_fraction"), 0)

        with_boots_state = deepcopy(battle_state)
        # A4 is Alakazam after alphabetical sorting.
        with_boots_state["available_switches"][1]["item"] = "heavydutyboots"
        with_boots = candidate_feature_matrix(with_boots_state)
        self.assertEqual(feature(with_boots, 4, "switch_entry_damage_fraction"), 0.0)

    def test_toxic_spikes_are_not_counted_as_spikes_damage(self) -> None:
        battle_state = state(forced_switch=True)
        battle_state["player_conditions"] = "toxicspikes"
        matrix = candidate_feature_matrix(battle_state)

        self.assertEqual(feature(matrix, 4, "switch_entry_damage_fraction"), 0.0)
        self.assertEqual(feature(matrix, 4, "switch_hazard_toxic_spikes"), 1.0)

    def test_side_screen_is_not_labeled_as_a_hazard(self) -> None:
        battle_state = state(can_tera=False)
        screen = move("reflect")
        screen.update({"category": "status", "base_power": 0})
        battle_state["player_active_pokemon"]["moves"] = [screen]
        matrix = candidate_feature_matrix(battle_state)

        self.assertEqual(feature(matrix, 0, "sets_hazard"), 0.0)

    def test_cache_round_trip_is_memory_mapped_and_versioned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_file = root / "tiny.jsonl"
            rows = [
                {"state": state(forced_switch=True), "action_id": 4, "legal_action_ids": [4, 5]},
                {"state": state(can_tera=False), "action_id": 0},
            ]
            data_file.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            cache = root / "tiny.mechanics-v1.npy"
            metadata = build_feature_cache(
                data_file,
                cache,
                schema="mechanics-v1",
                progress_every=0,
            )
            dataset = MechanicsCacheDataset(
                JsonlOffsetDataset(data_file),
                cache,
                mechanics_schema="mechanics-v1",
            )

            self.assertTrue(cache_is_current(data_file, cache, schema="mechanics-v1"))
            self.assertEqual(metadata["rows"], 2)
            self.assertEqual(dataset[0]["_mechanics_features"].shape, (13, MECHANICS_FEATURE_COUNT))
            self.assertEqual(np.load(cache, mmap_mode="r").dtype, np.float16)


if __name__ == "__main__":
    unittest.main()
