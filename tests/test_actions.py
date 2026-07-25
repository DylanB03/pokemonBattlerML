from __future__ import annotations

import unittest

from pokemon_battler.actions import (
    action_label,
    describe_action,
    legal_action_ids,
    parse_action_label,
)
from pokemon_battler.prompting import render_prompt
from tests.helpers import state


class ActionTests(unittest.TestCase):
    def test_normal_state_has_moves_switches_and_tera(self) -> None:
        battle_state = state()
        self.assertEqual(
            legal_action_ids(battle_state),
            [0, 1, 2, 3, 4, 5, 9, 10, 11, 12],
        )
        self.assertEqual(describe_action(battle_state, 0)["name"], "ironhead")
        self.assertEqual(describe_action(battle_state, 4)["species"], "alakazam")
        self.assertEqual(describe_action(battle_state, 5)["species"], "charizard")
        self.assertEqual(describe_action(battle_state, 12)["name"], "thunderbolt")

    def test_forced_switch_only_exposes_living_switch_candidates(self) -> None:
        battle_state = state(forced_switch=True)
        # Fainted Pokémon are absent from available_switches in UniversalState.
        battle_state["available_switches"] = [battle_state["available_switches"][0]]
        battle_state["player_active_pokemon"]["hp_pct"] = 0.0
        battle_state["player_active_pokemon"]["status"] = "fnt"
        self.assertEqual(legal_action_ids(battle_state), [4])
        self.assertEqual(describe_action(battle_state, 4)["species"], "charizard")

    def test_prompt_contains_dynamic_semantic_mapping(self) -> None:
        prompt = render_prompt(state())
        self.assertIn("<AVAILABLE_SWITCH action_id=A4>", prompt)
        self.assertIn("<A4> universal_action=4 type=switch species=alakazam", prompt)
        self.assertTrue(prompt.endswith("<ACTION>\n"))

    def test_action_label_round_trip(self) -> None:
        for action_id in range(13):
            self.assertEqual(parse_action_label(action_label(action_id)), action_id)
        with self.assertRaises(ValueError):
            parse_action_label("A13")


if __name__ == "__main__":
    unittest.main()
