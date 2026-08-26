from __future__ import annotations

import unittest

from pokemon_battler.core.actions import (
    action_label,
    describe_action,
    legal_action_ids,
    parse_action_label,
    pp_aware_legal_action_ids,
)
from pokemon_battler.core.prompting import render_prompt, render_prompt_sections
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

    def test_compact_prompt_keeps_candidate_mapping(self) -> None:
        sections = render_prompt_sections(state(), "compact-v1")
        prompt = sections.text
        self.assertIn("LEGAL\nA0|move|ironhead|tera=0", prompt)
        self.assertIn("A4|switch|alakazam", prompt)
        self.assertTrue(prompt.endswith("ANSWER\n"))

    def test_action_label_round_trip(self) -> None:
        for action_id in range(13):
            self.assertEqual(parse_action_label(action_label(action_id)), action_id)
        with self.assertRaises(ValueError):
            parse_action_label("A13")

    def test_pp_aware_mask_removes_normal_and_tera_versions(self) -> None:
        battle_state = state()
        iron_head = next(
            move for move in battle_state["player_active_pokemon"]["moves"]
            if move["name"] == "ironhead"
        )
        iron_head["current_pp"] = 0
        refined = pp_aware_legal_action_ids(battle_state)
        self.assertNotIn(0, refined)
        self.assertNotIn(9, refined)
        battle_state["prepared_legal_action_ids"] = refined
        self.assertEqual(legal_action_ids(battle_state), refined)


if __name__ == "__main__":
    unittest.main()
