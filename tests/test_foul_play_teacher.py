from __future__ import annotations

import unittest
from types import SimpleNamespace

from pokemon_battler.foul_play_teacher_bridge import (
    _choice_action_id,
    aggregate_mcts_policy,
    aggregate_mcts_targets,
)
from tests.helpers import state


class FoulPlayTeacherTests(unittest.TestCase):
    def test_mcts_policy_aggregates_every_sample_and_does_not_apply_top_filter(self) -> None:
        first = SimpleNamespace(
            total_visits=100,
            side_one=[
                SimpleNamespace(move_choice="protect", visits=75),
                SimpleNamespace(move_choice="switch alakazam", visits=25),
            ],
        )
        second = SimpleNamespace(
            total_visits=200,
            side_one=[
                SimpleNamespace(move_choice="protect", visits=100),
                SimpleNamespace(move_choice="switch alakazam", visits=100),
                SimpleNamespace(move_choice="quickattack", visits=0),
            ],
        )
        policy, visits = aggregate_mcts_policy(
            [(first, 0.25, 0), (second, 0.75, 1)]
        )
        self.assertEqual(visits, 300)
        self.assertAlmostEqual(policy["protect"], 0.5625)
        self.assertAlmostEqual(policy["switch alakazam"], 0.4375)
        self.assertEqual(policy["quickattack"], 0.0)

    def test_choices_map_to_internal_candidates_without_text_generation(self) -> None:
        observation = state()
        self.assertEqual(_choice_action_id(observation, "protect"), 1)
        self.assertEqual(_choice_action_id(observation, "protect-tera"), 10)
        self.assertEqual(_choice_action_id(observation, "switch alakazam"), 4)
        self.assertEqual(_choice_action_id(observation, "switch charizard"), 5)

    def test_mcts_targets_preserve_per_action_win_values(self) -> None:
        result = SimpleNamespace(
            total_visits=100,
            side_one=[
                SimpleNamespace(move_choice="protect", visits=60, total_score=42.0),
                SimpleNamespace(
                    move_choice="switch alakazam", visits=40, total_score=20.0
                ),
            ],
        )
        policy, values, root, visits = aggregate_mcts_targets([(result, 1.0, 0)])
        self.assertEqual(visits, 100)
        self.assertAlmostEqual(policy["protect"], 0.6)
        self.assertAlmostEqual(values["protect"], 0.7)
        self.assertAlmostEqual(values["switch alakazam"], 0.5)
        self.assertAlmostEqual(root or 0.0, 0.62)


if __name__ == "__main__":
    unittest.main()
