from __future__ import annotations

import unittest
from collections import Counter

from pokemon_battler.evaluation_utils import ActionMetrics, select_evaluation_dataset
from tests.helpers import state


class RowDataset:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class EvaluationUtilityTests(unittest.TestCase):
    def test_hash_sampling_is_reproducible_and_not_just_the_head(self) -> None:
        rows = [
            {"battle_id": f"battle-{index}", "turn_index": index, "source": "test"}
            for index in range(20)
        ]
        dataset = RowDataset(rows)
        first, first_metadata = select_evaluation_dataset(
            dataset,
            max_examples=5,
            mode="hash",
            seed=7,
        )
        second, second_metadata = select_evaluation_dataset(
            dataset,
            max_examples=5,
            mode="hash",
            seed=7,
        )
        self.assertEqual(first.indices, second.indices)
        self.assertEqual(
            first_metadata["selected_indices_sha256"],
            second_metadata["selected_indices_sha256"],
        )
        self.assertNotEqual(first.indices, list(range(5)))

    def test_metrics_separate_type_and_oracle_type_accuracy(self) -> None:
        battle_state = state()
        row = {
            "state": battle_state,
            "action_id": 4,
            "legal_action_ids": [0, 1, 2, 3, 4, 5, 9, 10, 11, 12],
            "battle_id": "battle",
            "battle_date": "2026-01-01",
        }
        scores = {action_id: -float(action_id) for action_id in row["legal_action_ids"]}
        scores[0] = 10.0
        scores[4] = 9.0
        metrics = ActionMetrics(train_action_counts=Counter({0: 10, 4: 5}))
        metrics.add(0, row, scores)
        report = metrics.report()
        self.assertEqual(report["accuracy"], 0.0)
        self.assertEqual(report["action_type_accuracy"], 0.0)
        self.assertEqual(report["oracle_type_accuracy"], 1.0)
        self.assertEqual(report["baselines"]["uniform_legal_expected_accuracy"], 0.1)


if __name__ == "__main__":
    unittest.main()

