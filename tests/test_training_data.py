from __future__ import annotations

import unittest

import torch

from pokemon_battler.training_data import (
    CandidateCollator,
    MechanicsCollator,
    PolicyCollator,
    SFTCollator,
    state_with_row_context,
)
from tests.helpers import state


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, value: str, add_special_tokens: bool = True) -> list[int]:
        prefix = [2] if add_special_tokens else []
        return prefix + [3 + ord(character) for character in value]


class CollatorTests(unittest.TestCase):
    def test_legacy_row_metadata_is_added_without_mutating_state(self) -> None:
        battle_state = state()
        enriched = state_with_row_context({"state": battle_state, "turn_index": 17})
        self.assertEqual(enriched["turn_index"], 17)
        self.assertEqual(enriched["player_remaining"], 3)
        self.assertNotIn("turn_index", battle_state)

    def test_only_target_tokens_receive_loss(self) -> None:
        collator = SFTCollator(FakeTokenizer(), max_length=20_000)
        batch = collator(
            [
                {
                    "state": state(),
                    "action_id": 0,
                    "target": "A0",
                }
            ]
        )
        labels = batch["labels"][0].tolist()
        supervised = [value for value in labels if value != -100]
        self.assertEqual(supervised, FakeTokenizer().encode("A0", False) + [1])
        self.assertGreater(labels.count(-100), len(supervised))

    def test_target_mismatch_is_rejected(self) -> None:
        collator = SFTCollator(FakeTokenizer(), max_length=20_000)
        with self.assertRaises(ValueError):
            collator([{"state": state(), "action_id": 0, "target": "A1"}])

    def test_policy_collator_masks_illegal_actions(self) -> None:
        collator = PolicyCollator(FakeTokenizer(), max_length=20_000)
        batch = collator(
            [
                {
                    "state": state(forced_switch=True),
                    "action_id": 4,
                    "legal_action_ids": [4, 5],
                }
            ]
        )
        self.assertEqual(batch["action_ids"].tolist(), [4])
        self.assertEqual(
            torch.nonzero(batch["legal_action_mask"][0], as_tuple=False)
            .flatten()
            .tolist(),
            [4, 5],
        )

    def test_candidate_collator_records_only_legal_marker_positions(self) -> None:
        collator = CandidateCollator(
            FakeTokenizer(),
            max_length=20_000,
            prompt_format="compact-v1",
        )
        batch = collator(
            [
                {
                    "state": state(forced_switch=True),
                    "action_id": 4,
                    "legal_action_ids": [4, 5],
                }
            ]
        )
        positions = batch["candidate_positions"][0]
        self.assertGreaterEqual(int(positions[4]), 0)
        self.assertGreaterEqual(int(positions[5]), 0)
        self.assertTrue(torch.all(positions[[0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12]] == -1))
        self.assertLess(int(positions.max()), int(batch["attention_mask"].sum()))

    def test_mechanics_collator_adds_numeric_and_identity_features(self) -> None:
        collator = MechanicsCollator(FakeTokenizer(), max_length=20_000)
        batch = collator(
            [
                {
                    "state": state(forced_switch=True),
                    "action_id": 4,
                    "legal_action_ids": [4, 5],
                }
            ]
        )
        self.assertEqual(batch["mechanics_features"].shape[:2], (1, 13))
        self.assertEqual(batch["mechanics_identity_ids"].shape[:2], (1, 13))
        self.assertEqual(batch["action_ids"].tolist(), [4])
        self.assertEqual(
            torch.nonzero(batch["legal_action_mask"][0], as_tuple=False).flatten().tolist(),
            [4, 5],
        )


if __name__ == "__main__":
    unittest.main()
