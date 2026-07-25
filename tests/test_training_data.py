from __future__ import annotations

import unittest

from pokemon_battler.training_data import SFTCollator
from tests.helpers import state


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, value: str, add_special_tokens: bool = True) -> list[int]:
        prefix = [2] if add_special_tokens else []
        return prefix + [3 + ord(character) for character in value]


class CollatorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

