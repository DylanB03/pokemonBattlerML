from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from pokemon_battler.data.frozen_cache import (
    DATA_FILENAME,
    FROZEN_CACHE_SCHEMA,
    FrozenCacheDataset,
)


class FrozenCacheTests(unittest.TestCase):
    def test_cache_dataset_validates_kind_and_row_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            (cache / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema": FROZEN_CACHE_SCHEMA,
                        "kind": "teacher",
                        "rows": 2,
                    }
                ),
                encoding="utf-8",
            )
            torch.save(
                {
                    "qwen_state_hidden": torch.zeros((2, 8), dtype=torch.float16),
                    "legal_action_mask": torch.ones((2, 13), dtype=torch.bool),
                },
                cache / DATA_FILENAME,
            )
            dataset = FrozenCacheDataset(cache, expected_kind="teacher")
            self.assertEqual(len(dataset), 2)
            self.assertEqual(tuple(dataset[0]["qwen_state_hidden"].shape), (8,))
            with self.assertRaisesRegex(ValueError, "Expected a 'replay' cache"):
                FrozenCacheDataset(cache, expected_kind="replay")
