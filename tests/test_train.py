from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from pokemon_battler.train import (
    _evaluate_model,
    _training_class_weights,
    learning_rate_multiplier,
)


class HiddenStateModel(torch.nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        del attention_mask, output_hidden_states, use_cache, logits_to_keep
        hidden = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).float()
        return SimpleNamespace(hidden_states=(hidden,))


class TrainTests(unittest.TestCase):
    def test_policy_validation_reports_action_metrics(self) -> None:
        model = HiddenStateModel()
        head = torch.nn.Linear(4, 13, bias=False)
        torch.nn.init.zeros_(head.weight)
        batch = {
            "input_ids": torch.tensor([[0, 1], [0, 2]]),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
            "action_ids": torch.tensor([0, 1]),
            "legal_action_mask": torch.tensor(
                [
                    [True, True] + [False] * 11,
                    [True, True] + [False] * 11,
                ]
            ),
        }
        metrics = _evaluate_model(
            model,
            head,
            [batch],
            torch.device("cpu"),
            torch.float32,
            None,
            "logits_to_keep",
            "policy-head",
        )
        self.assertEqual(metrics["validation_accuracy"], 0.5)
        self.assertEqual(metrics["validation_top_k_accuracy"]["top_2"], 1.0)
        self.assertAlmostEqual(metrics["validation_mrr"], 0.75)
        self.assertEqual(metrics["validation_prediction_counts"], {"A0": 2})

    def test_cosine_schedule_keeps_configured_floor(self) -> None:
        midpoint = learning_rate_multiplier(
            500,
            warmup_updates=0,
            scheduler_updates=1000,
            scheduler_name="cosine",
            min_lr_ratio=0.05,
        )
        end = learning_rate_multiplier(
            1000,
            warmup_updates=0,
            scheduler_updates=1000,
            scheduler_name="cosine",
            min_lr_ratio=0.05,
        )
        self.assertAlmostEqual(midpoint, 0.525)
        self.assertAlmostEqual(end, 0.05)

    def test_constant_schedule_only_uses_warmup(self) -> None:
        warmup = learning_rate_multiplier(
            5,
            warmup_updates=10,
            scheduler_updates=1000,
            scheduler_name="constant-with-warmup",
            min_lr_ratio=0.0,
        )
        steady = learning_rate_multiplier(
            500,
            warmup_updates=10,
            scheduler_updates=1000,
            scheduler_name="constant-with-warmup",
            min_lr_ratio=0.0,
        )
        self.assertEqual(warmup, 0.5)
        self.assertEqual(steady, 1.0)

    def test_optional_class_weights_are_finite_and_capped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.jsonl"
            rows = [{"action_id": 0}] * 9 + [{"action_id": 9}]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            tensor, values = _training_class_weights(
                str(path),
                "sqrt-inverse",
                3.0,
                torch.device("cpu"),
            )
        self.assertIsNotNone(tensor)
        assert values is not None
        self.assertTrue(torch.isfinite(tensor).all())
        self.assertGreater(values[9], values[0])


if __name__ == "__main__":
    unittest.main()
