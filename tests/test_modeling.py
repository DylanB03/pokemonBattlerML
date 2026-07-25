from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from pokemon_battler.modeling import (
    assistant_only_loss,
    indexed_logits_parameter,
    score_legal_actions,
)
from tests.helpers import state


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, value: str, add_special_tokens: bool = True) -> list[int]:
        prefix = [2] if add_special_tokens else []
        return prefix + [3 + ord(character) for character in value]


class LimitedLogitsModel:
    def __init__(self) -> None:
        self.logits_to_keep: int | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int = 0,
    ) -> SimpleNamespace:
        del attention_mask
        self.logits_to_keep = logits_to_keep
        logits = torch.zeros(
            (input_ids.shape[0], logits_to_keep, 100),
            dtype=torch.float32,
        )
        return SimpleNamespace(logits=logits)

    __call__ = forward


class IndexedLossModel:
    def __init__(self) -> None:
        self.positions: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        self.positions = logits_to_keep
        logits = torch.zeros(
            (input_ids.shape[0], logits_to_keep.numel(), 10),
            dtype=torch.float32,
            requires_grad=True,
        )
        return SimpleNamespace(logits=logits)

    __call__ = forward


class ModelingTests(unittest.TestCase):
    def test_action_scoring_requests_only_suffix_logits(self) -> None:
        model = LimitedLogitsModel()
        scores = score_legal_actions(
            model,
            FakeTokenizer(),
            state(),
            torch.device("cpu"),
            max_length=20_000,
        )

        # A10-A12 tokenize to three characters plus EOS, the longest candidate.
        self.assertEqual(model.logits_to_keep, 5)
        self.assertEqual(set(scores), {0, 1, 2, 3, 4, 5, 9, 10, 11, 12})
        self.assertAlmostEqual(scores[0], -3 * math.log(100), places=5)
        self.assertAlmostEqual(scores[10], -4 * math.log(100), places=5)

    def test_assistant_loss_projects_only_supervised_positions(self) -> None:
        model = IndexedLossModel()
        batch = {
            "input_ids": torch.tensor([[2, 3, 4, 5, 6]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "labels": torch.tensor([[-100, -100, -100, 5, 6]]),
        }
        parameter = indexed_logits_parameter(model)
        loss = assistant_only_loss(model, batch, logits_parameter=parameter)

        self.assertEqual(parameter, "logits_to_keep")
        self.assertEqual(model.positions.tolist(), [2, 3])
        self.assertAlmostEqual(float(loss.item()), math.log(10), places=5)
        loss.backward()


if __name__ == "__main__":
    unittest.main()
