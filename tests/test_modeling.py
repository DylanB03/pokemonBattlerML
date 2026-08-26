from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from pokemon_battler.models.modeling import (
    MechanicsHead,
    assistant_only_loss,
    indexed_logits_parameter,
    masked_candidate_logits,
    masked_mechanics_logits,
    masked_policy_logits,
    policy_head_loss,
    score_legal_actions,
)
from pokemon_battler.core.mechanics import MECHANICS_FEATURE_COUNT as V1_FEATURE_COUNT
from pokemon_battler.core.mechanics_v2 import (
    MECHANICS_FEATURE_COUNT as V2_FEATURE_COUNT,
    MECHANICS_IDENTITY_COUNT,
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


class HiddenStateModel:
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        self.assertions = (output_hidden_states, logits_to_keep)
        hidden = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).float()
        return SimpleNamespace(hidden_states=(hidden,))

    __call__ = forward


class FeatureScorer(torch.nn.Module):
    def forward(
        self,
        state_hidden: torch.Tensor,
        mechanics_features: torch.Tensor,
    ) -> torch.Tensor:
        del state_hidden
        return mechanics_features[..., 0]


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

    def test_policy_head_removes_illegal_logits_before_loss(self) -> None:
        model = HiddenStateModel()
        head = torch.nn.Linear(4, 13, bias=False)
        torch.nn.init.zeros_(head.weight)
        with torch.no_grad():
            head.weight[1, 2] = 3.0
            head.weight[5, 2] = 100.0
        batch = {
            "input_ids": torch.tensor([[0, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
            "action_ids": torch.tensor([1]),
            "legal_action_mask": torch.tensor(
                [
                    [
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ]
                ]
            ),
        }
        logits = masked_policy_logits(
            model,
            head,
            batch,
            logits_parameter="logits_to_keep",
        )
        self.assertEqual(model.assertions, (True, 1))
        self.assertTrue(torch.isneginf(logits[0, 5]))
        self.assertEqual(int(logits.argmax(dim=1).item()), 1)
        loss = policy_head_loss(
            model,
            head,
            batch,
            logits_parameter="logits_to_keep",
        )
        self.assertLess(float(loss.item()), 0.1)

    def test_candidate_head_shares_scorer_and_masks_illegal_logits(self) -> None:
        model = HiddenStateModel()
        scorer = torch.nn.Linear(4, 1, bias=False)
        torch.nn.init.zeros_(scorer.weight)
        with torch.no_grad():
            scorer.weight[0, 2] = 3.0
        batch = {
            "input_ids": torch.tensor([[0, 1, 2, 3]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "candidate_positions": torch.tensor([[-1, 2, 1] + [-1] * 10]),
            "legal_action_mask": torch.tensor(
                [[False, True, True] + [False] * 10]
            ),
        }
        logits = masked_candidate_logits(
            model,
            scorer,
            batch,
            logits_parameter="logits_to_keep",
        )
        self.assertEqual(int(logits.argmax(dim=1).item()), 1)
        self.assertTrue(torch.isneginf(logits[0, 0]))

    def test_mechanics_head_scores_numeric_features_and_masks_illegal_actions(self) -> None:
        features = torch.zeros((1, 13, 1))
        features[0, 1, 0] = 3.0
        features[0, 5, 0] = 100.0
        batch = {
            "input_ids": torch.tensor([[0, 1]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "mechanics_features": features,
            "legal_action_mask": torch.tensor(
                [[True, True] + [False] * 11]
            ),
        }
        logits = masked_mechanics_logits(
            HiddenStateModel(),
            FeatureScorer(),
            batch,
            logits_parameter="logits_to_keep",
        )
        self.assertEqual(int(logits.argmax(dim=1).item()), 1)
        self.assertTrue(torch.isneginf(logits[0, 5]))

    def test_mechanics_head_preserves_v1_and_accepts_v2_identities(self) -> None:
        v1 = MechanicsHead(16, schema="mechanics-v1")
        v2 = MechanicsHead(16, schema="mechanics-v2")
        state_hidden = torch.zeros((2, 16))

        v1_logits = v1(state_hidden, torch.zeros((2, 13, V1_FEATURE_COUNT)))
        v2_logits = v2(
            state_hidden,
            torch.zeros((2, 13, V2_FEATURE_COUNT)),
            torch.zeros((2, 13, MECHANICS_IDENTITY_COUNT), dtype=torch.long),
        )

        self.assertEqual(v1_logits.shape, (2, 13))
        self.assertEqual(v2_logits.shape, (2, 13))
        self.assertFalse(any(key.startswith("identity_embeddings") for key in v1.state_dict()))
        self.assertTrue(any(key.startswith("identity_embeddings") for key in v2.state_dict()))


if __name__ == "__main__":
    unittest.main()
