from __future__ import annotations

import unittest

import torch

from pokemon_battler.baseline import FeatureCollator, HashedActionPolicy
from pokemon_battler.mechanics_baseline import MechanicsOnlyCollator, MechanicsOnlyPolicy
from tests.helpers import state


class BaselineTests(unittest.TestCase):
    def test_hashed_policy_scores_only_legal_actions(self) -> None:
        battle_state = state(forced_switch=True)
        batch = FeatureCollator(num_buckets=1024)(
            [
                {
                    "state": battle_state,
                    "action_id": 4,
                    "legal_action_ids": [4, 5],
                }
            ]
        )
        model = HashedActionPolicy(num_buckets=1024, embedding_dim=8, hidden_dim=16)
        logits = model(batch)
        self.assertEqual(logits.shape, (1, 13))
        self.assertEqual(
            torch.nonzero(torch.isfinite(logits[0]), as_tuple=False).flatten().tolist(),
            [4, 5],
        )
        loss = torch.nn.functional.cross_entropy(logits, batch["action_ids"])
        loss.backward()

    def test_mechanics_only_policy_scores_only_legal_actions(self) -> None:
        batch = MechanicsOnlyCollator()(
            [
                {
                    "state": state(forced_switch=True),
                    "action_id": 4,
                    "legal_action_ids": [4, 5],
                }
            ]
        )
        model = MechanicsOnlyPolicy(hidden_size=16)
        logits = model(batch)
        self.assertEqual(logits.shape, (1, 13))
        self.assertEqual(
            torch.nonzero(torch.isfinite(logits[0]), as_tuple=False).flatten().tolist(),
            [4, 5],
        )
        torch.nn.functional.cross_entropy(logits, batch["action_ids"]).backward()


if __name__ == "__main__":
    unittest.main()
