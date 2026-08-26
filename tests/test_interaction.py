from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import torch

from pokemon_battler.data.interaction_cache import (
    InteractionCacheDataset,
    build_interaction_cache,
    interaction_cache_is_current,
)
from pokemon_battler.models.interaction_features import build_interaction_features
from pokemon_battler.models.interaction_modeling import (
    InteractionPolicyHead,
    interaction_policy_loss,
)
from pokemon_battler.data.prepare import SplitConfig, prepare_dataset
from pokemon_battler.data.training_data import (
    InteractionCollator,
    JsonlOffsetDataset,
)
from tests.helpers import state, terminal_state


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, value: str, add_special_tokens: bool = True) -> list[int]:
        prefix = [2] if add_special_tokens else []
        return prefix + [3 + ord(character) for character in value]


def prepared_dataset(root: Path) -> Path:
    raw = root / "raw"
    raw.mkdir()
    first = state()
    second = state()
    second["player_active_pokemon"]["hp_pct"] = 0.5
    second["player_prev_move"]["name"] = "thunderbolt"
    source = raw / "battle-1_1800_a_vs_b_01-02-2025_WIN.json"
    source.write_text(
        json.dumps(
            {
                "states": [first, second, terminal_state()],
                "actions": [0, 1, -1],
            }
        ),
        encoding="utf-8",
    )
    data_dir = root / "data"
    prepare_dataset(
        [raw],
        data_dir,
        split_config=SplitConfig(
            mode="chronological",
            validation_start=date(2025, 2, 1),
            test_start=date(2025, 3, 1),
        ),
    )
    return data_dir / "train.jsonl"


class InteractionTests(unittest.TestCase):
    def test_features_and_cache_have_fixed_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = prepared_dataset(Path(temp_dir))
            dataset = JsonlOffsetDataset(data_file)
            features = build_interaction_features(dataset[1])
            self.assertEqual((len(features["global_numeric"]),), (30,))
            self.assertEqual(
                (len(features["pokemon_numeric"]), len(features["pokemon_numeric"][0])),
                (12, 50),
            )
            self.assertEqual(
                (
                    len(features["candidate_numeric"]),
                    len(features["candidate_numeric"][0]),
                ),
                (13, 207),
            )
            self.assertEqual(
                (len(features["history_numeric"]), len(features["history_numeric"][0])),
                (4, 12),
            )
            metadata = build_interaction_cache(data_file)
            self.assertEqual(metadata["prepared_schema_version"], 3)
            cached = InteractionCacheDataset(
                dataset,
                data_file.with_name("train.interaction-v1"),
            )
            self.assertIn("_interaction_features", cached[0])

    def test_interaction_collator_requires_schema_three(self) -> None:
        collator = InteractionCollator(FakeTokenizer(), max_length=20_000)
        with self.assertRaisesRegex(ValueError, "prepared schema 3"):
            collator(
                [
                    {
                        "schema_version": 2,
                        "state": state(),
                        "action_id": 0,
                        "legal_action_ids": [0],
                    }
                ]
            )

    def test_value_weight_reduces_repeated_long_battle_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = prepared_dataset(Path(temp_dir))
            row = JsonlOffsetDataset(data_file)[0]
            collator = InteractionCollator(FakeTokenizer(), max_length=20_000)
            short = dict(row, battle_decision_count=8)
            long = dict(row, battle_decision_count=128)
            weights = collator([short, long])["value_weights"]
            self.assertTrue(torch.allclose(weights, torch.tensor([4.0, 0.25])))

    def test_cache_rejects_a_feature_schema_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = prepared_dataset(Path(temp_dir))
            cache_dir = data_file.with_name("train.interaction-v1")
            build_interaction_cache(data_file, cache_dir)
            metadata_path = cache_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["schema_fingerprint"] = "stale"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertFalse(interaction_cache_is_current(data_file, cache_dir))
            with self.assertRaisesRegex(ValueError, "feature schema"):
                InteractionCacheDataset(JsonlOffsetDataset(data_file), cache_dir)

    def test_head_normalizes_only_legal_actions_and_backpropagates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = prepared_dataset(Path(temp_dir))
            dataset = JsonlOffsetDataset(data_file)
            cache_dir = data_file.with_name("train.interaction-v1")
            build_interaction_cache(data_file, cache_dir)
            cached = InteractionCacheDataset(dataset, cache_dir)
            batch = InteractionCollator(FakeTokenizer(), max_length=20_000)(
                [cached[0], cached[1]]
            )
            head = InteractionPolicyHead(
                16,
                d_model=32,
                attention_heads=4,
                layers=1,
                feedforward_size=64,
                dropout=0.0,
                qwen_mode="none",
            )
            outputs = head(torch.zeros((2, 16)), batch)
            self.assertEqual(tuple(outputs["action_value_logits"].shape), (2, 13))
            self.assertTrue(
                torch.isneginf(
                    outputs["action_value_logits"].masked_select(
                        ~batch["legal_action_mask"]
                    )
                ).all()
            )
            probabilities = outputs["action_log_probs"].exp()
            self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(2)))
            self.assertTrue(
                torch.isneginf(
                    outputs["action_log_probs"].masked_select(~batch["legal_action_mask"])
                ).all()
            )
            loss, parts = interaction_policy_loss(
                outputs,
                batch,
                family_weights=torch.ones(3),
            )
            self.assertEqual(set(parts), {"policy_loss", "family_aux_loss", "value_loss"})
            loss.backward()
            self.assertIsNotNone(head.candidate_scorer.weight.grad)

    def test_roster_order_is_not_a_position_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = prepared_dataset(Path(temp_dir))
            dataset = JsonlOffsetDataset(data_file)
            batch = InteractionCollator(FakeTokenizer(), max_length=20_000)(
                [dataset[0]]
            )
            head = InteractionPolicyHead(
                16,
                d_model=32,
                attention_heads=4,
                layers=1,
                feedforward_size=64,
                dropout=0.0,
                qwen_mode="none",
            ).eval()
            with torch.inference_mode():
                original = head(torch.zeros((1, 16)), batch)["action_log_probs"]
                permutation = torch.tensor([2, 0, 1, 3, 4, 5])
                permuted = {key: value.clone() for key, value in batch.items()}
                for key in (
                    "interaction_pokemon_numeric",
                    "interaction_pokemon_ids",
                    "interaction_pokemon_mask",
                ):
                    permuted[key][:, :6] = batch[key][:, permutation]
                inverse = torch.empty_like(permutation)
                inverse[permutation] = torch.arange(6)
                slots = batch["interaction_candidate_actor_slot"]
                valid = slots >= 0
                permuted["interaction_candidate_actor_slot"][valid] = inverse[
                    slots[valid]
                ]
                reordered = head(torch.zeros((1, 16)), permuted)["action_log_probs"]
            self.assertTrue(torch.allclose(original, reordered, atol=1e-5, rtol=1e-5))

    def test_masked_structured_padding_cannot_change_a_legal_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = prepared_dataset(Path(temp_dir))
            batch = InteractionCollator(FakeTokenizer(), max_length=20_000)(
                [JsonlOffsetDataset(data_file)[0]]
            )
            head = InteractionPolicyHead(
                16,
                d_model=32,
                attention_heads=4,
                layers=1,
                feedforward_size=64,
                dropout=0.0,
                qwen_mode="none",
            ).eval()
            changed = {key: value.clone() for key, value in batch.items()}
            pokemon_padding = ~changed["interaction_pokemon_mask"]
            history_padding = ~changed["interaction_history_mask"]
            changed["interaction_pokemon_numeric"][pokemon_padding] = 1000
            changed["interaction_pokemon_ids"][pokemon_padding] = 1
            changed["interaction_history_numeric"][history_padding] = 1000
            changed["interaction_history_ids"][history_padding] = 1
            with torch.inference_mode():
                original = head(torch.zeros((1, 16)), batch)["action_log_probs"]
                padded = head(torch.zeros((1, 16)), changed)["action_log_probs"]
            self.assertTrue(torch.allclose(original, padded, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
