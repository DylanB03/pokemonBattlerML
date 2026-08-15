from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from pokemon_battler.frozen_cache import checkpoint_signature
from pokemon_battler.residual_cache import RESIDUAL_ARRAY_SPECS, RESIDUAL_CACHE_SCHEMA
from pokemon_battler.residual_modeling import (
    ResidualPolicyHead,
    load_champion_scorer,
    load_residual_head,
)
from pokemon_battler.residual_pipeline import build_parser
from pokemon_battler.residual_train import train_residual_policy
from pokemon_battler.trajectory_cache import ARRAY_SPECS, TRAJECTORY_CACHE_SCHEMA


class ResidualPolicyTests(unittest.TestCase):
    def test_zero_initialized_head_is_the_champion_policy(self) -> None:
        torch.manual_seed(3)
        head = ResidualPolicyHead(8, hidden_size=12, dropout=0.0)
        global_embedding = torch.randn(4, 8)
        candidates = torch.randn(4, 13, 8)
        legal = torch.rand(4, 13) > 0.3
        legal[:, 0] = True
        champion_logits = torch.randn(4, 13).masked_fill(~legal, float("-inf"))
        champion = torch.log_softmax(champion_logits, dim=1)
        output = head(global_embedding, candidates, legal, champion)
        torch.testing.assert_close(output["action_log_probs"], champion, atol=1e-6, rtol=0)
        self.assertEqual(int(torch.count_nonzero(output["logit_deltas"])), 0)

    def test_champion_scorer_loads_only_frozen_policy_projections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            state = {
                "candidate_scorer.weight": torch.randn(1, 8),
                "candidate_scorer.bias": torch.randn(1),
                "family_scorer.weight": torch.randn(3, 8),
                "family_scorer.bias": torch.randn(3),
                "unrelated.weight": torch.randn(2, 2),
            }
            save_file(state, checkpoint / "interaction_head.safetensors")
            scorer = load_champion_scorer(
                checkpoint, d_model=8, device=torch.device("cpu")
            )
            candidates = torch.randn(2, 13, 8)
            global_embedding = torch.randn(2, 8)
            legal = torch.ones(2, 13, dtype=torch.bool)
            output = scorer(global_embedding, candidates, legal)
            self.assertEqual(tuple(output.shape), (2, 13))
            torch.testing.assert_close(output.exp().sum(1), torch.ones(2))

    def test_pipeline_defaults_reuse_cache_and_require_two_battle_gates(self) -> None:
        args = build_parser().parse_args(["--output-dir", "outputs/test-residual"])
        self.assertEqual(args.teacher_train_rows, 8_000)
        self.assertEqual(args.replay_train_rows, 32_000)
        self.assertEqual(args.pilot_games, 50)
        self.assertEqual(args.final_games, 100)
        self.assertFalse(args.skip_battle_evaluation)

    def test_tiny_cached_training_writes_reloadable_residual(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            save_file(
                {
                    "candidate_scorer.weight": torch.zeros(1, 8),
                    "candidate_scorer.bias": torch.zeros(1),
                    "family_scorer.weight": torch.zeros(3, 8),
                    "family_scorer.bias": torch.zeros(3),
                },
                source / "interaction_head.safetensors",
            )
            signature = checkpoint_signature(source)

            def teacher_cache(path: Path, rows: int, seed: int) -> None:
                path.mkdir()
                rng = np.random.default_rng(seed)
                candidates = rng.normal(size=(rows, 13, 8)).astype(np.float16)
                targets = candidates[:, :, 0].argmax(1)
                teacher = np.full((rows, 13), 0.01 / 12, dtype=np.float32)
                teacher[np.arange(rows), targets] = 0.99
                arrays = {
                    "global": rng.normal(size=(rows, 8)).astype(np.float16),
                    "candidates": candidates,
                    "legal": np.ones((rows, 13), dtype=np.uint8),
                    "champion_log_probs": np.full(
                        (rows, 13), -np.log(13), dtype=np.float32
                    ),
                    "teacher_probabilities": teacher,
                    "teacher_actions": targets.astype(np.int8),
                    "teacher_confidence": np.full(rows, 0.99, dtype=np.float32),
                }
                for name, values in arrays.items():
                    np.save(path / RESIDUAL_ARRAY_SPECS[name][0], values)
                (path / "metadata.json").write_text(
                    json.dumps(
                        {
                            "schema": RESIDUAL_CACHE_SCHEMA,
                            "checkpoint_signature": signature,
                            "rows": rows,
                            "d_model": 8,
                        }
                    ),
                    encoding="utf-8",
                )

            def replay_cache(path: Path, rows: int, seed: int) -> None:
                path.mkdir()
                rng = np.random.default_rng(seed)
                arrays = {
                    "global": rng.normal(size=(rows, 8)).astype(np.float16),
                    "candidates": rng.normal(size=(rows, 13, 8)).astype(np.float16),
                    "legal": np.ones((rows, 13), dtype=np.uint8),
                    "actions": np.zeros(rows, dtype=np.int8),
                    "rewards": np.zeros(rows, dtype=np.float32),
                    "dones": np.ones(rows, dtype=np.uint8),
                    "transition_steps": np.ones(rows, dtype=np.int16),
                    "previous_actions": np.full(rows, 13, dtype=np.int8),
                    "previous_rewards": np.zeros(rows, dtype=np.float32),
                    "outcomes": np.ones(rows, dtype=np.int8),
                }
                for name, values in arrays.items():
                    np.save(path / ARRAY_SPECS[name][0], values)
                (path / "spans.json").write_text(
                    json.dumps(
                        [
                            {"trajectory_id": str(index), "start": index, "end": index + 1}
                            for index in range(rows)
                        ]
                    ),
                    encoding="utf-8",
                )
                (path / "metadata.json").write_text(
                    json.dumps(
                        {
                            "schema": TRAJECTORY_CACHE_SCHEMA,
                            "checkpoint_signature": signature,
                            "rows": rows,
                            "trajectories": rows,
                            "d_model": 8,
                            "reward_gamma": 0.99,
                        }
                    ),
                    encoding="utf-8",
                )

            teacher_train = root / "teacher-train"
            teacher_validation = root / "teacher-validation"
            replay_train = root / "replay-train"
            replay_validation = root / "replay-validation"
            teacher_cache(teacher_train, 96, 1)
            teacher_cache(teacher_validation, 48, 2)
            replay_cache(replay_train, 64, 3)
            replay_cache(replay_validation, 32, 4)
            output = root / "candidate"
            with redirect_stdout(io.StringIO()):
                report = train_residual_policy(
                    checkpoint=source,
                    teacher_train_cache=teacher_train,
                    teacher_validation_cache=teacher_validation,
                    replay_train_cache=replay_train,
                    replay_validation_cache=replay_validation,
                    output_dir=output,
                    epochs=3,
                    batch_size=32,
                    learning_rate=1e-3,
                    hidden_size=16,
                    dropout=0.0,
                    replay_train_rows=64,
                    replay_validation_rows=32,
                    early_stopping_patience=3,
                    minimum_teacher_kl_gain=-1.0,
                    minimum_teacher_agreement_gain=-1.0,
                    maximum_replay_kl=10.0,
                    maximum_replay_action_change=1.0,
                )
            self.assertLessEqual(
                abs(report["identity_max_log_probability_difference"]), 1e-6
            )
            self.assertTrue(report["offline_gate"]["passed"])
            loaded = load_residual_head(output, torch.device("cpu"))
            self.assertEqual(loaded.d_model, 8)


if __name__ == "__main__":
    unittest.main()
