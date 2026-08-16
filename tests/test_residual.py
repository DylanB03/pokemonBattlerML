from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from pokemon_battler.frozen_cache import checkpoint_signature
from pokemon_battler.policy_suite import _should_promote
from pokemon_battler.residual_cache import RESIDUAL_ARRAY_SPECS, RESIDUAL_CACHE_SCHEMA
from pokemon_battler.residual_modeling import (
    ResidualPolicyHead,
    load_champion_scorer,
    load_residual_head,
)
from pokemon_battler.residual_pipeline import (
    _manifest_teams,
    _validate_source_and_replay_caches,
    build_parser,
    run as run_residual_pipeline,
)
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

    def test_final_gate_rejects_a_positive_delta_with_inconclusive_interval(self) -> None:
        self.assertFalse(
            _should_promote(
                0.04,
                [-0.01, 0.09],
                promotion_margin=0.0,
                minimum_interval_lower=0.0,
            )
        )
        self.assertFalse(
            _should_promote(
                0.04,
                [0.0, 0.08],
                promotion_margin=0.0,
                minimum_interval_lower=0.0,
            )
        )
        self.assertTrue(
            _should_promote(
                0.04,
                [0.01, 0.08],
                promotion_margin=0.0,
                minimum_interval_lower=0.0,
            )
        )

    def test_preflight_rejects_a_replay_cache_from_another_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "interaction_head.safetensors").write_bytes(b"head")
            for split in ("train", "validation"):
                cache = root / split
                cache.mkdir()
                (cache / "metadata.json").write_text(
                    json.dumps(
                        {
                            "schema": TRAJECTORY_CACHE_SCHEMA,
                            "checkpoint_signature": "wrong",
                            "rows": 10,
                            "d_model": 8,
                        }
                    ),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "not encoded by the selected"):
                _validate_source_and_replay_caches(
                    checkpoint, root / "train", root / "validation"
                )

    def test_preflight_rejects_silently_stacking_over_a_residual_champion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "interaction_head.safetensors").write_bytes(b"head")
            (checkpoint / "residual_head.safetensors").write_bytes(b"residual")
            (checkpoint / "residual_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "plain interaction champion"):
                _validate_source_and_replay_caches(
                    checkpoint, root / "unused-train", root / "unused-validation"
                )

    @staticmethod
    def _selected_teacher_references() -> list[SimpleNamespace]:
        teams = _manifest_teams(
            Path("examples/opponent-pools/gen9ou-foul-play.txt")
        )
        training = [
            team
            for team in teams
            if team.name not in {"gen9ou6.txt", "gen9ou7.txt", "gen9ou8.txt"}
        ]
        return [SimpleNamespace(team=str(team)) for team in training]

    def test_pipeline_promotes_only_after_both_gates_and_writes_absolute_pointer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            args = build_parser().parse_args(["--output-dir", str(output)])
            references = self._selected_teacher_references()
            with (
                patch(
                    "pokemon_battler.residual_pipeline.select_disjoint_teacher_rows",
                    return_value=(references, references, {"strategy": "test"}),
                ),
                patch("pokemon_battler.residual_pipeline._copy_references"),
                patch(
                    "pokemon_battler.residual_pipeline.build_residual_teacher_cache",
                    return_value={"rows": 1},
                ),
                patch(
                    "pokemon_battler.residual_pipeline.train_residual_policy",
                    return_value={"offline_gate": {"passed": True}},
                ),
                patch(
                    "pokemon_battler.residual_pipeline.run_policy_suite",
                    side_effect=[{"promoted": True}, {"promoted": True}],
                ) as suite,
                patch("pokemon_battler.residual_pipeline._release_memory"),
                redirect_stdout(io.StringIO()),
            ):
                report = run_residual_pipeline(args)
            candidate = (output / "03-residual-candidate").resolve()
            self.assertTrue(report["promoted"])
            self.assertEqual(report["selected_checkpoint"], str(candidate))
            self.assertEqual(
                (output / "selected_checkpoint.txt").read_text().strip(),
                str(candidate),
            )
            self.assertEqual(suite.call_count, 2)
            pilot_args = suite.call_args_list[0].args[0]
            final_args = suite.call_args_list[1].args[0]
            self.assertEqual(pilot_args.minimum_delta_interval_lower, -0.05)
            self.assertEqual(final_args.minimum_delta_interval_lower, 0.0)

    def test_pipeline_failure_restores_absolute_champion_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            args = build_parser().parse_args(["--output-dir", str(output)])
            references = self._selected_teacher_references()
            with (
                patch(
                    "pokemon_battler.residual_pipeline.select_disjoint_teacher_rows",
                    return_value=(references, references, {"strategy": "test"}),
                ),
                patch("pokemon_battler.residual_pipeline._copy_references"),
                patch(
                    "pokemon_battler.residual_pipeline.build_residual_teacher_cache",
                    return_value={"rows": 1},
                ),
                patch(
                    "pokemon_battler.residual_pipeline.train_residual_policy",
                    side_effect=RuntimeError("deliberate failure"),
                ),
                patch("pokemon_battler.residual_pipeline._release_memory"),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
                    run_residual_pipeline(args)
            champion = args.checkpoint.resolve()
            self.assertEqual(
                (output / "selected_checkpoint.txt").read_text().strip(),
                str(champion),
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["selected_checkpoint"], str(champion))

    def test_offline_or_pilot_rejection_never_selects_the_candidate(self) -> None:
        for offline_passed, suite_results, expected_decision in (
            (False, [], "offline_gate_failed"),
            (True, [{"promoted": False}], "heldout_pilot_failed"),
        ):
            with self.subTest(expected_decision=expected_decision):
                with tempfile.TemporaryDirectory() as temporary:
                    output = Path(temporary) / "run"
                    args = build_parser().parse_args(["--output-dir", str(output)])
                    references = self._selected_teacher_references()
                    with (
                        patch(
                            "pokemon_battler.residual_pipeline.select_disjoint_teacher_rows",
                            return_value=(
                                references,
                                references,
                                {"strategy": "test"},
                            ),
                        ),
                        patch("pokemon_battler.residual_pipeline._copy_references"),
                        patch(
                            "pokemon_battler.residual_pipeline.build_residual_teacher_cache",
                            return_value={"rows": 1},
                        ),
                        patch(
                            "pokemon_battler.residual_pipeline.train_residual_policy",
                            return_value={
                                "offline_gate": {"passed": offline_passed}
                            },
                        ),
                        patch(
                            "pokemon_battler.residual_pipeline.run_policy_suite",
                            side_effect=suite_results,
                        ) as suite,
                        patch("pokemon_battler.residual_pipeline._release_memory"),
                        redirect_stdout(io.StringIO()),
                    ):
                        report = run_residual_pipeline(args)
                    champion = args.checkpoint.resolve()
                    self.assertFalse(report["promoted"])
                    self.assertEqual(report["selected_checkpoint"], str(champion))
                    self.assertEqual(report["promotion_decision"], expected_decision)
                    self.assertEqual(suite.call_count, len(suite_results))
                    self.assertEqual(
                        (output / "selected_checkpoint.txt").read_text().strip(),
                        str(champion),
                    )

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
