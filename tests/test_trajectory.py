from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pokemon_battler.data.frozen_cache import checkpoint_signature
from pokemon_battler.models.interaction_features import validate_interaction_row
from pokemon_battler.data.prepare import ReplayMetadata
from pokemon_battler.data.trajectory_cache import ARRAY_SPECS, TRAJECTORY_CACHE_SCHEMA
from pokemon_battler.models.trajectory_modeling import (
    TrajectoryPolicyHead,
    load_trajectory_head,
)
from pokemon_battler.data.trajectory_prepare import (
    PREVIOUS_ACTION_SENTINEL,
    trajectory_rows,
)
from pokemon_battler.pipelines.trajectory_pipeline import build_parser as build_pipeline_parser
from pokemon_battler.training.trajectory_train import (
    SequenceWindowDataset,
    collate_sequence_windows,
    trajectory_iql_loss,
    train_trajectory_policy,
)
from tests.helpers import state, terminal_state


class TrajectoryTests(unittest.TestCase):
    def test_pipeline_defaults_to_whole_trajectory_ablation_and_battle_gate(self) -> None:
        args = build_pipeline_parser().parse_args(["--output-dir", "outputs/test"])
        self.assertEqual(args.trajectory_sample_rate, 0.02)
        self.assertEqual(args.epochs, 8)
        self.assertEqual(args.games, 100)
        self.assertFalse(args.skip_battle_evaluation)

    def test_schema_four_keeps_consecutive_valid_decisions_and_terminal_reward(self) -> None:
        first = state()
        missing = deepcopy(first)
        missing["opponent_active_pokemon"]["hp_pct"] = 0.55
        second = deepcopy(missing)
        second["player_active_pokemon"]["hp_pct"] = 0.60
        terminal = terminal_state()
        terminal["opponent_active_pokemon"]["hp_pct"] = 0.0
        metadata = ReplayMetadata("battle-1", 1800, date(2025, 1, 1), "WIN")
        rows = trajectory_rows(
            "one-pov.json",
            {"states": [first, missing, second, terminal], "actions": [0, -1, 1, -1]},
            metadata,
            "train",
            Counter(),
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["trajectory_position"] for row in rows], [0, 1])
        self.assertEqual(rows[0]["transition_steps"], 2)
        self.assertEqual(rows[0]["next_turn_index"], 2)
        self.assertFalse(rows[0]["done"])
        self.assertTrue(rows[1]["done"])
        self.assertGreater(rows[1]["reward"], 1.0)
        self.assertEqual(rows[0]["previous_action_id"], PREVIOUS_ACTION_SENTINEL)
        self.assertEqual(rows[1]["previous_action_id"], 0)
        self.assertAlmostEqual(rows[1]["previous_reward"], rows[0]["reward"])
        validate_interaction_row(rows[0])

    def test_pov_identifier_does_not_merge_two_sides_of_one_battle(self) -> None:
        metadata = ReplayMetadata("same-battle", 1800, None, "WIN")
        trajectory = {"states": [state(), terminal_state()], "actions": [0, -1]}
        first = trajectory_rows("p1.json", trajectory, metadata, "train", Counter())
        second = trajectory_rows("p2.json", trajectory, metadata, "train", Counter())
        self.assertNotEqual(first[0]["trajectory_id"], second[0]["trajectory_id"])

    def test_sequence_windows_include_burn_in_and_one_step_lookahead(self) -> None:
        rows = 6
        cache = SimpleNamespace(
            spans=[{"trajectory_id": "a", "start": 0, "end": rows}],
            arrays={
                "global": np.zeros((rows, 8), dtype=np.float16),
                "candidates": np.zeros((rows, 13, 8), dtype=np.float16),
                "legal": np.ones((rows, 13), dtype=np.uint8),
                "actions": np.zeros(rows, dtype=np.int8),
                "rewards": np.zeros(rows, dtype=np.float32),
                "dones": np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8),
                "transition_steps": np.ones(rows, dtype=np.int16),
                "previous_actions": np.zeros(rows, dtype=np.int8),
                "previous_rewards": np.zeros(rows, dtype=np.float32),
                "outcomes": np.ones(rows, dtype=np.int8),
            },
        )
        dataset = SequenceWindowDataset(cache, sequence_length=3, burn_in=1)
        self.assertEqual(len(dataset), 2)
        first, second = dataset[0], dataset[1]
        self.assertEqual(len(first["actions"]), 4)
        self.assertEqual(first["loss_mask"].tolist(), [True, True, True, False])
        self.assertEqual(second["loss_mask"].tolist(), [False, True, True, True])
        batch = collate_sequence_windows([first, second])
        self.assertEqual(tuple(batch["candidates"].shape), (2, 4, 13, 8))

    def test_memoryless_and_recurrent_heads_train_against_next_state_targets(self) -> None:
        batch_size, turns, d_model = 2, 4, 16
        batch = {
            "global": torch.randn(batch_size, turns, d_model),
            "candidates": torch.randn(batch_size, turns, 13, d_model),
            "legal": torch.ones(batch_size, turns, 13, dtype=torch.bool),
            "actions": torch.randint(0, 13, (batch_size, turns)),
            "rewards": torch.zeros(batch_size, turns),
            "dones": torch.tensor([[False, False, False, True]] * batch_size),
            "transition_steps": torch.ones(batch_size, turns),
            "previous_actions": torch.randint(0, 14, (batch_size, turns)),
            "previous_rewards": torch.zeros(batch_size, turns),
            "loss_mask": torch.ones(batch_size, turns, dtype=torch.bool),
            "has_next": torch.tensor([[True, True, True, False]] * batch_size),
        }
        for memory_type in ("none", "gru"):
            model = TrajectoryPolicyHead(
                d_model,
                memory_type=memory_type,
                hidden_size=24,
                recurrent_layers=2,
                dropout=0.0,
            )
            outputs = model(
                batch["global"],
                batch["candidates"],
                batch["legal"],
                batch["previous_actions"],
                batch["previous_rewards"],
            )
            loss, parts = trajectory_iql_loss(
                outputs,
                outputs,
                batch,
                gamma=0.99,
                expectile=0.7,
                advantage_temperature=0.2,
                maximum_advantage_weight=20.0,
                behavior_clone_weight=0.1,
                behavior_clone_only=False,
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(parts["q_loss"]))
            loss.backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_tiny_cached_training_writes_a_reloadable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            signature = checkpoint_signature(source)

            def write_cache(path: Path, trajectories: int) -> None:
                path.mkdir()
                length = trajectories * 3
                rng = np.random.default_rng(7)
                arrays = {
                    "global": rng.normal(size=(length, 8)).astype(np.float16),
                    "candidates": rng.normal(size=(length, 13, 8)).astype(np.float16),
                    "legal": np.ones((length, 13), dtype=np.uint8),
                    "actions": np.arange(length, dtype=np.int8) % 13,
                    "rewards": np.tile([0.01, 0.02, 1.0], trajectories).astype(
                        np.float32
                    ),
                    "dones": np.tile([0, 0, 1], trajectories).astype(np.uint8),
                    "transition_steps": np.ones(length, dtype=np.int16),
                    "previous_actions": np.tile(
                        [PREVIOUS_ACTION_SENTINEL, 0, 1], trajectories
                    ).astype(np.int8),
                    "previous_rewards": np.tile([0.0, 0.01, 0.02], trajectories).astype(
                        np.float32
                    ),
                    "outcomes": np.ones(length, dtype=np.int8),
                }
                for name, array in arrays.items():
                    np.save(path / ARRAY_SPECS[name][0], array)
                spans = [
                    {"trajectory_id": str(index), "start": index * 3, "end": index * 3 + 3}
                    for index in range(trajectories)
                ]
                (path / "spans.json").write_text(json.dumps(spans), encoding="utf-8")
                (path / "metadata.json").write_text(
                    json.dumps(
                        {
                            "schema": TRAJECTORY_CACHE_SCHEMA,
                            "checkpoint_signature": signature,
                            "rows": length,
                            "trajectories": trajectories,
                            "d_model": 8,
                            "reward_gamma": 0.99,
                        }
                    ),
                    encoding="utf-8",
                )

            train_cache = root / "train-cache"
            validation_cache = root / "validation-cache"
            write_cache(train_cache, 3)
            write_cache(validation_cache, 2)
            output = root / "trained"
            with redirect_stdout(io.StringIO()):
                report = train_trajectory_policy(
                    source_checkpoint=source,
                    train_cache=train_cache,
                    validation_cache=validation_cache,
                    output_dir=output,
                    memory_type="gru",
                    epochs=2,
                    behavior_clone_epochs=1,
                    sequence_length=3,
                    burn_in=1,
                    batch_size=2,
                    hidden_size=12,
                    recurrent_layers=1,
                    dropout=0.0,
                    log_steps=100,
                )
            self.assertEqual(report["updates"], 4)
            loaded = load_trajectory_head(output, torch.device("cpu"))
            self.assertEqual(loaded.hidden_size, 12)


if __name__ == "__main__":
    unittest.main()
