from __future__ import annotations

import tempfile
import unittest

from pokemon_battler.pipelines.experiment import _train_arguments, build_parser
from pokemon_battler.pipelines.interaction_experiment import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    build_parser as build_interaction_parser,
)


class ExperimentTests(unittest.TestCase):
    def test_recommended_run_is_one_full_mechanics_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = build_parser().parse_args(["--output-dir", temp_dir])
            train_args = _train_arguments(args)
        self.assertEqual(train_args.objective, "mechanics-head")
        self.assertEqual(train_args.prompt_format, "mechanics-v2")
        self.assertTrue(train_args.train_mechanics_cache.endswith("train.mechanics-v2.npy"))
        self.assertEqual(train_args.epochs, 1)
        self.assertIsNone(train_args.max_steps)
        self.assertIsNone(train_args.eval_batches)
        self.assertEqual(train_args.validation_sample_mode, "hash")
        self.assertEqual(train_args.min_lr_ratio, 0.05)
        self.assertEqual(train_args.attn_implementation, "sdpa")
        self.assertEqual(train_args.early_stopping_patience, 4)
        self.assertEqual(train_args.early_stopping_min_delta, 0.002)

    def test_interaction_run_owns_data_preparation_and_training_defaults(self) -> None:
        args = build_interaction_parser().parse_args([])
        self.assertEqual(args.data_dir, DEFAULT_DATA_DIR)
        self.assertEqual(args.output_dir, DEFAULT_OUTPUT_DIR)
        self.assertEqual(args.sample_rate, 0.02)
        self.assertEqual(args.interaction_d_model, 384)
        self.assertEqual(args.interaction_layers, 4)
        self.assertEqual(args.family_aux_weight, 0.25)
        self.assertEqual(args.value_loss_weight, 0.25)
        self.assertFalse(args.skip_overfit_gate)


if __name__ == "__main__":
    unittest.main()
