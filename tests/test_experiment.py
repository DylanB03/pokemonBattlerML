from __future__ import annotations

import tempfile
import unittest

from pokemon_battler.experiment import _train_arguments, build_parser


class ExperimentTests(unittest.TestCase):
    def test_recommended_run_is_one_full_mechanics_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = build_parser().parse_args(["--output-dir", temp_dir])
            train_args = _train_arguments(args)
        self.assertEqual(train_args.objective, "mechanics-head")
        self.assertEqual(train_args.prompt_format, "mechanics-v1")
        self.assertTrue(train_args.train_mechanics_cache.endswith("train.mechanics-v1.npy"))
        self.assertEqual(train_args.epochs, 1)
        self.assertIsNone(train_args.max_steps)
        self.assertIsNone(train_args.eval_batches)
        self.assertEqual(train_args.validation_sample_mode, "hash")
        self.assertEqual(train_args.min_lr_ratio, 0.05)
        self.assertEqual(train_args.attn_implementation, "sdpa")
        self.assertEqual(train_args.early_stopping_patience, 4)
        self.assertEqual(train_args.early_stopping_min_delta, 0.002)


if __name__ == "__main__":
    unittest.main()
