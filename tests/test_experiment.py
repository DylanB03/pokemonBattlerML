from __future__ import annotations

import tempfile
import unittest

from pokemon_battler.experiment import _train_arguments, build_parser


class ExperimentTests(unittest.TestCase):
    def test_recommended_run_is_one_full_candidate_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = build_parser().parse_args(["--output-dir", temp_dir])
            train_args = _train_arguments(args)
        self.assertEqual(train_args.objective, "candidate-head")
        self.assertEqual(train_args.prompt_format, "compact-v1")
        self.assertEqual(train_args.epochs, 1)
        self.assertIsNone(train_args.max_steps)
        self.assertIsNone(train_args.eval_batches)
        self.assertEqual(train_args.validation_sample_mode, "hash")
        self.assertEqual(train_args.min_lr_ratio, 0.05)
        self.assertEqual(train_args.attn_implementation, "sdpa")


if __name__ == "__main__":
    unittest.main()
