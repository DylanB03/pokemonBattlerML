from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pokemon_battler.gated_pipeline import _cached_command, build_parser
from pokemon_battler.policy_ablation import _configuration_rows


class GatedPipelineTests(unittest.TestCase):
    def test_default_command_uses_completed_artifacts_and_one_output(self) -> None:
        args = build_parser().parse_args(["--output-dir", "outputs/test-gated"])
        self.assertEqual(
            args.checkpoint,
            Path("outputs/public-learning/positive-winrate-1000/batch-005/candidate"),
        )
        command = _cached_command(
            python="python",
            checkpoint=args.checkpoint,
            cache_dir=Path("cache"),
            output_dir=Path("candidate"),
            variant="policy-only",
            args=args,
            family_weight=0.0,
            action_value_weight=0.0,
            replay=True,
        )
        self.assertEqual(command.count("--seed"), 1)
        self.assertIn("--replay-validation-cache", command)

    def test_preview_ablations_exist_only_when_a_preview_head_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            self.assertEqual(len(_configuration_rows(checkpoint, 0.35)), 2)
            (checkpoint / "team_preview_head.safetensors").touch()
            rows = _configuration_rows(checkpoint, 0.35)
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[-1]["name"], "preview-q-blend")
