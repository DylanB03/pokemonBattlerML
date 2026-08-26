from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from pokemon_battler.showdown.foul_play_worker import _fallback_decision
from pokemon_battler.showdown.parallel_teacher_collect import _merge_traces, _worker_command
from pokemon_battler.evaluation.policy_suite import _checkpoint_preview_enabled, _indexed_results
from pokemon_battler.evaluation.structured_blend_sweep import _normalized_weights, _weight_slug
from pokemon_battler.models.structured_modeling import set_structured_blend_weight


class ParallelPipelineTests(unittest.TestCase):
    def test_foul_play_search_failure_uses_a_legal_request_fallback(self) -> None:
        request = {
            "active": [
                {
                    "moves": [
                        {"id": "recover", "pp": 0, "disabled": False},
                        {"id": "thunderbolt", "pp": 12, "disabled": False},
                    ]
                }
            ],
            "side": {
                "pokemon": [
                    {"active": True, "condition": "100/100"},
                    {"active": False, "condition": "100/100"},
                ]
            },
        }
        self.assertEqual(
            _fallback_decision(request, 7), ["/choose move thunderbolt", "7"]
        )
        request["forceSwitch"] = [True]
        self.assertEqual(_fallback_decision(request, 8), ["/switch 2", "8"])
        self.assertEqual(
            _fallback_decision({}, 9, team_preview=True), ["/switch 1", "9"]
        )

    def test_teacher_workers_get_unique_ports_users_and_shared_advisor(self) -> None:
        args = Namespace(
            team_file=Path("fixed.txt"),
            enemy_policy="foul-play",
            seed=42,
            battle_format="gen9ou",
            showdown_dir=Path("showdown"),
            server_port=8000,
            server_startup_timeout=60,
            opponents_dir=Path("opponents"),
            opponent_startup_timeout=90,
            foul_play_search_time_ms=250,
            foul_play_parallelism=1,
            foul_play_search_threads=1,
            progress_interval=2,
            battle_stall_timeout=600,
            student_action_probability=0.7,
            enemy_team_file=[Path("one.txt"), Path("two.txt")],
            enemy_team_dir=None,
            enemy_foul_play_search_time_ms=250,
            no_bootstrap_server=False,
            no_bootstrap_opponents=False,
        )
        command = _worker_command(
            args,
            worker=3,
            games=25,
            output_dir=Path("worker-03"),
            advisor_url="http://127.0.0.1:8765/predict",
        )
        self.assertIn("8003", command)
        self.assertIn("PBFoulTeachW03", command)
        self.assertIn("PBFoulEnemyW03", command)
        self.assertIn("http://127.0.0.1:8765/predict", command)
        self.assertNotIn("--student-checkpoint", command)

    def test_trace_merge_prefixes_otherwise_colliding_local_battle_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            shards = []
            for worker in range(2):
                directory = root / f"worker-{worker:02d}"
                directory.mkdir()
                (directory / "foul_play_teacher.jsonl").write_text(
                    json.dumps(
                        {
                            "battle_id": "battle-gen9ou-1",
                            "decision_phase": "team_preview" if worker == 0 else "turn",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                shards.append(directory)
            output = root / "merged.jsonl"
            counts = _merge_traces(shards, output)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(counts, {"rows": 2, "turn_rows": 1, "preview_rows": 1})
            self.assertEqual(
                [row["battle_id"] for row in rows],
                ["worker-00:battle-gen9ou-1", "worker-01:battle-gen9ou-1"],
            )

    def test_promotion_alignment_uses_worker_game_and_team(self) -> None:
        summary = {
            "battles": [
                {
                    "opponent": "PBFoulEvalW01",
                    "opponent_game_index": 2,
                    "scheduled_enemy_team_file": "team-b.txt",
                    "won": True,
                }
            ]
        }
        self.assertEqual(
            _indexed_results(summary),
            {("pbfoulevalw01", 2, "team-b.txt"): 1},
        )

    def test_policy_suite_disables_missing_preview_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory)
            self.assertFalse(_checkpoint_preview_enabled(checkpoint, True))
            (checkpoint / "team_preview_head.safetensors").touch()
            self.assertTrue(_checkpoint_preview_enabled(checkpoint, True))
            self.assertFalse(_checkpoint_preview_enabled(checkpoint, False))

    def test_structured_blend_sweep_normalizes_and_persists_selection(self) -> None:
        self.assertEqual(_normalized_weights([1.0, 0.5, 0.5]), [0.5, 1.0])
        self.assertEqual(_weight_slug(0.25), "0p25")
        with self.assertRaises(ValueError):
            _normalized_weights([-0.1])
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory)
            (checkpoint / "training_config.json").write_text(
                json.dumps({"structured_policy_blend_weight": 0.5}), encoding="utf-8"
            )
            set_structured_blend_weight(checkpoint, 0.75)
            metadata = json.loads(
                (checkpoint / "training_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["structured_policy_blend_weight"], 0.75)
            self.assertEqual(metadata["selected_structured_blend_weight"], 0.75)


if __name__ == "__main__":
    unittest.main()
