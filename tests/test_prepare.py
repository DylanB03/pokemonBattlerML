from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from datetime import date
from pathlib import Path

from pokemon_battler.prepare import (
    SplitConfig,
    choose_split,
    iter_trajectories,
    parse_replay_metadata,
    prepare_dataset,
)
from tests.helpers import state, terminal_state


class PrepareTests(unittest.TestCase):
    def test_reads_lz4_and_tar_inputs(self) -> None:
        import lz4.frame

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = json.dumps({"states": [], "actions": []}).encode("utf-8")

            lz4_path = root / "trajectory.json.lz4"
            lz4_path.write_bytes(lz4.frame.compress(payload))
            lz4_rows = list(iter_trajectories([lz4_path]))
            self.assertEqual(lz4_rows[0][1]["actions"], [])

            tar_path = root / "gen9ou.tar"
            with tarfile.open(tar_path, "w") as archive:
                info = tarfile.TarInfo("gen9ou/trajectory.json")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            tar_rows = list(iter_trajectories([tar_path]))
            self.assertEqual(tar_rows[0][1]["states"], [])

    def test_filename_metadata(self) -> None:
        metadata = parse_replay_metadata(
            "gen9ou.tar:gen9ou/[gen9ou]battle-123_1720_foo_vs_bar_01-02-2025_WIN.json.lz4"
        )
        self.assertEqual(metadata.battle_id, "[gen9ou]battle-123")
        self.assertEqual(metadata.rating, 1720)
        self.assertEqual(metadata.battle_date, date(2025, 1, 2))
        self.assertEqual(metadata.outcome, "WIN")

    def test_hash_split_is_grouped_by_battle(self) -> None:
        config = SplitConfig(seed=7)
        first = choose_split("same-battle", date(2025, 1, 1), config)
        second = choose_split("same-battle", date(2026, 1, 1), config)
        self.assertEqual(first, second)

    def test_chronological_split(self) -> None:
        config = SplitConfig(
            mode="chronological",
            validation_start=date(2025, 1, 1),
            test_start=date(2025, 2, 1),
        )
        self.assertEqual(choose_split("a", date(2024, 12, 31), config), "train")
        self.assertEqual(choose_split("b", date(2025, 1, 15), config), "validation")
        self.assertEqual(choose_split("c", date(2025, 2, 1), config), "test")

    def test_prepare_skips_missing_and_illegal_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            trajectory = {
                "states": [state(), state(), state(), terminal_state()],
                # Valid, missing, illegal for this state, then terminal placeholder.
                "actions": [0, -1, 8, -1],
            }
            source = (
                raw_dir
                / "[gen9ou]battle-123_1720_foo_vs_bar_01-02-2025_WIN.json"
            )
            source.write_text(json.dumps(trajectory), encoding="utf-8")

            report = prepare_dataset(
                [raw_dir],
                root / "prepared",
                split_config=SplitConfig(
                    mode="chronological",
                    validation_start=date(2025, 2, 1),
                    test_start=date(2025, 3, 1),
                ),
                min_rating=1600,
            )
            self.assertEqual(report["examples_per_split"]["train"], 1)
            self.assertEqual(report["counters"]["missing_actions"], 1)
            self.assertEqual(report["counters"]["actions_not_recoverably_legal"], 1)

            rows = [
                json.loads(line)
                for line in (root / "prepared" / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(rows[0]["target"], "A0")
            self.assertIn(0, rows[0]["legal_action_ids"])
            self.assertEqual(rows[0]["state"]["turn_index"], 0)
            self.assertEqual(rows[0]["state"]["player_remaining"], 3)
            self.assertEqual(rows[0]["legal_mask_quality"], "recoverable")
            self.assertEqual(
                rows[0]["state"]["recent_move_history"],
                [{"player": "protect", "opponent": "saltcure"}],
            )

    def test_preparation_history_does_not_leak_future_reveals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            first = state()
            second = state()
            second["opponent_active_pokemon"]["name"] = "dragapult"
            second["opponent_active_pokemon"]["base_species"] = "dragapult"
            trajectory = {
                "states": [first, second, terminal_state()],
                "actions": [0, 0, -1],
            }
            source = raw_dir / "battle-9_1800_a_vs_b_01-02-2025_WIN.json"
            source.write_text(json.dumps(trajectory), encoding="utf-8")
            prepare_dataset(
                [raw_dir],
                root / "prepared",
                split_config=SplitConfig(
                    mode="chronological",
                    validation_start=date(2025, 2, 1),
                    test_start=date(2025, 3, 1),
                ),
            )
            rows = [
                json.loads(line)
                for line in (root / "prepared" / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            first_names = {
                pokemon["name"]
                for pokemon in rows[0]["state"]["opponent_revealed_pokemon"]
            }
            second_names = {
                pokemon["name"]
                for pokemon in rows[1]["state"]["opponent_revealed_pokemon"]
            }
            self.assertNotIn("dragapult", first_names)
            self.assertIn("dragapult", second_names)


if __name__ == "__main__":
    unittest.main()
