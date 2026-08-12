from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from pokemon_battler.teacher_collect import (
    _enemy_schedule,
    _foul_play_winners,
    _latest_foul_play_record,
    _validate_showdown_teams,
    build_parser,
    run,
)
from tests.test_team_pool import TEAM_ONE


class TeacherCollectTests(unittest.TestCase):
    def test_cli_separates_fixed_teacher_team_from_enemy_pool(self) -> None:
        args = build_parser().parse_args(
            [
                "--team-file",
                "fixed.txt",
                "--enemy-team-file",
                "enemy-one.txt",
                "--enemy-team-file",
                "enemy-two.txt",
                "--output-dir",
                "teacher-output",
            ]
        )
        self.assertEqual(args.team_file, Path("fixed.txt"))
        self.assertEqual(
            args.enemy_team_file,
            [Path("enemy-one.txt"), Path("enemy-two.txt")],
        )
        self.assertEqual(args.enemy_policy, "foul-play")

    def test_smart_enemy_schedule_randomizes_without_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = []
            for name, team in (
                ("one.txt", TEAM_ONE),
                ("two.txt", TEAM_ONE.replace("Pikachu", "Raichu")),
                ("three.txt", TEAM_ONE.replace("Pikachu", "Zapdos")),
            ):
                path = root / name
                path.write_text(team, encoding="utf-8")
                paths.append(path)

            schedule, report = _enemy_schedule(paths, games=7, seed=9)

            self.assertEqual(len(schedule), 7)
            self.assertEqual(len(set(schedule[:3])), 3)
            self.assertEqual(len(set(schedule[3:6])), 3)
            self.assertEqual(len(report["selections"]), 7)

    def test_foul_play_progress_reads_completed_winners(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            log = Path(temporary_directory) / "opponent.log"
            log.write_text(
                "INFO     Winner: PBFoulPlay\n"
                "INFO     W: 1 L: 0\n"
                "INFO     Winner: PBFoulPlayEnemy\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _foul_play_winners(log),
                ["PBFoulPlay", "PBFoulPlayEnemy"],
            )
            with log.open("a", encoding="utf-8") as stream:
                stream.write("INFO     W: 1\tL: 1\n")
            self.assertEqual(_latest_foul_play_record(log), (1, 1))

    def test_legality_preflight_reports_rejected_team_before_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable = root / "pokemon-showdown"
            executable.touch()
            team = root / "obsolete.txt"
            team.write_text(TEAM_ONE, encoding="utf-8")
            with (
                patch("pokemon_battler.teacher_collect.shutil.which", return_value="node"),
                patch(
                    "pokemon_battler.teacher_collect.subprocess.run",
                    return_value=CompletedProcess(
                        args=[],
                        returncode=1,
                        stdout="",
                        stderr="Tera Blast is banned.",
                    ),
                ),
                self.assertRaisesRegex(ValueError, "Tera Blast is banned"),
            ):
                _validate_showdown_teams(root, "gen9ou", [team])

    def test_collection_refuses_a_single_enemy_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixed = root / "fixed.txt"
            enemy = root / "enemy.txt"
            fixed.write_text(TEAM_ONE, encoding="utf-8")
            enemy.write_text(TEAM_ONE.replace("Pikachu", "Pichu"), encoding="utf-8")
            output = root / "output"
            args = build_parser().parse_args(
                [
                    "--team-file",
                    str(fixed),
                    "--enemy-team-file",
                    str(enemy),
                    "--output-dir",
                    str(output),
                ]
            )
            with self.assertRaisesRegex(ValueError, "at least 2 distinct team compositions"):
                run(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
