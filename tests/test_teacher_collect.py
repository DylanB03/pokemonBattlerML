from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pokemon_battler.teacher_collect import build_parser, run
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
