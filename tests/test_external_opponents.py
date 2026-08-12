from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pokemon_battler.external_opponents import (
    EXTERNAL_OPPONENTS,
    ExternalOpponentProcess,
)
from pokemon_battler.live_eval import build_parser


class ExternalOpponentTests(unittest.TestCase):
    def _manager(
        self,
        root: Path,
        opponent: str,
    ) -> ExternalOpponentProcess:
        team_file = root / "team.txt"
        team_file.write_text("Pikachu\n- Thunderbolt\n", encoding="utf-8")
        output_dir = root / "report"
        output_dir.mkdir()
        return ExternalOpponentProcess(
            opponent,
            opponents_dir=root / "opponents",
            output_dir=output_dir,
            team_file=team_file,
            battle_format="gen9ou",
            games=20,
            server_port=8000,
            bootstrap=False,
        )

    def test_cli_exposes_all_three_pinned_opponents(self) -> None:
        for opponent in EXTERNAL_OPPONENTS:
            with self.subTest(opponent=opponent):
                args = build_parser().parse_args(["--opponent", opponent])
                self.assertEqual(args.opponent, opponent)
                self.assertEqual(len(EXTERNAL_OPPONENTS[opponent].revision), 40)

    def test_only_foul_play_initiates_challenges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            one_step = self._manager(root, "pokechamp-one-step")
            self.assertFalse(one_step.challenges_player)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            foul_play = self._manager(root, "foul-play")
            self.assertTrue(foul_play.challenges_player)
            foul_play.start_challenges()
            self.assertEqual(foul_play.start_path.read_text(encoding="utf-8"), "PBPolicy\n")

    def test_foul_play_command_uses_isolated_environment_and_player_challenges(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manager = self._manager(root, "foul-play")
            checkout = root / "opponents" / "foul-play"
            (checkout / ".venv" / "bin").mkdir(parents=True)
            (checkout / ".venv" / "bin" / "python").touch()
            (checkout / ".venv" / ".pokemon-battler-dependencies-v1").write_text(
                "poke-engine==0.0.48\n",
                encoding="utf-8",
            )
            (checkout / "fp" / "teams" / "teams").mkdir(parents=True)
            manager.checkout = checkout

            command, _ = manager._foul_play_command()

            self.assertTrue(Path(command[0]).is_absolute())
            self.assertEqual(command[command.index("--bot-mode") + 1], "challenge_user")
            self.assertEqual(command[command.index("--user-to-challenge") + 1], "PBPolicy")
            self.assertIn("--start-file", command)
            self.assertEqual(
                Path(command[command.index("--teacher-trace") + 1]),
                manager.teacher_trace_path.resolve(),
            )
            copied_team = (
                checkout / "fp" / "teams" / "teams" / "pokemon-battler-opponent.txt"
            )
            self.assertEqual(copied_team.read_text(encoding="utf-8"), "Pikachu\n- Thunderbolt\n")

    def test_metadata_records_source_search_and_preview_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = self._manager(Path(temporary_directory), "foul-play")
            metadata = manager.metadata()
            self.assertEqual(metadata["search_time_ms"], 100)
            self.assertEqual(metadata["team_preview"], "published-search-preview")
            self.assertEqual(metadata["license"], "GPL-3.0")
            self.assertEqual(metadata["teacher_trace"], str(manager.teacher_trace_path))


if __name__ == "__main__":
    unittest.main()
