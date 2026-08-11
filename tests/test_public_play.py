from __future__ import annotations

import asyncio
import contextlib
import io
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pokemon_battler.public_play import (
    OPPONENT_KEY,
    PASSWORD_KEY,
    USERNAME_KEY,
    _public_summary,
    _run_matchmaking,
    _validate_args,
    build_parser,
    load_public_environment,
    resolve_checkpoint,
    run,
)


class PublicPlayTests(unittest.TestCase):
    def test_env_file_loads_credentials_and_optional_opponent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / ".env"
            env_file.write_text(
                "\n".join(
                    (
                        f'{USERNAME_KEY}="Public Policy Bot"',
                        f'{PASSWORD_KEY}="secret # value"',
                        f"{OPPONENT_KEY}=HumanTester",
                    )
                ),
                encoding="utf-8",
            )
            loaded = load_public_environment(env_file, environ={})
            self.assertEqual(loaded.account.username, "Public Policy Bot")
            self.assertEqual(loaded.account.password, "secret # value")
            self.assertEqual(loaded.opponent, "HumanTester")

    def test_process_environment_overrides_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / ".env"
            env_file.write_text(
                f"{USERNAME_KEY}=FileBot\n{PASSWORD_KEY}=file-secret\n",
                encoding="utf-8",
            )
            loaded = load_public_environment(
                env_file,
                environ={USERNAME_KEY: "ProcessBot", PASSWORD_KEY: "process-secret"},
            )
            self.assertEqual(loaded.account.username, "ProcessBot")
            self.assertEqual(loaded.account.password, "process-secret")

    def test_missing_env_values_fail_without_exposing_a_password(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / ".env"
            env_file.write_text(f"{USERNAME_KEY}=Bot\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, PASSWORD_KEY):
                load_public_environment(env_file, environ={})

    def test_selected_checkpoint_pointer_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "candidate"
            checkpoint.mkdir()
            pointer = root / "selected_checkpoint.txt"
            pointer.write_text(str(checkpoint) + "\n", encoding="utf-8")
            self.assertEqual(resolve_checkpoint(pointer), checkpoint.resolve())

    def test_public_cli_defaults_to_a_login_only_probe(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.mode, "login")
        self.assertFalse(args.learn)
        self.assertIsNone(args.sample_actions)
        self.assertTrue(args.start_timer)
        self.assertFalse(hasattr(args, "session_timeout"))

    def test_ladder_waits_for_requested_games_without_a_wall_clock_timeout(
        self,
    ) -> None:
        player = SimpleNamespace(ladder=AsyncMock())
        asyncio.run(
            _run_matchmaking(
                player,
                mode="ladder",
                opponent=None,
                games=50,
            )
        )
        player.ladder.assert_awaited_once_with(50)

    def test_public_summary_does_not_count_an_active_battle_as_a_tie(self) -> None:
        finished_win = SimpleNamespace(
            battle_tag="battle-win",
            finished=True,
            won=True,
            lost=False,
            turn=12,
            opponent_username="WinnerOpponent",
            rating=1010,
            opponent_rating=1000,
        )
        finished_loss = SimpleNamespace(
            battle_tag="battle-loss",
            finished=True,
            won=False,
            lost=True,
            turn=20,
            opponent_username="LossOpponent",
            rating=1040,
            opponent_rating=1050,
        )
        active = SimpleNamespace(
            battle_tag="battle-active",
            finished=False,
            won=None,
            lost=None,
            turn=8,
            opponent_username="ActiveOpponent",
            rating=None,
            opponent_rating=None,
        )
        player = SimpleNamespace(
            battles={
                battle.battle_tag: battle
                for battle in (finished_win, finished_loss, active)
            },
            username="PublicBot",
            decision_count=10,
            fallback_count=0,
            inference_latencies=[],
        )
        args = SimpleNamespace(
            mode="ladder",
            battle_format="gen9ou",
            team_file=Path("team.txt"),
            games=50,
            sample_actions=False,
            sampling_temperature=1.0,
            team_preview="random",
        )
        summary = _public_summary(
            args,
            checkpoint=Path("checkpoint"),
            player=player,
            rollout={},
            error=None,
        )
        self.assertEqual(summary["started_games"], 3)
        self.assertEqual(summary["finished_games"], 2)
        self.assertEqual(summary["unfinished_games"], 1)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertEqual(summary["ties"], 0)
        self.assertEqual(summary["win_rate"], 0.5)
        self.assertFalse(summary["battles"][-1]["finished"])

    def test_login_artifacts_never_contain_the_password(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env"
            secret = "unique-never-write-this-secret"
            env_file.write_text(
                f"{USERNAME_KEY}=ArtifactTestBot\n{PASSWORD_KEY}={secret}\n",
                encoding="utf-8",
            )
            output_dir = root / "login-output"
            args = build_parser().parse_args(
                [
                    "--env-file",
                    str(env_file),
                    "--mode",
                    "login",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            probe = {
                "schema": "public-showdown-login-v1",
                "account": "ArtifactTestBot",
                "server": "wss://example.invalid",
                "logged_in": True,
                "error": None,
            }
            with patch(
                "pokemon_battler.public_play._probe_public_login",
                new=AsyncMock(return_value=probe),
            ), contextlib.redirect_stdout(io.StringIO()):
                run(args)
            artifacts = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output_dir.iterdir()
                if path.is_file()
            )
            self.assertNotIn(secret, artifacts)

    def test_learning_batch_trains_and_promotes_a_separate_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env"
            env_file.write_text(
                f"{USERNAME_KEY}=LearningBot\n{PASSWORD_KEY}=secret\n",
                encoding="utf-8",
            )
            checkpoint = root / "initial"
            checkpoint.mkdir()
            output_dir = root / "learning-output"
            args = build_parser().parse_args(
                [
                    "--env-file",
                    str(env_file),
                    "--mode",
                    "accept",
                    "--opponent",
                    "HumanTester",
                    "--games",
                    "16",
                    "--learn",
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(output_dir),
                    "--promotion-games",
                    "2",
                ]
            )
            public_summary = {
                "finished_games": 16,
                "fallbacks": 0,
                "rollout": {"pending_battles": 0, "decisions": 24},
            }

            def fake_train(**kwargs):
                kwargs["output_dir"].mkdir(parents=True)
                return {"schema": "qwen-ppo-update-v1"}

            promotion = {"wins": 2, "losses": 0, "ties": 0, "score": 1.0}
            with (
                patch(
                    "pokemon_battler.public_play._play_public_batch",
                    new=AsyncMock(return_value=public_summary),
                ),
                patch(
                    "pokemon_battler.public_play.train_ppo_rollouts",
                    side_effect=fake_train,
                ),
                patch(
                    "pokemon_battler.public_play._promotion_evaluation",
                    return_value=promotion,
                ),
                patch("pokemon_battler.public_play._release_models"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                summary = run(args)

            candidate = output_dir / "batch-001" / "candidate"
            self.assertEqual(summary["selected_checkpoint"], str(candidate.resolve()))
            self.assertTrue(candidate.is_dir())
            self.assertEqual(
                (output_dir / "selected_checkpoint.txt")
                .read_text(encoding="utf-8")
                .strip(),
                str(candidate.resolve()),
            )

    def test_accept_requires_an_allowlisted_opponent(self) -> None:
        args = build_parser().parse_args(["--mode", "accept"])
        args.sample_actions = False
        with self.assertRaisesRegex(ValueError, "requires --opponent"):
            _validate_args(args, None)

    def test_learning_enables_only_unit_temperature_sampled_rollouts(self) -> None:
        args = build_parser().parse_args(
            ["--mode", "accept", "--learn", "--games", "16"]
        )
        args.sample_actions = True
        _validate_args(args, "HumanTester")

        args.sampling_temperature = 0.8
        with self.assertRaisesRegex(ValueError, "temperature 1.0"):
            _validate_args(args, "HumanTester")

        args.sampling_temperature = 1.0
        args.sample_actions = False
        with self.assertRaisesRegex(ValueError, "requires sampled actions"):
            _validate_args(args, "HumanTester")

    def test_small_learning_batch_warns_but_remains_available_for_smoke_tests(self) -> None:
        args = build_parser().parse_args(
            ["--mode", "accept", "--learn", "--games", "2"]
        )
        args.sample_actions = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _validate_args(args, "HumanTester")
        self.assertEqual(len(caught), 1)
        self.assertIn("very noisy PPO batch", str(caught[0].message))


if __name__ == "__main__":
    unittest.main()
