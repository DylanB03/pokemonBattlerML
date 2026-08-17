from __future__ import annotations

import asyncio
import contextlib
import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pokemon_battler.public_play import (
    OPPONENT_KEY,
    PASSWORD_KEY,
    PublicBattleProgress,
    USERNAME_KEY,
    _format_duration,
    _print_public_summary,
    _public_summary,
    _public_summary_from_trace,
    _run_matchmaking,
    _validate_args,
    build_parser,
    load_public_environment,
    resolve_checkpoint,
    run,
)


class PublicPlayTests(unittest.TestCase):
    def test_public_progress_prints_and_deduplicates_cumulative_results(self) -> None:
        progress = PublicBattleProgress(requested_games=3)
        terminal = io.StringIO()
        win = {
            "event": "battle_finished",
            "battle_id": "battle-win",
            "won": True,
            "lost": False,
            "opponent": "FirstOpponent",
            "turns": 14,
        }
        loss = {
            "event": "battle_finished",
            "battle_id": "battle-loss",
            "won": False,
            "lost": True,
            "opponent": "SecondOpponent",
            "turns": 22,
        }
        with contextlib.redirect_stdout(terminal):
            progress.record(win)
            progress.record(win)
            progress.record(loss)

        output = terminal.getvalue()
        self.assertIn("[public 1/3] WIN vs FirstOpponent", output)
        self.assertIn("record 1-0-0 | win rate 100.0% | turns 14", output)
        self.assertIn("[public 2/3] LOSS vs SecondOpponent", output)
        self.assertIn("record 1-1-0 | win rate 50.0% | turns 22", output)
        self.assertEqual(progress.completed_games, 2)

    def test_public_final_summary_is_compact(self) -> None:
        terminal = io.StringIO()
        with contextlib.redirect_stdout(terminal):
            _print_public_summary(
                {
                    "requested_games": 5,
                    "finished_games": 4,
                    "wins": 2,
                    "losses": 1,
                    "ties": 1,
                    "fallbacks": 0,
                    "unfinished_games": 1,
                }
            )
        self.assertEqual(
            terminal.getvalue().strip().splitlines(),
            [
                "[public summary] completed 4/5 | record 2-1-1 | win rate 50.0% | "
                "fallbacks 0 | unfinished 1",
                "[public ELO] unavailable | rated updates 0/4 "
                "(challenge games may be unrated)",
            ],
        )
        self.assertEqual(_format_duration(941.9), "15m 42s")

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

    def test_frozen_campaign_supports_multiple_reported_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env"
            env_file.write_text(
                f"{USERNAME_KEY}=FrozenBot\n{PASSWORD_KEY}=secret\n",
                encoding="utf-8",
            )
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            output_dir = root / "frozen-campaign"
            args = build_parser().parse_args(
                [
                    "--env-file",
                    str(env_file),
                    "--mode",
                    "ladder",
                    "--games",
                    "100",
                    "--batches",
                    "2",
                    "--concurrent-games",
                    "4",
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            public_summary = {
                "finished_games": 100,
                "wins": 55,
                "losses": 45,
                "ties": 0,
                "decisions": 200,
                "fallbacks": 0,
                "battles": [],
                "rollout": {"pending_battles": 0, "decisions": 200},
            }
            with (
                patch(
                    "pokemon_battler.public_play._play_public_batch",
                    new=AsyncMock(side_effect=[public_summary, public_summary]),
                ) as play,
                patch("pokemon_battler.public_play._release_models"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                summary = run(args)

            self.assertEqual(play.await_count, 2)
            self.assertEqual(summary["campaign"]["completed_batches"], 2)
            self.assertEqual(summary["campaign"]["public"]["finished_games"], 200)
            self.assertEqual(
                summary["campaign"]["stop_reason"], "completed_requested_batches"
            )
            self.assertEqual(summary["selected_checkpoint"], str(checkpoint.resolve()))
            self.assertFalse(summary["learning_enabled"])

    def test_public_concurrency_must_be_positive(self) -> None:
        args = build_parser().parse_args(
            ["--mode", "ladder", "--concurrent-games", "0"]
        )
        args.sample_actions = False
        with self.assertRaisesRegex(ValueError, "concurrent-games"):
            _validate_args(args, None)

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
            rating_updates={
                "battle-win": {
                    "before": 1010,
                    "after": 1040,
                    "change": 30,
                    "result": "winning",
                },
                "battle-loss": {
                    "before": 1040,
                    "after": 1018,
                    "change": -22,
                    "result": "losing",
                },
            },
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
        self.assertEqual(summary["rating"]["rated_games"], 2)
        self.assertEqual(summary["rating"]["start_elo"], 1010)
        self.assertEqual(summary["rating"]["end_elo"], 1018)
        self.assertEqual(summary["rating"]["net_change"], 8)
        self.assertEqual(summary["rating"]["elo_gained"], 30)
        self.assertEqual(summary["rating"]["elo_lost"], -22)
        self.assertEqual(summary["rating"]["peak_elo"], 1040)
        self.assertEqual(summary["rating"]["minimum_elo"], 1010)
        self.assertTrue(summary["rating"]["complete"])

    def test_trace_summary_recovers_completed_games_and_policy_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "decisions.jsonl"
            rows = [
                {
                    "event": "decision",
                    "battle_id": "battle-one",
                    "fallback_reason": None,
                    "showdown_turn": 3,
                    "prediction": {"latency_seconds": 0.1},
                },
                {
                    "event": "decision",
                    "battle_id": "battle-one",
                    "fallback_reason": "test fallback",
                    "showdown_turn": 4,
                    "prediction": {"latency_seconds": 0.2},
                },
                {
                    "event": "battle_finished",
                    "battle_id": "battle-one",
                    "won": True,
                    "lost": False,
                    "turns": 4,
                    "opponent": "Opponent",
                    "rating": 1100,
                    "opponent_rating": 1110,
                    "rating_update": {
                        "before": 1100,
                        "after": 1120,
                        "change": 20,
                        "result": "winning",
                    },
                },
            ]
            trace.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            args = build_parser().parse_args(
                ["--mode", "ladder", "--games", "100", "--team-preview", "random"]
            )
            args.sample_actions = False
            summary = _public_summary_from_trace(
                args,
                checkpoint=Path("checkpoint"),
                account="PublicBot",
                trace_path=trace,
            )

            self.assertEqual(summary["finished_games"], 1)
            self.assertEqual(summary["wins"], 1)
            self.assertEqual(summary["decisions"], 2)
            self.assertEqual(summary["fallbacks"], 1)
            self.assertEqual(summary["rating"]["net_change"], 20)
            self.assertAlmostEqual(summary["inference_latency_seconds"]["mean"], 0.15)

    def test_frozen_resume_finishes_partial_batch_then_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env"
            env_file.write_text(
                f"{USERNAME_KEY}=ResumeBot\n{PASSWORD_KEY}=secret\n",
                encoding="utf-8",
            )
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            output_dir = root / "campaign"
            public_dir = output_dir / "batch-001" / "public"
            public_dir.mkdir(parents=True)
            trace = public_dir / "decisions.jsonl"
            trace.write_text(
                json.dumps(
                    {
                        "event": "battle_finished",
                        "battle_id": "prior-battle",
                        "won": True,
                        "lost": False,
                        "turns": 8,
                        "opponent": "PriorOpponent",
                        "rating_update": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                [
                    "--env-file",
                    str(env_file),
                    "--mode",
                    "ladder",
                    "--games",
                    "2",
                    "--batches",
                    "2",
                    "--concurrent-games",
                    "2",
                    "--team-preview",
                    "random",
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(output_dir),
                    "--resume",
                ]
            )
            (output_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "account": "ResumeBot",
                        "mode": "ladder",
                        "games": 2,
                        "batches": 2,
                        "battle_format": "gen9ou",
                        "team_preview": "random",
                        "learn": False,
                        "initial_checkpoint": str(checkpoint.resolve()),
                        "team_file": str(args.team_file),
                    }
                ),
                encoding="utf-8",
            )
            calls: list[int] = []

            def fake_play(*_positional, **kwargs):
                games = int(kwargs["games_to_play"])
                calls.append(games)
                destination = kwargs["output_dir"]
                destination.mkdir(parents=True, exist_ok=True)
                with (destination / "decisions.jsonl").open("a", encoding="utf-8") as stream:
                    for index in range(games):
                        stream.write(
                            json.dumps(
                                {
                                    "event": "battle_finished",
                                    "battle_id": f"session-{len(calls)}-{index}",
                                    "won": False,
                                    "lost": True,
                                    "turns": 10,
                                    "opponent": "NewOpponent",
                                    "rating_update": None,
                                }
                            )
                            + "\n"
                        )
                return {"rollout": {}}

            with (
                patch(
                    "pokemon_battler.public_play._play_public_batch",
                    new=AsyncMock(side_effect=fake_play),
                ),
                patch("pokemon_battler.public_play._release_models"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                summary = run(args)

            self.assertEqual(calls, [1, 2])
            self.assertEqual(summary["campaign"]["completed_batches"], 2)
            self.assertEqual(summary["campaign"]["public"]["finished_games"], 4)
            self.assertEqual(summary["batches"][0]["public"]["wins"], 1)
            self.assertEqual(summary["batches"][0]["public"]["losses"], 1)
            config = json.loads(
                (output_dir / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["resume_count"], 1)

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
                "wins": 8,
                "losses": 8,
                "ties": 0,
                "decisions": 24,
                "fallbacks": 0,
                "battles": [],
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
            self.assertEqual(summary["campaign"]["model_improvement"]["ppo_candidates_promoted"], 1)

    def test_campaign_stops_before_training_when_public_score_is_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env"
            env_file.write_text(
                f"{USERNAME_KEY}=LearningBot\n{PASSWORD_KEY}=secret\n",
                encoding="utf-8",
            )
            checkpoint = root / "initial"
            checkpoint.mkdir()
            output_dir = root / "positive-campaign"
            args = build_parser().parse_args(
                [
                    "--env-file",
                    str(env_file),
                    "--mode",
                    "ladder",
                    "--games",
                    "16",
                    "--batches",
                    "3",
                    "--stop-win-rate",
                    "0.5",
                    "--learn",
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            public_summary = {
                "finished_games": 16,
                "wins": 9,
                "losses": 7,
                "ties": 0,
                "decisions": 24,
                "fallbacks": 0,
                "battles": [],
                "rollout": {"pending_battles": 0, "decisions": 24},
            }
            train = Mock()
            promotion = Mock()
            with (
                patch(
                    "pokemon_battler.public_play._play_public_batch",
                    new=AsyncMock(return_value=public_summary),
                ) as play,
                patch("pokemon_battler.public_play.train_ppo_rollouts", train),
                patch("pokemon_battler.public_play._promotion_evaluation", promotion),
                patch("pokemon_battler.public_play._release_models"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                summary = run(args)

            play.assert_awaited_once()
            train.assert_not_called()
            promotion.assert_not_called()
            self.assertEqual(summary["campaign"]["stop_reason"], "positive_win_rate")
            self.assertTrue(summary["campaign"]["positive_win_rate_reached"])
            self.assertEqual(summary["campaign"]["completed_batches"], 1)
            self.assertEqual(summary["campaign"]["public"]["wins"], 9)
            self.assertEqual(summary["campaign"]["maximum_public_games"], 48)
            self.assertEqual(
                summary["batches"][0]["training_skipped_reason"],
                "positive_win_rate_reached",
            )
            self.assertTrue((output_dir / "campaign_summary.json").is_file())

    def test_each_promoted_candidate_is_the_next_ppo_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env"
            env_file.write_text(
                f"{USERNAME_KEY}=LearningBot\n{PASSWORD_KEY}=secret\n",
                encoding="utf-8",
            )
            initial = root / "initial"
            initial.mkdir()
            output_dir = root / "chained-campaign"
            args = build_parser().parse_args(
                [
                    "--env-file",
                    str(env_file),
                    "--mode",
                    "ladder",
                    "--games",
                    "16",
                    "--batches",
                    "2",
                    "--learn",
                    "--checkpoint",
                    str(initial),
                    "--output-dir",
                    str(output_dir),
                    "--promotion-games",
                    "2",
                ]
            )
            public_summary = {
                "finished_games": 16,
                "wins": 8,
                "losses": 8,
                "ties": 0,
                "decisions": 24,
                "fallbacks": 0,
                "battles": [],
                "rollout": {"pending_battles": 0, "decisions": 24},
            }
            training_sources: list[Path] = []

            def fake_train(**kwargs):
                source = kwargs["checkpoint"].resolve()
                training_sources.append(source)
                kwargs["output_dir"].mkdir(parents=True)
                return {
                    "schema": "qwen-ppo-update-v1",
                    "source_checkpoint": str(source),
                    "updates": 2,
                    "approximate_kl": 0.005,
                }

            promotion = {"wins": 2, "losses": 0, "ties": 0, "score": 1.0}
            with (
                patch(
                    "pokemon_battler.public_play._play_public_batch",
                    new=AsyncMock(side_effect=[public_summary, public_summary]),
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

            first_candidate = output_dir / "batch-001" / "candidate"
            second_candidate = output_dir / "batch-002" / "candidate"
            self.assertEqual(
                training_sources,
                [initial.resolve(), first_candidate.resolve()],
            )
            self.assertEqual(
                summary["selected_checkpoint"], str(second_candidate.resolve())
            )
            improvement = summary["campaign"]["model_improvement"]
            self.assertEqual(improvement["ppo_candidates_trained"], 2)
            self.assertEqual(improvement["ppo_candidates_promoted"], 2)
            self.assertEqual(improvement["ppo_updates"], 4)
            self.assertFalse(improvement["selected_checkpoint_has_public_batch"])
            self.assertEqual(
                improvement["checkpoint_chain"],
                [
                    str(initial.resolve()),
                    str(first_candidate.resolve()),
                    str(second_candidate.resolve()),
                ],
            )

    def test_accept_requires_an_allowlisted_opponent(self) -> None:
        args = build_parser().parse_args(["--mode", "accept"])
        args.sample_actions = False
        with self.assertRaisesRegex(ValueError, "requires --opponent"):
            _validate_args(args, None)

    def test_stop_win_rate_requires_learning_and_a_valid_threshold(self) -> None:
        args = build_parser().parse_args(["--mode", "ladder", "--stop-win-rate", "0.5"])
        args.sample_actions = False
        with self.assertRaisesRegex(ValueError, "requires --learn"):
            _validate_args(args, None)

        args = build_parser().parse_args(
            ["--mode", "ladder", "--learn", "--stop-win-rate", "1.0"]
        )
        args.sample_actions = True
        with self.assertRaisesRegex(ValueError, r"must be in \[0, 1\)"):
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
