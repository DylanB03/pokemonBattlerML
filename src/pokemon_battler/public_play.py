from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from dotenv import dotenv_values
from poke_env import AccountConfiguration, RandomPlayer, ShowdownServerConfiguration

from pokemon_battler.league import QwenLeague
from pokemon_battler.live_eval import (
    DEFAULT_TEAM,
    _latency_summary,
    _read_team,
    _wilson_interval,
)
from pokemon_battler.live_policy import (
    DecisionTraceWriter,
    InteractionPlayer,
    InteractionPolicyRuntime,
)
from pokemon_battler.poke_env_compat import (
    close_poke_env_clients,
    install_safe_poke_env_shutdown,
)
from pokemon_battler.reinforcement import WinTrajectoryBuffer
from pokemon_battler.rl_training import train_ppo_rollouts
from pokemon_battler.win_experiment import _promotion_evaluation


PUBLIC_SCHEMA = "public-showdown-session-v1"
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_CHECKPOINT_POINTER = Path("outputs/qwen-win-pilot-1/selected_checkpoint.txt")
USERNAME_KEY = "POKEMON_SHOWDOWN_USERNAME"
PASSWORD_KEY = "POKEMON_SHOWDOWN_PASSWORD"
OPPONENT_KEY = "POKEMON_SHOWDOWN_OPPONENT"
PUBLIC_MODES = ("login", "accept", "challenge", "ladder")


@dataclass(frozen=True)
class PublicEnvironment:
    account: AccountConfiguration
    opponent: str | None


def load_public_environment(
    env_file: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> PublicEnvironment:
    """Load public credentials without placing their values in run artifacts."""
    source = Path(env_file)
    if not source.is_file():
        raise FileNotFoundError(
            f"Public-play environment file does not exist: {source}. "
            "Copy .env.example to .env and fill in the registered account."
        )
    values = {
        key: value
        for key, value in dotenv_values(source).items()
        if value is not None
    }
    process_environment = os.environ if environ is None else environ
    for key in (USERNAME_KEY, PASSWORD_KEY, OPPONENT_KEY):
        if key in process_environment:
            values[key] = process_environment[key]

    username = str(values.get(USERNAME_KEY, "")).strip()
    password = str(values.get(PASSWORD_KEY, ""))
    opponent = str(values.get(OPPONENT_KEY, "")).strip() or None
    missing = [
        key
        for key, value in ((USERNAME_KEY, username), (PASSWORD_KEY, password))
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required value(s) in {source}: {', '.join(missing)}")
    return PublicEnvironment(
        account=AccountConfiguration(username, password),
        opponent=opponent,
    )


def resolve_checkpoint(checkpoint: Path | None) -> Path:
    """Resolve either a checkpoint directory or a selected_checkpoint.txt pointer."""
    source = checkpoint or DEFAULT_CHECKPOINT_POINTER
    source = source.expanduser()
    if source.is_file():
        target_text = source.read_text(encoding="utf-8").strip()
        if not target_text:
            raise ValueError(f"Checkpoint pointer is empty: {source}")
        target = Path(target_text).expanduser()
        if not target.is_absolute():
            from_working_directory = target
            from_pointer_directory = source.parent / target
            target = (
                from_working_directory
                if from_working_directory.exists()
                else from_pointer_directory
            )
    else:
        target = source
    if not target.is_dir():
        raise FileNotFoundError(
            f"Public policy checkpoint does not exist: {target}. Pass --checkpoint "
            "a checkpoint directory or selected_checkpoint.txt file."
        )
    return target.resolve()


def _release_models() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


async def _wait_for_login(player: Any, timeout: float) -> None:
    async def wait_on_client_loop() -> None:
        await player.ps_client.logged_in.wait()

    concurrent = asyncio.run_coroutine_threadsafe(
        wait_on_client_loop(), player.ps_client.loop
    )
    try:
        await asyncio.wait_for(asyncio.wrap_future(concurrent), timeout=timeout)
    except TimeoutError as exc:
        concurrent.cancel()
        raise TimeoutError(
            f"Pokémon Showdown login for {player.username!r} did not complete "
            f"within {timeout:g} seconds. Check the .env credentials and account name."
        ) from exc


async def _run_matchmaking(
    player: InteractionPlayer,
    *,
    mode: str,
    opponent: str | None,
    games: int,
) -> None:
    if mode == "login":
        return
    if mode == "accept":
        assert opponent is not None
        operation = player.accept_challenges(opponent, games)
    elif mode == "challenge":
        assert opponent is not None
        operation = player.send_challenges(opponent, games)
    elif mode == "ladder":
        operation = player.ladder(games)
    else:  # pragma: no cover - protected by argparse and validation
        raise ValueError(f"Unknown public mode: {mode}")
    # A wall-clock deadline here can cancel poke-env's matchmaking coroutine while
    # a rated battle is active. Let the finite game-count operation finish so the
    # client never abandons a battle merely because the overall run took too long.
    await operation


def _public_summary(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    player: InteractionPlayer,
    rollout: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    battles = list(player.battles.values())
    finished_battles = [battle for battle in battles if battle.finished]
    wins = sum(battle.won is True for battle in finished_battles)
    losses = sum(battle.lost is True for battle in finished_battles)
    ties = len(finished_battles) - wins - losses
    return {
        "schema": PUBLIC_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "account": player.username,
        "checkpoint": str(checkpoint),
        "battle_format": args.battle_format,
        "team_file": str(args.team_file),
        "requested_games": 0 if args.mode == "login" else args.games,
        "started_games": len(battles),
        "finished_games": len(finished_battles),
        "unfinished_games": len(battles) - len(finished_battles),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / len(finished_battles) if finished_battles else None,
        "win_rate_wilson_95": _wilson_interval(wins, len(finished_battles)),
        "decisions": player.decision_count,
        "fallbacks": player.fallback_count,
        "fallback_rate": (
            player.fallback_count / player.decision_count
            if player.decision_count
            else 0.0
        ),
        "sample_actions": args.sample_actions,
        "sampling_temperature": args.sampling_temperature,
        "team_preview_policy": args.team_preview,
        "inference_latency_seconds": _latency_summary(player.inference_latencies),
        "rollout": rollout,
        "error": error,
        "battles": [
            {
                "battle_id": battle.battle_tag,
                "finished": battle.finished,
                "won": battle.won,
                "lost": battle.lost,
                "turns": battle.turn,
                "opponent": battle.opponent_username,
                "rating": battle.rating,
                "opponent_rating": battle.opponent_rating,
            }
            for battle in battles
        ],
    }


async def _play_public_batch(
    args: argparse.Namespace,
    *,
    account: AccountConfiguration,
    opponent: str | None,
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    buffer = WinTrajectoryBuffer(gamma=args.gamma, gae_lambda=args.gae_lambda)

    def receive(event: dict[str, Any]) -> None:
        if event["event"] == "decision":
            buffer.record_decision(
                event["battle_id"],
                event["observation"],
                action_id=event["action_id"],
                old_log_probability=event["old_log_probability"],
                value_probability=event["value_probability"],
            )
        elif event["event"] == "battle_finished":
            buffer.finish_battle(
                event["battle_id"], won=event["won"], lost=event["lost"]
            )

    runtime = InteractionPolicyRuntime(
        checkpoint,
        model_name=args.model,
        max_length=args.max_length,
        prompt_format=args.prompt_format,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    player = InteractionPlayer(
        runtime,
        account_configuration=account,
        trace_writer=DecisionTraceWriter(output_dir / "decisions.jsonl"),
        decision_callback=receive,
        fail_fast=args.fail_fast,
        sample_actions=args.sample_actions,
        sampling_temperature=args.sampling_temperature,
        team_preview_policy=args.team_preview,
        battle_format=args.battle_format,
        max_concurrent_battles=1,
        save_replays=str(output_dir / "replays"),
        server_configuration=ShowdownServerConfiguration,
        start_timer_on_battle_start=args.start_timer,
        team=_read_team(args.team_file),
    )
    session_error: Exception | None = None
    close_error: Exception | None = None
    try:
        await _wait_for_login(player, args.login_timeout)
        await _run_matchmaking(
            player,
            mode=args.mode,
            opponent=opponent,
            games=args.games,
        )
    except Exception as exc:
        session_error = exc
    finally:
        rollout = buffer.write_jsonl(output_dir / "rollouts.jsonl")
        try:
            await close_poke_env_clients(player.ps_client)
        except Exception as exc:
            close_error = exc

    error = session_error or close_error
    summary = _public_summary(
        args,
        checkpoint=checkpoint,
        player=player,
        rollout=rollout,
        error=f"{type(error).__name__}: {error}" if error is not None else None,
    )
    (output_dir / "public_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if error is not None:
        raise RuntimeError(
            f"Public session failed; partial artifacts were saved to {output_dir}"
        ) from error
    return summary


async def _probe_public_login(
    account: AccountConfiguration,
    *,
    login_timeout: float,
) -> dict[str, Any]:
    """Verify credentials and the public socket without loading the policy model."""
    player = RandomPlayer(
        account_configuration=account,
        battle_format="gen9ou",
        max_concurrent_battles=1,
        server_configuration=ShowdownServerConfiguration,
    )
    error: Exception | None = None
    try:
        await _wait_for_login(player, login_timeout)
    except Exception as exc:
        error = exc
    try:
        await close_poke_env_clients(player.ps_client)
    except Exception as exc:
        if error is None:
            error = exc
    summary = {
        "schema": "public-showdown-login-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "account": player.username,
        "server": ShowdownServerConfiguration.websocket_url,
        "logged_in": error is None,
        "error": f"{type(error).__name__}: {error}" if error is not None else None,
    }
    return summary


def _validate_args(args: argparse.Namespace, opponent: str | None) -> None:
    if args.games <= 0 or args.batches <= 0:
        raise ValueError("--games and --batches must be positive")
    if args.mode in {"accept", "challenge"} and not opponent:
        raise ValueError(
            f"--mode {args.mode} requires --opponent or {OPPONENT_KEY} in .env"
        )
    if args.mode == "login" and args.learn:
        raise ValueError("--learn cannot be used with --mode login")
    if args.batches > 1 and not args.learn:
        raise ValueError("Multiple --batches require --learn")
    if args.login_timeout <= 0:
        raise ValueError("--login-timeout must be positive")
    if args.sampling_temperature <= 0:
        raise ValueError("--sampling-temperature must be positive")
    if not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        raise ValueError("--gamma and --gae-lambda must be in [0, 1]")
    if args.learn:
        if not args.sample_actions:
            raise ValueError("--learn requires sampled actions; remove --no-sample-actions")
        if args.sampling_temperature != 1.0:
            raise ValueError(
                "Public PPO currently requires --sampling-temperature 1.0 so saved "
                "and recomputed policy probabilities match exactly"
            )
        positive = {
            "promotion_games": args.promotion_games,
            "ppo_epochs": args.ppo_epochs,
            "ppo_batch_size": args.ppo_batch_size,
            "ppo_gradient_accumulation_steps": args.ppo_gradient_accumulation_steps,
            "concurrent_games": args.concurrent_games,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"These learning arguments must be positive: {', '.join(invalid)}")
        if not 0 <= args.promotion_threshold <= 1:
            raise ValueError("--promotion-threshold must be in [0, 1]")
        if args.games < 16:
            warnings.warn(
                "Fewer than 16 public games is a very noisy PPO batch; use this only "
                "as a wiring test.",
                stacklevel=2,
            )


def _promotion_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        promotion_games=args.promotion_games,
        battle_format=args.battle_format,
        showdown_dir=args.showdown_dir,
        server_port=args.server_port,
        no_bootstrap_server=args.no_bootstrap_server,
        server_startup_timeout=args.server_startup_timeout,
        concurrent_games=args.concurrent_games,
        model=args.model,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    public_environment = load_public_environment(args.env_file)
    opponent = args.opponent or public_environment.opponent
    if args.sample_actions is None:
        args.sample_actions = bool(args.learn)
    _validate_args(args, opponent)
    checkpoint = None if args.mode == "login" else resolve_checkpoint(args.checkpoint)
    if args.mode != "login" and not args.team_file.is_file():
        raise FileNotFoundError(args.team_file)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_root = Path("outputs/public-learning" if args.learn else "reports/public")
    output_dir = args.output_dir or default_root / timestamp
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Public output directory is not empty: {output_dir}. Use a new directory "
            "so checkpoints and traces are never overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: _serialize(value)
        for key, value in vars(args).items()
        if key not in {"password"}
    } | {
        "account": public_environment.account.username,
        "opponent": opponent,
        "initial_checkpoint": str(checkpoint) if checkpoint is not None else None,
        "credential_source": str(args.env_file),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.mode == "login":
        login_summary = asyncio.run(
            _probe_public_login(
                public_environment.account,
                login_timeout=args.login_timeout,
            )
        )
        summary = {
            "schema": "public-showdown-run-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "login",
            "learning_enabled": False,
            "batches": [],
            "login": login_summary,
            "selected_checkpoint": None,
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Saved public Showdown login probe to {output_dir}")
        if not login_summary["logged_in"]:
            raise RuntimeError(
                "Public Pokémon Showdown login probe failed; see summary.json"
            )
        return summary

    assert checkpoint is not None

    league: QwenLeague | None = None
    if args.learn:
        league = QwenLeague(output_dir / "league.json")
        league.initialize(checkpoint, entry_id="public-initial")
    champion_checkpoint = checkpoint
    batches: list[dict[str, Any]] = []
    for batch_number in range(1, args.batches + 1):
        batch_dir = output_dir / f"batch-{batch_number:03d}"
        public_summary = asyncio.run(
            _play_public_batch(
                args,
                account=public_environment.account,
                opponent=opponent,
                checkpoint=champion_checkpoint,
                output_dir=batch_dir / "public",
            )
        )
        batch_report: dict[str, Any] = {
            "batch": batch_number,
            "public": public_summary,
            "training": None,
            "promotion": None,
            "league_result": None,
        }
        _release_models()

        if args.learn:
            if public_summary["finished_games"] != args.games:
                raise RuntimeError(
                    "Refusing to train because the public batch did not finish every "
                    "requested game"
                )
            if public_summary["fallbacks"]:
                raise RuntimeError(
                    "Refusing to train on a public batch containing policy fallbacks"
                )
            rollout = public_summary["rollout"]
            if rollout["pending_battles"] or not rollout["decisions"]:
                raise RuntimeError(
                    "Refusing to train on incomplete or empty public trajectories"
                )
            candidate_checkpoint = batch_dir / "candidate"
            training = train_ppo_rollouts(
                checkpoint=champion_checkpoint,
                rollout_file=batch_dir / "public" / "rollouts.jsonl",
                output_dir=candidate_checkpoint,
                model_name=args.model,
                epochs=args.ppo_epochs,
                batch_size=args.ppo_batch_size,
                gradient_accumulation_steps=args.ppo_gradient_accumulation_steps,
                qwen_learning_rate=args.qwen_learning_rate,
                head_learning_rate=args.head_learning_rate,
                clip_ratio=args.ppo_clip,
                value_clip=args.value_clip,
                value_coefficient=args.value_coefficient,
                entropy_coefficient=args.entropy_coefficient,
                target_kl=args.target_kl,
                dtype_name=args.dtype or "auto",
                load_in_4bit=(
                    True if args.load_in_4bit is None else args.load_in_4bit
                ),
                local_files_only=(
                    True if args.local_files_only is None else args.local_files_only
                ),
                attn_implementation=args.attn_implementation or "sdpa",
                seed=args.seed + batch_number,
                rollout_source="public-showdown",
            )
            batch_report["training"] = training
            _release_models()
            promotion = _promotion_evaluation(
                _promotion_args(args),
                candidate_checkpoint=candidate_checkpoint,
                champion_checkpoint=champion_checkpoint,
                actor_team=args.team_file,
                opponent_team=args.team_file,
                output_dir=batch_dir / "promotion",
            )
            batch_report["promotion"] = promotion
            assert league is not None
            league_result = league.record_candidate(
                candidate_id=f"public-batch-{batch_number:03d}",
                checkpoint=candidate_checkpoint,
                wins=int(promotion["wins"]),
                losses=int(promotion["losses"]),
                ties=int(promotion["ties"]),
                promotion_threshold=args.promotion_threshold,
            )
            batch_report["league_result"] = league_result
            champion_checkpoint = Path(league.champion["checkpoint"])
            _release_models()

        batches.append(batch_report)
        (batch_dir / "batch_summary.json").write_text(
            json.dumps(batch_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "selected_checkpoint.txt").write_text(
            str(champion_checkpoint.resolve()) + "\n", encoding="utf-8"
        )

    summary = {
        "schema": "public-showdown-run-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "learning_enabled": args.learn,
        "batches": batches,
        "selected_checkpoint": str(champion_checkpoint.resolve()),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved public Showdown run to {output_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Qwen interaction policy on the public Pokémon Showdown server, "
            "save complete trajectories, and optionally train/promote between batches."
        )
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--mode", choices=PUBLIC_MODES, default="login")
    parser.add_argument("--opponent")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--battle-format", default="gen9ou")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--team-preview", choices=("first", "random"), default="random")
    parser.add_argument(
        "--sample-actions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults off for evaluation and on with --learn.",
    )
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument(
        "--start-timer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--login-timeout", type=float, default=30.0)
    parser.add_argument("--fail-fast", action="store_true")

    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-batch-size", type=int, default=2)
    parser.add_argument("--ppo-gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.005)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--qwen-learning-rate", type=float, default=2e-6)
    parser.add_argument("--head-learning-rate", type=float, default=2e-5)
    parser.add_argument("--promotion-games", type=int, default=40)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--showdown-dir", type=Path, default=Path("data/pokemon-showdown"))
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-startup-timeout", type=float, default=60.0)
    parser.add_argument("--no-bootstrap-server", action="store_true")
    parser.add_argument("--concurrent-games", type=int, default=2)
    parser.add_argument("--model")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--prompt-format")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    install_safe_poke_env_shutdown()
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
