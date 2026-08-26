from __future__ import annotations

import argparse
import gc
import json
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import torch

from pokemon_battler.data.interaction_cache import (
    build_interaction_cache,
    default_interaction_cache_path,
    interaction_cache_is_current,
)
from pokemon_battler.models.interaction_features import PREPARED_SCHEMA_VERSION
from pokemon_battler.training.league import QwenLeague
from pokemon_battler.showdown.poke_env_compat import install_safe_poke_env_shutdown
from pokemon_battler.data.prepare import SplitConfig, prepare_dataset
from pokemon_battler.training.rl_training import train_offline_outcomes, train_ppo_rollouts
from pokemon_battler.showdown.self_play import run_qwen_match
from pokemon_battler.showdown.live_eval import _wilson_interval


DEFAULT_INITIAL_CHECKPOINT = Path("outputs/interaction-v1-1epoch/policy/final")
DEFAULT_RAW_INPUT = Path("data/raw/metamon/gen9ou.tar.gz")
DEFAULT_DATA_DIR = Path("data/gen9ou-interaction-v1")
DEFAULT_OUTPUT_DIR = Path("outputs/qwen-win-v1")
DEFAULT_TEAM = Path("examples/teams/gen9ou-balance.txt")


def _release_models() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _default_team_rotations(source: Path, output_dir: Path) -> list[Path]:
    """Create one legal fixed-team file per possible lead from the bundled team."""
    blocks = [block.strip() for block in source.read_text(encoding="utf-8").split("\n\n")]
    blocks = [block for block in blocks if block]
    if len(blocks) != 6:
        raise ValueError(f"Expected six Pokémon in the default team: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for lead in range(6):
        path = output_dir / f"lead-{lead + 1}.txt"
        ordered = [blocks[lead], *blocks[:lead], *blocks[lead + 1 :]]
        path.write_text("\n\n".join(ordered) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def _ensure_data(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    train_file = args.data_dir / "train.jsonl"
    report_path = args.data_dir / "prepare_report.json"
    if train_file.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("schema_version", -1)) != PREPARED_SCHEMA_VERSION:
            raise ValueError(
                f"Prepared data at {args.data_dir} is not schema "
                f"{PREPARED_SCHEMA_VERSION}; choose a new --data-dir"
            )
        split = report.get("split_config") or {}
        expected = {
            "inputs": [str(args.raw_input)],
            "battle_format": args.battle_format,
            "min_rating": args.min_rating,
            "outcome": "both",
            "sample_rate": args.sample_rate,
        }
        mismatches = [
            key for key, expected_value in expected.items()
            if report.get(key) != expected_value
        ]
        if (
            split.get("mode") != "chronological"
            or split.get("validation_start") != args.validation_start.isoformat()
            or split.get("test_start") != args.test_start.isoformat()
            or int(split.get("seed", -1)) != args.seed
        ):
            mismatches.append("split_config")
        if mismatches:
            raise ValueError(
                f"Prepared data at {args.data_dir} does not match this run "
                f"({', '.join(mismatches)}); choose a new --data-dir"
            )
    else:
        if args.data_dir.exists() and any(args.data_dir.iterdir()):
            raise FileExistsError(
                f"Incomplete prepared data directory: {args.data_dir}. "
                "Choose a new --data-dir instead of mixing datasets."
            )
        if not args.raw_input.is_file():
            raise FileNotFoundError(
                f"Neither prepared data nor the raw replay archive exists: "
                f"{args.raw_input}"
            )
        report = prepare_dataset(
            [args.raw_input],
            args.data_dir,
            split_config=SplitConfig(
                mode="chronological",
                seed=args.seed,
                validation_start=args.validation_start,
                test_start=args.test_start,
            ),
            battle_format=args.battle_format,
            min_rating=args.min_rating,
            outcome="both",
            sample_rate=args.sample_rate,
            progress_every=args.prepare_progress_every,
        )
    cache = default_interaction_cache_path(train_file)
    if not interaction_cache_is_current(train_file, cache):
        build_interaction_cache(
            train_file,
            cache,
            overwrite=args.rebuild_cache,
            progress_every=args.cache_progress_every,
        )
    return train_file, cache, report


def _promotion_evaluation(
    args: argparse.Namespace,
    *,
    candidate_checkpoint: Path,
    champion_checkpoint: Path,
    actor_team: Path,
    opponent_team: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate both team orientations so lead/team assignment cannot select a model."""
    forward_games = (args.promotion_games + 1) // 2
    reverse_games = args.promotion_games // 2

    def play(label: str, games: int, own_team: Path, other_team: Path) -> dict[str, Any]:
        return run_qwen_match(
            actor_checkpoint=candidate_checkpoint,
            opponent_checkpoint=champion_checkpoint,
            actor_team_file=own_team,
            opponent_team_file=other_team,
            games=games,
            output_dir=output_dir / label,
            battle_format=args.battle_format,
            showdown_dir=args.showdown_dir,
            server_port=args.server_port,
            bootstrap_server=not args.no_bootstrap_server,
            server_startup_timeout=args.server_startup_timeout,
            concurrent_games=args.concurrent_games,
            sample_actor=False,
            sample_opponent=False,
            collect_actor=False,
            model_name=args.model,
            dtype_name=args.dtype,
            load_in_4bit=args.load_in_4bit,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )

    matches = [play("forward", forward_games, actor_team, opponent_team)]
    _release_models()
    if reverse_games:
        matches.append(play("reverse", reverse_games, opponent_team, actor_team))
        _release_models()
    wins = sum(int(match["wins"]) for match in matches)
    losses = sum(int(match["losses"]) for match in matches)
    ties = sum(int(match["ties"]) for match in matches)
    games = wins + losses + ties
    summary = {
        "schema": "qwen-paired-promotion-v1",
        "candidate_checkpoint": str(candidate_checkpoint),
        "champion_checkpoint": str(champion_checkpoint),
        "games": games,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "score": (wins + 0.5 * ties) / games,
        "win_rate_wilson_95": _wilson_interval(wins, games),
        "orientations": matches,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.initial_checkpoint.is_dir():
        raise FileNotFoundError(args.initial_checkpoint)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Win-training output directory is not empty: {args.output_dir}. "
            "Use a new directory so every post-trained model is retained."
        )
    if args.iterations < 0:
        raise ValueError("--iterations cannot be negative")
    positive = {
        "rollout_games": args.rollout_games,
        "promotion_games": args.promotion_games,
        "concurrent_games": args.concurrent_games,
        "ppo_epochs": args.ppo_epochs,
        "ppo_batch_size": args.ppo_batch_size,
        "ppo_gradient_accumulation_steps": args.ppo_gradient_accumulation_steps,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if not 0 <= args.promotion_threshold <= 1:
        raise ValueError("--promotion-threshold must be in [0, 1]")
    if args.rollout_temperature <= 0:
        raise ValueError("--rollout-temperature must be positive")
    requested_team_files = args.team_file or [DEFAULT_TEAM]
    for path in requested_team_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.skip_offline:
        train_file = cache = None
        dataset_report = None
    else:
        train_file, cache, dataset_report = _ensure_data(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def serialize_config(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, list):
            return [serialize_config(item) for item in value]
        return value

    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                key: serialize_config(value)
                for key, value in vars(args).items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    team_files = (
        list(args.team_file)
        if args.team_file
        else _default_team_rotations(DEFAULT_TEAM, args.output_dir / "team-pool")
    )

    if args.skip_offline:
        starting_checkpoint = args.initial_checkpoint
        offline_report = None
    else:
        assert train_file is not None and cache is not None
        starting_checkpoint = args.output_dir / "offline"
        offline_report = train_offline_outcomes(
            checkpoint=args.initial_checkpoint,
            train_file=train_file,
            interaction_cache=cache,
            output_dir=starting_checkpoint,
            model_name=args.model,
            epochs=args.offline_epochs,
            max_steps=args.offline_max_steps,
            batch_size=args.offline_batch_size,
            gradient_accumulation_steps=args.offline_gradient_accumulation_steps,
            qwen_learning_rate=args.qwen_learning_rate,
            head_learning_rate=args.head_learning_rate,
            expectile=args.expectile,
            advantage_temperature=args.advantage_temperature,
            max_advantage_weight=args.max_advantage_weight,
            behavior_clone_weight=args.behavior_clone_weight,
            dtype_name=args.dtype,
            load_in_4bit=args.load_in_4bit,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
            seed=args.seed,
        )
        _release_models()

    league = QwenLeague(args.output_dir / "league.json")
    league.initialize(
        starting_checkpoint,
        entry_id="behavior-cloning" if args.skip_offline else "offline-warm-start",
    )
    if not args.skip_offline:
        league.add_reference(
            args.initial_checkpoint,
            entry_id="behavior-cloning",
        )
    iterations: list[dict[str, Any]] = []
    for iteration in range(1, args.iterations + 1):
        champion = league.champion
        opponent = league.sample_opponent(seed=args.seed + iteration)
        iteration_dir = args.output_dir / f"iteration-{iteration:03d}"
        actor_team = team_files[(iteration - 1) % len(team_files)]
        opponent_team = team_files[iteration % len(team_files)]
        rollout_summary = run_qwen_match(
            actor_checkpoint=Path(champion["checkpoint"]),
            opponent_checkpoint=Path(opponent["checkpoint"]),
            actor_team_file=actor_team,
            opponent_team_file=opponent_team,
            games=args.rollout_games,
            output_dir=iteration_dir / "rollout",
            battle_format=args.battle_format,
            showdown_dir=args.showdown_dir,
            server_port=args.server_port,
            bootstrap_server=not args.no_bootstrap_server,
            server_startup_timeout=args.server_startup_timeout,
            concurrent_games=args.concurrent_games,
            sample_actor=True,
            sample_opponent=True,
            temperature=args.rollout_temperature,
            collect_actor=True,
            model_name=args.model,
            dtype_name=args.dtype,
            load_in_4bit=args.load_in_4bit,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )
        _release_models()
        candidate_checkpoint = iteration_dir / "candidate"
        ppo_report = train_ppo_rollouts(
            checkpoint=Path(champion["checkpoint"]),
            rollout_file=iteration_dir / "rollout" / "rollouts.jsonl",
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
            dtype_name=args.dtype,
            load_in_4bit=args.load_in_4bit,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
            seed=args.seed + iteration,
        )
        _release_models()
        evaluation = _promotion_evaluation(
            args,
            candidate_checkpoint=candidate_checkpoint,
            champion_checkpoint=Path(champion["checkpoint"]),
            actor_team=actor_team,
            opponent_team=opponent_team,
            output_dir=iteration_dir / "promotion",
        )
        candidate_id = f"iteration-{iteration:03d}"
        league_result = league.record_candidate(
            candidate_id=candidate_id,
            checkpoint=candidate_checkpoint,
            wins=int(evaluation["wins"]),
            losses=int(evaluation["losses"]),
            ties=int(evaluation["ties"]),
            promotion_threshold=args.promotion_threshold,
        )
        iteration_report = {
            "iteration": iteration,
            "training_checkpoint": champion,
            "rollout_opponent": opponent,
            "teams": [str(actor_team), str(opponent_team)],
            "rollout": rollout_summary,
            "ppo": ppo_report,
            "promotion": evaluation,
            "league_result": league_result,
        }
        iterations.append(iteration_report)
        (iteration_dir / "iteration_summary.json").write_text(
            json.dumps(iteration_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    summary = {
        "schema": "qwen-win-experiment-v1",
        "dataset": dataset_report,
        "offline": offline_report,
        "iterations": iterations,
        "selected_checkpoint": league.champion["checkpoint"],
        "league_manifest": str(league.path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "selected_checkpoint.txt").write_text(
        str(league.champion["checkpoint"]) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train Qwen for battle wins with offline outcome learning and "
            "local-Showdown self-play PPO. Showdown executes mechanics only."
        )
    )
    parser.add_argument("--initial-checkpoint", type=Path, default=DEFAULT_INITIAL_CHECKPOINT)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_RAW_INPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model")
    parser.add_argument("--team-file", type=Path, action="append")
    parser.add_argument("--battle-format", default="gen9ou")
    parser.add_argument("--min-rating", type=int, default=1600)
    parser.add_argument("--sample-rate", type=float, default=0.02)
    parser.add_argument("--validation-start", type=date.fromisoformat, default=date(2026, 1, 1))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2026, 4, 1))
    parser.add_argument("--prepare-progress-every", type=int, default=10_000)
    parser.add_argument("--cache-progress-every", type=int, default=10_000)
    parser.add_argument("--rebuild-cache", action="store_true")

    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--offline-epochs", type=int, default=1)
    parser.add_argument("--offline-max-steps", type=int, default=1000)
    parser.add_argument("--offline-batch-size", type=int, default=1)
    parser.add_argument("--offline-gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--advantage-temperature", type=float, default=0.1)
    parser.add_argument("--max-advantage-weight", type=float, default=20.0)
    parser.add_argument("--behavior-clone-weight", type=float, default=0.1)

    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--rollout-games", type=int, default=64)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--ppo-batch-size", type=int, default=2)
    parser.add_argument("--ppo-gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--qwen-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--promotion-games", type=int, default=40)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)

    parser.add_argument("--showdown-dir", type=Path, default=Path("data/pokemon-showdown"))
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-startup-timeout", type=float, default=60.0)
    parser.add_argument("--no-bootstrap-server", action="store_true")
    parser.add_argument("--concurrent-games", type=int, default=4)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--no-4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(load_in_4bit=True, local_files_only=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    install_safe_poke_env_shutdown()
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
