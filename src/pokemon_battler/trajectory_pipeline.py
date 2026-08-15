from __future__ import annotations

import argparse
import gc
import json
from argparse import Namespace
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import torch

from pokemon_battler.gated_pipeline import DEFAULT_CHAMPION, DEFAULT_ENEMY_MANIFEST
from pokemon_battler.live_eval import DEFAULT_TEAM
from pokemon_battler.policy_suite import run as run_policy_suite
from pokemon_battler.prepare import SplitConfig
from pokemon_battler.team_pool import resolve_team_pool
from pokemon_battler.trajectory_cache import build_trajectory_cache
from pokemon_battler.trajectory_prepare import prepare_trajectory_dataset
from pokemon_battler.trajectory_train import train_trajectory_policy

DEFAULT_RAW_DATA = Path("data/raw/metamon/gen9ou.tar.gz")


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _manifest_teams(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    teams = [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return resolve_team_pool(teams, None, minimum_teams=3)


def _suite_arguments(
    *,
    candidate: Path,
    champion: Path,
    output_dir: Path,
    team_file: Path,
    enemy_teams: Sequence[Path],
    games: int,
    seed: int,
    concurrent_games: int,
    search_time_ms: int,
    showdown_dir: Path,
    opponents_dir: Path,
    server_port: int,
) -> Namespace:
    return Namespace(
        candidate=candidate,
        champion=champion,
        team_file=team_file,
        enemy_team_file=list(enemy_teams),
        enemy_team_dir=None,
        games=games,
        seed=seed,
        output_dir=output_dir,
        showdown_dir=showdown_dir,
        opponents_dir=opponents_dir,
        server_port=server_port,
        foul_play_search_time_ms=search_time_ms,
        foul_play_parallelism=1,
        foul_play_search_threads=1,
        concurrent_games=concurrent_games,
        promotion_margin=0.0,
        candidate_action_value_weight=None,
        champion_action_value_weight=None,
        candidate_preview=True,
        champion_preview=True,
        no_bootstrap_server=False,
        no_bootstrap_opponents=False,
    )


def _write_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (args.input, args.checkpoint, args.team_file):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.games <= 0 or args.concurrent_games <= 0:
        raise ValueError("games and concurrent_games must be positive")
    args.output_dir.mkdir(parents=True)
    enemy_teams = _manifest_teams(args.enemy_team_manifest)
    data_dir = args.output_dir / "01-trajectories"
    cache_dir = args.output_dir / "02-encoded-cache"
    memoryless_dir = args.output_dir / "03-memoryless"
    recurrent_dir = args.output_dir / "04-recurrent"
    architecture_eval_dir = args.output_dir / "05-recurrent-vs-memoryless"
    champion_eval_dir = args.output_dir / "06-selected-vs-champion"
    manifest: dict[str, Any] = {
        "schema": "trajectory-win-pipeline-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "source_checkpoint": str(args.checkpoint),
        "raw_data": str(args.input),
        "fixed_player_team": str(args.team_file),
        "enemy_teams": [str(path) for path in enemy_teams],
        "phases": {},
    }
    _write_manifest(args.output_dir, manifest)
    try:
        manifest["phases"]["prepare"] = prepare_trajectory_dataset(
            [args.input],
            data_dir,
            split_config=SplitConfig(
                mode="chronological",
                seed=args.seed,
                validation_start=args.validation_start,
                test_start=args.test_start,
            ),
            battle_format="gen9ou",
            min_rating=args.min_rating,
            outcome="both",
            trajectory_sample_rate=args.trajectory_sample_rate,
            max_trajectories_per_split=args.max_trajectories_per_split,
            reward_gamma=args.gamma,
            progress_every=args.progress_every,
        )
        _write_manifest(args.output_dir, manifest)
        cache_dir.mkdir()
        for split in ("train", "validation"):
            manifest["phases"][f"cache_{split}"] = build_trajectory_cache(
                checkpoint=args.checkpoint,
                data_file=data_dir / f"{split}.jsonl",
                output_dir=cache_dir / split,
                batch_size=args.cache_batch_size,
                dtype_name=args.dtype,
                load_in_4bit=args.load_in_4bit,
                local_files_only=args.local_files_only,
                attn_implementation=args.attn_implementation,
            )
            _release_memory()
            _write_manifest(args.output_dir, manifest)
        common = {
            "source_checkpoint": args.checkpoint,
            "train_cache": cache_dir / "train",
            "validation_cache": cache_dir / "validation",
            "epochs": args.epochs,
            "behavior_clone_epochs": args.behavior_clone_epochs,
            "sequence_length": args.sequence_length,
            "burn_in": args.burn_in,
            "batch_size": args.train_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gamma": args.gamma,
            "expectile": args.expectile,
            "advantage_temperature": args.advantage_temperature,
            "maximum_advantage_weight": args.maximum_advantage_weight,
            "behavior_clone_weight": args.behavior_clone_weight,
            "target_ema": args.target_ema,
            "hidden_size": args.hidden_size,
            "recurrent_layers": args.recurrent_layers,
            "dropout": args.dropout,
            "seed": args.seed,
            "log_steps": args.log_steps,
        }
        manifest["phases"]["memoryless_train"] = train_trajectory_policy(
            **common,
            output_dir=memoryless_dir,
            memory_type="none",
        )
        _release_memory()
        _write_manifest(args.output_dir, manifest)
        manifest["phases"]["recurrent_train"] = train_trajectory_policy(
            **common,
            output_dir=recurrent_dir,
            memory_type="gru",
        )
        _release_memory()
        _write_manifest(args.output_dir, manifest)

        selected = recurrent_dir
        if not args.skip_battle_evaluation:
            architecture = run_policy_suite(
                _suite_arguments(
                    candidate=recurrent_dir,
                    champion=memoryless_dir,
                    output_dir=architecture_eval_dir,
                    team_file=args.team_file,
                    enemy_teams=enemy_teams,
                    games=args.games,
                    seed=args.seed + 1000,
                    concurrent_games=args.concurrent_games,
                    search_time_ms=args.search_time_ms,
                    showdown_dir=args.showdown_dir,
                    opponents_dir=args.opponents_dir,
                    server_port=args.server_port,
                )
            )
            manifest["phases"]["architecture_evaluation"] = architecture
            if float(architecture["win_rate_delta"]) < 0:
                selected = memoryless_dir
            _release_memory()
            champion = run_policy_suite(
                _suite_arguments(
                    candidate=selected,
                    champion=args.checkpoint,
                    output_dir=champion_eval_dir,
                    team_file=args.team_file,
                    enemy_teams=enemy_teams,
                    games=args.games,
                    seed=args.seed + 2000,
                    concurrent_games=args.concurrent_games,
                    search_time_ms=args.search_time_ms,
                    showdown_dir=args.showdown_dir,
                    opponents_dir=args.opponents_dir,
                    server_port=args.server_port,
                )
            )
            manifest["phases"]["champion_evaluation"] = champion
            if not champion["promoted"]:
                selected = args.checkpoint
        else:
            recurrent_objective = float(
                manifest["phases"]["recurrent_train"]["best_validation_objective"]
            )
            memoryless_objective = float(
                manifest["phases"]["memoryless_train"]["best_validation_objective"]
            )
            if memoryless_objective < recurrent_objective:
                selected = memoryless_dir
        (args.output_dir / "selected_checkpoint.txt").write_text(
            str(selected) + "\n", encoding="utf-8"
        )
        manifest["selected_checkpoint"] = str(selected)
        manifest["status"] = "complete"
        _write_manifest(args.output_dir, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(args.output_dir, manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare whole trajectories, freeze Qwen once, train memoryless and "
            "recurrent next-state IQL policies, and compare them in held-out battles."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--enemy-team-manifest", type=Path, default=DEFAULT_ENEMY_MANIFEST)
    parser.add_argument("--min-rating", type=int, default=1600)
    parser.add_argument("--trajectory-sample-rate", type=float, default=0.02)
    parser.add_argument("--max-trajectories-per-split", type=int)
    parser.add_argument("--validation-start", type=date.fromisoformat, default=date(2026, 1, 1))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2026, 4, 1))
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--behavior-clone-epochs", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--advantage-temperature", type=float, default=0.2)
    parser.add_argument("--maximum-advantage-weight", type=float, default=20.0)
    parser.add_argument("--behavior-clone-weight", type=float, default=0.1)
    parser.add_argument("--target-ema", type=float, default=0.995)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--recurrent-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--concurrent-games", type=int, default=4)
    parser.add_argument("--search-time-ms", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-steps", type=int, default=50)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--showdown-dir", type=Path, default=Path("data/pokemon-showdown"))
    parser.add_argument("--opponents-dir", type=Path, default=Path("data/opponents"))
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--skip-battle-evaluation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
