from __future__ import annotations

import argparse
import gc
import json
from argparse import Namespace
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from pokemon_battler.frozen_cache import checkpoint_signature
from pokemon_battler.gated_pipeline import (
    DEFAULT_CHAMPION,
    DEFAULT_ENEMY_MANIFEST,
    DEFAULT_TEACHER_FILES,
)
from pokemon_battler.live_eval import DEFAULT_TEAM
from pokemon_battler.modeling import has_interaction_head
from pokemon_battler.policy_suite import run as run_policy_suite
from pokemon_battler.residual_cache import build_residual_teacher_cache
from pokemon_battler.residual_modeling import has_residual_head
from pokemon_battler.residual_train import train_residual_policy
from pokemon_battler.selective_data import (
    _copy_references,
    select_disjoint_teacher_rows,
)
from pokemon_battler.team_pool import resolve_team_pool
from pokemon_battler.trajectory_cache import TRAJECTORY_CACHE_SCHEMA
from pokemon_battler.trajectory_modeling import has_trajectory_head

DEFAULT_REPLAY_TRAIN_CACHE = Path(
    "outputs/trajectory-iql-v1/02-encoded-cache/train"
)
DEFAULT_REPLAY_VALIDATION_CACHE = Path(
    "outputs/trajectory-iql-v1/02-encoded-cache/validation"
)


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _manifest_teams(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    entries = [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return resolve_team_pool(entries, None, minimum_teams=3)


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
    minimum_delta_interval_lower: float,
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
        minimum_delta_interval_lower=minimum_delta_interval_lower,
        candidate_action_value_weight=0.0,
        champion_action_value_weight=0.0,
        candidate_preview=False,
        champion_preview=False,
        no_bootstrap_server=False,
        no_bootstrap_opponents=False,
    )


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_selection(output_dir: Path, checkpoint: Path) -> None:
    (output_dir / "selected_checkpoint.txt").write_text(
        str(checkpoint.resolve()) + "\n", encoding="utf-8"
    )


def _validate_source_and_replay_caches(
    checkpoint: Path,
    train_cache: Path,
    validation_cache: Path,
) -> dict[str, Any]:
    if not has_interaction_head(checkpoint):
        raise ValueError("Residual training requires an interaction-policy checkpoint")
    if has_trajectory_head(checkpoint) or has_residual_head(checkpoint):
        raise ValueError(
            "This pipeline currently requires a plain interaction champion; stacking "
            "a new residual over an auxiliary policy would not reproduce its deployed "
            "starting distribution."
        )
    signature = checkpoint_signature(checkpoint)
    cache_reports: dict[str, Any] = {}
    d_models: set[int] = set()
    for split, cache_dir in (("train", train_cache), ("validation", validation_cache)):
        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != TRAJECTORY_CACHE_SCHEMA:
            raise ValueError(f"{split} replay cache uses an incompatible schema")
        if metadata.get("checkpoint_signature") != signature:
            raise ValueError(
                f"{split} replay cache was not encoded by the selected champion; "
                "refusing to build teacher embeddings against mismatched features"
            )
        d_model = int(metadata["d_model"])
        d_models.add(d_model)
        cache_reports[split] = {
            "path": str(cache_dir.resolve()),
            "rows": int(metadata["rows"]),
            "d_model": d_model,
            "checkpoint_signature": str(metadata["checkpoint_signature"]),
        }
    if len(d_models) != 1:
        raise ValueError("Train and validation replay caches use different embedding sizes")
    return {
        "source_checkpoint": str(checkpoint.resolve()),
        "checkpoint_signature": signature,
        "replay_caches": cache_reports,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    teacher_files = list(args.teacher_data or DEFAULT_TEACHER_FILES)
    required = [
        args.checkpoint,
        args.team_file,
        args.enemy_team_manifest,
        args.replay_train_cache,
        args.replay_validation_cache,
        *teacher_files,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.pilot_games <= 0 or args.final_games <= 0:
        raise ValueError("Battle gate game counts must be positive")
    all_enemy_teams = _manifest_teams(args.enemy_team_manifest)
    source_checkpoint = args.checkpoint.resolve()
    preflight = _validate_source_and_replay_caches(
        source_checkpoint, args.replay_train_cache, args.replay_validation_cache
    )
    args.output_dir.mkdir(parents=True)
    selected_checkpoint = source_checkpoint
    _write_selection(args.output_dir, selected_checkpoint)
    data_dir = args.output_dir / "01-selected-teacher"
    cache_dir = args.output_dir / "02-teacher-cache"
    candidate_dir = args.output_dir / "03-residual-candidate"
    pilot_dir = args.output_dir / "04-heldout-pilot"
    final_dir = args.output_dir / "05-full-gate"
    manifest: dict[str, Any] = {
        "schema": "champion-residual-pipeline-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "source_checkpoint": str(source_checkpoint),
        "selected_checkpoint": str(selected_checkpoint),
        "safety_contract": {
            "initial_policy": "exact champion distribution",
            "source_checkpoint_is_immutable": True,
            "battle_gate_required_for_promotion": True,
        },
        "phases": {"preflight": preflight},
    }
    _write_manifest(args.output_dir, manifest)
    try:
        train_rows, validation_rows, selection_report = select_disjoint_teacher_rows(
            teacher_files,
            train_limit=args.teacher_train_rows,
            validation_limit=args.teacher_validation_rows,
            seed=args.seed,
            validation_fraction=args.teacher_validation_fraction,
            battle_cap=args.trajectory_cap,
        )
        data_dir.mkdir()
        train_file = data_dir / "teacher-train.jsonl"
        validation_file = data_dir / "teacher-validation.jsonl"
        _copy_references(train_rows, train_file)
        _copy_references(validation_rows, validation_file)
        training_team_paths = {Path(row.team).resolve() for row in train_rows}
        heldout_teams = [
            team for team in all_enemy_teams if team.resolve() not in training_team_paths
        ]
        if len(heldout_teams) < 3:
            raise ValueError(
                "The battle pilot needs at least three enemy teams absent from teacher training"
            )
        manifest["phases"]["selection"] = selection_report | {
            "teacher_files": [str(path) for path in teacher_files],
            "training_enemy_teams": sorted(str(path) for path in training_team_paths),
            "heldout_battle_enemy_teams": [str(path) for path in heldout_teams],
        }
        _write_manifest(args.output_dir, manifest)

        cache_dir.mkdir()
        manifest["phases"]["teacher_train_cache"] = build_residual_teacher_cache(
            checkpoint=source_checkpoint,
            data_file=train_file,
            output_dir=cache_dir / "train",
            batch_size=args.cache_batch_size,
            dtype_name=args.dtype,
            load_in_4bit=args.load_in_4bit,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )
        _release_memory()
        _write_manifest(args.output_dir, manifest)
        manifest["phases"]["teacher_validation_cache"] = build_residual_teacher_cache(
            checkpoint=source_checkpoint,
            data_file=validation_file,
            output_dir=cache_dir / "validation",
            batch_size=args.cache_batch_size,
            dtype_name=args.dtype,
            load_in_4bit=args.load_in_4bit,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )
        _release_memory()
        _write_manifest(args.output_dir, manifest)

        training_report = train_residual_policy(
            checkpoint=source_checkpoint,
            teacher_train_cache=cache_dir / "train",
            teacher_validation_cache=cache_dir / "validation",
            replay_train_cache=args.replay_train_cache,
            replay_validation_cache=args.replay_validation_cache,
            output_dir=candidate_dir,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            maximum_gradient_norm=args.maximum_gradient_norm,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            maximum_logit_delta=args.maximum_logit_delta,
            rehearsal_weight=args.rehearsal_weight,
            residual_penalty_weight=args.residual_penalty_weight,
            replay_train_rows=args.replay_train_rows,
            replay_validation_rows=args.replay_validation_rows,
            early_stopping_patience=args.early_stopping_patience,
            minimum_teacher_kl_gain=args.minimum_teacher_kl_gain,
            minimum_teacher_agreement_gain=args.minimum_teacher_agreement_gain,
            maximum_replay_kl=args.maximum_replay_kl,
            maximum_replay_action_change=args.maximum_replay_action_change,
            seed=args.seed,
        )
        manifest["phases"]["residual_training"] = training_report
        _release_memory()
        _write_manifest(args.output_dir, manifest)

        promotion_decision = "offline_gate_failed"
        if args.skip_battle_evaluation:
            promotion_decision = "battle_evaluation_skipped"
        elif training_report["offline_gate"]["passed"]:
            pilot = run_policy_suite(
                _suite_arguments(
                    candidate=candidate_dir,
                    champion=source_checkpoint,
                    output_dir=pilot_dir,
                    team_file=args.team_file,
                    enemy_teams=heldout_teams,
                    games=args.pilot_games,
                    seed=args.seed + 1000,
                    concurrent_games=args.concurrent_games,
                    search_time_ms=args.search_time_ms,
                    showdown_dir=args.showdown_dir,
                    opponents_dir=args.opponents_dir,
                    server_port=args.server_port,
                    minimum_delta_interval_lower=-0.05,
                )
            )
            manifest["phases"]["heldout_pilot"] = pilot
            _release_memory()
            _write_manifest(args.output_dir, manifest)
            if pilot["promoted"]:
                promotion_decision = "full_battle_gate_failed"
                final = run_policy_suite(
                    _suite_arguments(
                        candidate=candidate_dir,
                        champion=source_checkpoint,
                        output_dir=final_dir,
                        team_file=args.team_file,
                        enemy_teams=all_enemy_teams,
                        games=args.final_games,
                        seed=args.seed + 2000,
                        concurrent_games=args.concurrent_games,
                        search_time_ms=args.search_time_ms,
                        showdown_dir=args.showdown_dir,
                        opponents_dir=args.opponents_dir,
                        server_port=args.server_port,
                        minimum_delta_interval_lower=0.0,
                    )
                )
                manifest["phases"]["full_battle_gate"] = final
                if final["promoted"]:
                    selected_checkpoint = candidate_dir.resolve()
                    promotion_decision = "promoted"
            else:
                promotion_decision = "heldout_pilot_failed"
        manifest["selected_checkpoint"] = str(selected_checkpoint)
        manifest["candidate_checkpoint"] = str(candidate_dir.resolve())
        manifest["promoted"] = selected_checkpoint == candidate_dir.resolve()
        manifest["promotion_decision"] = promotion_decision
        manifest["status"] = "complete"
        _write_selection(args.output_dir, selected_checkpoint)
        _write_manifest(args.output_dir, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["selected_checkpoint"] = str(source_checkpoint)
        _write_selection(args.output_dir, source_checkpoint)
        _write_manifest(args.output_dir, manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Learn a bounded correction over the current champion from Foul Play "
            "distributions, regularize it on broad replay, and promote only after "
            "paired held-out battle gates."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--teacher-data", type=Path, action="append")
    parser.add_argument("--replay-train-cache", type=Path, default=DEFAULT_REPLAY_TRAIN_CACHE)
    parser.add_argument(
        "--replay-validation-cache",
        type=Path,
        default=DEFAULT_REPLAY_VALIDATION_CACHE,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--enemy-team-manifest", type=Path, default=DEFAULT_ENEMY_MANIFEST)
    parser.add_argument("--teacher-train-rows", type=int, default=8_000)
    parser.add_argument("--teacher-validation-rows", type=int, default=2_000)
    parser.add_argument("--teacher-validation-fraction", type=float, default=0.2)
    parser.add_argument("--trajectory-cap", type=int, default=24)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--maximum-gradient-norm", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--maximum-logit-delta", type=float, default=1.5)
    parser.add_argument("--rehearsal-weight", type=float, default=0.5)
    parser.add_argument("--residual-penalty-weight", type=float, default=0.01)
    parser.add_argument("--replay-train-rows", type=int, default=32_000)
    parser.add_argument("--replay-validation-rows", type=int, default=8_000)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--minimum-teacher-kl-gain", type=float, default=0.02)
    parser.add_argument("--minimum-teacher-agreement-gain", type=float, default=0.02)
    parser.add_argument("--maximum-replay-kl", type=float, default=0.05)
    parser.add_argument("--maximum-replay-action-change", type=float, default=0.15)
    parser.add_argument("--pilot-games", type=int, default=50)
    parser.add_argument("--final-games", type=int, default=100)
    parser.add_argument("--concurrent-games", type=int, default=4)
    parser.add_argument("--search-time-ms", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
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
