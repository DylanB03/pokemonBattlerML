from __future__ import annotations

import argparse
import json
import os
import shutil
from argparse import Namespace
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pokemon_battler.gated_pipeline import DEFAULT_CHAMPION
from pokemon_battler.interaction_cache import ARRAY_SPECS
from pokemon_battler.live_eval import DEFAULT_TEAM
from pokemon_battler.metamon_assets import (
    DEFAULT_SELFPLAY_SUBSETS,
    DEFAULT_TEAM_SETS,
    download_metamon_assets,
)
from pokemon_battler.parallel_interaction_cache import (
    build_parallel_interaction_caches,
)
from pokemon_battler.parallel_trajectory_prepare import (
    prepare_trajectory_dataset_parallel,
)
from pokemon_battler.policy_suite import run as run_policy_suite
from pokemon_battler.prepare import SplitConfig
from pokemon_battler.structured_train import train_structured_policy
from pokemon_battler.team_manifest import build_team_manifests


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, path)


def _remove_streamed_selfplay_archives(paths: Sequence[str]) -> dict[str, Any]:
    removed: list[str] = []
    removed_bytes = 0
    for value in paths:
        path = Path(value)
        if not path.name.lower().endswith(".tar.lz4") or not path.is_file():
            continue
        removed_bytes += path.stat().st_size
        path.unlink()
        removed.append(str(path))
    return {
        "removed": removed,
        "removed_gib": round(removed_bytes / 1024**3, 2),
    }


def _estimated_interaction_cache_bytes(rows: int) -> int:
    feature_bytes = sum(
        np.dtype(dtype).itemsize * int(np.prod(tail))
        for dtype, tail in ARRAY_SPECS.values()
    )
    target_bytes = np.dtype(np.int8).itemsize * 2 + np.dtype(np.int16).itemsize
    return rows * (feature_bytes + target_bytes)


def _load_complete(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("status") == "complete" else None


def _evaluation_arguments(
    args: argparse.Namespace,
    candidate: Path,
    enemy_teams: Sequence[Path],
    output_dir: Path,
) -> Namespace:
    return Namespace(
        candidate=candidate,
        champion=args.checkpoint,
        team_file=args.team_file,
        enemy_team_file=list(enemy_teams),
        enemy_team_dir=None,
        games=args.games,
        seed=args.seed + 10_000,
        output_dir=output_dir,
        showdown_dir=args.showdown_dir,
        opponents_dir=args.opponents_dir,
        server_port=args.server_port,
        foul_play_search_time_ms=args.search_time_ms,
        foul_play_parallelism=1,
        foul_play_search_threads=1,
        concurrent_games=args.concurrent_games,
        promotion_margin=0.0,
        minimum_delta_interval_lower=-0.05,
        candidate_action_value_weight=None,
        champion_action_value_weight=None,
        candidate_preview=True,
        champion_preview=True,
        no_bootstrap_server=False,
        no_bootstrap_opponents=False,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers <= 0 or args.concurrent_games <= 0:
        raise ValueError("workers and concurrent-games must be positive")
    if args.maximum_prepared_gib <= 0 or args.maximum_cache_gib <= 0:
        raise ValueError("storage limits must be positive")
    if not args.checkpoint.is_dir() or not args.team_file.is_file():
        raise FileNotFoundError("The source checkpoint or fixed player team is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = args.output_dir / "run_manifest.json"
    manifest: dict[str, Any]
    if run_manifest_path.is_file():
        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            return manifest
    else:
        disk = shutil.disk_usage(args.output_dir)
        manifest = {
            "schema": "large-offline-qwen-sidecar-pipeline-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "source_checkpoint": str(args.checkpoint),
            "workers": args.workers,
            "concurrent_games": args.concurrent_games,
            "free_disk_gib_at_start": round(disk.free / 1024**3, 1),
            "phases": {},
        }
        _write_manifest(run_manifest_path, manifest)
    selected_pointer = args.output_dir / "selected_checkpoint.txt"
    if not selected_pointer.is_file():
        selected_pointer.write_text(str(args.checkpoint.resolve()) + "\n", encoding="utf-8")

    try:
        assets_manifest = args.assets_root / "assets.json"
        if args.skip_download:
            if not assets_manifest.is_file():
                raise FileNotFoundError(f"--skip-download requires an existing {assets_manifest}")
            assets = json.loads(assets_manifest.read_text(encoding="utf-8"))
        else:
            assets = download_metamon_assets(
                args.assets_root,
                selfplay_subsets=args.selfplay_subsets or DEFAULT_SELFPLAY_SUBSETS,
                team_sets=args.team_sets or DEFAULT_TEAM_SETS,
                battle_format="gen9ou",
                selfplay_revision=args.selfplay_revision,
                teams_revision=args.teams_revision,
                keep_compressed=args.keep_compressed,
            )
        manifest["phases"]["assets"] = assets
        _write_manifest(run_manifest_path, manifest)

        prepared_dir = args.output_dir / "01-prepared"
        prepared = prepare_trajectory_dataset_parallel(
            [Path(path) for path in assets["selfplay"]],
            prepared_dir,
            split_config=SplitConfig(
                mode="hash",
                seed=args.seed,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
            ),
            battle_format="gen9ou",
            min_rating=None,
            outcome="both",
            trajectory_sample_rate=args.trajectory_sample_rate,
            reward_gamma=0.99,
            workers=args.workers,
            shard_trajectories=args.shard_trajectories,
            progress_every=args.progress_every,
            resume=True,
            require_outcome=True,
            maximum_unmapped_fraction=args.maximum_unmapped_fraction,
            maximum_output_bytes=int(args.maximum_prepared_gib * 1024**3),
        )
        manifest["phases"]["prepare"] = prepared["summary"]
        prepared_rows = sum(
            int(value)
            for value in prepared["summary"]["transitions_per_split"].values()
        )
        estimated_cache_bytes = _estimated_interaction_cache_bytes(prepared_rows)
        manifest["phases"]["storage_guard"] = {
            "prepared_limit_gib": args.maximum_prepared_gib,
            "cache_limit_gib": args.maximum_cache_gib,
            "prepared_rows": prepared_rows,
            "estimated_cache_gib": round(estimated_cache_bytes / 1024**3, 2),
        }
        _write_manifest(run_manifest_path, manifest)
        maximum_cache_bytes = int(args.maximum_cache_gib * 1024**3)
        if estimated_cache_bytes > maximum_cache_bytes:
            raise RuntimeError(
                "Estimated interaction cache exceeds its storage limit: "
                f"{estimated_cache_bytes / 1024**3:.2f} GiB required, maximum is "
                f"{args.maximum_cache_gib:.2f} GiB. Reduce --trajectory-sample-rate "
                "or increase --maximum-cache-gib."
            )

        cache_dir = args.output_dir / "02-interaction-cache"
        cache = build_parallel_interaction_caches(
            prepared_dir,
            cache_dir,
            workers=args.workers,
            progress_every=args.progress_every,
        )
        manifest["phases"]["interaction_cache"] = {
            "rows": cache["rows"],
            "shards": len(cache["shards"]),
            "elapsed_seconds": cache["elapsed_seconds"],
        }
        _write_manifest(run_manifest_path, manifest)

        team_manifest_dir = args.output_dir / "03-team-manifests"
        team_report_path = team_manifest_dir / "report.json"
        if team_report_path.is_file():
            team_report = json.loads(team_report_path.read_text(encoding="utf-8"))
        else:
            team_report = build_team_manifests(
                [Path(path) for path in assets["teams"].values()],
                team_manifest_dir,
                workers=args.workers,
                seed=args.seed,
                maximum_per_split=args.maximum_teams_per_split,
            )
        manifest["phases"]["team_corpus"] = team_report
        _write_manifest(run_manifest_path, manifest)

        candidate_dir = args.output_dir / "04-candidate"
        training_report_path = candidate_dir / "structured_training_report.json"
        if training_report_path.is_file():
            training = json.loads(training_report_path.read_text(encoding="utf-8"))
        else:
            training = train_structured_policy(
                source_checkpoint=args.checkpoint,
                prepared_dir=prepared_dir,
                cache_dir=cache_dir,
                output_dir=candidate_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                family_weight=args.family_weight,
                expectile=args.expectile,
                advantage_temperature=args.advantage_temperature,
                maximum_advantage_weight=args.maximum_advantage_weight,
                behavior_clone_weight=args.behavior_clone_weight,
                q_weight=args.q_weight,
                value_weight=args.value_weight,
                blend_weight=args.blend_weight,
                num_workers=args.workers,
                device_name=args.device_name,
                seed=args.seed,
                log_steps=args.log_steps,
            )
        manifest["phases"]["training"] = training
        _write_manifest(run_manifest_path, manifest)

        selected = candidate_dir
        if not args.skip_battle_evaluation:
            evaluation_dir = args.output_dir / "05-heldout-evaluation"
            summary_path = evaluation_dir / "summary.json"
            if summary_path.is_file():
                evaluation = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                heldout_manifest = Path(team_report["manifests"]["test"])
                enemy_teams = [
                    Path(line)
                    for line in heldout_manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                evaluation = run_policy_suite(
                    _evaluation_arguments(args, candidate_dir, enemy_teams, evaluation_dir)
                )
            manifest["phases"]["heldout_evaluation"] = evaluation
            if not evaluation["promoted"]:
                selected = args.checkpoint
        selected = selected.resolve()
        manifest["selected_checkpoint"] = str(selected)
        selected_pointer.write_text(str(selected) + "\n", encoding="utf-8")
        if not args.keep_compressed:
            manifest["phases"]["source_cleanup"] = _remove_streamed_selfplay_archives(
                assets["selfplay"]
            )
        manifest.pop("error", None)
        manifest["status"] = "complete"
        _write_manifest(run_manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest
    except BaseException as exc:
        manifest["status"] = "interrupted"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        try:
            _write_manifest(run_manifest_path, manifest)
        except OSError as manifest_error:
            print(
                json.dumps(
                    {
                        "phase": "manifest-write-failed",
                        "original_error": manifest["error"],
                        "manifest_error": (
                            f"{type(manifest_error).__name__}: {manifest_error}"
                        ),
                    }
                ),
                flush=True,
            )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download broad Metamon self-play, prepare/cache it with four CPU workers, "
            "train a structured sidecar on the current Qwen champion, and gate it in battles."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, default=Path("data/metamon-large"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--selfplay-subset", dest="selfplay_subsets", action="append")
    parser.add_argument("--team-set", dest="team_sets", action="append")
    parser.add_argument("--selfplay-revision", default="main")
    parser.add_argument("--teams-revision", default="v5")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--keep-compressed",
        action="store_true",
        help="Retain downloaded self-play archives after a successful pipeline run.",
    )
    parser.add_argument("--trajectory-sample-rate", type=float, default=0.005)
    parser.add_argument("--maximum-prepared-gib", type=float, default=32.0)
    parser.add_argument("--maximum-cache-gib", type=float, default=16.0)
    parser.add_argument("--maximum-unmapped-fraction", type=float, default=0.01)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-trajectories", type=int, default=2_000)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--maximum-teams-per-split", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--family-weight", type=float, default=0.0)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--advantage-temperature", type=float, default=0.1)
    parser.add_argument("--maximum-advantage-weight", type=float, default=20.0)
    parser.add_argument("--behavior-clone-weight", type=float, default=0.1)
    parser.add_argument("--q-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--blend-weight", type=float, default=0.5)
    parser.add_argument(
        "--device", dest="device_name", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--concurrent-games", type=int, default=4)
    parser.add_argument("--search-time-ms", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-battle-evaluation", action="store_true")
    parser.add_argument("--showdown-dir", type=Path, default=Path("data/pokemon-showdown"))
    parser.add_argument("--opponents-dir", type=Path, default=Path("data/opponents"))
    parser.add_argument("--server-port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
