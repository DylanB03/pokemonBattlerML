from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pokemon_battler.live_eval import DEFAULT_TEAM
from pokemon_battler.team_pool import resolve_team_pool

DEFAULT_CHAMPION = Path(
    "outputs/public-learning/positive-winrate-1000/batch-005/candidate"
)
DEFAULT_DIAGNOSTIC_CANDIDATE = Path("outputs/qwen-dagger-v1/round-00/candidate")
DEFAULT_TEACHER_FILES = (
    Path("outputs/qwen-dagger-v1/round-00/teacher/foul_play_teacher.jsonl"),
    Path("outputs/qwen-dagger-v1/round-01/teacher/foul_play_teacher.jsonl"),
)
DEFAULT_ENEMY_MANIFEST = Path("examples/opponent-pools/gen9ou-foul-play.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference ablations, held-out expert validation, selective cached "
            "distillation, offline gates, and one paired battle evaluation."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument(
        "--diagnostic-candidate", type=Path, default=DEFAULT_DIAGNOSTIC_CANDIDATE
    )
    parser.add_argument("--teacher-data", type=Path, action="append")
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--enemy-team-file", type=Path, action="append", default=[])
    parser.add_argument("--enemy-team-dir", type=Path)
    parser.add_argument("--enemy-team-manifest", type=Path, default=DEFAULT_ENEMY_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-teams", type=int, default=3)
    parser.add_argument("--diagnostic-games", type=int, default=50)
    parser.add_argument("--evaluation-games", type=int, default=100)
    parser.add_argument("--search-time-ms", type=int, default=250)
    parser.add_argument("--concurrent-games", type=int, default=4)
    parser.add_argument("--teacher-train-rows", type=int, default=8_000)
    parser.add_argument("--teacher-validation-rows", type=int, default=2_000)
    parser.add_argument(
        "--replay-train-data",
        type=Path,
        default=Path("data/gen9ou-interaction-v1/train.jsonl"),
    )
    parser.add_argument(
        "--replay-validation-data",
        type=Path,
        default=Path("data/gen9ou-interaction-v1/validation.jsonl"),
    )
    parser.add_argument("--replay-train-rows", type=int, default=4_000)
    parser.add_argument("--replay-validation-rows", type=int, default=2_000)
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--minimum-overfit-agreement", type=float, default=0.85)
    parser.add_argument("--minimum-teacher-agreement-gain", type=float, default=0.03)
    parser.add_argument("--maximum-replay-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--showdown-dir", type=Path, default=Path("data/pokemon-showdown"))
    parser.add_argument("--opponents-dir", type=Path, default=Path("data/opponents"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run(command: list[str], *, dry_run: bool) -> None:
    print("$ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _team_arguments(paths: Sequence[Path]) -> list[str]:
    arguments: list[str] = []
    for path in paths:
        arguments.extend(("--enemy-team-file", str(path)))
    return arguments


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _all_enemy_teams(args: argparse.Namespace) -> list[Path]:
    paths = list(args.enemy_team_file)
    if args.enemy_team_manifest is not None:
        if not args.enemy_team_manifest.is_file():
            raise FileNotFoundError(args.enemy_team_manifest)
        paths.extend(
            Path(line.strip())
            for line in args.enemy_team_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return resolve_team_pool(
        paths, args.enemy_team_dir, minimum_teams=args.holdout_teams + 3
    )


def _cached_command(
    *,
    python: str,
    checkpoint: Path,
    cache_dir: Path,
    output_dir: Path,
    variant: str,
    args: argparse.Namespace,
    family_weight: float,
    action_value_weight: float,
    replay: bool,
    epochs: int | None = None,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    teacher_validation_cache: Path | None = None,
) -> list[str]:
    command = [
        python,
        "-m",
        "pokemon_battler.cached_distill",
        "--checkpoint",
        str(checkpoint),
        "--teacher-train-cache",
        str(cache_dir / "teacher-train"),
        "--teacher-validation-cache",
        str(teacher_validation_cache or cache_dir / "teacher-validation"),
        "--output-dir",
        str(output_dir),
        "--variant",
        variant,
        "--epochs",
        str(epochs or args.head_epochs),
        "--batch-size",
        str(args.head_batch_size),
        "--family-aux-weight",
        str(family_weight),
        "--action-value-weight",
        str(action_value_weight),
        "--action-value-loss-type",
        "ranking",
        "--minimum-teacher-agreement-gain",
        str(args.minimum_teacher_agreement_gain),
        "--maximum-replay-accuracy-drop",
        str(args.maximum_replay_accuracy_drop),
        "--seed",
        str(args.seed),
    ]
    if replay:
        command.extend(
            (
                "--replay-train-cache",
                str(cache_dir / "replay-train"),
                "--replay-validation-cache",
                str(cache_dir / "replay-validation"),
            )
        )
    else:
        command.extend(("--rehearsal-weight", "0"))
    if train_limit is not None:
        command.extend(("--train-limit", str(train_limit)))
    if validation_limit is not None:
        command.extend(("--validation-limit", str(validation_limit)))
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    positive = (
        args.diagnostic_games,
        args.evaluation_games,
        args.concurrent_games,
        args.teacher_train_rows,
        args.teacher_validation_rows,
        args.replay_train_rows,
        args.replay_validation_rows,
        args.cache_batch_size,
        args.head_batch_size,
        args.head_epochs,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Game, row, batch, and epoch counts must be positive")
    teacher_files = list(args.teacher_data or DEFAULT_TEACHER_FILES)
    for path in (args.checkpoint, args.diagnostic_candidate, *teacher_files):
        if not path.exists():
            raise FileNotFoundError(path)
    teams = sorted(_all_enemy_teams(args), key=lambda path: str(path))
    if not 2 <= args.holdout_teams < len(teams) - 2:
        raise ValueError("--holdout-teams must leave at least three non-held-out teams")
    holdout = teams[-args.holdout_teams :]
    args.output_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": "gated-qwen-improvement-pipeline-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "champion_checkpoint": str(args.checkpoint),
        "diagnostic_candidate": str(args.diagnostic_candidate),
        "teacher_sources": [str(path) for path in teacher_files],
        "fixed_player_team": str(args.team_file),
        "heldout_enemy_teams": [str(path) for path in holdout],
        "gates": {},
        "phases": {},
    }
    _write_manifest(args.output_dir, manifest)
    python = sys.executable
    expert_dir = args.output_dir / "01-expert-ceiling"
    ablation_dir = args.output_dir / "02-inference-ablation"
    selected_data_dir = args.output_dir / "03-selected-data"
    cache_dir = args.output_dir / "04-frozen-cache"
    overfit_dir = args.output_dir / "05-overfit-gate"
    variants_dir = args.output_dir / "06-objective-variants"
    evaluation_dir = args.output_dir / "07-paired-evaluation"
    try:
        _run(
            [
                python,
                "-m",
                "pokemon_battler.parallel_teacher_collect",
                "--enemy-policy",
                "foul-play",
                "--team-file",
                str(args.team_file),
                *_team_arguments(holdout),
                "--games",
                str(args.diagnostic_games),
                "--concurrent-games",
                str(args.concurrent_games),
                "--seed",
                str(args.seed + 100),
                "--foul-play-search-time-ms",
                str(args.search_time_ms),
                "--enemy-foul-play-search-time-ms",
                str(args.search_time_ms),
                "--showdown-dir",
                str(args.showdown_dir),
                "--opponents-dir",
                str(args.opponents_dir),
                "--server-port",
                str(args.server_port),
                "--output-dir",
                str(expert_dir),
            ],
            dry_run=args.dry_run,
        )
        _run(
            [
                python,
                "-m",
                "pokemon_battler.policy_ablation",
                "--candidate",
                str(args.diagnostic_candidate),
                "--champion",
                str(args.checkpoint),
                "--team-file",
                str(args.team_file),
                *_team_arguments(holdout),
                "--games",
                str(args.diagnostic_games),
                "--concurrent-games",
                str(args.concurrent_games),
                "--seed",
                str(args.seed + 200),
                "--foul-play-search-time-ms",
                str(args.search_time_ms),
                "--showdown-dir",
                str(args.showdown_dir),
                "--opponents-dir",
                str(args.opponents_dir),
                "--server-port",
                str(args.server_port),
                "--output-dir",
                str(ablation_dir),
            ],
            dry_run=args.dry_run,
        )
        if args.dry_run:
            start_checkpoint = args.diagnostic_candidate
            manifest["phases"]["dry_run"] = True
        else:
            expert = json.loads((expert_dir / "summary.json").read_text(encoding="utf-8"))
            ablation = json.loads((ablation_dir / "summary.json").read_text(encoding="utf-8"))
            policy_only = next(
                row for row in ablation["configurations"] if row["name"] == "policy-only"
            )
            start_checkpoint = (
                args.diagnostic_candidate
                if float(policy_only["win_rate_delta"]) >= 0
                else args.checkpoint
            )
            manifest["phases"]["expert_ceiling"] = {
                "teacher_win_rate": expert["teacher_win_rate"],
                "interval": expert["teacher_win_rate_wilson_95"],
                "summary": str(expert_dir / "summary.json"),
            }
            manifest["phases"]["inference_ablation"] = {
                "best_configuration": ablation["best_configuration"]["name"],
                "policy_only_delta": policy_only["win_rate_delta"],
                "summary": str(ablation_dir / "summary.json"),
            }
        manifest["training_start_checkpoint"] = str(start_checkpoint)
        _write_manifest(args.output_dir, manifest)

        selection_command = [
            python,
            "-m",
            "pokemon_battler.selective_data",
        ]
        for path in teacher_files:
            selection_command.extend(("--teacher-data", str(path)))
        selection_command.extend(
            (
                "--teacher-validation-data",
                str(expert_dir / "foul_play_teacher.jsonl"),
                "--replay-train-data",
                str(args.replay_train_data),
                "--replay-validation-data",
                str(args.replay_validation_data),
                "--teacher-train-rows",
                str(args.teacher_train_rows),
                "--teacher-validation-rows",
                str(args.teacher_validation_rows),
                "--replay-train-rows",
                str(args.replay_train_rows),
                "--replay-validation-rows",
                str(args.replay_validation_rows),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(selected_data_dir),
            )
        )
        _run(selection_command, dry_run=args.dry_run)
        _run(
            [
                python,
                "-m",
                "pokemon_battler.frozen_cache",
                "--checkpoint",
                str(start_checkpoint),
                "--teacher-train",
                str(selected_data_dir / "teacher-train.jsonl"),
                "--teacher-validation",
                str(selected_data_dir / "teacher-validation.jsonl"),
                "--replay-train",
                str(selected_data_dir / "replay-train.jsonl"),
                "--replay-validation",
                str(selected_data_dir / "replay-validation.jsonl"),
                "--batch-size",
                str(args.cache_batch_size),
                "--output-dir",
                str(cache_dir),
            ],
            dry_run=args.dry_run,
        )
        _run(
            _cached_command(
                python=python,
                checkpoint=start_checkpoint,
                cache_dir=cache_dir,
                output_dir=overfit_dir,
                variant="overfit-256",
                args=args,
                family_weight=0.25,
                action_value_weight=0.0,
                replay=False,
                epochs=25,
                train_limit=256,
                validation_limit=256,
                teacher_validation_cache=cache_dir / "teacher-train",
            )
            + ["--learning-rate", "0.0003"],
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            overfit = json.loads(
                (overfit_dir / "cached_distillation_report.json").read_text(encoding="utf-8")
            )
            overfit_agreement = float(
                overfit["teacher_after"]["teacher_top1_agreement"]
            )
            manifest["gates"]["overfit"] = {
                "required_agreement": args.minimum_overfit_agreement,
                "observed_agreement": overfit_agreement,
                "passed": overfit_agreement >= args.minimum_overfit_agreement,
            }
            _write_manifest(args.output_dir, manifest)
            if overfit_agreement < args.minimum_overfit_agreement:
                manifest["status"] = "stopped-at-overfit-gate"
                _write_manifest(args.output_dir, manifest)
                print(json.dumps(manifest, indent=2, sort_keys=True))
                return manifest

        variants = (
            ("policy-only", 0.0, 0.0),
            ("policy-family", 0.25, 0.0),
            ("policy-family-q-ranking", 0.25, 0.10),
        )
        variant_reports = []
        for name, family_weight, q_weight in variants:
            output = variants_dir / name
            _run(
                _cached_command(
                    python=python,
                    checkpoint=start_checkpoint,
                    cache_dir=cache_dir,
                    output_dir=output,
                    variant=name,
                    args=args,
                    family_weight=family_weight,
                    action_value_weight=q_weight,
                    replay=True,
                ),
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                variant_reports.append(
                    json.loads(
                        (output / "cached_distillation_report.json").read_text(
                            encoding="utf-8"
                        )
                    )
                )
        if args.dry_run:
            manifest["status"] = "dry-run-complete"
            _write_manifest(args.output_dir, manifest)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return manifest
        eligible = [
            report for report in variant_reports if report["eligible_for_battle_evaluation"]
        ]
        if not eligible:
            manifest["status"] = "stopped-at-offline-gate"
            manifest["gates"]["offline"] = {
                "passed": False,
                "variants": [
                    {
                        "name": report["variant"],
                        "teacher_agreement_gain": report["teacher_agreement_gain"],
                        "teacher_kl_improvement": report["teacher_kl_improvement"],
                        "replay_accuracy_drop": report["replay_accuracy_drop"],
                    }
                    for report in variant_reports
                ],
            }
            _write_manifest(args.output_dir, manifest)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return manifest
        eligible.sort(
            key=lambda report: (
                float(report["teacher_agreement_gain"])
                - float(report["replay_accuracy_drop"]),
                float(report["teacher_kl_improvement"]),
            ),
            reverse=True,
        )
        finalist_report = eligible[0]
        finalist = variants_dir / str(finalist_report["variant"])
        manifest["gates"]["offline"] = {
            "passed": True,
            "selected_variant": finalist_report["variant"],
            "teacher_agreement_gain": finalist_report["teacher_agreement_gain"],
            "teacher_kl_improvement": finalist_report["teacher_kl_improvement"],
            "replay_accuracy_drop": finalist_report["replay_accuracy_drop"],
        }
        _write_manifest(args.output_dir, manifest)
        _run(
            [
                python,
                "-m",
                "pokemon_battler.policy_suite",
                "--candidate",
                str(finalist),
                "--champion",
                str(start_checkpoint),
                "--team-file",
                str(args.team_file),
                *_team_arguments(holdout),
                "--games",
                str(args.evaluation_games),
                "--concurrent-games",
                str(args.concurrent_games),
                "--seed",
                str(args.seed + 300),
                "--foul-play-search-time-ms",
                str(args.search_time_ms),
                "--candidate-action-value-weight",
                "0",
                "--champion-action-value-weight",
                "0",
                "--no-candidate-preview",
                "--no-champion-preview",
                "--showdown-dir",
                str(args.showdown_dir),
                "--opponents-dir",
                str(args.opponents_dir),
                "--server-port",
                str(args.server_port),
                "--output-dir",
                str(evaluation_dir),
            ],
            dry_run=False,
        )
        evaluation = json.loads((evaluation_dir / "summary.json").read_text(encoding="utf-8"))
        selected_checkpoint = finalist if evaluation["promoted"] else start_checkpoint
        manifest["gates"]["battle"] = {
            "passed": bool(evaluation["promoted"]),
            "candidate_win_rate": evaluation["candidate"]["win_rate"],
            "champion_win_rate": evaluation["champion"]["win_rate"],
            "win_rate_delta": evaluation["win_rate_delta"],
            "paired_bootstrap_delta_95": evaluation["paired_bootstrap_delta_95"],
            "summary": str(evaluation_dir / "summary.json"),
        }
        manifest["selected_checkpoint"] = str(selected_checkpoint)
        manifest["status"] = "complete-promoted" if evaluation["promoted"] else "complete-rejected"
        (args.output_dir / "selected_checkpoint.txt").write_text(
            str(selected_checkpoint) + "\n", encoding="utf-8"
        )
        _write_manifest(args.output_dir, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_manifest(args.output_dir, manifest)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
