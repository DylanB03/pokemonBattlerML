from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pokemon_battler.showdown.live_eval import DEFAULT_TEAM
from pokemon_battler.data.team_pool import resolve_team_pool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run expert collection, corrected distillation, student-state DAgger, "
            "learned preview training, and held-out Foul Play promotion end to end."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--enemy-team-file", type=Path, action="append", default=[])
    parser.add_argument("--enemy-team-dir", type=Path)
    parser.add_argument("--enemy-team-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--games-per-round", type=int, default=200)
    parser.add_argument("--evaluation-games", type=int, default=100)
    parser.add_argument("--holdout-teams", type=int, default=3)
    parser.add_argument("--search-time-ms", type=int, default=250)
    parser.add_argument("--concurrent-games", type=int, default=4)
    parser.add_argument("--distillation-epochs", type=int, default=2)
    parser.add_argument(
        "--rehearsal-data", type=Path, default=Path("data/gen9ou-interaction-v1/train.jsonl")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--showdown-dir", type=Path, default=Path("data/pokemon-showdown"))
    parser.add_argument("--opponents-dir", type=Path, default=Path("data/opponents"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run(command: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _team_arguments(team_files: Sequence[Path]) -> list[str]:
    result: list[str] = []
    for path in team_files:
        result.extend(("--enemy-team-file", str(path)))
    return result


def _merge_traces(paths: Sequence[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output.open("wb") as destination:
        for path in paths:
            with path.open("rb") as source:
                for line in source:
                    if line.strip():
                        destination.write(line)
                        rows += 1
    return rows


def _student_probability(round_index: int, rounds: int) -> float:
    if round_index == 0:
        return 0.0
    return min(1.0, 0.35 + 0.65 * round_index / max(rounds - 1, 1))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (
        args.rounds <= 0
        or args.games_per_round <= 0
        or args.evaluation_games <= 0
        or args.concurrent_games <= 0
    ):
        raise ValueError("Round and game counts must be positive")
    enemy_team_files = list(args.enemy_team_file)
    if args.enemy_team_manifest is not None:
        if not args.enemy_team_manifest.is_file():
            raise FileNotFoundError(args.enemy_team_manifest)
        enemy_team_files.extend(
            Path(line.strip())
            for line in args.enemy_team_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    all_teams = resolve_team_pool(
        enemy_team_files, args.enemy_team_dir, minimum_teams=args.holdout_teams + 3
    )
    if not 2 <= args.holdout_teams < len(all_teams) - 1:
        raise ValueError("--holdout-teams must leave at least three training teams")
    # This split is persisted and never reshuffled between rounds. No teacher
    # state from the held-out compositions can enter the aggregate training file.
    ordered = sorted(all_teams, key=lambda path: str(path))
    holdout = ordered[-args.holdout_teams :]
    training = ordered[: -args.holdout_teams]
    args.output_dir.mkdir(parents=True)
    manifest = {
        "schema": "qwen-win-pipeline-v1",
        "source_checkpoint": str(args.checkpoint),
        "fixed_player_team": str(args.team_file),
        "training_enemy_teams": [str(path) for path in training],
        "heldout_enemy_teams": [str(path) for path in holdout],
        "ppo_enabled": False,
        "concurrent_games": args.concurrent_games,
        "rounds": [],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    champion = args.checkpoint
    traces: list[Path] = []
    python = sys.executable
    for round_index in range(args.rounds):
        round_dir = args.output_dir / f"round-{round_index:02d}"
        teacher_dir = round_dir / "teacher"
        distilled_dir = round_dir / "distilled"
        candidate_dir = round_dir / "candidate"
        evaluation_dir = round_dir / "heldout-evaluation"
        probability = _student_probability(round_index, args.rounds)
        collect = [
            python,
            "-m",
            "pokemon_battler.showdown.parallel_teacher_collect",
            "--enemy-policy",
            "foul-play",
            "--team-file",
            str(args.team_file),
            *_team_arguments(training),
            "--games",
            str(args.games_per_round),
            "--concurrent-games",
            str(args.concurrent_games),
            "--seed",
            str(args.seed + round_index),
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
            str(teacher_dir),
        ]
        if probability > 0:
            collect.extend(
                (
                    "--student-checkpoint",
                    str(champion),
                    "--student-action-probability",
                    str(probability),
                )
            )
        _run(collect, dry_run=args.dry_run)
        trace = teacher_dir / "foul_play_teacher.jsonl"
        traces.append(trace)
        aggregate = round_dir / "teacher_aggregate.jsonl"
        aggregate_rows = 0 if args.dry_run else _merge_traces(traces, aggregate)

        distill = [
            python,
            "-m",
            "pokemon_battler.training.distill",
            "--checkpoint",
            str(champion),
            "--teacher-data",
            str(aggregate),
            "--output-dir",
            str(distilled_dir),
            "--epochs",
            str(args.distillation_epochs),
            "--freeze-qwen",
            "--hard-target-weight",
            "0",
            "--confidence-power",
            "0",
            "--confident-disagreement-weight",
            "0",
            "--tera-weight",
            "2",
        ]
        if args.rehearsal_data.is_file():
            distill.extend(
                ("--rehearsal-data", str(args.rehearsal_data), "--rehearsal-weight", "0.1")
            )
        _run(distill, dry_run=args.dry_run)
        _run(
            [
                python,
                "-m",
                "pokemon_battler.training.preview_train",
                "--checkpoint",
                str(distilled_dir),
                "--teacher-data",
                str(aggregate),
                "--output-dir",
                str(candidate_dir),
            ],
            dry_run=args.dry_run,
        )
        _run(
            [
                python,
                "-m",
                "pokemon_battler.evaluation.policy_suite",
                "--candidate",
                str(candidate_dir),
                "--champion",
                str(champion),
                "--team-file",
                str(args.team_file),
                *_team_arguments(holdout),
                "--games",
                str(args.evaluation_games),
                "--concurrent-games",
                str(args.concurrent_games),
                "--seed",
                str(args.seed + 10_000),
                "--foul-play-search-time-ms",
                str(args.search_time_ms),
                "--showdown-dir",
                str(args.showdown_dir),
                "--opponents-dir",
                str(args.opponents_dir),
                "--server-port",
                str(args.server_port),
                "--output-dir",
                str(evaluation_dir),
            ],
            dry_run=args.dry_run,
        )
        evaluation = (
            {"promoted": False, "dry_run": True}
            if args.dry_run
            else json.loads((evaluation_dir / "summary.json").read_text())
        )
        if evaluation.get("promoted"):
            champion = candidate_dir
        round_report = {
            "round": round_index,
            "student_action_probability": probability,
            "teacher_trace": str(trace),
            "aggregate_teacher_rows": aggregate_rows,
            "candidate": str(candidate_dir),
            "champion_after_round": str(champion),
            "promotion": evaluation,
        }
        manifest["rounds"].append(round_report)
        manifest["current_champion"] = str(champion)
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"phase": "pipeline-round", **round_report}, indent=2), flush=True)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
