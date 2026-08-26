from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pokemon_battler.showdown.live_eval import _wilson_interval
from pokemon_battler.showdown.live_policy import InteractionPolicyRuntime
from pokemon_battler.showdown.policy_advisor import PolicyAdvisorServer
from pokemon_battler.showdown.teacher_collect import build_parser as build_teacher_parser


def build_parser():
    parser = build_teacher_parser()
    parser.description = (
        "Collect independent Foul Play teacher/DAgger shards concurrently while "
        "sharing one Qwen advisor."
    )
    parser.add_argument("--concurrent-games", type=int, default=4)
    return parser


def _append_option(command: list[str], name: str, value: Any | None) -> None:
    if value is not None:
        command.extend((name, str(value)))


def _worker_command(
    args: Any,
    *,
    worker: int,
    games: int,
    output_dir: Path,
    advisor_url: str | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pokemon_battler.showdown.teacher_collect",
        "--team-file",
        str(args.team_file),
        "--enemy-policy",
        args.enemy_policy,
        "--games",
        str(games),
        "--seed",
        str(args.seed + worker),
        "--battle-format",
        args.battle_format,
        "--showdown-dir",
        str(args.showdown_dir),
        "--server-port",
        str(args.server_port + worker),
        "--server-startup-timeout",
        str(args.server_startup_timeout),
        "--opponents-dir",
        str(args.opponents_dir),
        "--opponent-startup-timeout",
        str(args.opponent_startup_timeout),
        "--foul-play-search-time-ms",
        str(args.foul_play_search_time_ms),
        "--foul-play-parallelism",
        str(args.foul_play_parallelism),
        "--foul-play-search-threads",
        str(args.foul_play_search_threads),
        "--progress-interval",
        str(args.progress_interval),
        "--battle-stall-timeout",
        str(args.battle_stall_timeout),
        "--teacher-username",
        f"PBFoulTeachW{worker:02d}",
        "--foul-play-enemy-username",
        f"PBFoulEnemyW{worker:02d}",
        "--student-action-probability",
        str(args.student_action_probability),
        "--output-dir",
        str(output_dir),
    ]
    for team_file in args.enemy_team_file:
        command.extend(("--enemy-team-file", str(team_file)))
    _append_option(command, "--enemy-team-dir", args.enemy_team_dir)
    _append_option(
        command,
        "--enemy-foul-play-search-time-ms",
        args.enemy_foul_play_search_time_ms,
    )
    _append_option(command, "--student-advisor-url", advisor_url)
    if args.no_bootstrap_server:
        command.append("--no-bootstrap-server")
    if args.no_bootstrap_opponents:
        command.append("--no-bootstrap-opponents")
    return command


def _merge_traces(shard_dirs: Sequence[Path], output_path: Path) -> dict[str, int]:
    counts = {"rows": 0, "turn_rows": 0, "preview_rows": 0}
    with output_path.open("w", encoding="utf-8") as destination:
        for worker, directory in enumerate(shard_dirs):
            trace = directory / "foul_play_teacher.jsonl"
            with trace.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["battle_id"] = f"worker-{worker:02d}:{row['battle_id']}"
                    row["collection_worker"] = worker
                    destination.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                    counts["rows"] += 1
                    if row.get("decision_phase") == "team_preview":
                        counts["preview_rows"] += 1
                    else:
                        counts["turn_rows"] += 1
    return counts


def run(args: Any) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.games <= 0 or args.concurrent_games <= 0:
        raise ValueError("--games and --concurrent-games must be positive")
    workers = min(args.concurrent_games, args.games)
    if args.server_port + workers - 1 > 65535:
        raise ValueError("Concurrent Showdown worker ports exceed 65535")
    if args.keep_server:
        raise ValueError("--keep-server is incompatible with parallel collection")
    if args.student_advisor_url is not None:
        raise ValueError(
            "parallel collection manages its own shared advisor; use --student-checkpoint"
        )
    if args.student_action_probability > 0 and args.student_checkpoint is None:
        raise ValueError("A positive student action probability requires --student-checkpoint")
    args.output_dir.mkdir(parents=True)
    shard_dirs = [args.output_dir / f"worker-{worker:02d}" for worker in range(workers)]
    game_counts = [
        args.games // workers + int(worker < args.games % workers) for worker in range(workers)
    ]

    advisor_context: Any = nullcontext(None)
    if args.student_checkpoint is not None:
        runtime = InteractionPolicyRuntime(args.student_checkpoint)
        advisor_context = PolicyAdvisorServer(runtime, port=args.student_advisor_port)

    processes: list[subprocess.Popen[Any]] = []
    with advisor_context as advisor:
        advisor_url = advisor.url if advisor is not None else None
        commands = [
            _worker_command(
                args,
                worker=worker,
                games=game_counts[worker],
                output_dir=shard_dirs[worker],
                advisor_url=advisor_url,
            )
            for worker in range(workers)
        ]
        (args.output_dir / "worker_commands.json").write_text(
            json.dumps(commands, indent=2) + "\n", encoding="utf-8"
        )
        try:
            for worker, command in enumerate(commands):
                print(
                    f"[parallel teacher] starting worker {worker + 1}/{workers} "
                    f"for {game_counts[worker]} games on port {args.server_port + worker}",
                    flush=True,
                )
                processes.append(subprocess.Popen(command))
            failures = []
            for worker, process in enumerate(processes):
                return_code = process.wait()
                if return_code:
                    failures.append((worker, return_code))
            if failures:
                raise RuntimeError(f"Parallel teacher workers failed: {failures}")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()

    shard_summaries = [
        json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        for directory in shard_dirs
    ]
    trace_path = args.output_dir / "foul_play_teacher.jsonl"
    trace_counts = _merge_traces(shard_dirs, trace_path)
    finished = sum(int(summary["finished_games"]) for summary in shard_summaries)
    wins = sum(int(summary["teacher_wins"]) for summary in shard_summaries)
    losses = sum(int(summary["enemy_wins"]) for summary in shard_summaries)
    ties = sum(int(summary["ties"]) for summary in shard_summaries)
    summary = {
        "schema": "parallel-fixed-team-foul-play-teacher-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_games": args.games,
        "finished_games": finished,
        "concurrent_games": workers,
        "teacher_wins": wins,
        "enemy_wins": losses,
        "ties": ties,
        "teacher_win_rate": wins / finished if finished else None,
        "teacher_win_rate_wilson_95": _wilson_interval(wins, finished),
        "teacher_team_fixed": True,
        "enemy_teams_randomized": True,
        "student_checkpoint": (
            str(args.student_checkpoint) if args.student_checkpoint is not None else None
        ),
        "student_action_probability": args.student_action_probability,
        "shared_qwen_advisor": args.student_checkpoint is not None,
        "teacher_trace": str(trace_path),
        "teacher_trace_counts": trace_counts,
        "workers": shard_summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
