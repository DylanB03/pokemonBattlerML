from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    ServerConfiguration,
    SimpleHeuristicsPlayer,
)

from pokemon_battler.external_opponents import ExternalOpponentProcess
from pokemon_battler.live_eval import DEFAULT_TEAM, _wilson_interval
from pokemon_battler.live_policy import InteractionPolicyRuntime
from pokemon_battler.observations import canonicalize_observation
from pokemon_battler.poke_env_compat import (
    close_poke_env_clients,
    install_safe_poke_env_shutdown,
)
from pokemon_battler.policy_advisor import PolicyAdvisorServer
from pokemon_battler.showdown_server import LocalShowdownServer
from pokemon_battler.team_pool import ShuffledTeamPool, resolve_team_pool

TEACHER_USERNAME = "PBFoulPlay"
ENEMY_USERNAME = "PBTeacherEnemy"
FOUL_PLAY_ENEMY_USERNAME = "PBFoulPlayEnemy"
ENEMY_POLICIES = {
    "random": RandomPlayer,
    "max-power": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
}
ENEMY_POLICY_NAMES = ("foul-play", *ENEMY_POLICIES)
_FOUL_PLAY_WINNER = re.compile(r"^INFO\s+Winner:\s*(.+?)\s*$", re.MULTILINE)
_FOUL_PLAY_RECORD = re.compile(r"^INFO\s+W:\s*(\d+)\s+L:\s*(\d+)\s*$", re.MULTILINE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Foul Play MCTS targets with the deployment team fixed and "
            "a randomized enemy OU team on every local battle."
        )
    )
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--enemy-team-file", type=Path, action="append", default=[])
    parser.add_argument("--enemy-team-dir", type=Path)
    parser.add_argument("--enemy-policy", choices=ENEMY_POLICY_NAMES, default="foul-play")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--battle-format", default="gen9ou")
    parser.add_argument("--showdown-dir", type=Path, default=Path("data/pokemon-showdown"))
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--no-bootstrap-server", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--server-startup-timeout", type=float, default=60.0)
    parser.add_argument("--opponents-dir", type=Path, default=Path("data/opponents"))
    parser.add_argument("--no-bootstrap-opponents", action="store_true")
    parser.add_argument("--opponent-startup-timeout", type=float, default=90.0)
    parser.add_argument("--foul-play-search-time-ms", type=int, default=250)
    parser.add_argument("--foul-play-parallelism", type=int, default=1)
    parser.add_argument("--foul-play-search-threads", type=int, default=1)
    parser.add_argument(
        "--enemy-foul-play-search-time-ms",
        type=int,
        help="Enemy Foul Play search budget; defaults to the teacher's budget.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=2.0,
        help="Seconds between smart-vs-smart progress checks.",
    )
    parser.add_argument(
        "--battle-stall-timeout",
        type=float,
        default=600.0,
        help="Fail if neither Foul Play log changes for this many seconds.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        help="Run DAgger: let this Qwen policy control a fraction of teacher states.",
    )
    parser.add_argument(
        "--student-action-probability",
        type=float,
        default=0.0,
        help="Probability that the student, rather than Foul Play, acts (0..1).",
    )
    parser.add_argument("--student-advisor-port", type=int, default=8765)
    parser.add_argument(
        "--student-advisor-url",
        help="Use an already loaded local Qwen advisor instead of loading a checkpoint.",
    )
    parser.add_argument("--teacher-username", default=TEACHER_USERNAME)
    parser.add_argument("--foul-play-enemy-username", default=FOUL_PLAY_ENEMY_USERNAME)
    return parser


def _validate_showdown_teams(
    showdown_dir: Path,
    battle_format: str,
    team_files: Sequence[Path],
) -> None:
    """Reject obsolete or otherwise illegal teams before a challenge can hang."""
    node = shutil.which("node")
    executable = showdown_dir.resolve() / "pokemon-showdown"
    if node is None or not executable.is_file():
        raise FileNotFoundError(
            "Team legality preflight requires Node.js and the local Pokémon "
            f"Showdown checkout at {showdown_dir}"
        )
    failures: list[str] = []
    for team_file in team_files:
        result = subprocess.run(
            [node, str(executable), "validate-team", battle_format],
            cwd=showdown_dir,
            input=team_file.read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            reason = (result.stderr or result.stdout).strip() or "validation failed"
            failures.append(f"{team_file}: {reason}")
    if failures:
        raise ValueError(
            "Showdown rejected these team files before collection:\n- " + "\n- ".join(failures)
        )


def _foul_play_winners(log_path: Path) -> list[str]:
    if not log_path.is_file():
        return []
    return _FOUL_PLAY_WINNER.findall(log_path.read_text(encoding="utf-8"))


def _latest_foul_play_record(log_path: Path) -> tuple[int, int]:
    if not log_path.is_file():
        return 0, 0
    with log_path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 65536))
        tail = stream.read().decode("utf-8", errors="replace")
    matches = _FOUL_PLAY_RECORD.findall(tail)
    if not matches:
        return 0, 0
    wins, losses = matches[-1]
    return int(wins), int(losses)


def _file_modified_at(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


def _finalize_teacher_trace(
    trace_path: Path,
    battles: Sequence[dict[str, Any]],
) -> dict[str, int]:
    """Attach outcome/team targets and normalize every collected observation."""
    if not trace_path.is_file():
        return {"rows": 0, "turn_rows": 0, "preview_rows": 0}
    rows: list[dict[str, Any]] = []
    order: list[str] = []
    decision_counts: dict[str, int] = {}
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            battle_id = str(row.get("battle_id") or "")
            if battle_id and battle_id not in order:
                order.append(battle_id)
            if row.get("decision_phase") != "team_preview":
                decision_counts[battle_id] = decision_counts.get(battle_id, 0) + 1
            rows.append(row)
    if len(order) != len(battles):
        raise RuntimeError(
            "Teacher trace battle order does not align with completed games: "
            f"trace={len(order)}, games={len(battles)}"
        )
    metadata = {battle_id: dict(battle) for battle_id, battle in zip(order, battles)}
    temporary = trace_path.with_suffix(trace_path.suffix + ".tmp")
    preview_rows = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            battle_id = str(row.get("battle_id") or "")
            battle = metadata[battle_id]
            result = str(battle.get("result") or "")
            row["outcome"] = (
                "WIN" if result == "teacher-win" else "LOSS" if result == "enemy-win" else "TIE"
            )
            row["battle_decision_count"] = decision_counts.get(battle_id, 0)
            row["enemy_team_file"] = battle.get("enemy_team_file")
            if row.get("decision_phase") == "team_preview":
                preview_rows += 1
            else:
                row = canonicalize_observation(row)
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    temporary.replace(trace_path)
    return {
        "rows": len(rows),
        "turn_rows": len(rows) - preview_rows,
        "preview_rows": preview_rows,
    }


async def _collect(
    args: argparse.Namespace,
    output_dir: Path,
    manager: ExternalOpponentProcess,
    enemy_pool: ShuffledTeamPool,
) -> dict[str, Any]:
    server_configuration = ServerConfiguration(
        f"ws://localhost:{args.server_port}/showdown/websocket",
        LocalhostServerConfiguration.authentication_url,
    )
    enemy_class = ENEMY_POLICIES[args.enemy_policy]
    enemy = enemy_class(
        account_configuration=AccountConfiguration(ENEMY_USERNAME, None),
        battle_format=args.battle_format,
        max_concurrent_battles=1,
        save_replays=str(output_dir / "replays"),
        server_configuration=server_configuration,
        team=enemy_pool,
    )
    try:
        deadline = asyncio.get_running_loop().time() + 10.0
        while not enemy.ps_client.logged_in.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"{ENEMY_USERNAME} did not log in within 10 seconds")
            await asyncio.sleep(0.05)
        manager.start_challenges()
        await enemy.accept_challenges(args.teacher_username, args.games)
        manager.ensure_success()

        battles = list(enemy.battles.values())
        enemy_wins = sum(battle.won is True for battle in battles)
        teacher_wins = sum(battle.lost is True for battle in battles)
        ties = len(battles) - enemy_wins - teacher_wins
        if len(enemy_pool.selections) != len(battles):
            raise RuntimeError(
                "Enemy team selections do not align one-to-one with finished battles"
            )
        battle_results: list[dict[str, Any]] = []
        for battle, selection in zip(battles, enemy_pool.selections):
            result = (
                "teacher-win"
                if battle.lost is True
                else "enemy-win"
                if battle.won is True
                else "tie"
            )
            selection.update(
                {
                    "battle_id": battle.battle_tag,
                    "result": result,
                    "turns": int(getattr(battle, "turn", 0) or 0),
                }
            )
            battle_results.append(
                {
                    "battle_id": battle.battle_tag,
                    "enemy_team_file": selection["team_file"],
                    "result": result,
                    "turns": int(getattr(battle, "turn", 0) or 0),
                }
            )
        trace_path = manager.teacher_trace_path
        trace_counts = _finalize_teacher_trace(trace_path, battle_results)
        if trace_path.is_file():
            with trace_path.open(encoding="utf-8") as stream:
                teacher_examples = sum(1 for line in stream if line.strip())
        else:
            teacher_examples = 0
        pool_report = enemy_pool.report()
        (output_dir / "enemy_team_selections.json").write_text(
            json.dumps(pool_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "schema": "fixed-team-foul-play-teacher-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "battle_format": args.battle_format,
            "teacher": manager.metadata(),
            "teacher_team_file": str(args.team_file.resolve()),
            "teacher_team_fixed": True,
            "enemy_policy": args.enemy_policy,
            "enemy_teams_randomized": True,
            "enemy_team_pool": pool_report,
            "requested_games": args.games,
            "finished_games": len(battles),
            "teacher_wins": teacher_wins,
            "enemy_wins": enemy_wins,
            "ties": ties,
            "teacher_win_rate": teacher_wins / len(battles) if battles else None,
            "teacher_win_rate_wilson_95": _wilson_interval(teacher_wins, len(battles)),
            "teacher_examples": teacher_examples,
            "teacher_trace_counts": trace_counts,
            "teacher_trace": str(trace_path),
            "battles": battle_results,
        }
    finally:
        await close_poke_env_clients(enemy.ps_client)


def _enemy_schedule(
    enemy_team_files: Sequence[Path],
    *,
    games: int,
    seed: int,
) -> tuple[list[Path], dict[str, Any]]:
    pool = ShuffledTeamPool(enemy_team_files, seed=seed)
    schedule: list[Path] = []
    for _ in range(games):
        pool.yield_team()
        schedule.append(Path(pool.selections[-1]["team_file"]))
    return schedule, pool.report()


def _collect_foul_play_vs_foul_play(
    args: argparse.Namespace,
    output_dir: Path,
    teacher: ExternalOpponentProcess,
    enemy: ExternalOpponentProcess,
    pool_report: dict[str, Any],
) -> dict[str, Any]:
    teacher.start()
    enemy.start()
    last_completed = -1
    last_activity = time.monotonic()
    last_log_times = (0.0, 0.0)
    while True:
        teacher_wins, enemy_wins = _latest_foul_play_record(teacher.log_path)
        completed = teacher_wins + enemy_wins
        log_times = (
            _file_modified_at(teacher.log_path),
            _file_modified_at(enemy.log_path),
        )
        if log_times != last_log_times:
            last_activity = time.monotonic()
            last_log_times = log_times
        if completed != last_completed:
            print(
                f"[teacher {completed}/{args.games}] Foul Play vs Foul Play | "
                f"fixed-team teacher {teacher_wins}-{enemy_wins}",
                flush=True,
            )
            last_completed = completed
        teacher_done = teacher.process is not None and teacher.process.poll() is not None
        enemy_done = enemy.process is not None and enemy.process.poll() is not None
        if teacher_done or enemy_done:
            teacher.ensure_success()
            enemy.ensure_success()
            break
        if time.monotonic() - last_activity > args.battle_stall_timeout:
            raise TimeoutError(
                "Foul Play vs Foul Play made no log progress for "
                f"{args.battle_stall_timeout:g} seconds; inspect "
                f"{teacher.log_path} and {enemy.log_path}"
            )
        time.sleep(args.progress_interval)

    winners = _foul_play_winners(teacher.log_path)
    if len(winners) != args.games:
        raise RuntimeError(f"Teacher recorded {len(winners)} finished games, expected {args.games}")
    teacher_wins = sum(winner == args.teacher_username for winner in winners)
    enemy_wins = sum(winner == args.foul_play_enemy_username for winner in winners)
    ties = len(winners) - teacher_wins - enemy_wins
    selections = list(pool_report["selections"])
    battle_results: list[dict[str, Any]] = []
    for index, selection in enumerate(selections):
        winner = winners[index]
        selection["result"] = (
            "teacher-win"
            if winner == args.teacher_username
            else "enemy-win"
            if winner == args.foul_play_enemy_username
            else "tie"
        )
        battle_results.append(
            {
                "enemy_team_file": selection["team_file"],
                "result": selection["result"],
            }
        )
    trace_path = teacher.teacher_trace_path
    trace_counts = _finalize_teacher_trace(trace_path, battle_results)
    with trace_path.open(encoding="utf-8") as stream:
        teacher_examples = sum(1 for line in stream if line.strip())
    pool_report["selections"] = selections
    (output_dir / "enemy_team_selections.json").write_text(
        json.dumps(pool_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema": "fixed-team-foul-play-teacher-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "battle_format": args.battle_format,
        "teacher": teacher.metadata(),
        "teacher_team_file": str(args.team_file.resolve()),
        "teacher_team_fixed": True,
        "enemy": enemy.metadata(),
        "enemy_policy": "foul-play",
        "enemy_teams_randomized": True,
        "enemy_team_pool": pool_report,
        "requested_games": args.games,
        "finished_games": len(winners),
        "teacher_wins": teacher_wins,
        "enemy_wins": enemy_wins,
        "ties": ties,
        "teacher_win_rate": teacher_wins / len(winners) if winners else None,
        "teacher_win_rate_wilson_95": _wilson_interval(teacher_wins, len(winners)),
        "teacher_examples": teacher_examples,
        "teacher_trace_counts": trace_counts,
        "teacher_trace": str(trace_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.games <= 0:
        raise ValueError("--games must be positive")
    if not 1 <= args.server_port <= 65535:
        raise ValueError("--server-port must be between 1 and 65535")
    if not 0 <= args.student_action_probability <= 1:
        raise ValueError("--student-action-probability must be between zero and one")
    if (
        args.student_action_probability > 0
        and args.student_checkpoint is None
        and args.student_advisor_url is None
    ):
        raise ValueError(
            "A positive student action probability requires --student-checkpoint "
            "or --student-advisor-url"
        )
    if args.student_checkpoint is not None and args.student_advisor_url is not None:
        raise ValueError("Use only one of --student-checkpoint and --student-advisor-url")
    positive_search = {
        "--opponent-startup-timeout": args.opponent_startup_timeout,
        "--server-startup-timeout": args.server_startup_timeout,
        "--foul-play-search-time-ms": args.foul_play_search_time_ms,
        "--foul-play-parallelism": args.foul_play_parallelism,
        "--foul-play-search-threads": args.foul_play_search_threads,
        "--progress-interval": args.progress_interval,
        "--battle-stall-timeout": args.battle_stall_timeout,
    }
    if args.enemy_foul_play_search_time_ms is not None:
        positive_search["--enemy-foul-play-search-time-ms"] = args.enemy_foul_play_search_time_ms
    invalid = [name for name, value in positive_search.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if not args.team_file.is_file():
        raise FileNotFoundError(f"Fixed model team does not exist: {args.team_file}")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    enemy_team_files = resolve_team_pool(
        args.enemy_team_file,
        args.enemy_team_dir,
        minimum_teams=2,
    )
    server = LocalShowdownServer(
        args.showdown_dir,
        port=args.server_port,
        bootstrap=not args.no_bootstrap_server,
        startup_timeout=args.server_startup_timeout,
        log_path=args.output_dir / "showdown.log",
        stop_on_exit=not args.keep_server,
    )
    server.prepare()
    _validate_showdown_teams(
        args.showdown_dir,
        args.battle_format,
        [args.team_file, *enemy_team_files],
    )
    enemy_pool = ShuffledTeamPool(enemy_team_files, seed=args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True)

    def serialized(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [serialized(item) for item in value]
        return value

    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {key: serialized(value) for key, value in vars(args).items()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    advisor = None
    if args.student_checkpoint is not None:
        advisor = PolicyAdvisorServer(
            InteractionPolicyRuntime(args.student_checkpoint),
            port=args.student_advisor_port,
        )
    manager = ExternalOpponentProcess(
        "foul-play",
        opponents_dir=args.opponents_dir,
        output_dir=args.output_dir,
        team_file=args.team_file,
        battle_format=args.battle_format,
        games=args.games,
        server_port=args.server_port,
        bootstrap=not args.no_bootstrap_opponents,
        startup_timeout=args.opponent_startup_timeout,
        challenger=(
            args.foul_play_enemy_username if args.enemy_policy == "foul-play" else ENEMY_USERNAME
        ),
        foul_play_search_time_ms=args.foul_play_search_time_ms,
        foul_play_parallelism=args.foul_play_parallelism,
        foul_play_search_threads=args.foul_play_search_threads,
        username=args.teacher_username,
        student_advisor_url=(advisor.url if advisor is not None else args.student_advisor_url),
        student_action_probability=args.student_action_probability,
        dagger_seed=args.seed,
    )
    manager.prepare()
    if advisor is not None:
        advisor.__enter__()
    try:
        with server:
            if args.enemy_policy == "foul-play":
                enemy_schedule, pool_report = _enemy_schedule(
                    enemy_team_files,
                    games=args.games,
                    seed=args.seed,
                )
                enemy_output = args.output_dir / "enemy"
                enemy_output.mkdir()
                enemy_manager = ExternalOpponentProcess(
                    "foul-play",
                    opponents_dir=args.opponents_dir,
                    output_dir=enemy_output,
                    team_file=enemy_team_files[0],
                    battle_format=args.battle_format,
                    games=args.games,
                    server_port=args.server_port,
                    bootstrap=not args.no_bootstrap_opponents,
                    startup_timeout=args.opponent_startup_timeout,
                    challenger=args.teacher_username,
                    foul_play_search_time_ms=(
                        args.enemy_foul_play_search_time_ms or args.foul_play_search_time_ms
                    ),
                    foul_play_parallelism=args.foul_play_parallelism,
                    foul_play_search_threads=args.foul_play_search_threads,
                    username=args.foul_play_enemy_username,
                    foul_play_mode="accept_challenge",
                    foul_play_team_files=enemy_schedule,
                    capture_teacher_trace=False,
                )
                with manager, enemy_manager:
                    summary = _collect_foul_play_vs_foul_play(
                        args,
                        args.output_dir,
                        manager,
                        enemy_manager,
                        pool_report,
                    )
            else:
                with manager:
                    summary = asyncio.run(_collect(args, args.output_dir, manager, enemy_pool))
    finally:
        if advisor is not None:
            advisor.__exit__(None, None, None)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved fixed-team teacher collection to {args.output_dir}")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    install_safe_poke_env_shutdown()
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
