from __future__ import annotations

import argparse
import asyncio
import json
import random
from argparse import Namespace
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pokemon_battler.external_opponents import ExternalOpponentProcess
from pokemon_battler.live_eval import DEFAULT_TEAM, PLAYER_USERNAME, _run_battles
from pokemon_battler.poke_env_compat import install_safe_poke_env_shutdown
from pokemon_battler.showdown_server import LocalShowdownServer
from pokemon_battler.team_pool import ShuffledTeamPool, resolve_team_pool
from pokemon_battler.team_preview import has_team_preview_head


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare candidate and champion on the same held-out Foul Play suite."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--enemy-team-file", type=Path, action="append", default=[])
    parser.add_argument("--enemy-team-dir", type=Path)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--showdown-dir", type=Path, default=Path("data/pokemon-showdown"))
    parser.add_argument("--opponents-dir", type=Path, default=Path("data/opponents"))
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--foul-play-search-time-ms", type=int, default=250)
    parser.add_argument("--foul-play-parallelism", type=int, default=1)
    parser.add_argument("--foul-play-search-threads", type=int, default=1)
    parser.add_argument("--concurrent-games", type=int, default=4)
    parser.add_argument("--promotion-margin", type=float, default=0.0)
    parser.add_argument("--minimum-delta-interval-lower", type=float, default=-0.05)
    parser.add_argument("--candidate-action-value-weight", type=float)
    parser.add_argument("--champion-action-value-weight", type=float)
    parser.add_argument(
        "--candidate-preview", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--champion-preview", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--no-bootstrap-server", action="store_true")
    parser.add_argument("--no-bootstrap-opponents", action="store_true")
    return parser


def _schedule(team_files: list[Path], games: int, seed: int) -> list[Path]:
    pool = ShuffledTeamPool(team_files, seed=seed)
    schedule = []
    for _ in range(games):
        pool.yield_team()
        schedule.append(Path(pool.selections[-1]["team_file"]))
    return schedule


def _checkpoint_preview_enabled(checkpoint: Path, requested: bool) -> bool:
    """Use learned preview only when the requested checkpoint actually provides it."""
    return bool(requested and has_team_preview_head(checkpoint))


def _difference_interval(
    candidate: list[int], champion: list[int], *, seed: int, samples: int = 10000
) -> list[float]:
    if len(candidate) != len(champion) or not candidate:
        return [0.0, 0.0]
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        indices = [rng.randrange(len(candidate)) for _ in candidate]
        differences.append(
            sum(candidate[index] - champion[index] for index in indices) / len(indices)
        )
    differences.sort()
    return [differences[int(samples * 0.025)], differences[int(samples * 0.975)]]


def _should_promote(
    delta: float,
    interval: Sequence[float],
    *,
    promotion_margin: float,
    minimum_interval_lower: float,
) -> bool:
    return delta > promotion_margin and interval[0] > minimum_interval_lower


def _arguments(
    checkpoint: Path,
    team_file: Path,
    games: int,
    port: int,
    *,
    action_value_weight: float | None = None,
    load_preview_head: bool = True,
    team_preview_policy: str = "learned",
) -> Namespace:
    return Namespace(
        checkpoint=checkpoint,
        model=None,
        team_file=team_file,
        opponent_team_file=team_file,
        opponent="foul-play",
        games=games,
        battle_format="gen9ou",
        server_port=port,
        max_length=None,
        prompt_format=None,
        dtype=None,
        load_in_4bit=None,
        local_files_only=None,
        attn_implementation=None,
        action_value_weight=action_value_weight,
        load_preview_head=load_preview_head,
        team_preview_policy=team_preview_policy,
        fail_fast=True,
    )


def _run_policy(
    args: argparse.Namespace,
    checkpoint: Path,
    schedule: list[Path],
    output_dir: Path,
    *,
    action_value_weight: float | None = None,
    load_preview_head: bool = True,
    team_preview_policy: str = "learned",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True)
    workers = min(args.concurrent_games, args.games)
    schedules = [schedule[worker::workers] for worker in range(workers)]
    managers = []
    for worker, worker_schedule in enumerate(schedules):
        worker_output = output_dir / f"worker-{worker:02d}"
        worker_output.mkdir()
        manager = ExternalOpponentProcess(
            "foul-play",
            opponents_dir=args.opponents_dir,
            output_dir=worker_output,
            team_file=worker_schedule[0],
            battle_format="gen9ou",
            games=len(worker_schedule),
            server_port=args.server_port,
            bootstrap=not args.no_bootstrap_opponents,
            challenger=PLAYER_USERNAME,
            foul_play_search_time_ms=args.foul_play_search_time_ms,
            foul_play_parallelism=args.foul_play_parallelism,
            foul_play_search_threads=args.foul_play_search_threads,
            foul_play_team_files=worker_schedule,
            capture_teacher_trace=False,
            username=f"PBFoulEvalW{worker:02d}",
        )
        manager.prepare()
        managers.append(manager)
    with ExitStack() as stack:
        for manager in managers:
            stack.enter_context(manager)
        return asyncio.run(
            _run_battles(
                _arguments(
                    checkpoint,
                    args.team_file,
                    args.games,
                    args.server_port,
                    action_value_weight=action_value_weight,
                    load_preview_head=load_preview_head,
                    team_preview_policy=team_preview_policy,
                ),
                output_dir,
                external_opponent=managers,
            )
        )


def _indexed_results(summary: dict[str, Any]) -> dict[tuple[str, int, str], int]:
    return {
        (
            str(row["opponent"]).lower(),
            int(row["opponent_game_index"]),
            str(row["scheduled_enemy_team_file"]),
        ): int(bool(row["won"]))
        for row in summary["battles"]
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.games <= 0 or args.concurrent_games <= 0:
        raise ValueError("--games and --concurrent-games must be positive")
    teams = resolve_team_pool(args.enemy_team_file, args.enemy_team_dir, minimum_teams=3)
    schedule = _schedule(teams, args.games, args.seed)
    args.output_dir.mkdir(parents=True)
    candidate_action_value_weight = getattr(args, "candidate_action_value_weight", None)
    champion_action_value_weight = getattr(args, "champion_action_value_weight", None)
    candidate_preview = _checkpoint_preview_enabled(
        args.candidate, getattr(args, "candidate_preview", True)
    )
    champion_preview = _checkpoint_preview_enabled(
        args.champion, getattr(args, "champion_preview", True)
    )
    with LocalShowdownServer(
        args.showdown_dir,
        port=args.server_port,
        bootstrap=not args.no_bootstrap_server,
        log_path=args.output_dir / "showdown.log",
    ):
        candidate = _run_policy(
            args,
            args.candidate,
            schedule,
            args.output_dir / "candidate",
            action_value_weight=candidate_action_value_weight,
            load_preview_head=candidate_preview,
            team_preview_policy="learned" if candidate_preview else "first",
        )
        champion = _run_policy(
            args,
            args.champion,
            schedule,
            args.output_dir / "champion",
            action_value_weight=champion_action_value_weight,
            load_preview_head=champion_preview,
            team_preview_policy="learned" if champion_preview else "first",
        )
    candidate_index = _indexed_results(candidate)
    champion_index = _indexed_results(champion)
    if candidate_index.keys() != champion_index.keys():
        raise RuntimeError("Candidate and champion held-out schedules did not align")
    keys = sorted(candidate_index)
    candidate_results = [candidate_index[key] for key in keys]
    champion_results = [champion_index[key] for key in keys]
    candidate_rate = sum(candidate_results) / max(len(candidate_results), 1)
    champion_rate = sum(champion_results) / max(len(champion_results), 1)
    delta = candidate_rate - champion_rate
    interval = _difference_interval(candidate_results, champion_results, seed=args.seed)
    minimum_interval_lower = float(
        getattr(args, "minimum_delta_interval_lower", -0.05)
    )
    promoted = _should_promote(
        delta,
        interval,
        promotion_margin=args.promotion_margin,
        minimum_interval_lower=minimum_interval_lower,
    )
    report = {
        "schema": "heldout-foul-play-promotion-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "enemy_teams": [str(path) for path in teams],
        "schedule": [str(path) for path in schedule],
        "candidate": candidate,
        "champion": champion,
        "win_rate_delta": delta,
        "paired_bootstrap_delta_95": interval,
        "promotion_margin": args.promotion_margin,
        "minimum_delta_interval_lower": minimum_interval_lower,
        "concurrent_games": min(args.concurrent_games, args.games),
        "candidate_inference": {
            "action_value_weight": candidate_action_value_weight,
            "preview": candidate_preview,
        },
        "champion_inference": {
            "action_value_weight": champion_action_value_weight,
            "preview": champion_preview,
        },
        "promoted": promoted,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    install_safe_poke_env_shutdown()
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
