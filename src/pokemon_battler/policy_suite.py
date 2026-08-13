from __future__ import annotations

import argparse
import asyncio
import json
import random
from argparse import Namespace
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pokemon_battler.external_opponents import ExternalOpponentProcess
from pokemon_battler.live_eval import DEFAULT_TEAM, PLAYER_USERNAME, _run_battles
from pokemon_battler.poke_env_compat import install_safe_poke_env_shutdown
from pokemon_battler.showdown_server import LocalShowdownServer
from pokemon_battler.team_pool import ShuffledTeamPool, resolve_team_pool


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
    parser.add_argument("--promotion-margin", type=float, default=0.0)
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


def _arguments(checkpoint: Path, team_file: Path, games: int, port: int) -> Namespace:
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
        fail_fast=True,
    )


def _run_policy(
    args: argparse.Namespace,
    checkpoint: Path,
    schedule: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True)
    manager = ExternalOpponentProcess(
        "foul-play",
        opponents_dir=args.opponents_dir,
        output_dir=output_dir,
        team_file=schedule[0],
        battle_format="gen9ou",
        games=args.games,
        server_port=args.server_port,
        bootstrap=not args.no_bootstrap_opponents,
        challenger=PLAYER_USERNAME,
        foul_play_search_time_ms=args.foul_play_search_time_ms,
        foul_play_parallelism=args.foul_play_parallelism,
        foul_play_search_threads=args.foul_play_search_threads,
        foul_play_team_files=schedule,
        capture_teacher_trace=False,
    )
    manager.prepare()
    with manager:
        return asyncio.run(
            _run_battles(
                _arguments(checkpoint, args.team_file, args.games, args.server_port),
                output_dir,
                external_opponent=manager,
            )
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    teams = resolve_team_pool(args.enemy_team_file, args.enemy_team_dir, minimum_teams=3)
    schedule = _schedule(teams, args.games, args.seed)
    args.output_dir.mkdir(parents=True)
    with LocalShowdownServer(
        args.showdown_dir,
        port=args.server_port,
        bootstrap=not args.no_bootstrap_server,
        log_path=args.output_dir / "showdown.log",
    ):
        candidate = _run_policy(args, args.candidate, schedule, args.output_dir / "candidate")
        champion = _run_policy(args, args.champion, schedule, args.output_dir / "champion")
    candidate_results = [int(bool(row["won"])) for row in candidate["battles"]]
    champion_results = [int(bool(row["won"])) for row in champion["battles"]]
    candidate_rate = sum(candidate_results) / max(len(candidate_results), 1)
    champion_rate = sum(champion_results) / max(len(champion_results), 1)
    delta = candidate_rate - champion_rate
    interval = _difference_interval(candidate_results, champion_results, seed=args.seed)
    promoted = delta > args.promotion_margin and interval[0] >= -0.05
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
