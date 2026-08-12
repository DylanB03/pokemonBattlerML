from __future__ import annotations

import argparse
import asyncio
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

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
from pokemon_battler.poke_env_compat import (
    close_poke_env_clients,
    install_safe_poke_env_shutdown,
)
from pokemon_battler.showdown_server import LocalShowdownServer
from pokemon_battler.team_pool import ShuffledTeamPool, resolve_team_pool

TEACHER_USERNAME = "PBFoulPlay"
ENEMY_USERNAME = "PBTeacherEnemy"
ENEMY_POLICIES = {
    "random": RandomPlayer,
    "max-power": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
}


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
    parser.add_argument(
        "--enemy-policy", choices=tuple(ENEMY_POLICIES), default="heuristic"
    )
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


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
        await enemy.accept_challenges(TEACHER_USERNAME, args.games)
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
            "teacher_win_rate_wilson_95": _wilson_interval(
                teacher_wins, len(battles)
            ),
            "teacher_examples": teacher_examples,
            "teacher_trace": str(trace_path),
            "battles": battle_results,
        }
    finally:
        await close_poke_env_clients(enemy.ps_client)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.games <= 0:
        raise ValueError("--games must be positive")
    if not 1 <= args.server_port <= 65535:
        raise ValueError("--server-port must be between 1 and 65535")
    positive_search = {
        "--opponent-startup-timeout": args.opponent_startup_timeout,
        "--server-startup-timeout": args.server_startup_timeout,
        "--foul-play-search-time-ms": args.foul_play_search_time_ms,
        "--foul-play-parallelism": args.foul_play_parallelism,
        "--foul-play-search-threads": args.foul_play_search_threads,
    }
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
        challenger=ENEMY_USERNAME,
        foul_play_search_time_ms=args.foul_play_search_time_ms,
        foul_play_parallelism=args.foul_play_parallelism,
        foul_play_search_threads=args.foul_play_search_threads,
    )
    manager.prepare()
    with LocalShowdownServer(
        args.showdown_dir,
        port=args.server_port,
        bootstrap=not args.no_bootstrap_server,
        startup_timeout=args.server_startup_timeout,
        log_path=args.output_dir / "showdown.log",
        stop_on_exit=not args.keep_server,
    ):
        with manager:
            summary = asyncio.run(_collect(args, args.output_dir, manager, enemy_pool))
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
