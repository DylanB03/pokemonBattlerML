from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
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

from pokemon_battler.external_opponents import (
    EXTERNAL_OPPONENTS,
    ExternalOpponentProcess,
)
from pokemon_battler.live_policy import (
    DecisionTraceWriter,
    DeterministicPreviewMixin,
    InteractionPlayer,
    InteractionPolicyRuntime,
)
from pokemon_battler.poke_env_compat import (
    close_poke_env_clients,
    install_safe_poke_env_shutdown,
)
from pokemon_battler.showdown_server import LocalShowdownServer

DEFAULT_CHECKPOINT = Path("outputs/interaction-v1-1epoch/policy/final")
DEFAULT_TEAM = Path("examples/teams/gen9ou-balance.txt")


class DeterministicRandomPlayer(DeterministicPreviewMixin, RandomPlayer):
    pass


class DeterministicMaxBasePowerPlayer(DeterministicPreviewMixin, MaxBasePowerPlayer):
    pass


class DeterministicSimpleHeuristicsPlayer(
    DeterministicPreviewMixin,
    SimpleHeuristicsPlayer,
):
    pass


OPPONENTS = {
    "random": DeterministicRandomPlayer,
    "max-power": DeterministicMaxBasePowerPlayer,
    "heuristic": DeterministicSimpleHeuristicsPlayer,
}
ALL_OPPONENTS = (*OPPONENTS, *EXTERNAL_OPPONENTS)
PLAYER_USERNAME = "PBPolicy"


def _read_team(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Showdown team file does not exist: {path}")
    team = path.read_text(encoding="utf-8").strip()
    if not team:
        raise ValueError(f"Showdown team file is empty: {path}")
    return team


def _wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> list[float]:
    if games <= 0:
        return [0.0, 0.0]
    proportion = wins / games
    denominator = 1 + z * z / games
    center = (proportion + z * z / (2 * games)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / games + z * z / (4 * games * games))
        / denominator
    )
    return [max(center - margin, 0.0), min(center + margin, 1.0)]


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(math.ceil(0.95 * len(ordered)) - 1, len(ordered) - 1)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the trained interaction policy through complete battles on a local "
            "Pokémon Showdown server."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model")
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--opponent-team-file", type=Path)
    parser.add_argument("--opponent", choices=ALL_OPPONENTS, default="heuristic")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--battle-format", default="gen9ou")
    parser.add_argument("--showdown-dir", type=Path, default=Path("data/pokemon-showdown"))
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--no-bootstrap-server", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--server-startup-timeout", type=float, default=60.0)
    parser.add_argument("--opponents-dir", type=Path, default=Path("data/opponents"))
    parser.add_argument("--no-bootstrap-opponents", action="store_true")
    parser.add_argument("--opponent-startup-timeout", type=float, default=90.0)
    parser.add_argument("--foul-play-search-time-ms", type=int, default=100)
    parser.add_argument("--foul-play-parallelism", type=int, default=1)
    parser.add_argument("--foul-play-search-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--prompt-format")
    parser.add_argument(
        "--action-value-weight",
        type=float,
        help=(
            "Override the checkpoint's deployment Q-logit blend. Use 0 for the "
            "policy-only ablation."
        ),
    )
    parser.add_argument(
        "--load-preview-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load a learned lead head when the checkpoint contains one.",
    )
    parser.add_argument(
        "--team-preview-policy",
        choices=("learned", "first", "random"),
        default="learned",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


async def _run_battles(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    external_opponent: ExternalOpponentProcess | Sequence[ExternalOpponentProcess] | None = None,
) -> dict[str, Any]:
    external_opponents = (
        []
        if external_opponent is None
        else [external_opponent]
        if isinstance(external_opponent, ExternalOpponentProcess)
        else list(external_opponent)
    )
    team = _read_team(args.team_file)
    opponent_team = _read_team(args.opponent_team_file or args.team_file)
    runtime = InteractionPolicyRuntime(
        args.checkpoint,
        model_name=args.model,
        max_length=args.max_length,
        prompt_format=args.prompt_format,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
        action_value_weight=getattr(args, "action_value_weight", None),
        load_preview_head=getattr(args, "load_preview_head", True),
    )
    server_configuration = ServerConfiguration(
        f"ws://localhost:{args.server_port}/showdown/websocket",
        LocalhostServerConfiguration.authentication_url,
    )
    trace_writer = DecisionTraceWriter(output_dir / "decisions.jsonl")
    player = InteractionPlayer(
        runtime,
        account_configuration=AccountConfiguration(PLAYER_USERNAME, None),
        trace_writer=trace_writer,
        fail_fast=args.fail_fast,
        team_preview_policy=getattr(args, "team_preview_policy", "learned"),
        battle_format=args.battle_format,
        max_concurrent_battles=max(len(external_opponents), 1),
        save_replays=str(output_dir / "replays"),
        server_configuration=server_configuration,
        team=team,
    )
    opponent = None
    if not external_opponents:
        opponent_class = OPPONENTS[args.opponent]
        opponent = opponent_class(
            battle_format=args.battle_format,
            max_concurrent_battles=1,
            save_replays=False,
            server_configuration=server_configuration,
            team=opponent_team,
        )
    try:
        if not external_opponents:
            assert opponent is not None
            await player.battle_against(opponent, n_battles=args.games)
        else:
            if all(manager.challenges_player for manager in external_opponents):
                deadline = asyncio.get_running_loop().time() + 10.0
                while not player.ps_client.logged_in.is_set():
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError(f"{PLAYER_USERNAME} did not log in within 10 seconds")
                    await asyncio.sleep(0.05)
                for manager in external_opponents:
                    manager.start_challenges()
                await player.accept_challenges(
                    [manager.username for manager in external_opponents],
                    args.games,
                )
            else:
                if len(external_opponents) != 1:
                    raise ValueError("Concurrent external opponents must initiate their challenges")
                await player.send_challenges(external_opponents[0].username, args.games)
            for manager in external_opponents:
                manager.ensure_success()

        battles = list(player.battles.values())
        wins = sum(battle.won is True for battle in battles)
        losses = sum(battle.lost is True for battle in battles)
        ties = len(battles) - wins - losses
        foul_play_trace_examples = None
        if external_opponents and all(
            manager.spec.name == "foul-play" for manager in external_opponents
        ):
            foul_play_trace_examples = 0
            for manager in external_opponents:
                trace_path = manager.teacher_trace_path
                if trace_path.is_file():
                    with trace_path.open(encoding="utf-8") as stream:
                        foul_play_trace_examples += sum(1 for line in stream if line.strip())
        schedules = {
            manager.username: list(
                manager.foul_play_team_files or [manager.team_file] * manager.games
            )
            for manager in external_opponents
        }
        opponent_game_counts: dict[str, int] = {}
        battle_rows = []
        for battle in battles:
            opponent_name = str(battle.opponent_username)
            game_index = opponent_game_counts.get(opponent_name, 0)
            opponent_game_counts[opponent_name] = game_index + 1
            schedule = schedules.get(opponent_name, [])
            battle_rows.append(
                {
                    "battle_id": battle.battle_tag,
                    "won": battle.won,
                    "lost": battle.lost,
                    "turns": battle.turn,
                    "opponent": opponent_name,
                    "opponent_game_index": game_index,
                    "scheduled_enemy_team_file": (
                        str(schedule[game_index]) if game_index < len(schedule) else None
                    ),
                }
            )
        return {
            "schema": "local-showdown-eval-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(args.checkpoint),
            "model": runtime.model_name,
            "battle_format": args.battle_format,
            "opponent": args.opponent,
            "opponent_implementation": (
                (
                    external_opponents[0].metadata()
                    if len(external_opponents) == 1
                    else [manager.metadata() for manager in external_opponents]
                )
                if external_opponents
                else None
            ),
            "team_file": str(args.team_file),
            "opponent_team_file": str(args.opponent_team_file or args.team_file),
            "requested_games": args.games,
            "finished_games": len(battles),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": wins / len(battles) if battles else None,
            "foul_play_trace_examples": foul_play_trace_examples,
            "win_rate_wilson_95": _wilson_interval(wins, len(battles)),
            "decisions": player.decision_count,
            "fallbacks": player.fallback_count,
            "fallback_rate": (
                player.fallback_count / player.decision_count if player.decision_count else 0.0
            ),
            "inference_latency_seconds": _latency_summary(player.inference_latencies),
            "lead_policy": (
                "learned-team-preview-head"
                if runtime.preview_head is not None
                else "submitted-team-order-slot-1-fallback"
            ),
            "opponent_lead_policy": (
                external_opponents[0].metadata()["team_preview"]
                if external_opponents
                else "submitted-team-order-slot-1"
            ),
            "concurrent_games": max(len(external_opponents), 1),
            "battles": battle_rows,
        }
    finally:
        clients = [player.ps_client]
        if opponent is not None:
            clients.append(opponent.ps_client)
        await close_poke_env_clients(*clients)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.games <= 0:
        raise ValueError("--games must be positive")
    if not 1 <= args.server_port <= 65535:
        raise ValueError("--server-port must be between 1 and 65535")
    if args.opponent_startup_timeout <= 0:
        raise ValueError("--opponent-startup-timeout must be positive")
    if args.foul_play_search_time_ms <= 0:
        raise ValueError("--foul-play-search-time-ms must be positive")
    if args.foul_play_parallelism <= 0:
        raise ValueError("--foul-play-parallelism must be positive")
    if args.foul_play_search_threads <= 0:
        raise ValueError("--foul-play-search-threads must be positive")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or Path("reports/live") / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    server_log = output_dir / "showdown.log"
    external_opponent = (
        ExternalOpponentProcess(
            args.opponent,
            opponents_dir=args.opponents_dir,
            output_dir=output_dir,
            team_file=args.opponent_team_file or args.team_file,
            battle_format=args.battle_format,
            games=args.games,
            server_port=args.server_port,
            bootstrap=not args.no_bootstrap_opponents,
            startup_timeout=args.opponent_startup_timeout,
            challenger=PLAYER_USERNAME,
            foul_play_search_time_ms=args.foul_play_search_time_ms,
            foul_play_parallelism=args.foul_play_parallelism,
            foul_play_search_threads=args.foul_play_search_threads,
        )
        if args.opponent in EXTERNAL_OPPONENTS
        else None
    )
    if external_opponent is not None:
        external_opponent.prepare()

    with LocalShowdownServer(
        args.showdown_dir,
        port=args.server_port,
        bootstrap=not args.no_bootstrap_server,
        startup_timeout=args.server_startup_timeout,
        log_path=server_log,
        stop_on_exit=not args.keep_server,
    ):
        if external_opponent is None:
            summary = asyncio.run(_run_battles(args, output_dir))
        else:
            with external_opponent:
                summary = asyncio.run(
                    _run_battles(
                        args,
                        output_dir,
                        external_opponent=external_opponent,
                    )
                )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved live evaluation to {output_dir}")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    install_safe_poke_env_shutdown()
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
