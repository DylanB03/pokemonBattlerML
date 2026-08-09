from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from poke_env import (
    LocalhostServerConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    ServerConfiguration,
    SimpleHeuristicsPlayer,
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
        * math.sqrt(
            proportion * (1 - proportion) / games + z * z / (4 * games * games)
        )
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
    parser.add_argument("--opponent", choices=tuple(OPPONENTS), default="heuristic")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--battle-format", default="gen9ou")
    parser.add_argument("--showdown-dir", type=Path, default=Path("data/pokemon-showdown"))
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--no-bootstrap-server", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--server-startup-timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--prompt-format")
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


async def _run_battles(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
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
    )
    server_configuration = ServerConfiguration(
        f"ws://localhost:{args.server_port}/showdown/websocket",
        LocalhostServerConfiguration.authentication_url,
    )
    trace_writer = DecisionTraceWriter(output_dir / "decisions.jsonl")
    player = InteractionPlayer(
        runtime,
        trace_writer=trace_writer,
        fail_fast=args.fail_fast,
        battle_format=args.battle_format,
        max_concurrent_battles=1,
        save_replays=str(output_dir / "replays"),
        server_configuration=server_configuration,
        team=team,
    )
    opponent_class = OPPONENTS[args.opponent]
    opponent = opponent_class(
        battle_format=args.battle_format,
        max_concurrent_battles=1,
        save_replays=False,
        server_configuration=server_configuration,
        team=opponent_team,
    )
    try:
        await player.battle_against(opponent, n_battles=args.games)

        battles = list(player.battles.values())
        wins = sum(battle.won is True for battle in battles)
        losses = sum(battle.lost is True for battle in battles)
        ties = len(battles) - wins - losses
        return {
            "schema": "local-showdown-eval-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(args.checkpoint),
            "model": runtime.model_name,
            "battle_format": args.battle_format,
            "opponent": args.opponent,
            "team_file": str(args.team_file),
            "opponent_team_file": str(args.opponent_team_file or args.team_file),
            "requested_games": args.games,
            "finished_games": len(battles),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": wins / len(battles) if battles else None,
            "win_rate_wilson_95": _wilson_interval(wins, len(battles)),
            "decisions": player.decision_count,
            "fallbacks": player.fallback_count,
            "fallback_rate": (
                player.fallback_count / player.decision_count
                if player.decision_count
                else 0.0
            ),
            "inference_latency_seconds": _latency_summary(player.inference_latencies),
            "lead_policy": "submitted-team-order-slot-1",
            "battles": [
                {
                    "battle_id": battle.battle_tag,
                    "won": battle.won,
                    "lost": battle.lost,
                    "turns": battle.turn,
                    "opponent": battle.opponent_username,
                }
                for battle in battles
            ],
        }
    finally:
        await close_poke_env_clients(player.ps_client, opponent.ps_client)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.games <= 0:
        raise ValueError("--games must be positive")
    if not 1 <= args.server_port <= 65535:
        raise ValueError("--server-port must be between 1 and 65535")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
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
    with LocalShowdownServer(
        args.showdown_dir,
        port=args.server_port,
        bootstrap=not args.no_bootstrap_server,
        startup_timeout=args.server_startup_timeout,
        log_path=server_log,
        stop_on_exit=not args.keep_server,
    ):
        summary = asyncio.run(_run_battles(args, output_dir))
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
