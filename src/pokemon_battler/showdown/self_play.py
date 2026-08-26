from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poke_env import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration

from pokemon_battler.showdown.live_eval import _read_team, _wilson_interval
from pokemon_battler.showdown.live_policy import InteractionPlayer, InteractionPolicyRuntime
from pokemon_battler.showdown.poke_env_compat import close_poke_env_clients
from pokemon_battler.training.reinforcement import WinTrajectoryBuffer
from pokemon_battler.showdown.showdown_server import LocalShowdownServer


def _runtime_options(checkpoint: Path, options: dict[str, Any]) -> InteractionPolicyRuntime:
    return InteractionPolicyRuntime(
        checkpoint,
        model_name=options.get("model_name"),
        max_length=options.get("max_length"),
        dtype=options.get("dtype_name"),
        load_in_4bit=options.get("load_in_4bit"),
        local_files_only=options.get("local_files_only"),
        attn_implementation=options.get("attn_implementation"),
    )


async def _play_qwen_match(
    *,
    actor_checkpoint: Path,
    opponent_checkpoint: Path,
    actor_team: str,
    opponent_team: str,
    games: int,
    battle_format: str,
    server_port: int,
    concurrent_games: int,
    sample_actor: bool,
    sample_opponent: bool,
    temperature: float,
    collect_actor: bool,
    runtime_options: dict[str, Any],
) -> tuple[dict[str, Any], WinTrajectoryBuffer | None]:
    actor_runtime = _runtime_options(actor_checkpoint, runtime_options)
    opponent_runtime = _runtime_options(opponent_checkpoint, runtime_options)
    server_configuration = ServerConfiguration(
        f"ws://localhost:{server_port}/showdown/websocket",
        LocalhostServerConfiguration.authentication_url,
    )
    buffer = WinTrajectoryBuffer() if collect_actor else None

    def receive(event: dict[str, Any]) -> None:
        assert buffer is not None
        if event["event"] == "decision":
            buffer.record_decision(
                event["battle_id"],
                event["observation"],
                action_id=event["action_id"],
                old_log_probability=event["old_log_probability"],
                value_probability=event["value_probability"],
            )
        elif event["event"] == "battle_finished":
            buffer.finish_battle(
                event["battle_id"], won=event["won"], lost=event["lost"]
            )

    actor = InteractionPlayer(
        actor_runtime,
        account_configuration=AccountConfiguration("PBActor", None),
        battle_format=battle_format,
        max_concurrent_battles=concurrent_games,
        save_replays=False,
        server_configuration=server_configuration,
        team=actor_team,
        sample_actions=sample_actor,
        sampling_temperature=temperature,
        decision_callback=receive if collect_actor else None,
        fail_fast=True,
    )
    opponent = InteractionPlayer(
        opponent_runtime,
        account_configuration=AccountConfiguration("PBLeague", None),
        battle_format=battle_format,
        max_concurrent_battles=concurrent_games,
        save_replays=False,
        server_configuration=server_configuration,
        team=opponent_team,
        sample_actions=sample_opponent,
        sampling_temperature=temperature,
        fail_fast=True,
    )
    try:
        await actor.battle_against(opponent, n_battles=games)
        battles = list(actor.battles.values())
        wins = sum(battle.won is True for battle in battles)
        losses = sum(battle.lost is True for battle in battles)
        ties = len(battles) - wins - losses
        return (
            {
                "games": len(battles),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "score": (wins + 0.5 * ties) / len(battles) if battles else None,
                "win_rate_wilson_95": _wilson_interval(wins, len(battles)),
                "actor_decisions": actor.decision_count,
                "actor_fallbacks": actor.fallback_count,
                "opponent_decisions": opponent.decision_count,
                "opponent_fallbacks": opponent.fallback_count,
            },
            buffer,
        )
    finally:
        await close_poke_env_clients(actor.ps_client, opponent.ps_client)


def run_qwen_match(
    *,
    actor_checkpoint: Path,
    opponent_checkpoint: Path,
    actor_team_file: Path,
    opponent_team_file: Path,
    games: int,
    output_dir: Path,
    battle_format: str = "gen9ou",
    showdown_dir: Path = Path("data/pokemon-showdown"),
    server_port: int = 8000,
    bootstrap_server: bool = True,
    server_startup_timeout: float = 60.0,
    keep_server: bool = False,
    concurrent_games: int = 1,
    sample_actor: bool = False,
    sample_opponent: bool = False,
    temperature: float = 1.0,
    collect_actor: bool = False,
    model_name: str | None = None,
    max_length: int | None = None,
    dtype_name: str | None = None,
    load_in_4bit: bool | None = None,
    local_files_only: bool | None = None,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    """Play Qwen directly against a frozen Qwen policy on official Showdown."""
    if games <= 0 or concurrent_games <= 0:
        raise ValueError("games and concurrent_games must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    runtime_options = {
        "model_name": model_name,
        "max_length": max_length,
        "dtype_name": dtype_name,
        "load_in_4bit": load_in_4bit,
        "local_files_only": local_files_only,
        "attn_implementation": attn_implementation,
    }
    with LocalShowdownServer(
        showdown_dir,
        port=server_port,
        bootstrap=bootstrap_server,
        startup_timeout=server_startup_timeout,
        log_path=output_dir / "showdown.log",
        stop_on_exit=not keep_server,
    ):
        match, buffer = asyncio.run(
            _play_qwen_match(
                actor_checkpoint=actor_checkpoint,
                opponent_checkpoint=opponent_checkpoint,
                actor_team=_read_team(actor_team_file),
                opponent_team=_read_team(opponent_team_file),
                games=games,
                battle_format=battle_format,
                server_port=server_port,
                concurrent_games=concurrent_games,
                sample_actor=sample_actor,
                sample_opponent=sample_opponent,
                temperature=temperature,
                collect_actor=collect_actor,
                runtime_options=runtime_options,
            )
        )
    summary = {
        "schema": "qwen-self-play-match-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "actor_checkpoint": str(actor_checkpoint),
        "opponent_checkpoint": str(opponent_checkpoint),
        "actor_team_file": str(actor_team_file),
        "opponent_team_file": str(opponent_team_file),
        "sample_actor": sample_actor,
        "sample_opponent": sample_opponent,
        "temperature": temperature,
        **match,
    }
    if buffer is not None:
        rollout_report = buffer.write_jsonl(output_dir / "rollouts.jsonl")
        if rollout_report["pending_battles"]:
            raise RuntimeError("A self-play battle ended without a terminal callback")
        summary["rollouts"] = rollout_report
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
