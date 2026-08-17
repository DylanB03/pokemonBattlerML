from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from dotenv import dotenv_values
from poke_env import AccountConfiguration, RandomPlayer, ShowdownServerConfiguration

from pokemon_battler.league import QwenLeague
from pokemon_battler.live_eval import (
    DEFAULT_TEAM,
    _latency_summary,
    _read_team,
    _wilson_interval,
)
from pokemon_battler.live_policy import (
    DecisionTraceWriter,
    InteractionPlayer,
    InteractionPolicyRuntime,
)
from pokemon_battler.poke_env_compat import (
    close_poke_env_clients,
    install_safe_poke_env_shutdown,
)
from pokemon_battler.reinforcement import WinTrajectoryBuffer
from pokemon_battler.rl_training import train_ppo_rollouts
from pokemon_battler.win_experiment import _promotion_evaluation

PUBLIC_SCHEMA = "public-showdown-session-v2"
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_CHECKPOINT_POINTER = Path("outputs/qwen-win-pilot-1/selected_checkpoint.txt")
USERNAME_KEY = "POKEMON_SHOWDOWN_USERNAME"
PASSWORD_KEY = "POKEMON_SHOWDOWN_PASSWORD"
OPPONENT_KEY = "POKEMON_SHOWDOWN_OPPONENT"
PUBLIC_MODES = ("login", "accept", "challenge", "ladder")


@dataclass(frozen=True)
class PublicEnvironment:
    account: AccountConfiguration
    opponent: str | None


@dataclass
class PublicBattleProgress:
    """Print one compact cumulative result line per completed public battle."""

    requested_games: int
    completed_games: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    seen_battle_ids: set[str] = field(default_factory=set)

    def record(self, event: Mapping[str, Any]) -> None:
        battle_id = str(event["battle_id"])
        if battle_id in self.seen_battle_ids:
            return
        self.seen_battle_ids.add(battle_id)
        self.completed_games += 1
        if event["won"]:
            self.wins += 1
            result = "WIN"
        elif event["lost"]:
            self.losses += 1
            result = "LOSS"
        else:
            self.ties += 1
            result = "TIE"

        opponent = event.get("opponent") or "unknown opponent"
        turns = event.get("turns")
        turn_text = f" | turns {turns}" if turns is not None else ""
        rating_update = event.get("rating_update")
        rating_text = (
            f" | ELO {rating_update['before']}->{rating_update['after']} "
            f"({int(rating_update['change']):+d})"
            if rating_update is not None
            else ""
        )
        win_rate = self.wins / self.completed_games
        print(
            f"[public {self.completed_games}/{self.requested_games}] {result} "
            f"vs {opponent} | record {self.wins}-{self.losses}-{self.ties} | "
            f"win rate {win_rate:.1%}{turn_text}{rating_text}",
            flush=True,
        )


def load_public_environment(
    env_file: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> PublicEnvironment:
    """Load public credentials without placing their values in run artifacts."""
    source = Path(env_file)
    if not source.is_file():
        raise FileNotFoundError(
            f"Public-play environment file does not exist: {source}. "
            "Copy .env.example to .env and fill in the registered account."
        )
    values = {
        key: value
        for key, value in dotenv_values(source).items()
        if value is not None
    }
    process_environment = os.environ if environ is None else environ
    for key in (USERNAME_KEY, PASSWORD_KEY, OPPONENT_KEY):
        if key in process_environment:
            values[key] = process_environment[key]

    username = str(values.get(USERNAME_KEY, "")).strip()
    password = str(values.get(PASSWORD_KEY, ""))
    opponent = str(values.get(OPPONENT_KEY, "")).strip() or None
    missing = [
        key
        for key, value in ((USERNAME_KEY, username), (PASSWORD_KEY, password))
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required value(s) in {source}: {', '.join(missing)}")
    return PublicEnvironment(
        account=AccountConfiguration(username, password),
        opponent=opponent,
    )


def resolve_checkpoint(checkpoint: Path | None) -> Path:
    """Resolve either a checkpoint directory or a selected_checkpoint.txt pointer."""
    source = checkpoint or DEFAULT_CHECKPOINT_POINTER
    source = source.expanduser()
    if source.is_file():
        target_text = source.read_text(encoding="utf-8").strip()
        if not target_text:
            raise ValueError(f"Checkpoint pointer is empty: {source}")
        target = Path(target_text).expanduser()
        if not target.is_absolute():
            from_working_directory = target
            from_pointer_directory = source.parent / target
            target = (
                from_working_directory
                if from_working_directory.exists()
                else from_pointer_directory
            )
    else:
        target = source
    if not target.is_dir():
        raise FileNotFoundError(
            f"Public policy checkpoint does not exist: {target}. Pass --checkpoint "
            "a checkpoint directory or selected_checkpoint.txt file."
        )
    return target.resolve()


def _release_models() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


async def _wait_for_login(player: Any, timeout: float) -> None:
    async def wait_on_client_loop() -> None:
        await player.ps_client.logged_in.wait()

    concurrent = asyncio.run_coroutine_threadsafe(
        wait_on_client_loop(), player.ps_client.loop
    )
    try:
        await asyncio.wait_for(asyncio.wrap_future(concurrent), timeout=timeout)
    except TimeoutError as exc:
        concurrent.cancel()
        raise TimeoutError(
            f"Pokémon Showdown login for {player.username!r} did not complete "
            f"within {timeout:g} seconds. Check the .env credentials and account name."
        ) from exc


async def _run_matchmaking(
    player: InteractionPlayer,
    *,
    mode: str,
    opponent: str | None,
    games: int,
) -> None:
    if mode == "login":
        return
    if mode == "accept":
        assert opponent is not None
        operation = player.accept_challenges(opponent, games)
    elif mode == "challenge":
        assert opponent is not None
        operation = player.send_challenges(opponent, games)
    elif mode == "ladder":
        operation = player.ladder(games)
    else:  # pragma: no cover - protected by argparse and validation
        raise ValueError(f"Unknown public mode: {mode}")
    # A wall-clock deadline here can cancel poke-env's matchmaking coroutine while
    # a rated battle is active. Let the finite game-count operation finish so the
    # client never abandons a battle merely because the overall run took too long.
    await operation


def _rating_summary(battles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    finished_games = sum(bool(battle.get("finished")) for battle in battles)
    updates = [
        battle["rating_update"]
        for battle in battles
        if battle.get("finished") and battle.get("rating_update") is not None
    ]
    changes = [int(update["change"]) for update in updates]
    rated_wins = sum(update.get("result") == "winning" for update in updates)
    rated_losses = sum(update.get("result") == "losing" for update in updates)
    rated_ties = sum(update.get("result") == "tying" for update in updates)
    ratings = [
        rating
        for update in updates
        for rating in (int(update["before"]), int(update["after"]))
    ]
    start_elo = int(updates[0]["before"]) if updates else None
    end_elo = int(updates[-1]["after"]) if updates else None
    tracked_net_change = sum(changes)
    start_to_end_change = (
        end_elo - start_elo
        if start_elo is not None and end_elo is not None
        else None
    )
    return {
        "available": bool(updates),
        "rated_games": len(updates),
        "missing_finished_games": finished_games - len(updates),
        "complete": len(updates) == finished_games,
        "wins": rated_wins,
        "losses": rated_losses,
        "ties": rated_ties,
        "win_rate": rated_wins / len(updates) if updates else None,
        "start_elo": start_elo,
        "end_elo": end_elo,
        "net_change": tracked_net_change if updates else None,
        "start_to_end_change": start_to_end_change,
        "untracked_change": (
            start_to_end_change - tracked_net_change
            if start_to_end_change is not None
            else None
        ),
        "elo_gained": sum(change for change in changes if change > 0),
        "elo_lost": sum(change for change in changes if change < 0),
        "peak_elo": max(ratings) if ratings else None,
        "minimum_elo": min(ratings) if ratings else None,
    }


def _public_summary(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    player: InteractionPlayer,
    rollout: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    battles = list(player.battles.values())
    finished_battles = [battle for battle in battles if battle.finished]
    wins = sum(battle.won is True for battle in finished_battles)
    losses = sum(battle.lost is True for battle in finished_battles)
    ties = len(finished_battles) - wins - losses
    rating_updates = getattr(player, "rating_updates", {})
    battle_results = [
        {
            "battle_id": battle.battle_tag,
            "finished": battle.finished,
            "won": battle.won,
            "lost": battle.lost,
            "turns": battle.turn,
            "opponent": battle.opponent_username,
            # poke-env exposes the pre-battle ladder rating here. The exact
            # Showdown old -> new transition is stored separately below.
            "rating": battle.rating,
            "opponent_rating": battle.opponent_rating,
            "rating_update": rating_updates.get(battle.battle_tag),
        }
        for battle in battles
    ]
    return {
        "schema": PUBLIC_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "account": player.username,
        "checkpoint": str(checkpoint),
        "battle_format": args.battle_format,
        "team_file": str(args.team_file),
        "requested_games": 0 if args.mode == "login" else args.games,
        "started_games": len(battles),
        "finished_games": len(finished_battles),
        "unfinished_games": len(battles) - len(finished_battles),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / len(finished_battles) if finished_battles else None,
        "win_rate_wilson_95": _wilson_interval(wins, len(finished_battles)),
        "rating": _rating_summary(battle_results),
        "decisions": player.decision_count,
        "fallbacks": player.fallback_count,
        "fallback_rate": (
            player.fallback_count / player.decision_count
            if player.decision_count
            else 0.0
        ),
        "sample_actions": args.sample_actions,
        "sampling_temperature": args.sampling_temperature,
        "team_preview_policy": args.team_preview,
        "inference_latency_seconds": _latency_summary(player.inference_latencies),
        "rollout": rollout,
        "error": error,
        "battles": battle_results,
    }


def _public_summary_from_trace(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    account: str,
    trace_path: Path,
    rollout: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Rebuild a frozen public summary from its append-only decision trace."""
    battles: dict[str, dict[str, Any]] = {}
    decisions = 0
    fallbacks = 0
    latencies: list[float] = []
    if trace_path.is_file():
        with trace_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {trace_path} at line {line_number}"
                    ) from exc
                battle_id = str(row.get("battle_id") or "")
                if not battle_id:
                    raise ValueError(
                        f"Trace row {line_number} in {trace_path} has no battle_id"
                    )
                battle = battles.setdefault(
                    battle_id,
                    {
                        "battle_id": battle_id,
                        "finished": False,
                        "won": None,
                        "lost": None,
                        "turns": None,
                        "opponent": None,
                        "rating": None,
                        "opponent_rating": None,
                        "rating_update": None,
                    },
                )
                if row.get("event") == "decision":
                    decisions += 1
                    fallbacks += row.get("fallback_reason") is not None
                    prediction = row.get("prediction") or {}
                    latency = prediction.get("latency_seconds")
                    if latency is not None:
                        latencies.append(float(latency))
                    turn = row.get("showdown_turn")
                    if turn is not None:
                        battle["turns"] = max(int(turn), int(battle["turns"] or 0))
                elif row.get("event") == "battle_finished":
                    battle.update(
                        {
                            "finished": True,
                            "won": bool(row.get("won")),
                            "lost": bool(row.get("lost")),
                            "turns": row.get("turns"),
                            "opponent": row.get("opponent"),
                            "rating": row.get("rating"),
                            "opponent_rating": row.get("opponent_rating"),
                            "rating_update": row.get("rating_update"),
                        }
                    )
    battle_results = list(battles.values())
    finished_battles = [battle for battle in battle_results if battle["finished"]]
    wins = sum(battle["won"] is True for battle in finished_battles)
    losses = sum(battle["lost"] is True for battle in finished_battles)
    ties = len(finished_battles) - wins - losses
    return {
        "schema": PUBLIC_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "account": account,
        "checkpoint": str(checkpoint),
        "battle_format": args.battle_format,
        "team_file": str(args.team_file),
        "requested_games": args.games,
        "started_games": len(battle_results),
        "finished_games": len(finished_battles),
        "unfinished_games": len(battle_results) - len(finished_battles),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / len(finished_battles) if finished_battles else None,
        "win_rate_wilson_95": _wilson_interval(wins, len(finished_battles)),
        "rating": _rating_summary(battle_results),
        "decisions": decisions,
        "fallbacks": fallbacks,
        "fallback_rate": fallbacks / decisions if decisions else 0.0,
        "sample_actions": args.sample_actions,
        "sampling_temperature": args.sampling_temperature,
        "team_preview_policy": args.team_preview,
        "inference_latency_seconds": _latency_summary(latencies),
        "rollout": dict(rollout or {}),
        "error": error,
        "recovered_from_trace": True,
        "battles": battle_results,
    }


def _print_public_summary(summary: Mapping[str, Any]) -> None:
    finished = int(summary["finished_games"])
    requested = int(summary["requested_games"])
    wins = int(summary["wins"])
    losses = int(summary["losses"])
    ties = int(summary["ties"])
    win_rate = wins / finished if finished else 0.0
    print(
        f"[public summary] completed {finished}/{requested} | "
        f"record {wins}-{losses}-{ties} | win rate {win_rate:.1%} | "
        f"fallbacks {summary['fallbacks']} | unfinished {summary['unfinished_games']}",
        flush=True,
    )
    rating = summary.get("rating") or {}
    if rating.get("available"):
        print(
            f"[public ELO] {rating['start_elo']} -> {rating['end_elo']} | "
            f"net {int(rating['net_change']):+d} | gained "
            f"{int(rating['elo_gained']):+d} | lost {int(rating['elo_lost']):+d} | "
            f"peak {rating['peak_elo']} | low {rating['minimum_elo']} | "
            f"rated games {rating['rated_games']}/{finished}",
            flush=True,
        )
    else:
        print(
            f"[public ELO] unavailable | rated updates 0/{finished} "
            "(challenge games may be unrated)",
            flush=True,
        )


def _format_duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    minutes, remaining_seconds = divmod(rounded, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes}m {remaining_seconds}s"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def _result_score(summary: Mapping[str, Any]) -> float:
    games = int(summary["finished_games"])
    if not games:
        return 0.0
    return (int(summary["wins"]) + 0.5 * int(summary["ties"])) / games


def _campaign_summary(
    args: argparse.Namespace,
    batches: Sequence[Mapping[str, Any]],
    *,
    initial_checkpoint: Path,
    selected_checkpoint: Path,
    stop_reason: str,
) -> dict[str, Any]:
    public = [batch["public"] for batch in batches]
    finished = sum(int(item["finished_games"]) for item in public)
    wins = sum(int(item["wins"]) for item in public)
    losses = sum(int(item["losses"]) for item in public)
    ties = sum(int(item["ties"]) for item in public)
    decisions = sum(int(item["decisions"]) for item in public)
    fallbacks = sum(int(item["fallbacks"]) for item in public)
    rating = _rating_summary(
        [battle for item in public for battle in item.get("battles", [])]
    )
    promotions = [
        batch["promotion"] for batch in batches if batch.get("promotion") is not None
    ]
    training = [
        batch["training"] for batch in batches if batch.get("training") is not None
    ]
    promotion_games = sum(
        int(item["wins"]) + int(item["losses"]) + int(item["ties"])
        for item in promotions
    )
    promotion_wins = sum(int(item["wins"]) for item in promotions)
    promotion_ties = sum(int(item["ties"]) for item in promotions)
    batch_scores = [_result_score(item) for item in public]
    positive_reached = any(
        bool(batch.get("stop_condition", {}).get("reached")) for batch in batches
    )
    promoted = sum(
        bool((batch.get("league_result") or {}).get("promoted")) for batch in batches
    )
    public_score = (wins + 0.5 * ties) / finished if finished else 0.0
    promotion_score = (
        (promotion_wins + 0.5 * promotion_ties) / promotion_games
        if promotion_games
        else None
    )
    latest_public_checkpoint = (
        str(Path(batches[-1]["source_checkpoint"]).resolve()) if batches else None
    )
    selected_checkpoint_text = str(selected_checkpoint.resolve())
    return {
        "schema": "public-showdown-campaign-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stop_reason": stop_reason,
        "stop_win_rate": args.stop_win_rate,
        "positive_win_rate_reached": positive_reached,
        "planned_batches": args.batches,
        "completed_batches": len(batches),
        "games_per_batch": args.games,
        "maximum_public_games": args.games * args.batches,
        "public": {
            "finished_games": finished,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "score": public_score,
            "win_rate": wins / finished if finished else 0.0,
            "win_rate_wilson_95": _wilson_interval(wins, finished),
            "decisions": decisions,
            "fallbacks": fallbacks,
            "fallback_rate": fallbacks / decisions if decisions else 0.0,
            "rating": rating,
        },
        "model_improvement": {
            "initial_checkpoint": str(initial_checkpoint.resolve()),
            "selected_checkpoint": selected_checkpoint_text,
            "latest_public_checkpoint": latest_public_checkpoint,
            "selected_checkpoint_has_public_batch": (
                selected_checkpoint_text == latest_public_checkpoint
            ),
            "ppo_candidates_trained": len(training),
            "ppo_candidates_promoted": promoted,
            "ppo_candidates_rejected": len(training) - promoted,
            "ppo_updates": sum(int(item.get("updates", 0)) for item in training),
            "promotion_games": promotion_games,
            "promotion_wins": promotion_wins,
            "promotion_losses": sum(int(item["losses"]) for item in promotions),
            "promotion_ties": promotion_ties,
            "promotion_score": promotion_score,
            "first_public_batch_score": batch_scores[0] if batch_scores else None,
            "latest_public_batch_score": batch_scores[-1] if batch_scores else None,
            "public_batch_score_change": (
                batch_scores[-1] - batch_scores[0] if len(batch_scores) > 1 else None
            ),
            "checkpoint_chain": [
                str(initial_checkpoint.resolve()),
                *[
                    str(Path(batch["selected_checkpoint_after_batch"]).resolve())
                    for batch in batches
                    if (batch.get("league_result") or {}).get("promoted")
                ],
            ],
        },
        "batch_results": [
            {
                "batch": int(batch["batch"]),
                "source_checkpoint": batch["source_checkpoint"],
                "selected_checkpoint_after_batch": batch[
                    "selected_checkpoint_after_batch"
                ],
                "wins": int(batch["public"]["wins"]),
                "losses": int(batch["public"]["losses"]),
                "ties": int(batch["public"]["ties"]),
                "score": _result_score(batch["public"]),
                "rating": batch["public"].get("rating"),
                "positive_win_rate": bool(batch["stop_condition"]["reached"]),
                "candidate_trained": batch["training"] is not None,
                "candidate_promoted": bool(
                    (batch.get("league_result") or {}).get("promoted")
                ),
                "promotion_score": (
                    float(batch["promotion"]["score"])
                    if batch["promotion"] is not None
                    else None
                ),
            }
            for batch in batches
        ],
    }


def _print_campaign_summary(summary: Mapping[str, Any]) -> None:
    public = summary["public"]
    improvement = summary["model_improvement"]
    print(
        f"[campaign] {public['finished_games']}/{summary['maximum_public_games']} "
        f"public games | record {public['wins']}-{public['losses']}-{public['ties']} | "
        f"score {float(public['score']):.1%} | fallbacks {public['fallbacks']}",
        flush=True,
    )
    rating = public.get("rating") or {}
    if rating.get("available"):
        print(
            f"[campaign ELO] {rating['start_elo']} -> {rating['end_elo']} | "
            f"bot-game net {int(rating['net_change']):+d} | gained "
            f"{int(rating['elo_gained']):+d} | lost {int(rating['elo_lost']):+d} | "
            f"peak {rating['peak_elo']} | low {rating['minimum_elo']} | "
            f"rated games {rating['rated_games']}/{public['finished_games']}",
            flush=True,
        )
    print(
        f"[campaign models] PPO candidates {improvement['ppo_candidates_trained']} | "
        f"promoted {improvement['ppo_candidates_promoted']} | rejected "
        f"{improvement['ppo_candidates_rejected']} | PPO updates "
        f"{improvement['ppo_updates']} | stop {summary['stop_reason']}",
        flush=True,
    )


async def _play_public_batch(
    args: argparse.Namespace,
    *,
    account: AccountConfiguration,
    opponent: str | None,
    checkpoint: Path,
    output_dir: Path,
    games_to_play: int | None = None,
    resume_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    continuing = resume_summary is not None
    output_dir.mkdir(parents=True, exist_ok=continuing)
    buffer = WinTrajectoryBuffer(gamma=args.gamma, gae_lambda=args.gae_lambda)
    progress = PublicBattleProgress(args.games)
    if resume_summary is not None:
        progress.completed_games = int(resume_summary["finished_games"])
        progress.wins = int(resume_summary["wins"])
        progress.losses = int(resume_summary["losses"])
        progress.ties = int(resume_summary["ties"])
        progress.seen_battle_ids = {
            str(battle["battle_id"])
            for battle in resume_summary.get("battles", [])
            if battle.get("finished")
        }
    session_games = args.games if games_to_play is None else games_to_play
    if session_games <= 0:
        raise ValueError("games_to_play must be positive")

    def receive(event: dict[str, Any]) -> None:
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
            progress.record(event)

    runtime = InteractionPolicyRuntime(
        checkpoint,
        model_name=args.model,
        max_length=args.max_length,
        prompt_format=args.prompt_format,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    player = InteractionPlayer(
        runtime,
        account_configuration=account,
        trace_writer=DecisionTraceWriter(output_dir / "decisions.jsonl"),
        decision_callback=receive,
        fail_fast=args.fail_fast,
        sample_actions=args.sample_actions,
        sampling_temperature=args.sampling_temperature,
        team_preview_policy=args.team_preview,
        battle_format=args.battle_format,
        max_concurrent_battles=args.concurrent_games,
        save_replays=str(output_dir / "replays"),
        server_configuration=ShowdownServerConfiguration,
        start_timer_on_battle_start=args.start_timer,
        team=_read_team(args.team_file),
    )
    session_error: BaseException | None = None
    close_error: Exception | None = None
    try:
        await _wait_for_login(player, args.login_timeout)
        print(
            f"[public] logged in as {player.username} | mode {args.mode} | "
            f"games this session {session_games} | batch progress "
            f"{progress.completed_games}/{args.games}",
            flush=True,
        )
        await _run_matchmaking(
            player,
            mode=args.mode,
            opponent=opponent,
            games=session_games,
        )
    except (Exception, asyncio.CancelledError) as exc:
        session_error = exc
    finally:
        rollout = buffer.write_jsonl(output_dir / "rollouts.jsonl")
        try:
            await close_poke_env_clients(player.ps_client)
        except Exception as exc:
            close_error = exc

    error = session_error or close_error
    summary = _public_summary(
        args,
        checkpoint=checkpoint,
        player=player,
        rollout=rollout,
        error=f"{type(error).__name__}: {error}" if error is not None else None,
    )
    (output_dir / "public_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _print_public_summary(summary)
    if error is not None:
        raise RuntimeError(
            f"Public session failed; partial artifacts were saved to {output_dir}"
        ) from error
    return summary


async def _probe_public_login(
    account: AccountConfiguration,
    *,
    login_timeout: float,
) -> dict[str, Any]:
    """Verify credentials and the public socket without loading the policy model."""
    player = RandomPlayer(
        account_configuration=account,
        battle_format="gen9ou",
        max_concurrent_battles=1,
        server_configuration=ShowdownServerConfiguration,
    )
    error: Exception | None = None
    try:
        await _wait_for_login(player, login_timeout)
    except Exception as exc:
        error = exc
    try:
        await close_poke_env_clients(player.ps_client)
    except Exception as exc:
        if error is None:
            error = exc
    summary = {
        "schema": "public-showdown-login-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "account": player.username,
        "server": ShowdownServerConfiguration.websocket_url,
        "logged_in": error is None,
        "error": f"{type(error).__name__}: {error}" if error is not None else None,
    }
    return summary


def _validate_args(args: argparse.Namespace, opponent: str | None) -> None:
    if args.games <= 0 or args.batches <= 0:
        raise ValueError("--games and --batches must be positive")
    if args.concurrent_games <= 0:
        raise ValueError("--concurrent-games must be positive")
    if args.mode in {"accept", "challenge"} and not opponent:
        raise ValueError(
            f"--mode {args.mode} requires --opponent or {OPPONENT_KEY} in .env"
        )
    if args.mode == "login" and args.learn:
        raise ValueError("--learn cannot be used with --mode login")
    if args.resume and args.output_dir is None:
        raise ValueError("--resume requires an explicit --output-dir")
    if args.resume and (args.learn or args.mode == "login"):
        raise ValueError("--resume currently supports frozen public campaigns only")
    if args.stop_win_rate is not None:
        if not args.learn:
            raise ValueError("--stop-win-rate requires --learn")
        if not 0 <= args.stop_win_rate < 1:
            raise ValueError("--stop-win-rate must be in [0, 1)")
    if args.login_timeout <= 0:
        raise ValueError("--login-timeout must be positive")
    if args.sampling_temperature <= 0:
        raise ValueError("--sampling-temperature must be positive")
    if not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        raise ValueError("--gamma and --gae-lambda must be in [0, 1]")
    if args.learn:
        if not args.sample_actions:
            raise ValueError("--learn requires sampled actions; remove --no-sample-actions")
        if args.sampling_temperature != 1.0:
            raise ValueError(
                "Public PPO currently requires --sampling-temperature 1.0 so saved "
                "and recomputed policy probabilities match exactly"
            )
        positive = {
            "promotion_games": args.promotion_games,
            "ppo_epochs": args.ppo_epochs,
            "ppo_batch_size": args.ppo_batch_size,
            "ppo_gradient_accumulation_steps": args.ppo_gradient_accumulation_steps,
            "concurrent_games": args.concurrent_games,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"These learning arguments must be positive: {', '.join(invalid)}")
        if not 0 <= args.promotion_threshold <= 1:
            raise ValueError("--promotion-threshold must be in [0, 1]")
        if args.games < 16:
            warnings.warn(
                "Fewer than 16 public games is a very noisy PPO batch; use this only "
                "as a wiring test.",
                stacklevel=2,
            )


def _promotion_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        promotion_games=args.promotion_games,
        battle_format=args.battle_format,
        showdown_dir=args.showdown_dir,
        server_port=args.server_port,
        no_bootstrap_server=args.no_bootstrap_server,
        server_startup_timeout=args.server_startup_timeout,
        concurrent_games=args.concurrent_games,
        model=args.model,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )


def _validate_resume_configuration(
    args: argparse.Namespace,
    existing: Mapping[str, Any],
    *,
    checkpoint: Path,
    account: str,
) -> None:
    expected = {
        "mode": args.mode,
        "games": args.games,
        "batches": args.batches,
        "battle_format": args.battle_format,
        "team_preview": args.team_preview,
        "learn": False,
        "account": account,
    }
    mismatches = {
        key: (existing.get(key), value)
        for key, value in expected.items()
        if existing.get(key) != value
    }
    prior_checkpoint = existing.get("initial_checkpoint")
    if prior_checkpoint is None or Path(str(prior_checkpoint)).resolve() != checkpoint.resolve():
        mismatches["initial_checkpoint"] = (prior_checkpoint, str(checkpoint.resolve()))
    prior_team = Path(str(existing.get("team_file", "")))
    if prior_team.resolve() != args.team_file.resolve():
        mismatches["team_file"] = (str(prior_team), str(args.team_file))
    if mismatches:
        details = ", ".join(
            f"{key}: prior={prior!r}, requested={requested!r}"
            for key, (prior, requested) in sorted(mismatches.items())
        )
        raise ValueError(f"Resume arguments do not match the existing campaign: {details}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    public_environment = load_public_environment(args.env_file)
    opponent = args.opponent or public_environment.opponent
    if args.sample_actions is None:
        args.sample_actions = bool(args.learn)
    _validate_args(args, opponent)
    checkpoint = None if args.mode == "login" else resolve_checkpoint(args.checkpoint)
    if args.mode != "login" and not args.team_file.is_file():
        raise FileNotFoundError(args.team_file)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_root = Path("outputs/public-learning" if args.learn else "reports/public")
    output_dir = args.output_dir or default_root / timestamp
    output_has_files = output_dir.exists() and any(output_dir.iterdir())
    if output_has_files and not args.resume:
        raise FileExistsError(
            f"Public output directory is not empty: {output_dir}. Use a new directory "
            "so checkpoints and traces are never overwritten."
        )
    if args.resume and not output_has_files:
        raise FileNotFoundError(
            f"Resume output directory has no existing campaign: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if args.resume:
        if not config_path.is_file():
            raise FileNotFoundError(f"Resume campaign has no run_config.json: {output_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert checkpoint is not None
        _validate_resume_configuration(
            args,
            config,
            checkpoint=checkpoint,
            account=public_environment.account.username,
        )
        config["resume_count"] = int(config.get("resume_count", 0)) + 1
        config["last_resumed_at"] = datetime.now(timezone.utc).isoformat()
        config["concurrent_games"] = args.concurrent_games
    else:
        config = {
            key: _serialize(value)
            for key, value in vars(args).items()
            if key not in {"password"}
        } | {
            "account": public_environment.account.username,
            "opponent": opponent,
            "initial_checkpoint": str(checkpoint) if checkpoint is not None else None,
            "credential_source": str(args.env_file),
        }
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.mode == "login":
        login_summary = asyncio.run(
            _probe_public_login(
                public_environment.account,
                login_timeout=args.login_timeout,
            )
        )
        summary = {
            "schema": "public-showdown-run-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "login",
            "learning_enabled": False,
            "batches": [],
            "login": login_summary,
            "selected_checkpoint": None,
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        status = "success" if login_summary["logged_in"] else "failed"
        print(
            f"[login] {status} | account {login_summary['account']} | "
            f"server {login_summary['server']}",
            flush=True,
        )
        print(f"[run complete] summary {output_dir / 'summary.json'}", flush=True)
        if not login_summary["logged_in"]:
            raise RuntimeError(
                "Public Pokémon Showdown login probe failed; see summary.json"
            )
        return summary

    assert checkpoint is not None

    league: QwenLeague | None = None
    if args.learn:
        league = QwenLeague(output_dir / "league.json")
        league.initialize(checkpoint, entry_id="public-initial")
    initial_checkpoint = checkpoint
    champion_checkpoint = checkpoint
    batches: list[dict[str, Any]] = []
    if args.resume:
        for prior_batch_number in range(1, args.batches + 1):
            prior_path = (
                output_dir
                / f"batch-{prior_batch_number:03d}"
                / "batch_summary.json"
            )
            if not prior_path.is_file():
                break
            batches.append(json.loads(prior_path.read_text(encoding="utf-8")))
    stop_reason = (
        "completed_requested_batches"
        if len(batches) == args.batches
        else "in_progress"
    )
    campaign = _campaign_summary(
        args,
        batches,
        initial_checkpoint=initial_checkpoint,
        selected_checkpoint=champion_checkpoint,
        stop_reason=stop_reason,
    )
    for batch_number in range(1, args.batches + 1):
        if batch_number <= len(batches):
            continue
        batch_dir = output_dir / f"batch-{batch_number:03d}"
        batch_dir.mkdir(parents=True, exist_ok=args.resume)
        source_checkpoint = champion_checkpoint.resolve()
        public_dir = batch_dir / "public"
        trace_path = public_dir / "decisions.jsonl"
        prior_public: dict[str, Any] | None = None
        if args.resume and trace_path.is_file():
            prior_public = _public_summary_from_trace(
                args,
                checkpoint=source_checkpoint,
                account=public_environment.account.username,
                trace_path=trace_path,
            )
        finished_before = int(prior_public["finished_games"]) if prior_public else 0
        if finished_before > args.games:
            raise ValueError(
                f"Batch {batch_number} trace contains {finished_before} completed games, "
                f"more than the configured {args.games}"
            )
        games_remaining = args.games - finished_before
        print(
            f"[batch {batch_number}/{args.batches}] "
            f"{'resuming' if finished_before else 'starting'} public games | "
            f"progress {finished_before}/{args.games} | remaining {games_remaining} | "
            f"checkpoint {source_checkpoint}",
            flush=True,
        )
        session_summary: dict[str, Any] | None = None
        if games_remaining:
            session_summary = asyncio.run(
                _play_public_batch(
                    args,
                    account=public_environment.account,
                    opponent=opponent,
                    checkpoint=source_checkpoint,
                    output_dir=public_dir,
                    games_to_play=games_remaining,
                    resume_summary=prior_public,
                )
            )
        if args.resume:
            public_summary = _public_summary_from_trace(
                args,
                checkpoint=source_checkpoint,
                account=public_environment.account.username,
                trace_path=trace_path,
                rollout=(session_summary or {}).get("rollout"),
            )
            (public_dir / "public_summary.json").write_text(
                json.dumps(public_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _print_public_summary(public_summary)
        else:
            assert session_summary is not None
            public_summary = session_summary
        public_score = _result_score(public_summary)
        previous_score = (
            _result_score(batches[-1]["public"]) if batches else None
        )
        positive_reached = (
            args.stop_win_rate is not None and public_score > args.stop_win_rate
        )
        batch_report: dict[str, Any] = {
            "batch": batch_number,
            "source_checkpoint": str(source_checkpoint),
            "public": public_summary,
            "public_score": public_score,
            "public_score_change_from_previous_batch": (
                public_score - previous_score if previous_score is not None else None
            ),
            "stop_condition": {
                "threshold": args.stop_win_rate,
                "comparison": "strictly_greater",
                "reached": positive_reached,
            },
            "training": None,
            "training_skipped_reason": None,
            "promotion": None,
            "league_result": None,
            "model_improvement": None,
        }
        _release_models()

        if args.learn:
            if public_summary["finished_games"] != args.games:
                raise RuntimeError(
                    "Refusing to train because the public batch did not finish every "
                    "requested game"
                )
            if public_summary["fallbacks"]:
                raise RuntimeError(
                    "Refusing to train on a public batch containing policy fallbacks"
                )
            rollout = public_summary["rollout"]
            if rollout["pending_battles"] or not rollout["decisions"]:
                raise RuntimeError(
                    "Refusing to train on incomplete or empty public trajectories"
                )

        if args.learn and positive_reached:
            batch_report["training_skipped_reason"] = "positive_win_rate_reached"
            print(
                f"[learn] skipped | public score {public_score:.1%} is above "
                f"{args.stop_win_rate:.1%}; preserving the checkpoint that reached "
                "the target",
                flush=True,
            )
        elif args.learn:
            candidate_checkpoint = batch_dir / "candidate"
            print(
                f"[learn] training PPO candidate | decisions {rollout['decisions']} | "
                f"output {candidate_checkpoint}",
                flush=True,
            )
            training = train_ppo_rollouts(
                checkpoint=source_checkpoint,
                rollout_file=batch_dir / "public" / "rollouts.jsonl",
                output_dir=candidate_checkpoint,
                model_name=args.model,
                epochs=args.ppo_epochs,
                batch_size=args.ppo_batch_size,
                gradient_accumulation_steps=args.ppo_gradient_accumulation_steps,
                qwen_learning_rate=args.qwen_learning_rate,
                head_learning_rate=args.head_learning_rate,
                clip_ratio=args.ppo_clip,
                value_clip=args.value_clip,
                value_coefficient=args.value_coefficient,
                entropy_coefficient=args.entropy_coefficient,
                target_kl=args.target_kl,
                dtype_name=args.dtype or "auto",
                load_in_4bit=(
                    True if args.load_in_4bit is None else args.load_in_4bit
                ),
                local_files_only=(
                    True if args.local_files_only is None else args.local_files_only
                ),
                attn_implementation=args.attn_implementation or "sdpa",
                seed=args.seed + batch_number,
                rollout_source="public-showdown",
            )
            batch_report["training"] = training
            print(
                f"[learn] PPO complete | updates {training.get('updates', 'unknown')} | "
                f"approx KL {float(training.get('approximate_kl', 0.0)):.5f} | "
                f"elapsed {_format_duration(float(training.get('elapsed_seconds', 0.0)))}",
                flush=True,
            )
            _release_models()
            print(
                f"[promotion] candidate vs champion | games {args.promotion_games}",
                flush=True,
            )
            promotion = _promotion_evaluation(
                _promotion_args(args),
                candidate_checkpoint=candidate_checkpoint,
                champion_checkpoint=source_checkpoint,
                actor_team=args.team_file,
                opponent_team=args.team_file,
                output_dir=batch_dir / "promotion",
            )
            batch_report["promotion"] = promotion
            assert league is not None
            league_result = league.record_candidate(
                candidate_id=f"public-batch-{batch_number:03d}",
                checkpoint=candidate_checkpoint,
                wins=int(promotion["wins"]),
                losses=int(promotion["losses"]),
                ties=int(promotion["ties"]),
                promotion_threshold=args.promotion_threshold,
            )
            batch_report["league_result"] = league_result
            champion_checkpoint = Path(league.champion["checkpoint"])
            batch_report["model_improvement"] = {
                "source_checkpoint": str(source_checkpoint),
                "candidate_checkpoint": str(candidate_checkpoint.resolve()),
                "training_source_checkpoint": training.get("source_checkpoint"),
                "ppo_updates": int(training.get("updates", 0)),
                "approximate_kl": training.get("approximate_kl"),
                "ppo_policy_loss": training.get("ppo_policy_loss"),
                "ppo_value_loss": training.get("ppo_value_loss"),
                "ppo_total_loss": training.get("ppo_total_loss"),
                "promotion_wins": int(promotion["wins"]),
                "promotion_losses": int(promotion["losses"]),
                "promotion_ties": int(promotion["ties"]),
                "promotion_score": float(promotion["score"]),
                "promotion_threshold": args.promotion_threshold,
                "promoted": bool(league_result["promoted"]),
            }
            print(
                f"[promotion] record {promotion['wins']}-{promotion['losses']}-"
                f"{promotion['ties']} | score {float(promotion['score']):.1%} | "
                f"promoted {'yes' if league_result['promoted'] else 'no'}",
                flush=True,
            )
            _release_models()

        batch_report["selected_checkpoint_after_batch"] = str(
            champion_checkpoint.resolve()
        )
        batches.append(batch_report)
        (batch_dir / "batch_summary.json").write_text(
            json.dumps(batch_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "selected_checkpoint.txt").write_text(
            str(champion_checkpoint.resolve()) + "\n", encoding="utf-8"
        )
        if positive_reached:
            stop_reason = "positive_win_rate"
        elif batch_number == args.batches:
            stop_reason = (
                "maximum_public_games"
                if args.stop_win_rate is not None
                else "completed_requested_batches"
            )
        campaign = _campaign_summary(
            args,
            batches,
            initial_checkpoint=initial_checkpoint,
            selected_checkpoint=champion_checkpoint,
            stop_reason=stop_reason,
        )
        (output_dir / "campaign_summary.json").write_text(
            json.dumps(campaign, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _print_campaign_summary(campaign)
        if positive_reached:
            break

    summary = {
        "schema": "public-showdown-run-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "learning_enabled": args.learn,
        "batches": batches,
        "campaign": campaign,
        "selected_checkpoint": str(champion_checkpoint.resolve()),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[run complete] selected checkpoint {champion_checkpoint.resolve()}", flush=True)
    print(f"[run complete] summary {output_dir / 'summary.json'}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Qwen interaction policy on the public Pokémon Showdown server, "
            "save complete trajectories, and optionally train/promote between batches."
        )
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--mode", choices=PUBLIC_MODES, default="login")
    parser.add_argument("--opponent")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument(
        "--stop-win-rate",
        type=float,
        help=(
            "Stop before another PPO update when a completed public batch's score "
            "(wins plus half of ties) is strictly above this threshold."
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--battle-format", default="gen9ou")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted frozen campaign from its append-only trace.",
    )
    parser.add_argument(
        "--team-preview", choices=("learned", "first", "random"), default="learned"
    )
    parser.add_argument(
        "--sample-actions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults off for evaluation and on with --learn.",
    )
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument(
        "--start-timer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--login-timeout", type=float, default=30.0)
    parser.add_argument("--fail-fast", action="store_true")

    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-batch-size", type=int, default=2)
    parser.add_argument("--ppo-gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.005)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--qwen-learning-rate", type=float, default=2e-6)
    parser.add_argument("--head-learning-rate", type=float, default=2e-5)
    parser.add_argument("--promotion-games", type=int, default=40)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--showdown-dir", type=Path, default=Path("data/pokemon-showdown"))
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-startup-timeout", type=float, default=60.0)
    parser.add_argument("--no-bootstrap-server", action="store_true")
    parser.add_argument("--concurrent-games", type=int, default=2)
    parser.add_argument("--model")
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    install_safe_poke_env_shutdown()
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
