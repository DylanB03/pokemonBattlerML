from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pokemon_battler.actions import (
    action_label,
    pp_aware_legal_action_ids,
    recoverable_legal_action_ids,
)
from pokemon_battler.prepare import (
    ReplayMetadata,
    SplitConfig,
    _enrich_state,
    _explicit_legal_actions,
    _format_matches,
    _history_event,
    _history_snapshot,
    _observe_move_history,
    _observe_opponent,
    _observe_rosters,
    _pokemon_key,
    _roster_snapshot,
    choose_split,
    iter_trajectories,
    parse_replay_metadata,
)
from pokemon_battler.trajectory_rewards import (
    shaped_reward_from_event,
    terminal_reward,
)

TRAJECTORY_SCHEMA_VERSION = 4
DEFAULT_REWARD_GAMMA = 0.99
PREVIOUS_ACTION_SENTINEL = 13


def trajectory_id(source_name: str, metadata: ReplayMetadata) -> str:
    """Identify one POV, not merely one two-sided Showdown battle."""
    digest = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:20]
    return f"{metadata.battle_id}:{digest}"


def _trajectory_is_selected(
    battle_id: str,
    *,
    seed: int,
    sample_rate: float,
    sample_offset: float = 0.0,
) -> bool:
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1]")
    if sample_offset < 0 or sample_offset + sample_rate > 1:
        raise ValueError("sample_offset must be non-negative and offset + rate at most one")
    if sample_rate >= 1 and sample_offset == 0:
        return True
    digest = hashlib.sha256(
        f"trajectory-sample:{seed}:{battle_id}".encode("utf-8")
    ).digest()
    unit_interval = int.from_bytes(digest[:8], "big") / float(2**64)
    return sample_offset <= unit_interval < sample_offset + sample_rate


def _status_inflicted(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    side: str,
) -> bool:
    key = f"{side}_active_pokemon"
    before = previous.get(key) or {}
    after = current.get(key) or {}
    if _pokemon_key(before) != _pokemon_key(after):
        return False
    before_status = str(before.get("status") or "").lower()
    after_status = str(after.get("status") or "").lower()
    return before_status in {"", "none", "nostatus"} and after_status not in {
        "",
        "none",
        "nostatus",
        "fnt",
    }


def transition_event(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    event = _history_event(previous, current)
    event["player_status_inflicted"] = _status_inflicted(
        previous, current, side="player"
    )
    event["opponent_status_inflicted"] = _status_inflicted(
        previous, current, side="opponent"
    )
    return event


def _legal_action(
    state: dict[str, Any],
    action_id: int,
    counters: Counter[str],
) -> tuple[list[int], str] | None:
    try:
        recoverable = recoverable_legal_action_ids(state)
        pp_aware = pp_aware_legal_action_ids(state)
    except (KeyError, TypeError, ValueError):
        counters["invalid_states"] += 1
        return None
    if action_id not in recoverable:
        counters["actions_not_recoverably_legal"] += 1
        return None
    explicit = _explicit_legal_actions(state, recoverable)
    if explicit is not None and action_id in explicit:
        counters["exact_legal_masks"] += 1
        return explicit, "exact"
    if action_id in pp_aware and pp_aware != recoverable:
        counters["zero_pp_candidates_removed"] += len(recoverable) - len(pp_aware)
        return pp_aware, "pp-aware"
    return recoverable, "recoverable"


def trajectory_rows(
    source_name: str,
    trajectory: dict[str, Any],
    metadata: ReplayMetadata,
    split: str,
    counters: Counter[str],
    *,
    reward_gamma: float = DEFAULT_REWARD_GAMMA,
) -> list[dict[str, Any]]:
    """Prepare one complete POV as consecutive semi-Markov transitions."""
    states = trajectory.get("states")
    actions = trajectory.get("actions")
    if not isinstance(states, list) or not isinstance(actions, list):
        counters["malformed_trajectories"] += 1
        return []
    if len(states) < 2:
        counters["short_trajectories"] += 1
        return []
    if len(actions) < len(states) - 1:
        counters["length_mismatches"] += 1

    revealed_opponents: dict[str, dict[str, Any]] = {}
    recent_moves: list[dict[str, str]] = []
    player_roster: dict[str, dict[str, Any]] = {}
    opponent_roster: dict[str, dict[str, Any]] = {}
    player_order: list[str] = []
    opponent_order: list[str] = []
    history_events: list[dict[str, Any]] = []
    previous_state: dict[str, Any] | None = None
    prepared: list[dict[str, Any]] = []
    identifier = trajectory_id(source_name, metadata)

    decision_count = min(len(states) - 1, max(len(actions) - 1, 0))
    for raw_index in range(decision_count):
        state = states[raw_index]
        action_id = actions[raw_index]
        if not isinstance(state, dict) or not isinstance(action_id, int):
            counters["malformed_turns"] += 1
            continue
        if previous_state is not None:
            history_events.append(transition_event(previous_state, state))
            del history_events[:-4]
        _observe_rosters(
            state,
            previous_state,
            player_roster,
            player_order,
            opponent_roster,
            opponent_order,
        )
        _observe_opponent(state, revealed_opponents)
        _observe_move_history(state, recent_moves)
        previous_state = state
        if action_id == -1:
            counters["missing_actions"] += 1
            continue
        legal_result = _legal_action(state, action_id, counters)
        if legal_result is None:
            continue
        legal, legal_quality = legal_result
        enriched = _enrich_state(state, raw_index, revealed_opponents, recent_moves)
        enriched["prepared_legal_action_ids"] = legal
        prepared.append(
            {
                "schema_version": TRAJECTORY_SCHEMA_VERSION,
                "trajectory_id": identifier,
                "battle_id": metadata.battle_id,
                "battle_date": (
                    metadata.battle_date.isoformat() if metadata.battle_date else None
                ),
                "rating": metadata.rating,
                "outcome": metadata.outcome,
                "source": source_name,
                "turn_index": raw_index,
                "raw_state_index": raw_index,
                "battle_decision_count": max(len(states) - 1, 1),
                "split": split,
                "state": enriched,
                "player_roster": _roster_snapshot(
                    player_roster,
                    player_order,
                    enriched.get("player_active_pokemon"),
                    "player",
                ),
                "opponent_roster": _roster_snapshot(
                    opponent_roster,
                    opponent_order,
                    enriched.get("opponent_active_pokemon"),
                    "opponent",
                ),
                "history_events": _history_snapshot(history_events),
                "action_id": action_id,
                "target": action_label(action_id),
                "legal_action_ids": legal,
                "legal_mask_quality": legal_quality,
            }
        )

    if not prepared:
        counters["trajectories_without_valid_decisions"] += 1
        return []

    final_state_index = len(states) - 1
    for position, row in enumerate(prepared):
        raw_index = int(row["raw_state_index"])
        has_next = position + 1 < len(prepared)
        next_index = (
            int(prepared[position + 1]["raw_state_index"])
            if has_next
            else final_state_index
        )
        rewards: list[float] = []
        for index in range(raw_index, next_index):
            if index + 1 >= len(states):
                break
            before, after = states[index], states[index + 1]
            if not isinstance(before, dict) or not isinstance(after, dict):
                rewards.append(0.0)
                continue
            rewards.append(shaped_reward_from_event(transition_event(before, after)))
        done = not has_next
        if done:
            if not rewards:
                rewards.append(0.0)
            rewards[-1] += terminal_reward(metadata.outcome)
        discounted_reward = sum(
            (reward_gamma**index) * reward for index, reward in enumerate(rewards)
        )
        row["trajectory_position"] = position
        row["trajectory_length"] = len(prepared)
        row["next_turn_index"] = None if done else prepared[position + 1]["turn_index"]
        row["transition_steps"] = max(next_index - raw_index, 1)
        row["step_rewards"] = rewards
        row["reward_gamma"] = reward_gamma
        row["reward"] = discounted_reward
        row["done"] = done
        row["previous_action_id"] = (
            PREVIOUS_ACTION_SENTINEL
            if position == 0
            else int(prepared[position - 1]["action_id"])
        )
        row["previous_reward"] = (
            0.0 if position == 0 else float(prepared[position - 1]["reward"])
        )
    counters["transitions_written"] += len(prepared)
    counters["terminal_transitions"] += 1
    counters["transition_gap_steps"] += sum(
        int(row["transition_steps"]) for row in prepared
    )
    return prepared


def prepare_trajectory_dataset(
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    split_config: SplitConfig,
    battle_format: str | None = "gen9ou",
    min_rating: int | None = None,
    outcome: str = "both",
    trajectory_sample_rate: float = 1.0,
    trajectory_sample_offset: float = 0.0,
    max_trajectories_per_split: int | None = None,
    reward_gamma: float = DEFAULT_REWARD_GAMMA,
    progress_every: int = 10_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not 0 < trajectory_sample_rate <= 1:
        raise ValueError("trajectory_sample_rate must be in (0, 1]")
    if trajectory_sample_offset < 0 or trajectory_sample_offset + trajectory_sample_rate > 1:
        raise ValueError(
            "trajectory_sample_offset must be non-negative and offset + rate at most one"
        )
    if not 0 < reward_gamma <= 1:
        raise ValueError("reward_gamma must be in (0, 1]")
    if outcome not in {"both", "wins", "losses"}:
        raise ValueError("outcome must be one of: both, wins, losses")
    if max_trajectories_per_split is not None and max_trajectories_per_split <= 0:
        raise ValueError("max_trajectories_per_split must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / f"{name}.jsonl" for name in ("train", "validation", "test")}
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite: {', '.join(map(str, existing))}")
    counters: Counter[str] = Counter()
    rows_per_split: Counter[str] = Counter()
    trajectories_per_split: Counter[str] = Counter()
    battles_per_split: dict[str, set[str]] = {name: set() for name in paths}
    streams = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    try:
        for source_name, trajectory in iter_trajectories(inputs):
            counters["trajectories_seen"] += 1
            if progress_every and counters["trajectories_seen"] % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "phase": "trajectory-prepare",
                            "trajectories_seen": counters["trajectories_seen"],
                            "trajectories_written": sum(trajectories_per_split.values()),
                            "transitions_written": sum(rows_per_split.values()),
                        }
                    ),
                    flush=True,
                )
            metadata = parse_replay_metadata(source_name)
            if min_rating is not None and metadata.rating < min_rating:
                counters["rating_filtered"] += 1
                continue
            if outcome == "wins" and metadata.outcome != "WIN":
                counters["outcome_filtered"] += 1
                continue
            if outcome == "losses" and metadata.outcome != "LOSS":
                counters["outcome_filtered"] += 1
                continue
            if not _format_matches(trajectory, battle_format):
                counters["format_filtered"] += 1
                continue
            if not _trajectory_is_selected(
                metadata.battle_id,
                seed=split_config.seed,
                sample_rate=trajectory_sample_rate,
                sample_offset=trajectory_sample_offset,
            ):
                counters["trajectories_sampled_out"] += 1
                continue
            try:
                split = choose_split(metadata.battle_id, metadata.battle_date, split_config)
            except ValueError:
                counters["missing_split_date"] += 1
                continue
            if (
                max_trajectories_per_split is not None
                and trajectories_per_split[split] >= max_trajectories_per_split
            ):
                counters[f"{split}_trajectory_limit_filtered"] += 1
                continue
            rows = trajectory_rows(
                source_name,
                trajectory,
                metadata,
                split,
                counters,
                reward_gamma=reward_gamma,
            )
            if not rows:
                continue
            for row in rows:
                streams[split].write(json.dumps(row, separators=(",", ":")) + "\n")
            rows_per_split[split] += len(rows)
            trajectories_per_split[split] += 1
            battles_per_split[split].add(metadata.battle_id)
            if max_trajectories_per_split is not None and all(
                trajectories_per_split[name] >= max_trajectories_per_split
                for name in paths
            ):
                counters["stopped_at_trajectory_limits"] += 1
                break
    finally:
        for stream in streams.values():
            stream.close()

    overlap = (
        battles_per_split["train"] & battles_per_split["validation"]
        | battles_per_split["train"] & battles_per_split["test"]
        | battles_per_split["validation"] & battles_per_split["test"]
    )
    if overlap:
        raise AssertionError(f"Battle-level split leakage detected for {len(overlap)} battles")
    report = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "inputs": [str(path) for path in inputs],
        "output_dir": str(output_dir),
        "battle_format": battle_format,
        "min_rating": min_rating,
        "outcome": outcome,
        "trajectory_sample_rate": trajectory_sample_rate,
        "trajectory_sample_offset": trajectory_sample_offset,
        "max_trajectories_per_split": max_trajectories_per_split,
        "reward_gamma": reward_gamma,
        "split_config": {
            **asdict(split_config),
            "validation_start": (
                split_config.validation_start.isoformat()
                if split_config.validation_start
                else None
            ),
            "test_start": (
                split_config.test_start.isoformat() if split_config.test_start else None
            ),
        },
        "transitions_per_split": dict(rows_per_split),
        "trajectories_per_split": dict(trajectories_per_split),
        "battles_per_split": {
            split: len(values) for split, values in battles_per_split.items()
        },
        "counters": dict(counters),
    }
    (output_dir / "prepare_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected date in YYYY-MM-DD format") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare complete POV trajectories with real next-state transitions."
    )
    parser.add_argument("--input", dest="inputs", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", dest="battle_format", default="gen9ou")
    parser.add_argument("--min-rating", type=int)
    parser.add_argument("--outcome", choices=("both", "wins", "losses"), default="both")
    parser.add_argument("--trajectory-sample-rate", type=float, default=1.0)
    parser.add_argument("--trajectory-sample-offset", type=float, default=0.0)
    parser.add_argument("--max-trajectories-per-split", type=int)
    parser.add_argument("--reward-gamma", type=float, default=DEFAULT_REWARD_GAMMA)
    parser.add_argument("--split-mode", choices=("hash", "chronological"), default="hash")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-start", type=_iso_date)
    parser.add_argument("--test-start", type=_iso_date)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = prepare_trajectory_dataset(
        args.inputs,
        args.output_dir,
        split_config=SplitConfig(
            mode=args.split_mode,
            seed=args.seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            validation_start=args.validation_start,
            test_start=args.test_start,
        ),
        battle_format=args.battle_format,
        min_rating=args.min_rating,
        outcome=args.outcome,
        trajectory_sample_rate=args.trajectory_sample_rate,
        trajectory_sample_offset=args.trajectory_sample_offset,
        max_trajectories_per_split=args.max_trajectories_per_split,
        reward_gamma=args.reward_gamma,
        progress_every=args.progress_every,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
