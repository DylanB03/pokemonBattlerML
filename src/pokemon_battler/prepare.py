from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tarfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Sequence

from pokemon_battler.actions import (
    action_label,
    pp_aware_legal_action_ids,
    recoverable_legal_action_ids,
)

SUPPORTED_FILE_SUFFIXES = (".json", ".json.lz4")
SUPPORTED_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz")


@dataclass(frozen=True)
class TrajectorySource:
    name: str
    payload: bytes


@dataclass(frozen=True)
class ReplayMetadata:
    battle_id: str
    rating: int
    battle_date: date | None
    outcome: str | None


@dataclass(frozen=True)
class SplitConfig:
    mode: str = "hash"
    seed: int = 42
    train_fraction: float = 0.9
    validation_fraction: float = 0.05
    validation_start: date | None = None
    test_start: date | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"hash", "chronological"}:
            raise ValueError("split mode must be 'hash' or 'chronological'")
        if self.mode == "hash":
            if not 0 < self.train_fraction < 1:
                raise ValueError("train_fraction must be between 0 and 1")
            if not 0 <= self.validation_fraction < 1:
                raise ValueError("validation_fraction must be in [0, 1)")
            if self.train_fraction + self.validation_fraction >= 1:
                raise ValueError("train_fraction + validation_fraction must be less than 1")
        elif self.validation_start is None or self.test_start is None:
            raise ValueError(
                "chronological splitting requires validation_start and test_start"
            )
        elif self.validation_start >= self.test_start:
            raise ValueError("validation_start must be earlier than test_start")


def _has_suffix(name: str, suffixes: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in suffixes)


def _strip_data_suffix(name: str) -> str:
    basename = Path(name).name
    for suffix in (".json.lz4", ".json"):
        if basename.lower().endswith(suffix):
            return basename[: -len(suffix)]
    return basename


def _read_lz4(stream: BinaryIO) -> bytes:
    try:
        import lz4.frame
    except ImportError as exc:
        raise RuntimeError(
            "Reading .lz4 trajectories requires the 'lz4' package. "
            "Install the project dependencies first."
        ) from exc
    return lz4.frame.decompress(stream.read())


def _decode_payload(name: str, stream: BinaryIO) -> bytes:
    if name.lower().endswith(".lz4"):
        return _read_lz4(stream)
    return stream.read()


def _iter_tar(path: Path) -> Iterator[TrajectorySource]:
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile() or not _has_suffix(member.name, SUPPORTED_FILE_SUFFIXES):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with extracted:
                yield TrajectorySource(
                    name=f"{path.name}:{member.name}",
                    payload=_decode_payload(member.name, extracted),
                )


def _iter_path(path: Path) -> Iterator[TrajectorySource]:
    if path.is_dir():
        for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            if _has_suffix(child.name, SUPPORTED_TAR_SUFFIXES):
                yield from _iter_tar(child)
            elif _has_suffix(child.name, SUPPORTED_FILE_SUFFIXES):
                with child.open("rb") as stream:
                    yield TrajectorySource(
                        name=str(child),
                        payload=_decode_payload(child.name, stream),
                    )
        return

    if _has_suffix(path.name, SUPPORTED_TAR_SUFFIXES):
        yield from _iter_tar(path)
    elif _has_suffix(path.name, SUPPORTED_FILE_SUFFIXES):
        with path.open("rb") as stream:
            yield TrajectorySource(name=str(path), payload=_decode_payload(path.name, stream))
    else:
        raise ValueError(f"Unsupported input: {path}")


def iter_trajectories(paths: Iterable[Path]) -> Iterator[tuple[str, dict[str, Any]]]:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")
        for source in _iter_path(path):
            try:
                trajectory = json.loads(source.payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not decode trajectory {source.name}") from exc
            if not isinstance(trajectory, dict):
                raise ValueError(f"Trajectory must be a JSON object: {source.name}")
            yield source.name, trajectory


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    for pattern in ("%m-%d-%Y", "%m-%d-%Y-%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            continue
    return None


def parse_replay_metadata(source_name: str) -> ReplayMetadata:
    """
    Parse Metamon's ``battle_id_rating_p1_vs_p2_date_result`` filename.

    Parsing from both ends tolerates the one-extra-underscore player-name case
    handled by Metamon itself while keeping both POV files grouped by battle ID.
    """
    member_or_path = source_name
    for marker in (".tar.gz:", ".tgz:", ".tar:"):
        if marker in source_name:
            member_or_path = source_name.split(marker, 1)[1]
            break
    stem = _strip_data_suffix(member_or_path)
    parts = stem.split("_")

    battle_id = parts[0] if parts else stem
    rating = 1000
    if len(parts) > 1:
        try:
            rating = int(parts[1])
        except ValueError:
            pass

    battle_date = _parse_date(parts[-2] if len(parts) >= 2 else None)
    outcome = parts[-1].upper() if parts and parts[-1].upper() in {"WIN", "LOSS"} else None
    return ReplayMetadata(
        battle_id=battle_id,
        rating=rating,
        battle_date=battle_date,
        outcome=outcome,
    )


def choose_split(
    battle_id: str,
    battle_date: date | None,
    config: SplitConfig,
) -> str:
    if config.mode == "chronological":
        if battle_date is None:
            raise ValueError(
                f"Battle {battle_id!r} has no parseable date for chronological splitting"
            )
        assert config.validation_start is not None
        assert config.test_start is not None
        if battle_date < config.validation_start:
            return "train"
        if battle_date < config.test_start:
            return "validation"
        return "test"

    digest = hashlib.sha256(f"{config.seed}:{battle_id}".encode("utf-8")).digest()
    unit_interval = int.from_bytes(digest[:8], "big") / float(2**64)
    if unit_interval < config.train_fraction:
        return "train"
    if unit_interval < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"


def _sample_is_selected(battle_id: str, turn_index: int, seed: int, sample_rate: float) -> bool:
    if sample_rate >= 1:
        return True
    digest = hashlib.sha256(
        f"sample:{seed}:{battle_id}:{turn_index}".encode("utf-8")
    ).digest()
    unit_interval = int.from_bytes(digest[:8], "big") / float(2**64)
    return unit_interval < sample_rate


def _format_matches(trajectory: dict[str, Any], wanted_format: str | None) -> bool:
    if wanted_format is None:
        return True
    states = trajectory.get("states") or []
    if not states:
        return False
    observed = str(states[0].get("format", "")).replace("[", "").replace("]", "")
    observed = observed.replace(" ", "").lower()
    return observed == wanted_format.replace(" ", "").lower()


def _known_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() not in {
            "",
            "unknown",
            "unknownitem",
            "unknownability",
            "notype",
            "nomove",
        }
    return True


def _merge_revealed_pokemon(
    previous: dict[str, Any] | None,
    observed: dict[str, Any],
) -> dict[str, Any]:
    """Merge only information already visible by the current decision."""
    merged = copy.deepcopy(previous) if previous is not None else {}
    for key, value in observed.items():
        if key == "moves":
            continue
        if _known_value(value):
            merged[key] = copy.deepcopy(value)
    moves_by_name = {
        str(move.get("name", "")).lower(): copy.deepcopy(move)
        for move in merged.get("moves", [])
        if isinstance(move, dict) and _known_value(move.get("name"))
    }
    for move in observed.get("moves") or []:
        if not isinstance(move, dict) or not _known_value(move.get("name")):
            continue
        name = str(move["name"]).lower()
        combined_move = moves_by_name.get(name, {})
        for key, value in move.items():
            if _known_value(value):
                combined_move[key] = copy.deepcopy(value)
        moves_by_name[name] = combined_move
    merged["moves"] = list(moves_by_name.values())
    return merged


def _pokemon_key(pokemon: dict[str, Any] | None) -> str:
    if not pokemon:
        return ""
    value = pokemon.get("base_species") or pokemon.get("name") or ""
    return "".join(character for character in str(value).lower() if character.isalnum())


def _preview_pokemon(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        name = value.get("base_species") or value.get("name")
        if not name:
            return None
        return copy.deepcopy(value)
    if not value:
        return None
    return {"name": str(value), "base_species": str(value)}


def _observe_rosters(
    state: dict[str, Any],
    previous_state: dict[str, Any] | None,
    player_roster: dict[str, dict[str, Any]],
    player_order: list[str],
    opponent_roster: dict[str, dict[str, Any]],
    opponent_order: list[str],
) -> None:
    def add(
        roster: dict[str, dict[str, Any]],
        order: list[str],
        pokemon: dict[str, Any],
        *,
        revealed: bool,
    ) -> str:
        key = _pokemon_key(pokemon)
        if not key:
            return ""
        if key not in roster:
            roster[key] = {}
            order.append(key)
        roster[key] = _merge_revealed_pokemon(roster[key], pokemon)
        roster[key]["present"] = True
        roster[key]["revealed"] = bool(roster[key].get("revealed", False) or revealed)
        return key

    player_current: set[str] = set()
    for pokemon in [state.get("player_active_pokemon"), *(state.get("available_switches") or [])]:
        if isinstance(pokemon, dict):
            key = add(player_roster, player_order, pokemon, revealed=True)
            if key:
                player_current.add(key)
    # Once a player Pokémon has disappeared from both the active and switch lists,
    # the replay state no longer exposes it because it has fainted. Retain its slot.
    for key, pokemon in player_roster.items():
        if key not in player_current:
            pokemon["fainted"] = True
            pokemon["hp_pct"] = 0.0

    for preview in state.get("opponent_teampreview") or []:
        pokemon = _preview_pokemon(preview)
        if pokemon is not None:
            add(opponent_roster, opponent_order, pokemon, revealed=False)
    opponent = state.get("opponent_active_pokemon")
    if isinstance(opponent, dict):
        add(opponent_roster, opponent_order, opponent, revealed=True)

    if previous_state is not None:
        before_remaining = int(previous_state.get("opponents_remaining", 0) or 0)
        after_remaining = int(state.get("opponents_remaining", 0) or 0)
        before_active = previous_state.get("opponent_active_pokemon")
        before_key = _pokemon_key(before_active if isinstance(before_active, dict) else None)
        if after_remaining < before_remaining and before_key in opponent_roster:
            opponent_roster[before_key]["fainted"] = True
            opponent_roster[before_key]["hp_pct"] = 0.0


def _roster_snapshot(
    roster: dict[str, dict[str, Any]],
    order: list[str],
    active: dict[str, Any] | None,
    side: str,
) -> list[dict[str, Any]]:
    active_key = _pokemon_key(active)
    rows: list[dict[str, Any]] = []
    for slot, key in enumerate(order[:6]):
        pokemon = copy.deepcopy(roster[key])
        pokemon["slot"] = slot
        pokemon["side"] = side
        pokemon["active"] = key == active_key
        status = str(pokemon.get("status", "")).lower()
        hp = pokemon.get("hp_pct")
        pokemon["fainted"] = bool(
            pokemon.get("fainted", False)
            or status == "fnt"
            or isinstance(hp, (int, float)) and hp <= 0
        )
        rows.append(pokemon)
    return rows


def _player_remaining(state: dict[str, Any]) -> int:
    active = state.get("player_active_pokemon") or {}
    active_alive = float(active.get("hp_pct", 0) or 0) > 0 and active.get("status") != "fnt"
    return len(state.get("available_switches") or []) + int(active_alive)


def _move_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "unknown")
    return str(value or "unknown")


def _history_event(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    player_before = previous.get("player_active_pokemon") or {}
    player_after = current.get("player_active_pokemon") or {}
    opponent_before = previous.get("opponent_active_pokemon") or {}
    opponent_after = current.get("opponent_active_pokemon") or {}
    player_switched = _pokemon_key(player_before) != _pokemon_key(player_after)
    opponent_switched = _pokemon_key(opponent_before) != _pokemon_key(opponent_after)

    def hp_delta(before: dict[str, Any], after: dict[str, Any], switched: bool) -> float:
        if switched:
            return 0.0
        before_hp = before.get("hp_pct")
        after_hp = after.get("hp_pct")
        if not isinstance(before_hp, (int, float)) or not isinstance(after_hp, (int, float)):
            return 0.0
        return max(min(float(after_hp) - float(before_hp), 1.0), -1.0)

    player_fainted = _player_remaining(current) < _player_remaining(previous)
    opponent_fainted = int(current.get("opponents_remaining", 0) or 0) < int(
        previous.get("opponents_remaining", 0) or 0
    )
    return {
        "decision_offset": -1,
        "player_move": _move_name(current.get("player_prev_move")),
        "opponent_move": _move_name(current.get("opponent_prev_move")),
        "player_species_before": str(player_before.get("name") or "unknown"),
        "player_species_after": str(player_after.get("name") or "unknown"),
        "opponent_species_before": str(opponent_before.get("name") or "unknown"),
        "opponent_species_after": str(opponent_after.get("name") or "unknown"),
        "player_hp_delta": hp_delta(player_before, player_after, player_switched),
        "opponent_hp_delta": hp_delta(opponent_before, opponent_after, opponent_switched),
        "player_switched": player_switched,
        "opponent_switched": opponent_switched,
        "player_fainted": player_fainted,
        "opponent_fainted": opponent_fainted,
        "player_status_changed": player_before.get("status") != player_after.get("status"),
        "opponent_status_changed": (
            opponent_before.get("status") != opponent_after.get("status")
        ),
        "player_conditions_changed": (
            previous.get("player_conditions") != current.get("player_conditions")
        ),
        "opponent_conditions_changed": (
            previous.get("opponent_conditions") != current.get("opponent_conditions")
        ),
        "field_or_weather_changed": (
            previous.get("battle_field") != current.get("battle_field")
            or previous.get("weather") != current.get("weather")
        ),
    }


def _history_snapshot(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = copy.deepcopy(events[-4:])
    for index, event in enumerate(selected):
        event["decision_offset"] = index - len(selected)
    return selected


def _observe_opponent(
    state: dict[str, Any],
    revealed_opponents: dict[str, dict[str, Any]],
) -> None:
    opponent = state.get("opponent_active_pokemon") or {}
    opponent_name = str(opponent.get("name", "")).lower()
    if opponent_name:
        revealed_opponents[opponent_name] = _merge_revealed_pokemon(
            revealed_opponents.get(opponent_name),
            opponent,
        )


def _observe_move_history(
    state: dict[str, Any],
    recent_moves: list[dict[str, str]],
) -> None:
    def move_name(value: Any) -> str:
        if isinstance(value, dict):
            return str(value.get("name") or "nomove")
        return str(value or "nomove")

    entry = {
        "player": move_name(state.get("player_prev_move")),
        "opponent": move_name(state.get("opponent_prev_move")),
    }
    if entry == {"player": "nomove", "opponent": "nomove"}:
        return
    if not recent_moves or recent_moves[-1] != entry:
        recent_moves.append(entry)
        del recent_moves[:-4]


def _enrich_state(
    state: dict[str, Any],
    turn_index: int,
    revealed_opponents: dict[str, dict[str, Any]],
    recent_moves: list[dict[str, str]],
) -> dict[str, Any]:
    enriched = copy.deepcopy(state)
    enriched["turn_index"] = turn_index
    active = enriched.get("player_active_pokemon") or {}
    active_alive = float(active.get("hp_pct", 0) or 0) > 0 and active.get("status") != "fnt"
    enriched["player_remaining"] = len(enriched.get("available_switches") or []) + int(
        active_alive
    )
    enriched["opponent_revealed_pokemon"] = [
        copy.deepcopy(revealed_opponents[name]) for name in sorted(revealed_opponents)
    ]
    enriched["recent_move_history"] = copy.deepcopy(recent_moves)
    return enriched


def _explicit_legal_actions(
    state: dict[str, Any],
    recoverable: list[int],
) -> list[int] | None:
    for key in ("legal_action_ids", "valid_action_ids"):
        values = state.get(key)
        if not isinstance(values, list) or not values:
            continue
        if not all(isinstance(value, int) for value in values):
            continue
        legal = sorted(set(values))
        if set(legal).issubset(recoverable):
            return legal
    return None


def _trajectory_rows(
    source_name: str,
    trajectory: dict[str, Any],
    metadata: ReplayMetadata,
    split: str,
    seed: int,
    sample_rate: float,
    counters: Counter[str],
) -> Iterator[dict[str, Any]]:
    states = trajectory.get("states")
    actions = trajectory.get("actions")
    if not isinstance(states, list) or not isinstance(actions, list):
        counters["malformed_trajectories"] += 1
        return

    if len(states) < 2:
        counters["short_trajectories"] += 1
        return

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

    # Metamon deliberately pairs states[:-1] with actions[:-1]. The final state
    # is terminal and has no decision target.
    for turn_index, (state, action_id) in enumerate(zip(states[:-1], actions[:-1])):
        if not isinstance(state, dict) or not isinstance(action_id, int):
            counters["malformed_turns"] += 1
            continue
        if previous_state is not None:
            history_events.append(_history_event(previous_state, state))
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

        try:
            recoverable_legal = recoverable_legal_action_ids(state)
            pp_aware_legal = pp_aware_legal_action_ids(state)
        except (KeyError, TypeError, ValueError):
            counters["invalid_states"] += 1
            continue
        if action_id not in recoverable_legal:
            counters["actions_not_recoverably_legal"] += 1
            continue
        explicit_legal = _explicit_legal_actions(state, recoverable_legal)
        if explicit_legal is not None and action_id in explicit_legal:
            legal = explicit_legal
            legal_mask_quality = "exact"
            counters["exact_legal_masks"] += 1
        elif action_id in pp_aware_legal and pp_aware_legal != recoverable_legal:
            legal = pp_aware_legal
            legal_mask_quality = "pp-aware"
            counters["zero_pp_candidates_removed"] += len(recoverable_legal) - len(legal)
        else:
            legal = recoverable_legal
            legal_mask_quality = "recoverable"
        if not _sample_is_selected(metadata.battle_id, turn_index, seed, sample_rate):
            counters["sampled_out"] += 1
            continue

        state = _enrich_state(state, turn_index, revealed_opponents, recent_moves)
        state["prepared_legal_action_ids"] = legal

        yield {
            "schema_version": 3,
            "battle_id": metadata.battle_id,
            "battle_date": metadata.battle_date.isoformat() if metadata.battle_date else None,
            "rating": metadata.rating,
            "outcome": metadata.outcome,
            "source": source_name,
            "turn_index": turn_index,
            "battle_decision_count": max(len(states) - 1, 1),
            "split": split,
            "state": state,
            "player_roster": _roster_snapshot(
                player_roster,
                player_order,
                state.get("player_active_pokemon"),
                "player",
            ),
            "opponent_roster": _roster_snapshot(
                opponent_roster,
                opponent_order,
                state.get("opponent_active_pokemon"),
                "opponent",
            ),
            "history_events": _history_snapshot(history_events),
            "action_id": action_id,
            "target": action_label(action_id),
            "legal_action_ids": legal,
            "legal_mask_quality": legal_mask_quality,
        }


def prepare_dataset(
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    split_config: SplitConfig,
    battle_format: str | None = "gen9ou",
    min_rating: int | None = None,
    outcome: str = "both",
    sample_rate: float = 1.0,
    max_examples_per_split: int | None = None,
    progress_every: int = 10_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1]")
    if outcome not in {"both", "wins", "losses"}:
        raise ValueError("outcome must be one of: both, wins, losses")
    if max_examples_per_split is not None and max_examples_per_split <= 0:
        raise ValueError("max_examples_per_split must be positive")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        split: output_dir / f"{split}.jsonl"
        for split in ("train", "validation", "test")
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing dataset files: {names}")

    counters: Counter[str] = Counter()
    examples_per_split: Counter[str] = Counter()
    battles_per_split: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    streams = {
        split: path.open("w", encoding="utf-8")
        for split, path in output_paths.items()
    }
    try:
        for source_name, trajectory in iter_trajectories(inputs):
            counters["trajectories_seen"] += 1
            if (
                progress_every
                and counters["trajectories_seen"] % progress_every == 0
            ):
                print(
                    json.dumps(
                        {
                            "trajectories_seen": counters["trajectories_seen"],
                            "examples_written": counters["examples_written"],
                            "examples_per_split": {
                                split: examples_per_split[split]
                                for split in ("train", "validation", "test")
                            },
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

            try:
                split = choose_split(
                    metadata.battle_id,
                    metadata.battle_date,
                    split_config,
                )
            except ValueError:
                counters["missing_split_date"] += 1
                continue

            if (
                max_examples_per_split is not None
                and examples_per_split[split] >= max_examples_per_split
            ):
                counters[f"{split}_limit_filtered"] += 1
                continue

            wrote_battle = False
            for row in _trajectory_rows(
                source_name,
                trajectory,
                metadata,
                split,
                split_config.seed,
                sample_rate,
                counters,
            ):
                if (
                    max_examples_per_split is not None
                    and examples_per_split[split] >= max_examples_per_split
                ):
                    counters[f"{split}_limit_filtered"] += 1
                    break
                streams[split].write(json.dumps(row, separators=(",", ":")) + "\n")
                examples_per_split[split] += 1
                counters["examples_written"] += 1
                wrote_battle = True
            if wrote_battle:
                battles_per_split[split].add(metadata.battle_id)
            if max_examples_per_split is not None and all(
                examples_per_split[name] >= max_examples_per_split
                for name in ("train", "validation", "test")
            ):
                counters["stopped_at_split_limits"] += 1
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
        "schema_version": 3,
        "inputs": [str(path) for path in inputs],
        "output_dir": str(output_dir),
        "battle_format": battle_format,
        "min_rating": min_rating,
        "outcome": outcome,
        "sample_rate": sample_rate,
        "max_examples_per_split": max_examples_per_split,
        "progress_every": progress_every,
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
        "examples_per_split": {
            split: examples_per_split[split] for split in ("train", "validation", "test")
        },
        "battles_per_split": {
            split: len(battle_ids) for split, battle_ids in battles_per_split.items()
        },
        "counters": dict(counters),
    }
    report_path = output_dir / "prepare_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected date in YYYY-MM-DD format") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream Metamon trajectories into grouped turn-level SFT JSONL files."
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        required=True,
        help="A trajectory directory, .json/.json.lz4 file, or .tar[.gz]. Repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", default="gen9ou", help="Battle format filter.")
    parser.add_argument("--min-rating", type=int)
    parser.add_argument("--outcome", choices=("both", "wins", "losses"), default="both")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Deterministic turn-level sampling probability.",
    )
    parser.add_argument("--max-examples-per-split", type=int)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print progress every N trajectories; use 0 to disable.",
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--split-mode", choices=("hash", "chronological"), default="hash")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-start", type=_iso_date)
    parser.add_argument("--test-start", type=_iso_date)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    split_config = SplitConfig(
        mode=args.split_mode,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        validation_start=args.validation_start,
        test_start=args.test_start,
    )
    report = prepare_dataset(
        args.inputs,
        args.output_dir,
        split_config=split_config,
        battle_format=args.format,
        min_rating=args.min_rating,
        outcome=args.outcome,
        sample_rate=args.sample_rate,
        max_examples_per_split=args.max_examples_per_split,
        progress_every=args.progress_every,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
