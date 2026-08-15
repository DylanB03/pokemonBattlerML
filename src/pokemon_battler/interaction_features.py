from __future__ import annotations

import hashlib
from typing import Any

from poke_env.data import GenData

from pokemon_battler.actions import ACTION_COUNT, legal_action_ids, sorted_moves, sorted_switches
from pokemon_battler.mechanics import (
    _clamp,
    _generation,
    _hazard_entry,
    _normalized_id,
    _number,
    _status_name,
    _type_multiplier,
    _types,
)
from pokemon_battler.mechanics_v2 import (
    MECHANICS_FEATURE_COUNT,
    MECHANICS_IDENTITY_FIELDS,
    MECHANICS_IDENTITY_VOCAB_SIZES,
    _condition_values,
    _estimated_speed,
    _identity,
    _move_identity,
    _species_identity,
    candidate_feature_matrix,
    candidate_identity_matrix,
)

PREPARED_SCHEMA_VERSION = 3
SUPPORTED_PREPARED_SCHEMA_VERSIONS = (PREPARED_SCHEMA_VERSION, 4)
INTERACTION_CACHE_SCHEMA = "interaction-v1"
INTERACTION_MODEL_SCHEMA = "interaction-policy-v1"

GLOBAL_NUMERIC_NAMES = (
    "turn_fraction",
    "forced_switch",
    "can_tera",
    "player_remaining_fraction",
    "opponent_remaining_fraction",
    *(f"player_{name}" for name in (
        "stealth_rock",
        "spikes_layers",
        "toxic_spikes_layers",
        "sticky_web",
        "reflect",
        "light_screen",
        "aurora_veil",
        "tailwind",
        "safeguard",
    )),
    *(f"opponent_{name}" for name in (
        "stealth_rock",
        "spikes_layers",
        "toxic_spikes_layers",
        "sticky_web",
        "reflect",
        "light_screen",
        "aurora_veil",
        "tailwind",
        "safeguard",
    )),
    "trick_room",
    "player_team_hp_fraction",
    "opponent_revealed_hp_fraction",
    "opponent_roster_revealed_fraction",
    "legal_move_fraction",
    "legal_switch_fraction",
    "legal_tera_fraction",
)
GLOBAL_NUMERIC_COUNT = len(GLOBAL_NUMERIC_NAMES)

GLOBAL_ID_FIELDS = (
    ("format", "format"),
    ("weather", "weather"),
    ("terrain", "terrain"),
    ("legal_mask_quality", "legal_quality"),
)

POKEMON_NUMERIC_NAMES = (
    "present",
    "player_side",
    "active",
    "revealed",
    "fainted",
    "hp_known",
    "item_known",
    "ability_known",
    "hp_fraction",
    *(f"base_{stat}" for stat in ("hp", "atk", "def", "spa", "spd", "spe")),
    "estimated_effective_speed",
    "switch_entry_damage_fraction",
    *(f"{stat}_stage" for stat in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")),
    "known_moves_fraction",
    "terastallized",
    "tera_type_known",
    "known_defensive_worst",
    "known_move_immunity_fraction",
    "known_move_resistance_fraction",
    "known_move_weakness_fraction",
    "opponent_moves_known_fraction",
    "best_offensive_effectiveness",
    "best_damage_pressure",
    *(f"effect_{name}" for name in (
        "substitute",
        "protect",
        "leech_seed",
        "taunt",
        "encore",
        "disable",
        "heal_block",
        "salt_cure",
        "partial_trap",
        "yawn",
        "perish_song",
        "torment",
        "confusion",
        "recharge",
        "ingrain",
        "magnet_rise",
    )),
)
POKEMON_NUMERIC_COUNT = len(POKEMON_NUMERIC_NAMES)

POKEMON_ID_FIELDS = (
    ("species", "species"),
    ("item", "item"),
    ("ability", "ability"),
    ("type_1", "type"),
    ("type_2", "type"),
    ("tera_type", "type"),
    ("status", "status"),
    *((f"move_{index}", "move") for index in range(1, 5)),
)

HISTORY_NUMERIC_NAMES = (
    "recency",
    "player_hp_delta",
    "opponent_hp_delta",
    "player_switched",
    "opponent_switched",
    "player_fainted",
    "opponent_fainted",
    "player_status_changed",
    "opponent_status_changed",
    "player_conditions_changed",
    "opponent_conditions_changed",
    "field_or_weather_changed",
)
HISTORY_ID_FIELDS = (
    ("player_move", "move"),
    ("opponent_move", "move"),
    ("player_species_before", "species"),
    ("player_species_after", "species"),
    ("opponent_species_before", "species"),
    ("opponent_species_after", "species"),
)

INTERACTION_VOCAB_SIZES = {
    **MECHANICS_IDENTITY_VOCAB_SIZES,
    "format": 64,
    "legal_quality": 4,
}
INTERACTION_NAMESPACE_FIELDS = {
    "global": tuple(namespace for _, namespace in GLOBAL_ID_FIELDS),
    "pokemon": tuple(namespace for _, namespace in POKEMON_ID_FIELDS),
    "candidate": tuple(namespace for _, namespace in MECHANICS_IDENTITY_FIELDS),
    "history": tuple(namespace for _, namespace in HISTORY_ID_FIELDS),
}

_GEN9_POKEDEX = GenData.from_gen(9).pokedex
_LEGAL_QUALITY_IDS = {"recoverable": 1, "pp-aware": 2, "exact": 3}


def _hash_identity(namespace: str, value: Any, size: int) -> int:
    normalized = _normalized_id(value)
    if not normalized or normalized.startswith("unknown"):
        return 0
    digest = hashlib.blake2b(f"{namespace}:{normalized}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (size - 1) + 1


def identity(namespace: str, value: Any) -> int:
    if namespace in MECHANICS_IDENTITY_VOCAB_SIZES:
        return _identity(namespace, value)
    if namespace == "legal_quality":
        return _LEGAL_QUALITY_IDS.get(str(value), 0)
    return _hash_identity(namespace, value, INTERACTION_VOCAB_SIZES[namespace])


def _pokemon_key(pokemon: dict[str, Any]) -> str:
    return _normalized_id(pokemon.get("base_species") or pokemon.get("name"))


def _public_pokemon(pokemon: dict[str, Any]) -> dict[str, Any]:
    result = dict(pokemon)
    # Defense in depth for legacy teacher files: an unrevealed preview slot may
    # have been initialized at full HP by an opponent engine.  That is not
    # battle-time public information and must not become a learned feature.
    if result.get("side") == "opponent" and not bool(result.get("revealed")):
        result.update(
            {
                "hp_pct": None,
                "item": "unknownitem",
                "ability": "unknownability",
                "tera_type": "notype",
                "terastallized": False,
                "status": "nostatus",
                "effect": "noeffect",
                "moves": [],
            }
        )
        for stat in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"):
            result[f"{stat}_boost"] = 0
    species = _pokemon_key(result)
    data = _GEN9_POKEDEX.get(species, {})
    for stat, value in (data.get("baseStats") or {}).items():
        result.setdefault(f"base_{stat}", value)
    if not _types(result) and data.get("types"):
        result["types"] = list(data["types"])
    result.setdefault("lvl", 100)
    return result


def _known(value: Any, *, namespace: str) -> bool:
    normalized = _normalized_id(value)
    unknown = {
        "",
        "unknown",
        f"unknown{namespace}",
        "notype",
        "nomove",
        "nostatus",
        "noeffect",
    }
    return normalized not in unknown


def _roster_hp_mean(roster: list[dict[str, Any]], *, revealed_only: bool = False) -> float:
    values = [
        float(pokemon["hp_pct"])
        for pokemon in roster
        if (not revealed_only or pokemon.get("revealed"))
        and isinstance(pokemon.get("hp_pct"), (int, float))
    ]
    return sum(values) / len(values) if values else 0.0


def global_features(row: dict[str, Any]) -> tuple[list[float], list[int]]:
    state = row["state"]
    legal = {int(value) for value in row["legal_action_ids"]}
    player_conditions = _condition_values(state.get("player_conditions"))
    opponent_conditions = _condition_values(state.get("opponent_conditions"))
    player_roster = row["player_roster"]
    opponent_roster = row["opponent_roster"]
    values: dict[str, float] = {
        "turn_fraction": _clamp(_number(row.get("turn_index")) / 100.0),
        "forced_switch": float(bool(state.get("forced_switch"))),
        "can_tera": float(bool(state.get("can_tera"))),
        "player_remaining_fraction": _clamp(_number(state.get("player_remaining")) / 6.0),
        "opponent_remaining_fraction": _clamp(
            _number(state.get("opponents_remaining")) / 6.0
        ),
        "trick_room": float("trickroom" in _normalized_id(state.get("battle_field"))),
        "player_team_hp_fraction": _roster_hp_mean(player_roster),
        "opponent_revealed_hp_fraction": _roster_hp_mean(
            opponent_roster, revealed_only=True
        ),
        "opponent_roster_revealed_fraction": (
            sum(bool(pokemon.get("revealed")) for pokemon in opponent_roster)
            / max(len(opponent_roster), 1)
        ),
        "legal_move_fraction": sum(action in legal for action in range(4)) / 4.0,
        "legal_switch_fraction": sum(action in legal for action in range(4, 9)) / 5.0,
        "legal_tera_fraction": sum(action in legal for action in range(9, 13)) / 4.0,
    }
    values.update({f"player_{key}": value for key, value in player_conditions.items()})
    values.update({f"opponent_{key}": value for key, value in opponent_conditions.items()})
    numeric = [float(values[name]) for name in GLOBAL_NUMERIC_NAMES]
    ids = [
        identity("format", state.get("format")),
        identity("weather", state.get("weather")),
        identity("terrain", state.get("battle_field")),
        identity("legal_quality", row.get("legal_mask_quality")),
    ]
    return numeric, ids


def _matchup_features(
    pokemon: dict[str, Any],
    opposing: dict[str, Any],
    generation: int,
) -> dict[str, float]:
    opposing_moves = sorted_moves(opposing)
    multipliers: list[float] = []
    for move in opposing_moves:
        move_type = str(move.get("move_type") or "")
        multiplier, _ = _type_multiplier(
            move_type,
            pokemon,
            generation,
            attacker_ability=str(opposing.get("ability") or ""),
        )
        multipliers.append(multiplier)

    offensive: list[float] = []
    pressure: list[float] = []
    for move in sorted_moves(pokemon):
        multiplier, _ = _type_multiplier(
            str(move.get("move_type") or ""),
            opposing,
            generation,
            attacker_ability=str(pokemon.get("ability") or ""),
        )
        offensive.append(multiplier)
        power = max(_number(move.get("base_power")), 0.0)
        pressure.append(_clamp(power / 200.0) * _clamp(multiplier / 4.0))
    count = len(multipliers)
    return {
        "known_defensive_worst": _clamp(max(multipliers, default=1.0) / 4.0),
        "known_move_immunity_fraction": (
            sum(value == 0 for value in multipliers) / count if count else 0.0
        ),
        "known_move_resistance_fraction": (
            sum(0 < value < 1 for value in multipliers) / count if count else 0.0
        ),
        "known_move_weakness_fraction": (
            sum(value > 1 for value in multipliers) / count if count else 0.0
        ),
        "opponent_moves_known_fraction": count / 4.0,
        "best_offensive_effectiveness": _clamp(max(offensive, default=1.0) / 4.0),
        "best_damage_pressure": max(pressure, default=0.0),
    }


def pokemon_features(
    row: dict[str, Any],
) -> tuple[list[list[float]], list[list[int]], list[bool]]:
    state = row["state"]
    generation = _generation(state)
    rosters = [*row["player_roster"], *row["opponent_roster"]]
    roster_rows: list[dict[str, Any] | None] = [None] * 12
    for pokemon in rosters:
        side_offset = 0 if pokemon.get("side") == "player" else 6
        slot = int(pokemon.get("slot", -1))
        if 0 <= slot < 6:
            roster_rows[side_offset + slot] = _public_pokemon(pokemon)

    player_active = _public_pokemon(state["player_active_pokemon"])
    opponent_active = _public_pokemon(state["opponent_active_pokemon"])
    numeric_rows: list[list[float]] = []
    identity_rows: list[list[int]] = []
    mask: list[bool] = []
    effect_names = POKEMON_NUMERIC_NAMES[-16:]
    for index, raw_pokemon in enumerate(roster_rows):
        if raw_pokemon is None:
            numeric_rows.append([0.0] * POKEMON_NUMERIC_COUNT)
            identity_rows.append([0] * len(POKEMON_ID_FIELDS))
            mask.append(False)
            continue
        pokemon = raw_pokemon
        player_side = index < 6
        opposing = opponent_active if player_side else player_active
        hp = pokemon.get("hp_pct")
        status = _status_name(pokemon.get("status"))
        values: dict[str, float] = {
            "present": float(bool(pokemon.get("present", True))),
            "player_side": float(player_side),
            "active": float(bool(pokemon.get("active"))),
            "revealed": float(bool(pokemon.get("revealed", player_side))),
            "fainted": float(bool(pokemon.get("fainted")) or status == "fnt"),
            "hp_known": float(isinstance(hp, (int, float))),
            "item_known": float(_known(pokemon.get("item"), namespace="item")),
            "ability_known": float(_known(pokemon.get("ability"), namespace="ability")),
            "hp_fraction": _clamp(_number(hp)),
            "estimated_effective_speed": _clamp(
                _estimated_speed(
                    pokemon,
                    str(state.get("weather") or ""),
                    state.get("player_conditions" if player_side else "opponent_conditions"),
                )
                / 1000.0
            ),
            "switch_entry_damage_fraction": 0.0,
            "known_moves_fraction": min(len(sorted_moves(pokemon)), 4) / 4.0,
            "terastallized": float(bool(pokemon.get("terastallized"))),
            "tera_type_known": float(
                _known(pokemon.get("tera_type"), namespace="type")
            ),
        }
        for stat in ("hp", "atk", "def", "spa", "spd", "spe"):
            values[f"base_{stat}"] = _clamp(_number(pokemon.get(f"base_{stat}")) / 255.0)
        for stat in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"):
            values[f"{stat}_stage"] = _clamp(
                _number(pokemon.get(f"{stat}_boost")) / 6.0,
                -1.0,
                1.0,
            )
        if player_side and not pokemon.get("active") and not pokemon.get("fainted"):
            entry_damage, _, _ = _hazard_entry(state, pokemon, generation)
            values["switch_entry_damage_fraction"] = entry_damage
        values.update(_matchup_features(pokemon, opposing, generation))
        effect_text = _normalized_id(pokemon.get("effect"))
        for field_name in effect_names:
            effect_name = field_name.removeprefix("effect_")
            values[field_name] = float(_normalized_id(effect_name) in effect_text)
        numeric_rows.append([float(values[name]) for name in POKEMON_NUMERIC_NAMES])
        types = (*_types(pokemon), "", "")
        moves = [str(move.get("name") or "") for move in sorted_moves(pokemon)][:4]
        moves.extend([""] * (4 - len(moves)))
        identity_rows.append(
            [
                _species_identity(pokemon),
                identity("item", pokemon.get("item")),
                identity("ability", pokemon.get("ability")),
                identity("type", types[0]),
                identity("type", types[1]),
                identity("type", pokemon.get("tera_type")),
                identity("status", pokemon.get("status")),
                *(_move_identity(move) for move in moves),
            ]
        )
        mask.append(True)
    return numeric_rows, identity_rows, mask


def history_features(row: dict[str, Any]) -> tuple[list[list[float]], list[list[int]], list[bool]]:
    events = list(row.get("history_events") or [])[-4:]
    numeric_rows: list[list[float]] = []
    identity_rows: list[list[int]] = []
    mask: list[bool] = []
    padding = 4 - len(events)
    for _ in range(padding):
        numeric_rows.append([0.0] * len(HISTORY_NUMERIC_NAMES))
        identity_rows.append([0] * len(HISTORY_ID_FIELDS))
        mask.append(False)
    for event in events:
        numeric_rows.append(
            [
                _clamp((float(event.get("decision_offset", -1)) + 4.0) / 4.0),
                _clamp(_number(event.get("player_hp_delta")), -1.0, 1.0),
                _clamp(_number(event.get("opponent_hp_delta")), -1.0, 1.0),
                *(float(bool(event.get(name))) for name in HISTORY_NUMERIC_NAMES[3:]),
            ]
        )
        identity_rows.append(
            [
                _move_identity(event.get("player_move")),
                _move_identity(event.get("opponent_move")),
                _species_identity({"name": event.get("player_species_before")}),
                _species_identity({"name": event.get("player_species_after")}),
                _species_identity({"name": event.get("opponent_species_before")}),
                _species_identity({"name": event.get("opponent_species_after")}),
            ]
        )
        mask.append(True)
    return numeric_rows, identity_rows, mask


def candidate_actor_slots(row: dict[str, Any]) -> list[int]:
    state = row["state"]
    roster_by_key = {
        _pokemon_key(pokemon): int(pokemon["slot"])
        for pokemon in row["player_roster"]
    }
    active_slot = roster_by_key.get(_pokemon_key(state["player_active_pokemon"]), -1)
    switches = sorted_switches(state)
    slots = [-1] * ACTION_COUNT
    legal = {int(value) for value in row["legal_action_ids"]}
    for action_id in legal:
        if 4 <= action_id <= 8:
            slots[action_id] = roster_by_key.get(_pokemon_key(switches[action_id - 4]), -1)
        else:
            slots[action_id] = active_slot
        if slots[action_id] < 0:
            raise ValueError(f"Could not map legal A{action_id} to a player roster slot")
    return slots


def validate_interaction_observation(row: dict[str, Any]) -> None:
    """Validate the decision-time fields shared by training and live inference."""
    if int(row.get("schema_version", -1)) not in SUPPORTED_PREPARED_SCHEMA_VERSIONS:
        raise ValueError(
            "Interaction policy requires prepared schema 3 or trajectory schema 4; "
            f"found {row.get('schema_version')!r}"
        )
    required = {
        "state",
        "player_roster",
        "opponent_roster",
        "history_events",
        "legal_action_ids",
        "legal_mask_quality",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"Interaction row is missing: {', '.join(missing)}")
    legal = [int(value) for value in row["legal_action_ids"]]
    prepared = legal_action_ids(row["state"])
    if legal != prepared:
        raise ValueError("Interaction row legal_action_ids disagree with its state")
    candidate_actor_slots(row)


def validate_interaction_row(row: dict[str, Any]) -> None:
    """Validate a labelled interaction row used for training or offline evaluation."""
    validate_interaction_observation(row)
    if "action_id" not in row:
        raise ValueError("Interaction row is missing: action_id")
    legal = [int(value) for value in row["legal_action_ids"]]
    if int(row["action_id"]) not in legal:
        raise ValueError("Interaction row target is absent from legal_action_ids")


def _build_interaction_features(row: dict[str, Any]) -> dict[str, Any]:
    global_numeric, global_ids = global_features(row)
    pokemon_numeric, pokemon_ids, pokemon_mask = pokemon_features(row)
    history_numeric, history_ids, history_mask = history_features(row)
    state = row["state"]
    legal = {int(value) for value in row["legal_action_ids"]}
    return {
        "global_numeric": global_numeric,
        "global_ids": global_ids,
        "pokemon_numeric": pokemon_numeric,
        "pokemon_ids": pokemon_ids,
        "pokemon_mask": pokemon_mask,
        "candidate_numeric": candidate_feature_matrix(state),
        "candidate_ids": candidate_identity_matrix(state),
        "candidate_mask": [action_id in legal for action_id in range(ACTION_COUNT)],
        "candidate_actor_slot": candidate_actor_slots(row),
        "history_numeric": history_numeric,
        "history_ids": history_ids,
        "history_mask": history_mask,
    }


def build_interaction_observation_features(row: dict[str, Any]) -> dict[str, Any]:
    """Build policy inputs for an unlabelled live decision observation."""
    validate_interaction_observation(row)
    return _build_interaction_features(row)


def build_interaction_features(row: dict[str, Any]) -> dict[str, Any]:
    validate_interaction_row(row)
    return _build_interaction_features(row)


assert GLOBAL_NUMERIC_COUNT == 30
assert POKEMON_NUMERIC_COUNT == 50
assert MECHANICS_FEATURE_COUNT == 207
