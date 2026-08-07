from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from poke_env.battle import Effect, Field, SideCondition, Target, Weather
from poke_env.data import GenData

from pokemon_battler.actions import ACTION_COUNT, legal_action_ids, sorted_moves, sorted_switches
from pokemon_battler.mechanics import (
    BOOST_STATS,
    MECHANICS_FEATURE_NAMES as V1_FEATURE_NAMES,
    _clamp,
    _estimated_stat,
    _generation,
    _move_data,
    _normalized_id,
    _number,
    _stage_multiplier,
    _status_name,
    _type_multiplier,
    _types,
    candidate_feature_vector as v1_candidate_feature_vector,
)

MECHANICS_SCHEMA = "mechanics-v2"

TYPE_NAMES = (
    "bug",
    "dark",
    "dragon",
    "electric",
    "fairy",
    "fighting",
    "fire",
    "flying",
    "ghost",
    "grass",
    "ground",
    "ice",
    "normal",
    "poison",
    "psychic",
    "rock",
    "steel",
    "water",
)

SIDE_CONDITION_NAMES = (
    "stealth_rock",
    "spikes_layers",
    "toxic_spikes_layers",
    "sticky_web",
    "reflect",
    "light_screen",
    "aurora_veil",
    "tailwind",
    "safeguard",
)

MOVE_FLAG_NAMES = (
    "contact",
    "sound",
    "bullet",
    "pulse",
    "punch",
    "bite",
    "dance",
    "slicing",
    "wind",
    "powder",
    "reflectable",
    "bypasssub",
)

EXTRA_FEATURE_NAMES = (
    *(f"player_base_{stat}" for stat in ("hp", "atk", "def", "spa", "spd", "spe")),
    *(f"opponent_base_{stat}" for stat in ("hp", "atk", "def", "spa", "spd", "spe")),
    *(f"player_{stat}_stage" for stat in BOOST_STATS),
    *(f"opponent_{stat}_stage" for stat in BOOST_STATS),
    *(f"player_side_{name}" for name in SIDE_CONDITION_NAMES),
    *(f"opponent_side_{name}" for name in SIDE_CONDITION_NAMES),
    "field_trick_room",
    "actor_estimated_speed",
    "opponent_estimated_speed",
    "actor_faster_without_priority",
    "actor_moves_first_estimate",
    *(f"move_flag_{name}" for name in MOVE_FLAG_NAMES),
    "sets_stealth_rock",
    "sets_spikes",
    "sets_toxic_spikes",
    "sets_sticky_web",
    "sets_reflect",
    "sets_light_screen",
    "sets_aurora_veil",
    "sets_tailwind",
    "sets_safeguard",
    "sets_other_side_condition",
    "hazard_control_user_side",
    "hazard_control_both_sides",
    "swaps_side_conditions",
    "uses_target_defense",
    "uses_user_defense",
    "uses_target_attack",
    "variable_base_power",
    "conditional_priority",
    "item_interaction",
    "calls_another_move",
    "delayed_attack",
    "delayed_heal",
    "full_recovery",
    "cures_user_status",
    "requires_sleep",
    "creates_substitute",
    "removes_field_effect",
    "target_leech_seed",
    "target_taunt",
    "target_encore",
    "target_disable",
    "target_heal_block",
    "target_salt_cure",
    "target_partial_trap",
    "target_yawn",
    "target_perish_song",
    "target_torment",
    "self_locked_move",
    "other_volatile_effect",
    "candidate_known_defensive_worst",
    "candidate_known_move_immunity_fraction",
    "candidate_known_move_resistance_fraction",
    "candidate_known_move_weakness_fraction",
    "candidate_opponent_moves_known_fraction",
    "damage_accounts_for_special_rule",
    "damage_accounts_for_weather",
    "damage_accounts_for_terrain",
    "damage_accounts_for_item",
    "damage_accounts_for_ability",
)

MECHANICS_FEATURE_NAMES = (*V1_FEATURE_NAMES, *EXTRA_FEATURE_NAMES)
MECHANICS_FEATURE_COUNT = len(MECHANICS_FEATURE_NAMES)

_GEN9 = GenData.from_gen(9)
_MOVE_VOCAB = {name: index + 1 for index, name in enumerate(sorted(_GEN9.moves))}
_SPECIES_VOCAB = {name: index + 1 for index, name in enumerate(sorted(_GEN9.pokedex))}
_ABILITY_NAMES = sorted(
    {
        _normalized_id(ability)
        for pokemon in _GEN9.pokedex.values()
        for ability in (pokemon.get("abilities") or {}).values()
    }
)
_ABILITY_VOCAB = {name: index + 1 for index, name in enumerate(_ABILITY_NAMES)}
_ABILITY_FALLBACK_BUCKETS = 512
_TYPE_VOCAB = {name: index + 1 for index, name in enumerate(TYPE_NAMES)}
_STATUS_VOCAB = {
    name: index + 1
    for index, name in enumerate(("burn", "poison", "toxic", "paralysis", "sleep", "freeze"))
}


def _enum_vocab(enum_type: Any) -> dict[str, int]:
    names = sorted(
        _normalized_id(member.name)
        for member in enum_type
        if _normalized_id(member.name) != "unknown"
    )
    return {name: index + 1 for index, name in enumerate(names)}


_ENUM_VOCABS = {
    "weather": _enum_vocab(Weather),
    "terrain": _enum_vocab(Field),
    "effect": _enum_vocab(Effect),
    "side_condition": _enum_vocab(SideCondition),
    "target": _enum_vocab(Target),
}
_ENUM_FALLBACK_BUCKETS = {
    "weather": 16,
    "terrain": 32,
    "effect": 512,
    "side_condition": 64,
    "target": 32,
}

MECHANICS_IDENTITY_FIELDS = (
    ("candidate_move", "move"),
    ("actor_species", "species"),
    ("opponent_species", "species"),
    ("actor_item", "item"),
    ("actor_ability", "ability"),
    ("opponent_item", "item"),
    ("opponent_ability", "ability"),
    ("actor_type_1", "type"),
    ("actor_type_2", "type"),
    ("opponent_type_1", "type"),
    ("opponent_type_2", "type"),
    ("actor_tera_type", "type"),
    ("move_type", "type"),
    ("actor_status", "status"),
    ("opponent_status", "status"),
    ("weather", "weather"),
    ("terrain", "terrain"),
    ("actor_effect", "effect"),
    ("opponent_effect", "effect"),
    ("move_side_condition", "side_condition"),
    ("move_volatile", "effect"),
    ("move_target", "target"),
    ("previous_player_move", "move"),
    ("previous_opponent_move", "move"),
    *((f"actor_move_{index}", "move") for index in range(1, 5)),
    *((f"opponent_move_{index}", "move") for index in range(1, 5)),
)
MECHANICS_IDENTITY_NAMES = tuple(name for name, _ in MECHANICS_IDENTITY_FIELDS)
MECHANICS_IDENTITY_COUNT = len(MECHANICS_IDENTITY_NAMES)
MECHANICS_IDENTITY_VOCAB_SIZES = {
    "move": len(_MOVE_VOCAB) + 1,
    "species": len(_SPECIES_VOCAB) + 1,
    "item": 16_384,
    "ability": len(_ABILITY_VOCAB) + _ABILITY_FALLBACK_BUCKETS + 1,
    "type": len(_TYPE_VOCAB) + 1,
    "status": len(_STATUS_VOCAB) + 1,
    **{
        namespace: len(vocab) + _ENUM_FALLBACK_BUCKETS[namespace] + 1
        for namespace, vocab in _ENUM_VOCABS.items()
    },
}

_TARGET_DEFENSE_MOVES = {"psyshock", "psystrike", "secretsword"}
_USER_DEFENSE_MOVES = {"bodypress"}
_TARGET_ATTACK_MOVES = {"foulplay"}
_CONDITIONAL_PRIORITY_MOVES = {"suckerpunch", "thunderclap", "upperhand"}
_ITEM_INTERACTION_MOVES = {
    "acrobatics",
    "bestow",
    "corrosivegas",
    "covet",
    "fling",
    "incinerate",
    "knockoff",
    "poltergeist",
    "switcheroo",
    "thief",
    "trick",
}
_CALLS_MOVE = {
    "assist",
    "copycat",
    "mefirst",
    "metronome",
    "mirrormove",
    "naturepower",
    "sleeptalk",
}
_DELAYED_ATTACKS = {"doomdesire", "futuresight"}
_LOCKED_MOVES = {"outrage", "petaldance", "rollout", "thrash"}
_VARIABLE_POWER_MOVES = {
    "acrobatics",
    "brine",
    "dragonenergy",
    "electroball",
    "eruption",
    "facade",
    "flail",
    "foulplay",
    "grassknot",
    "gyroball",
    "heatcrash",
    "heavyslam",
    "hex",
    "infernalparade",
    "knockoff",
    "lastrespects",
    "lowkick",
    "powertrip",
    "reversal",
    "storedpower",
    "terablast",
    "barbbarrage",
    "venoshock",
    "waterspout",
}


def _effect_id(value: Any) -> str:
    if value is None:
        return ""
    return _normalized_id(getattr(value, "name", value))


def _hash_id(namespace: str, value: Any, size: int) -> int:
    normalized = _normalized_id(value)
    if normalized in {
        "",
        "unknown",
        "unknownitem",
        "unknownability",
        "noitem",
        "nomove",
        "noeffect",
        "noweather",
        "nofield",
        "notype",
        "nostatus",
    }:
        return 0
    digest = hashlib.blake2b(f"{namespace}:{normalized}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (size - 1) + 1


def _move_identity(value: Any) -> int:
    return _MOVE_VOCAB.get(_normalized_id(value), 0)


def _species_identity(pokemon: dict[str, Any]) -> int:
    for field in ("name", "base_species"):
        value = _SPECIES_VOCAB.get(_normalized_id(pokemon.get(field)))
        if value is not None:
            return value
    return 0


def _identity(namespace: str, value: Any) -> int:
    if namespace == "move":
        return _move_identity(value)
    if namespace == "type":
        return _TYPE_VOCAB.get(_effect_id(value), 0)
    if namespace == "status":
        status = _status_name(value)
        return _STATUS_VOCAB.get(status or "", 0)
    if namespace == "ability":
        normalized = _effect_id(value)
        exact = _ABILITY_VOCAB.get(normalized)
        if exact is not None:
            return exact
        fallback = _hash_id(namespace, normalized, _ABILITY_FALLBACK_BUCKETS + 1)
        return len(_ABILITY_VOCAB) + fallback if fallback else 0
    if namespace in _ENUM_VOCABS:
        normalized = _effect_id(value)
        exact = _ENUM_VOCABS[namespace].get(normalized)
        if exact is not None:
            return exact
        fallback_size = _ENUM_FALLBACK_BUCKETS[namespace] + 1
        fallback = _hash_id(namespace, normalized, fallback_size)
        return len(_ENUM_VOCABS[namespace]) + fallback if fallback else 0
    return _hash_id(namespace, _effect_id(value), MECHANICS_IDENTITY_VOCAB_SIZES[namespace])


def _previous_move_name(value: Any) -> str:
    return str(value.get("name", "")) if isinstance(value, dict) else str(value or "")


def candidate_identity_vector(state: dict[str, Any], action_id: int) -> list[int]:
    if action_id not in legal_action_ids(state):
        raise ValueError(f"A{action_id} is not legal in this state")
    active = state["player_active_pokemon"]
    opponent = state["opponent_active_pokemon"]
    if 4 <= action_id <= 8:
        actor = sorted_switches(state)[action_id - 4]
        row_move: dict[str, Any] | None = None
    else:
        actor = active
        move_index = action_id - 9 if action_id >= 9 else action_id
        row_move = sorted_moves(active)[move_index]
    generation = _generation(state)
    move = _move_data(str(row_move.get("name", "")), generation) if row_move else None
    actor_types = (*_types(actor), "", "")
    opponent_types = (*_types(opponent), "", "")
    actor_moves = [str(item.get("name", "")) for item in sorted_moves(actor)][:4]
    opponent_moves = [str(item.get("name", "")) for item in sorted_moves(opponent)][:4]
    actor_moves.extend([""] * (4 - len(actor_moves)))
    opponent_moves.extend([""] * (4 - len(opponent_moves)))
    values: dict[str, int] = {
        "candidate_move": _move_identity(row_move.get("name")) if row_move else 0,
        "actor_species": _species_identity(actor),
        "opponent_species": _species_identity(opponent),
        "actor_item": _identity("item", actor.get("item")),
        "actor_ability": _identity("ability", actor.get("ability")),
        "opponent_item": _identity("item", opponent.get("item")),
        "opponent_ability": _identity("ability", opponent.get("ability")),
        "actor_type_1": _identity("type", actor_types[0]),
        "actor_type_2": _identity("type", actor_types[1]),
        "opponent_type_1": _identity("type", opponent_types[0]),
        "opponent_type_2": _identity("type", opponent_types[1]),
        "actor_tera_type": _identity("type", actor.get("tera_type")),
        "move_type": _identity(
            "type",
            getattr(move, "type", row_move.get("move_type") if row_move else ""),
        ),
        "actor_status": _identity("status", actor.get("status")),
        "opponent_status": _identity("status", opponent.get("status")),
        "weather": _identity("weather", state.get("weather")),
        "terrain": _identity("terrain", state.get("battle_field")),
        "actor_effect": _identity("effect", actor.get("effect")),
        "opponent_effect": _identity("effect", opponent.get("effect")),
        "move_side_condition": _identity("side_condition", getattr(move, "side_condition", None)),
        "move_volatile": _identity("effect", getattr(move, "volatile_status", None)),
        "move_target": _identity("target", getattr(move, "target", None)),
        "previous_player_move": _move_identity(_previous_move_name(state.get("player_prev_move"))),
        "previous_opponent_move": _move_identity(
            _previous_move_name(state.get("opponent_prev_move"))
        ),
    }
    for index, name in enumerate(actor_moves, 1):
        values[f"actor_move_{index}"] = _move_identity(name)
    for index, name in enumerate(opponent_moves, 1):
        values[f"opponent_move_{index}"] = _move_identity(name)
    return [values[name] for name in MECHANICS_IDENTITY_NAMES]


def candidate_identity_matrix(state: dict[str, Any]) -> list[list[int]]:
    legal = set(legal_action_ids(state))
    return [
        candidate_identity_vector(state, action_id)
        if action_id in legal
        else [0] * MECHANICS_IDENTITY_COUNT
        for action_id in range(ACTION_COUNT)
    ]


def _condition_values(raw: Any) -> dict[str, float]:
    text = _normalized_id(raw)
    spikes = re.search(r"(?<!toxic)spikes([123])?", text)
    toxic = re.search(r"toxicspikes([12])?", text)
    return {
        "stealth_rock": float("stealthrock" in text),
        "spikes_layers": (float(spikes.group(1) or 1) / 3.0) if spikes else 0.0,
        "toxic_spikes_layers": (float(toxic.group(1) or 1) / 2.0) if toxic else 0.0,
        "sticky_web": float("stickyweb" in text),
        "reflect": float("reflect" in text),
        "light_screen": float("lightscreen" in text),
        "aurora_veil": float("auroraveil" in text),
        "tailwind": float("tailwind" in text),
        "safeguard": float("safeguard" in text),
    }


def _estimated_speed(pokemon: dict[str, Any], weather: str, conditions: Any) -> float:
    speed = _estimated_stat(pokemon.get("base_spe"), pokemon.get("lvl"))
    speed *= _stage_multiplier(pokemon.get("spe_boost"))
    status = _status_name(pokemon.get("status"))
    ability = _normalized_id(pokemon.get("ability"))
    item = _normalized_id(pokemon.get("item"))
    weather_id = _normalized_id(weather)
    if status == "paralysis" and ability != "quickfeet":
        speed *= 0.5
    if item == "choicescarf":
        speed *= 1.5
    if ability == "quickfeet" and status is not None:
        speed *= 1.5
    if (ability == "swiftswim" and "rain" in weather_id) or (
        ability == "chlorophyll" and "sun" in weather_id
    ) or (ability == "sandrush" and "sand" in weather_id) or (
        ability == "slushrush" and any(name in weather_id for name in ("snow", "hail"))
    ):
        speed *= 2.0
    if "tailwind" in _normalized_id(conditions):
        speed *= 2.0
    return speed


def _pokemon_weight(pokemon: dict[str, Any], generation: int) -> float | None:
    data = GenData.from_gen(generation).pokedex
    for field in ("name", "base_species"):
        entry = data.get(_normalized_id(pokemon.get(field)))
        if entry and entry.get("weightkg") is not None:
            return _number(entry["weightkg"])
    return None


def _variable_power(
    move_id: str,
    power: float,
    active: dict[str, Any],
    opponent: dict[str, Any],
    actor_speed: float,
    opponent_speed: float,
    generation: int,
    player_remaining: float,
) -> tuple[float, bool]:
    active_hp = _clamp(_number(active.get("hp_pct"), 1.0))
    opponent_hp = _clamp(_number(opponent.get("hp_pct"), 1.0))
    active_status = _status_name(active.get("status"))
    opponent_status = _status_name(opponent.get("status"))
    active_item = _normalized_id(active.get("item"))
    changed = move_id in _VARIABLE_POWER_MOVES
    if move_id in {"eruption", "waterspout", "dragonenergy"}:
        power = 150.0 * active_hp
    elif move_id in {"flail", "reversal"}:
        if active_hp <= 1 / 48:
            power = 200.0
        elif active_hp <= 1 / 5:
            power = 150.0
        elif active_hp <= 1 / 3:
            power = 100.0
        elif active_hp <= 1 / 2:
            power = 80.0
        else:
            power = 40.0
    elif move_id in {"hex", "venoshock", "barbbarrage", "infernalparade"} and opponent_status:
        power *= 2.0
    elif move_id == "facade" and active_status in {"burn", "poison", "toxic", "paralysis"}:
        power *= 2.0
    elif move_id == "brine" and opponent_hp <= 0.5:
        power *= 2.0
    elif move_id == "knockoff" and _normalized_id(opponent.get("item")) not in {
        "",
        "unknownitem",
        "noitem",
    }:
        power *= 1.5
    elif move_id == "acrobatics" and active_item in {"", "noitem"}:
        power *= 2.0
    elif move_id in {"storedpower", "powertrip"}:
        positive = sum(max(_number(active.get(f"{stat}_boost")), 0.0) for stat in BOOST_STATS)
        power = 20.0 + 20.0 * positive
    elif move_id == "gyroball":
        power = min(150.0, max(1.0, 25.0 * opponent_speed / max(actor_speed, 1.0) + 1.0))
    elif move_id == "electroball":
        ratio = actor_speed / max(opponent_speed, 1.0)
        if ratio >= 4:
            power = 150.0
        elif ratio >= 3:
            power = 120.0
        elif ratio >= 2:
            power = 80.0
        elif ratio > 1:
            power = 60.0
        else:
            power = 40.0
    elif move_id in {"lowkick", "grassknot"}:
        weight = _pokemon_weight(opponent, generation)
        if weight is not None:
            thresholds = ((10, 20), (25, 40), (50, 60), (100, 80), (200, 100))
            power = next((value for limit, value in thresholds if weight < limit), 120.0)
    elif move_id in {"heavyslam", "heatcrash"}:
        actor_weight = _pokemon_weight(active, generation)
        target_weight = _pokemon_weight(opponent, generation)
        if actor_weight is not None and target_weight:
            ratio = actor_weight / target_weight
            power = 120.0 if ratio >= 5 else 100.0 if ratio >= 4 else 80.0
            if ratio < 3:
                power = 60.0 if ratio >= 2 else 40.0
    elif move_id == "lastrespects":
        power = min(300.0, 50.0 + 50.0 * max(6.0 - player_remaining, 0.0))
    return power, changed


def _correct_stab(move_type: str, active: dict[str, Any], terastallize: bool) -> float:
    normalized_type = _normalized_id(move_type)
    original_types = _types(active)
    if not terastallize:
        return 1.5 if normalized_type in original_types else 1.0
    tera_type = _normalized_id(active.get("tera_type"))
    if normalized_type == tera_type:
        return 2.0 if normalized_type in original_types else 1.5
    # Terastallization retains STAB for the Pokémon's original types.
    return 1.5 if normalized_type in original_types else 1.0


def _correct_damage(
    state: dict[str, Any],
    action_id: int,
    values: dict[str, float],
    move: Any,
    row_move: dict[str, Any],
    actor_speed: float,
    opponent_speed: float,
    generation: int,
) -> None:
    active = state["player_active_pokemon"]
    opponent = state["opponent_active_pokemon"]
    move_id = move.id if move is not None else _normalized_id(row_move.get("name"))
    category = _normalized_id(
        getattr(getattr(move, "category", None), "name", row_move.get("category"))
    )
    move_type = _normalized_id(
        getattr(getattr(move, "type", None), "name", row_move.get("move_type"))
    )
    terastallize = action_id >= 9
    if terastallize and move_id == "terablast":
        move_type = _normalized_id(active.get("tera_type")) or move_type
        category = (
            "physical"
            if _number(active.get("base_atk")) > _number(active.get("base_spa"))
            else "special"
        )
    power = _number(getattr(move, "base_power", row_move.get("base_power")))
    power, variable = _variable_power(
        move_id,
        power,
        active,
        opponent,
        actor_speed,
        opponent_speed,
        generation,
        _number(state.get("player_remaining"), len(state.get("available_switches") or []) + 1),
    )
    values["variable_base_power"] = float(variable)
    values["base_power_fraction"] = _clamp(power / 200.0)
    if category not in {"physical", "special"} or power <= 0:
        return

    attack_source = active
    attack_key = "base_atk" if category == "physical" else "base_spa"
    attack_boost = "atk_boost" if category == "physical" else "spa_boost"
    defense_key = "base_def" if category == "physical" else "base_spd"
    defense_boost = "def_boost" if category == "physical" else "spd_boost"
    special_rule = False
    if move_id in _TARGET_DEFENSE_MOVES:
        defense_key, defense_boost, special_rule = "base_def", "def_boost", True
    elif move_id in _USER_DEFENSE_MOVES:
        attack_key, attack_boost, special_rule = "base_def", "def_boost", True
    elif move_id in _TARGET_ATTACK_MOVES:
        attack_source = opponent
        attack_key, attack_boost, special_rule = "base_atk", "atk_boost", True
    values["uses_target_defense"] = float(move_id in _TARGET_DEFENSE_MOVES)
    values["uses_user_defense"] = float(move_id in _USER_DEFENSE_MOVES)
    values["uses_target_attack"] = float(move_id in _TARGET_ATTACK_MOVES)
    values["damage_accounts_for_special_rule"] = float(special_rule or variable)

    attacker_stat = _estimated_stat(attack_source.get(attack_key), attack_source.get("lvl"))
    defender_stat = _estimated_stat(opponent.get(defense_key), opponent.get("lvl"))
    if _normalized_id(opponent.get("ability")) != "unaware":
        attacker_stat *= _stage_multiplier(attack_source.get(attack_boost))
    if _normalized_id(active.get("ability")) != "unaware":
        defender_stat *= _stage_multiplier(opponent.get(defense_boost))
    item = _normalized_id(active.get("item"))
    opponent_item = _normalized_id(opponent.get("item"))
    item_modifier = 1.0
    if item == "lifeorb":
        item_modifier *= 1.3
    elif item == "choiceband" and category == "physical":
        item_modifier *= 1.5
    elif item == "choicespecs" and category == "special":
        item_modifier *= 1.5
    if opponent_item == "assaultvest" and category == "special" and defense_key == "base_spd":
        defender_stat *= 1.5
    values["damage_accounts_for_item"] = float(
        item_modifier != 1.0 or opponent_item == "assaultvest"
    )

    flags = {_normalized_id(flag) for flag in (move.flags if move is not None else ())}
    ability = _normalized_id(active.get("ability"))
    ability_modifier = 1.0
    if ability == "technician" and power <= 60:
        ability_modifier *= 1.5
    if ability == "toughclaws" and "contact" in flags:
        ability_modifier *= 1.3
    if ability == "ironfist" and "punch" in flags:
        ability_modifier *= 1.2
    if ability == "strongjaw" and "bite" in flags:
        ability_modifier *= 1.5
    if ability == "sharpness" and "slicing" in flags:
        ability_modifier *= 1.5
    if ability == "megalauncher" and "pulse" in flags:
        ability_modifier *= 1.5
    if ability == "sheerforce" and getattr(move, "secondary", None):
        ability_modifier *= 1.3
    values["damage_accounts_for_ability"] = float(ability_modifier != 1.0)

    stab = _correct_stab(move_type, active, terastallize)
    if ability == "adaptability" and stab > 1.0:
        stab = 2.0
    values["stab_scaled"] = _clamp(stab / 2.0)
    multiplier, _ = _type_multiplier(
        move_type,
        opponent,
        generation,
        move_flags=set(move.flags) if move is not None else set(),
        attacker_ability=str(active.get("ability", "")),
    )
    level = max(_number(active.get("lvl"), 100.0), 1.0)
    base_damage = (
        (2.0 * level / 5.0 + 2.0)
        * power
        * attacker_stat
        / max(defender_stat, 1.0)
        / 50.0
        + 2.0
    )
    modifier = stab * multiplier * _number(getattr(move, "expected_hits", 1.0), 1.0)
    weather = _normalized_id(state.get("weather"))
    weather_modifier = 1.0
    if "rain" in weather:
        weather_modifier = 1.5 if move_type == "water" else 0.5 if move_type == "fire" else 1.0
    elif "sun" in weather:
        weather_modifier = 1.5 if move_type == "fire" else 0.5 if move_type == "water" else 1.0
    modifier *= weather_modifier
    values["damage_accounts_for_weather"] = float(weather_modifier != 1.0)
    terrain = _normalized_id(state.get("battle_field"))
    terrain_modifier = 1.0
    if ("electricterrain" in terrain and move_type == "electric") or (
        "grassyterrain" in terrain and move_type == "grass"
    ) or ("psychicterrain" in terrain and move_type == "psychic"):
        terrain_modifier = 1.3
    modifier *= terrain_modifier
    values["damage_accounts_for_terrain"] = float(terrain_modifier != 1.0)
    if (
        category == "physical"
        and _status_name(active.get("status")) == "burn"
        and move_id != "facade"
    ):
        modifier *= 0.5
    opponent_ability = _normalized_id(opponent.get("ability"))
    if opponent_ability == "multiscale" and _number(opponent.get("hp_pct"), 1.0) >= 1.0:
        modifier *= 0.5
    if opponent_ability in {"filter", "prismarmor", "solidrock"} and multiplier > 1.0:
        modifier *= 0.75
    if opponent_ability == "furcoat" and category == "physical":
        modifier *= 0.5
    if opponent_ability == "icescales" and category == "special":
        modifier *= 0.5
    if opponent_ability == "thickfat" and move_type in {"fire", "ice"}:
        modifier *= 0.5
    maximum_damage = base_damage * modifier * item_modifier * ability_modifier
    target_hp = _estimated_stat(opponent.get("base_hp"), opponent.get("lvl"), hp=True)
    target_hp *= max(_number(opponent.get("hp_pct"), 1.0), 0.01)
    maximum_fraction = maximum_damage / max(target_hp, 1.0)
    minimum_fraction = maximum_fraction * 0.85
    values.update(
        {
            "damage_estimate_available": 1.0,
            "damage_min_target_hp": _clamp(minimum_fraction / 2.0),
            "damage_max_target_hp": _clamp(maximum_fraction / 2.0),
            "ko_possible": float(maximum_fraction >= 1.0),
            "ko_guaranteed": float(minimum_fraction >= 1.0),
        }
    )


def _candidate_defense(
    state: dict[str, Any],
    action_id: int,
    actor: dict[str, Any],
    values: dict[str, float],
    generation: int,
) -> None:
    defender = actor
    if action_id >= 9:
        tera_type = _normalized_id(actor.get("tera_type"))
        if tera_type and tera_type != "notype":
            defender = actor | {"types": tera_type}
    multipliers: list[float] = []
    opponent = state["opponent_active_pokemon"]
    for row_move in sorted_moves(opponent):
        move = _move_data(str(row_move.get("name", "")), generation)
        category = _normalized_id(
            getattr(getattr(move, "category", None), "name", row_move.get("category"))
        )
        power = _number(getattr(move, "base_power", row_move.get("base_power")))
        if category not in {"physical", "special"} or power <= 0:
            continue
        move_type = _normalized_id(
            getattr(getattr(move, "type", None), "name", row_move.get("move_type"))
        )
        multiplier, _ = _type_multiplier(
            move_type,
            defender,
            generation,
            move_flags=set(move.flags) if move is not None else set(),
            attacker_ability=str(opponent.get("ability", "")),
        )
        multipliers.append(multiplier)
    if not multipliers:
        return
    total = len(multipliers)
    values.update(
        {
            "candidate_known_defensive_worst": _clamp(max(multipliers) / 4.0),
            "candidate_known_move_immunity_fraction": (
                sum(value == 0 for value in multipliers) / total
            ),
            "candidate_known_move_resistance_fraction": (
                sum(0 < value < 1 for value in multipliers) / total
            ),
            "candidate_known_move_weakness_fraction": (
                sum(value > 1 for value in multipliers) / total
            ),
            "candidate_opponent_moves_known_fraction": _clamp(total / 4.0),
        }
    )


def candidate_feature_vector(state: dict[str, Any], action_id: int) -> list[float]:
    if action_id not in legal_action_ids(state):
        raise ValueError(f"A{action_id} is not legal in this state")
    values = {name: 0.0 for name in MECHANICS_FEATURE_NAMES}
    values.update(dict(zip(V1_FEATURE_NAMES, v1_candidate_feature_vector(state, action_id))))
    active = state["player_active_pokemon"]
    opponent = state["opponent_active_pokemon"]
    actor = sorted_switches(state)[action_id - 4] if 4 <= action_id <= 8 else active
    generation = _generation(state)

    for owner_name, pokemon in (("player", active), ("opponent", opponent)):
        for stat in ("hp", "atk", "def", "spa", "spd", "spe"):
            values[f"{owner_name}_base_{stat}"] = _clamp(
                _number(pokemon.get(f"base_{stat}")) / 255.0
            )
        for stat in BOOST_STATS:
            values[f"{owner_name}_{stat}_stage"] = _clamp(
                _number(pokemon.get(f"{stat}_boost")) / 6.0,
                -1.0,
                1.0,
            )
    for side, raw in (
        ("player", state.get("player_conditions")),
        ("opponent", state.get("opponent_conditions")),
    ):
        for name, value in _condition_values(raw).items():
            values[f"{side}_side_{name}"] = value
    field = _normalized_id(state.get("battle_field"))
    values["field_trick_room"] = float("trickroom" in field)
    actor_conditions = state.get("player_conditions")
    actor_speed = _estimated_speed(actor, str(state.get("weather", "")), actor_conditions)
    opponent_speed = _estimated_speed(
        opponent,
        str(state.get("weather", "")),
        state.get("opponent_conditions"),
    )
    values["actor_estimated_speed"] = _clamp(actor_speed / 1_000.0)
    values["opponent_estimated_speed"] = _clamp(opponent_speed / 1_000.0)
    faster = actor_speed > opponent_speed
    values["actor_faster_without_priority"] = float(faster)

    if 4 <= action_id <= 8:
        values["actor_moves_first_estimate"] = float(faster != bool(values["field_trick_room"]))
        _candidate_defense(state, action_id, actor, values, generation)
    else:
        move_index = action_id - 9 if action_id >= 9 else action_id
        row_move = sorted_moves(active)[move_index]
        move = _move_data(str(row_move.get("name", "")), generation)
        move_id = move.id if move is not None else _normalized_id(row_move.get("name"))
        priority = _number(getattr(move, "priority", row_move.get("priority")))
        values["actor_moves_first_estimate"] = float(
            priority > 0 or (priority == 0 and faster != bool(values["field_trick_room"]))
        )
        flags = {_normalized_id(flag) for flag in (move.flags if move is not None else ())}
        for name in MOVE_FLAG_NAMES:
            values[f"move_flag_{name}"] = float(_normalized_id(name) in flags)
        side_condition = _effect_id(getattr(move, "side_condition", None))
        side_mapping = {
            "stealthrock": "sets_stealth_rock",
            "spikes": "sets_spikes",
            "toxicspikes": "sets_toxic_spikes",
            "stickyweb": "sets_sticky_web",
            "reflect": "sets_reflect",
            "lightscreen": "sets_light_screen",
            "auroraveil": "sets_aurora_veil",
            "tailwind": "sets_tailwind",
            "safeguard": "sets_safeguard",
        }
        if side_condition:
            values[side_mapping.get(side_condition, "sets_other_side_condition")] = 1.0
        values["hazard_control_user_side"] = float(move_id in {"mortalspin", "rapidspin"})
        values["hazard_control_both_sides"] = float(move_id in {"defog", "tidyup"})
        values["swaps_side_conditions"] = float(move_id == "courtchange")
        values["conditional_priority"] = float(move_id in _CONDITIONAL_PRIORITY_MOVES)
        values["item_interaction"] = float(move_id in _ITEM_INTERACTION_MOVES)
        values["calls_another_move"] = float(move_id in _CALLS_MOVE)
        values["delayed_attack"] = float(move_id in _DELAYED_ATTACKS)
        values["delayed_heal"] = float(move_id == "wish")
        values["full_recovery"] = float(move_id == "rest")
        values["cures_user_status"] = float(
            move_id
            in {
                "aromatherapy",
                "healingwish",
                "healbell",
                "junglehealing",
                "lunarblessing",
                "rest",
            }
        )
        values["requires_sleep"] = float(move_id in {"sleeptalk", "snore"})
        values["creates_substitute"] = float(move_id == "substitute")
        values["removes_field_effect"] = float(move_id in {"defog", "icespinner", "steelroller"})
        values["self_locked_move"] = float(move_id in _LOCKED_MOVES)
        volatile = _effect_id(getattr(move, "volatile_status", None))
        volatile_mapping = {
            "leechseed": "target_leech_seed",
            "taunt": "target_taunt",
            "encore": "target_encore",
            "disable": "target_disable",
            "healblock": "target_heal_block",
            "saltcure": "target_salt_cure",
            "partiallytrapped": "target_partial_trap",
            "yawn": "target_yawn",
            "perishsong": "target_perish_song",
            "torment": "target_torment",
        }
        if volatile:
            values[volatile_mapping.get(volatile, "other_volatile_effect")] = 1.0
        _correct_damage(
            state,
            action_id,
            values,
            move,
            row_move,
            actor_speed,
            opponent_speed,
            generation,
        )
        _candidate_defense(state, action_id, actor, values, generation)

    vector = [float(values[name]) for name in MECHANICS_FEATURE_NAMES]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Mechanics-v2 feature generation produced a non-finite value")
    return vector


def candidate_feature_matrix(state: dict[str, Any]) -> list[list[float]]:
    legal = set(legal_action_ids(state))
    return [
        candidate_feature_vector(state, action_id)
        if action_id in legal
        else [0.0] * MECHANICS_FEATURE_COUNT
        for action_id in range(ACTION_COUNT)
    ]
