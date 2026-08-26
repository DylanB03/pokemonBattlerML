from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

from poke_env.battle import Move, PokemonType
from poke_env.data import GenData

from pokemon_battler.core.actions import (
    ACTION_COUNT,
    legal_action_ids,
    sorted_moves,
    sorted_switches,
)
from pokemon_battler.showdown.poke_env_compat import install_safe_poke_env_shutdown

install_safe_poke_env_shutdown()

MECHANICS_SCHEMA = "mechanics-v1"
BOOST_STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
STATUS_NAMES = ("burn", "poison", "toxic", "paralysis", "sleep", "freeze")

# The order is checkpoint ABI. Additions require a new schema name so a saved head
# can never silently interpret a feature column with different semantics.
MECHANICS_FEATURE_NAMES = (
    "bias",
    "action_move",
    "action_switch",
    "action_tera",
    "forced_switch",
    "tera_available",
    "turn_fraction",
    "player_hp_fraction",
    "opponent_hp_fraction",
    "player_remaining_fraction",
    "opponent_remaining_fraction",
    "move_physical",
    "move_special",
    "move_status",
    "base_power_fraction",
    "accuracy",
    "priority_scaled",
    "pp_fraction",
    "stab_scaled",
    "effectiveness_scaled",
    "effectiveness_immune",
    "effectiveness_quarter",
    "effectiveness_half",
    "effectiveness_neutral",
    "effectiveness_double",
    "effectiveness_quadruple",
    "effectiveness_other",
    "ability_immunity_applied",
    "expected_hits_fraction",
    "damage_estimate_available",
    "damage_min_target_hp",
    "damage_max_target_hp",
    "ko_possible",
    "ko_guaranteed",
    "fixed_damage",
    "heal_fraction",
    "drain_fraction",
    "recoil_fraction",
    "self_destruct",
    "self_switch",
    "force_switch",
    "protect_move",
    "stalling_move",
    "breaks_protect",
    "high_critical_rate",
    "sets_hazard",
    "removes_hazard",
    "sets_weather",
    "sets_terrain",
    "custom_effect_unmodeled",
    *(f"target_status_{name}_chance" for name in STATUS_NAMES),
    "target_confusion_chance",
    "target_flinch_chance",
    *(f"self_{stat}_delta" for stat in BOOST_STATS),
    *(f"target_{stat}_delta" for stat in BOOST_STATS),
    "switch_hp_fraction",
    "switch_post_entry_hp_fraction",
    "switch_entry_damage_fraction",
    "switch_status_burn",
    "switch_status_poison",
    "switch_status_toxic",
    "switch_status_paralysis",
    "switch_status_sleep",
    "switch_status_freeze",
    "switch_base_hp",
    "switch_base_atk",
    "switch_base_def",
    "switch_base_spa",
    "switch_base_spd",
    "switch_base_spe",
    "switch_faster_by_base_stat",
    "switch_best_offensive_effectiveness",
    "switch_best_damage_pressure",
    "switch_known_defensive_worst",
    "switch_known_move_immunity_fraction",
    "switch_known_move_resistance_fraction",
    "switch_known_move_weakness_fraction",
    "switch_opponent_moves_known_fraction",
    "switch_hazard_toxic_spikes",
    "switch_hazard_sticky_web",
)
MECHANICS_FEATURE_COUNT = len(MECHANICS_FEATURE_NAMES)

_STATUS_KEYS = {
    "brn": "burn",
    "burn": "burn",
    "psn": "poison",
    "poison": "poison",
    "tox": "toxic",
    "toxic": "toxic",
    "par": "paralysis",
    "paralysis": "paralysis",
    "slp": "sleep",
    "sleep": "sleep",
    "frz": "freeze",
    "freeze": "freeze",
}
_HAZARD_MOVES = {
    "ceaselessedge",
    "gmaxstonesurge",
    "gmaxsteelsurge",
    "spikes",
    "stealthrock",
    "stickyweb",
    "stoneaxe",
    "toxicspikes",
}
_HAZARD_REMOVAL_MOVES = {
    "courtchange",
    "defog",
    "mortalspin",
    "rapidspin",
    "tidyup",
}
_FORCE_SWITCH_MOVES = {
    "circlethrow",
    "dragon tail",
    "dragontail",
    "roar",
    "whirlwind",
}
_ABILITY_TYPE_IMMUNITIES = {
    "eartheater": "ground",
    "flashfire": "fire",
    "levitate": "ground",
    "lightningrod": "electric",
    "motordrive": "electric",
    "sapsipper": "grass",
    "stormdrain": "water",
    "voltabsorb": "electric",
    "waterabsorb": "water",
    "wellbakedbody": "fire",
}
_ABILITY_BYPASS = {"moldbreaker", "teravolt", "turboblaze"}
_CUSTOM_EFFECT_FLAGS = {
    "basePowerCallback",
    "beforeMoveCallback",
    "beforeTurnCallback",
    "damageCallback",
    "hasCustomRecoil",
    "onAfterHit",
    "onAfterMove",
    "onAfterMoveSecondarySelf",
    "onBasePower",
    "onEffectiveness",
    "onHit",
    "onHitField",
    "onModifyMove",
    "onMoveFail",
    "onPrepareHit",
    "onTry",
    "onTryHit",
    "onTryHitSide",
    "onTryMove",
}


def _normalized_id(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(max(value, lower), upper)


def _types(pokemon: dict[str, Any]) -> tuple[str, ...]:
    raw = pokemon.get("types") or ""
    values = raw if isinstance(raw, (list, tuple)) else str(raw).split()
    return tuple(
        value
        for item in values
        if (value := _normalized_id(item)) not in {"", "notype", "unknown"}
    )


def _generation(state: dict[str, Any]) -> int:
    match = re.search(r"gen(\d+)", str(state.get("format", "gen9")), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 9


@lru_cache(maxsize=4096)
def _move_data(name: str, generation: int) -> Move | None:
    try:
        return Move(Move.retrieve_id(name), generation)
    except (KeyError, TypeError, ValueError):
        return None


@lru_cache(maxsize=256)
def _type_multiplier_cached(
    move_type: str,
    defender_types: tuple[str, ...],
    generation: int,
) -> float:
    if not move_type or not defender_types:
        return 1.0
    try:
        attack_type = PokemonType.from_name(move_type)
        target_types = [PokemonType.from_name(value) for value in defender_types]
        return float(
            attack_type.damage_multiplier(
                *target_types,
                type_chart=GenData.from_gen(generation).type_chart,
            )
        )
    except (KeyError, TypeError, ValueError):
        return 1.0


def _type_multiplier(
    move_type: str,
    defender: dict[str, Any],
    generation: int,
    *,
    move_flags: set[str] | None = None,
    attacker_ability: str = "",
) -> tuple[float, bool]:
    normalized_type = _normalized_id(move_type)
    value = _type_multiplier_cached(normalized_type, _types(defender), generation)
    ability = _normalized_id(defender.get("ability"))
    bypasses_ability = _normalized_id(attacker_ability) in _ABILITY_BYPASS
    ability_immunity = False
    if not bypasses_ability:
        ability_immunity = _ABILITY_TYPE_IMMUNITIES.get(ability) == normalized_type
        flags = move_flags or set()
        if ability == "soundproof" and "sound" in flags:
            ability_immunity = True
        elif ability == "bulletproof" and "bullet" in flags:
            ability_immunity = True
        elif ability == "windrider" and "wind" in flags:
            ability_immunity = True
        elif ability == "wonderguard" and 0 < value <= 1:
            ability_immunity = True
    return (0.0 if ability_immunity else value), ability_immunity


def _stage_multiplier(stage: Any) -> float:
    value = max(-6.0, min(6.0, _number(stage)))
    return (2.0 + value) / 2.0 if value >= 0 else 2.0 / (2.0 - value)


def _estimated_stat(base: Any, level: Any, *, hp: bool = False) -> float:
    # Replays omit EVs, IVs, and natures. Neutral 31-IV/0-EV stats make this a
    # reproducible pressure estimate, not an assertion of exact damage.
    base_value = max(_number(base, 1.0), 1.0)
    level_value = max(_number(level, 100.0), 1.0)
    core = math.floor((2.0 * base_value + 31.0) * level_value / 100.0)
    return core + level_value + 10.0 if hp else core + 5.0


def _tera_stab(
    move_type: str,
    active: dict[str, Any],
    terastallize: bool,
) -> float:
    normalized_type = _normalized_id(move_type)
    current_types = _types(active)
    if terastallize:
        tera_type = _normalized_id(active.get("tera_type"))
        if normalized_type == tera_type:
            return 2.0 if normalized_type in current_types else 1.5
        return 1.0
    return 1.5 if normalized_type in current_types else 1.0


def _boost_values() -> dict[str, float]:
    return {stat: 0.0 for stat in BOOST_STATS}


def _add_boosts(target: dict[str, float], boosts: Any, chance: float = 1.0) -> None:
    if not isinstance(boosts, dict):
        return
    for stat, amount in boosts.items():
        normalized = _normalized_id(stat)
        if normalized in target:
            target[normalized] += _number(amount) * chance


def _status_name(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "name", value)
    return _STATUS_KEYS.get(_normalized_id(raw))


def _condition_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items()).lower()
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value).lower()
    return str(value or "").lower()


def _hazard_entry(
    state: dict[str, Any],
    pokemon: dict[str, Any],
    generation: int,
) -> tuple[float, float, float]:
    conditions = _condition_text(state.get("player_conditions"))
    item = _normalized_id(pokemon.get("item"))
    ability = _normalized_id(pokemon.get("ability"))
    if item == "heavydutyboots" or ability == "magicguard":
        return 0.0, 0.0, 0.0

    types = set(_types(pokemon))
    grounded = "flying" not in types and ability != "levitate"
    damage = 0.0
    normalized_conditions = _normalized_id(conditions)
    if "stealthrock" in normalized_conditions:
        rock, _ = _type_multiplier("rock", pokemon, generation)
        damage += 0.125 * rock
    # `toxicspikes` must not be interpreted as ordinary Spikes damage.
    spikes_match = re.search(r"(?<!toxic)spikes([123])?", normalized_conditions)
    if spikes_match and grounded:
        layers = int(spikes_match.group(1) or 1)
        damage += {1: 0.125, 2: 1.0 / 6.0, 3: 0.25}[layers]
    toxic_spikes = float("toxicspikes" in normalized_conditions and grounded)
    sticky_web = float("stickyweb" in normalized_conditions and grounded)
    return _clamp(damage), toxic_spikes, sticky_web


def _base_features(state: dict[str, Any], action_id: int) -> dict[str, float]:
    active = state["player_active_pokemon"]
    opponent = state["opponent_active_pokemon"]
    values = {name: 0.0 for name in MECHANICS_FEATURE_NAMES}
    values.update(
        {
            "bias": 1.0,
            "action_move": float(action_id <= 3 or action_id >= 9),
            "action_switch": float(4 <= action_id <= 8),
            "action_tera": float(action_id >= 9),
            "forced_switch": float(bool(state.get("forced_switch", False))),
            "tera_available": float(bool(state.get("can_tera", False))),
            "turn_fraction": _clamp(_number(state.get("turn_index")) / 100.0),
            "player_hp_fraction": _clamp(_number(active.get("hp_pct"))),
            "opponent_hp_fraction": _clamp(_number(opponent.get("hp_pct"))),
            "player_remaining_fraction": _clamp(
                _number(
                    state.get("player_remaining"),
                    len(state.get("available_switches") or []) + 1,
                )
                / 6.0
            ),
            "opponent_remaining_fraction": _clamp(
                _number(state.get("opponents_remaining"), 6.0) / 6.0
            ),
        }
    )
    return values


def _effectiveness_features(values: dict[str, float], multiplier: float) -> None:
    values["effectiveness_scaled"] = _clamp(multiplier / 4.0)
    buckets = {
        0.0: "effectiveness_immune",
        0.25: "effectiveness_quarter",
        0.5: "effectiveness_half",
        1.0: "effectiveness_neutral",
        2.0: "effectiveness_double",
        4.0: "effectiveness_quadruple",
    }
    key = min(buckets, key=lambda candidate: abs(candidate - multiplier))
    if abs(key - multiplier) < 1e-6:
        values[buckets[key]] = 1.0
    else:
        values["effectiveness_other"] = 1.0


def _move_features(
    state: dict[str, Any],
    action_id: int,
    values: dict[str, float],
    generation: int,
) -> None:
    active = state["player_active_pokemon"]
    opponent = state["opponent_active_pokemon"]
    terastallize = action_id >= 9
    move_index = action_id - 9 if terastallize else action_id
    row_move = sorted_moves(active)[move_index]
    move = _move_data(str(row_move.get("name", "")), generation)
    move_id = move.id if move is not None else _normalized_id(row_move.get("name"))
    category = (
        move.category.name.lower()
        if move is not None
        else _normalized_id(row_move.get("category", "status"))
    )
    move_type = (
        move.type.name.lower()
        if move is not None
        else _normalized_id(row_move.get("move_type"))
    )
    if terastallize and move_id == "terablast":
        move_type = _normalized_id(active.get("tera_type")) or move_type
        category = (
            "physical"
            if _number(active.get("base_atk")) > _number(active.get("base_spa"))
            else "special"
        )
    base_power = _number(move.base_power if move is not None else row_move.get("base_power"))
    accuracy = _number(move.accuracy if move is not None else row_move.get("accuracy"), 1.0)
    priority = _number(move.priority if move is not None else row_move.get("priority"))
    current_pp = _number(row_move.get("current_pp"))
    max_pp = _number(row_move.get("max_pp"))
    flags = set(move.flags) if move is not None else set()
    multiplier, ability_immunity = _type_multiplier(
        move_type,
        opponent,
        generation,
        move_flags=flags,
        attacker_ability=str(active.get("ability", "")),
    )
    stab = _tera_stab(move_type, active, terastallize)
    expected_hits = _number(move.expected_hits if move is not None else 1.0, 1.0)

    values.update(
        {
            "move_physical": float(category == "physical"),
            "move_special": float(category == "special"),
            "move_status": float(category == "status"),
            "base_power_fraction": _clamp(base_power / 200.0),
            "accuracy": _clamp(accuracy),
            "priority_scaled": _clamp(priority / 7.0, -1.0, 1.0),
            "pp_fraction": _clamp(current_pp / max_pp) if max_pp > 0 else 0.0,
            "stab_scaled": _clamp(stab / 2.0),
            "ability_immunity_applied": float(ability_immunity),
            "expected_hits_fraction": _clamp(expected_hits / 5.0),
        }
    )
    _effectiveness_features(values, multiplier)

    if category in {"physical", "special"} and base_power > 0:
        attack_key = "base_atk" if category == "physical" else "base_spa"
        attack_boost = "atk_boost" if category == "physical" else "spa_boost"
        defense_key = "base_def" if category == "physical" else "base_spd"
        defense_boost = "def_boost" if category == "physical" else "spd_boost"
        attacker_stat = _estimated_stat(active.get(attack_key), active.get("lvl"))
        defender_stat = _estimated_stat(opponent.get(defense_key), opponent.get("lvl"))
        attacker_stat *= _stage_multiplier(active.get(attack_boost))
        defender_stat *= _stage_multiplier(opponent.get(defense_boost))
        level = max(_number(active.get("lvl"), 100.0), 1.0)
        base_damage = (
            (2.0 * level / 5.0 + 2.0)
            * base_power
            * attacker_stat
            / max(defender_stat, 1.0)
            / 50.0
            + 2.0
        )
        maximum_damage = base_damage * stab * multiplier * expected_hits
        if category == "physical" and _normalized_id(active.get("status")) in {"brn", "burn"}:
            maximum_damage *= 0.5
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
    if move is None:
        values["custom_effect_unmodeled"] = 1.0
        return

    fixed_damage = move.damage
    if fixed_damage:
        fixed = (
            _number(active.get("lvl"), 100.0)
            if fixed_damage == "level"
            else _number(fixed_damage)
        )
        target_hp = _estimated_stat(opponent.get("base_hp"), opponent.get("lvl"), hp=True)
        target_hp *= max(_number(opponent.get("hp_pct"), 1.0), 0.01)
        fraction = fixed / max(target_hp, 1.0)
        values.update(
            {
                "fixed_damage": 1.0,
                "damage_estimate_available": 1.0,
                "damage_min_target_hp": _clamp(fraction / 2.0),
                "damage_max_target_hp": _clamp(fraction / 2.0),
                "ko_possible": float(fraction >= 1.0),
                "ko_guaranteed": float(fraction >= 1.0),
            }
        )

    values.update(
        {
            "heal_fraction": _clamp(_number(move.heal)),
            "drain_fraction": _clamp(_number(move.drain)),
            "recoil_fraction": _clamp(_number(move.recoil)),
            "self_destruct": float(bool(move.self_destruct)),
            "self_switch": float(bool(move.self_switch)),
            "force_switch": float(bool(move.force_switch) or move_id in _FORCE_SWITCH_MOVES),
            "protect_move": float(bool(move.is_protect_move or move.is_side_protect_move)),
            "stalling_move": float(bool(move.stalling_move)),
            "breaks_protect": float(bool(move.breaks_protect)),
            "high_critical_rate": _clamp(_number(move.crit_ratio) / 6.0),
            "sets_hazard": float(move_id in _HAZARD_MOVES),
            "removes_hazard": float(move_id in _HAZARD_REMOVAL_MOVES),
            "sets_weather": float(move.weather is not None),
            "sets_terrain": float(move.terrain is not None),
            "custom_effect_unmodeled": float(bool(flags.intersection(_CUSTOM_EFFECT_FLAGS))),
        }
    )

    self_boosts = _boost_values()
    target_boosts = _boost_values()
    target_name = getattr(move.target, "name", "")
    primary_boost_target = self_boosts if target_name == "SELF" else target_boosts
    _add_boosts(primary_boost_target, move.boosts)
    _add_boosts(self_boosts, move.self_boost)

    direct_status = _status_name(move.status)
    if direct_status is not None:
        values[f"target_status_{direct_status}_chance"] = 1.0
    direct_volatile = _normalized_id(move.volatile_status)
    if direct_volatile == "confusion":
        values["target_confusion_chance"] = 1.0

    for secondary in move.secondary:
        if not isinstance(secondary, dict):
            continue
        chance = _clamp(_number(secondary.get("chance"), 100.0) / 100.0)
        secondary_status = _status_name(secondary.get("status"))
        if secondary_status is not None:
            key = f"target_status_{secondary_status}_chance"
            values[key] = max(values[key], chance)
        volatile = _normalized_id(secondary.get("volatileStatus"))
        if volatile == "confusion":
            values["target_confusion_chance"] = max(
                values["target_confusion_chance"], chance
            )
        elif volatile == "flinch":
            values["target_flinch_chance"] = max(values["target_flinch_chance"], chance)
        _add_boosts(target_boosts, secondary.get("boosts"), chance)
        self_effect = secondary.get("self")
        if isinstance(self_effect, dict):
            _add_boosts(self_boosts, self_effect.get("boosts"), chance)

    for stat in BOOST_STATS:
        values[f"self_{stat}_delta"] = _clamp(self_boosts[stat] / 6.0, -1.0, 1.0)
        values[f"target_{stat}_delta"] = _clamp(target_boosts[stat] / 6.0, -1.0, 1.0)


def _switch_features(
    state: dict[str, Any],
    action_id: int,
    values: dict[str, float],
    generation: int,
) -> None:
    pokemon = sorted_switches(state)[action_id - 4]
    opponent = state["opponent_active_pokemon"]
    entry_damage, toxic_spikes, sticky_web = _hazard_entry(state, pokemon, generation)
    hp = _clamp(_number(pokemon.get("hp_pct")))
    values.update(
        {
            "switch_hp_fraction": hp,
            "switch_post_entry_hp_fraction": _clamp(hp - entry_damage),
            "switch_entry_damage_fraction": entry_damage,
            "switch_hazard_toxic_spikes": toxic_spikes,
            "switch_hazard_sticky_web": sticky_web,
        }
    )
    status = _status_name(pokemon.get("status"))
    if status is not None:
        values[f"switch_status_{status}"] = 1.0
    for stat in ("hp", "atk", "def", "spa", "spd", "spe"):
        values[f"switch_base_{stat}"] = _clamp(
            _number(pokemon.get(f"base_{stat}")) / 255.0
        )
    values["switch_faster_by_base_stat"] = float(
        _number(pokemon.get("base_spe")) > _number(opponent.get("base_spe"))
    )

    offensive_effectiveness: list[float] = []
    offensive_pressure: list[float] = []
    for row_move in sorted_moves(pokemon):
        move = _move_data(str(row_move.get("name", "")), generation)
        move_type = (
            move.type.name.lower()
            if move is not None
            else _normalized_id(row_move.get("move_type"))
        )
        flags = set(move.flags) if move is not None else set()
        multiplier, _ = _type_multiplier(
            move_type,
            opponent,
            generation,
            move_flags=flags,
            attacker_ability=str(pokemon.get("ability", "")),
        )
        offensive_effectiveness.append(multiplier)
        power = _number(move.base_power if move is not None else row_move.get("base_power"))
        stab = 1.5 if _normalized_id(move_type) in _types(pokemon) else 1.0
        offensive_pressure.append(power * stab * multiplier / 400.0)
    if offensive_effectiveness:
        values["switch_best_offensive_effectiveness"] = _clamp(
            max(offensive_effectiveness) / 4.0
        )
        values["switch_best_damage_pressure"] = _clamp(max(offensive_pressure))

    defensive: list[float] = []
    for row_move in sorted_moves(opponent):
        move = _move_data(str(row_move.get("name", "")), generation)
        move_type = (
            move.type.name.lower()
            if move is not None
            else _normalized_id(row_move.get("move_type"))
        )
        flags = set(move.flags) if move is not None else set()
        multiplier, _ = _type_multiplier(
            move_type,
            pokemon,
            generation,
            move_flags=flags,
            attacker_ability=str(opponent.get("ability", "")),
        )
        defensive.append(multiplier)
    if defensive:
        total = len(defensive)
        values.update(
            {
                "switch_known_defensive_worst": _clamp(max(defensive) / 4.0),
                "switch_known_move_immunity_fraction": sum(value == 0 for value in defensive)
                / total,
                "switch_known_move_resistance_fraction": sum(0 < value < 1 for value in defensive)
                / total,
                "switch_known_move_weakness_fraction": sum(value > 1 for value in defensive)
                / total,
                "switch_opponent_moves_known_fraction": _clamp(total / 4.0),
            }
        )


def candidate_feature_vector(state: dict[str, Any], action_id: int) -> list[float]:
    """Return one normalized, name-free mechanics vector for a legal action."""
    if action_id not in legal_action_ids(state):
        raise ValueError(f"A{action_id} is not legal in this state")
    generation = _generation(state)
    values = _base_features(state, action_id)
    if 4 <= action_id <= 8:
        _switch_features(state, action_id, values, generation)
    else:
        _move_features(state, action_id, values, generation)
    vector = [float(values[name]) for name in MECHANICS_FEATURE_NAMES]
    if len(vector) != MECHANICS_FEATURE_COUNT or not all(math.isfinite(value) for value in vector):
        raise ValueError("Mechanics feature generation produced an invalid vector")
    return vector


def candidate_feature_matrix(state: dict[str, Any]) -> list[list[float]]:
    """Return a dense A0-A12 feature matrix; illegal rows remain all zero."""
    legal = set(legal_action_ids(state))
    return [
        candidate_feature_vector(state, action_id)
        if action_id in legal
        else [0.0] * MECHANICS_FEATURE_COUNT
        for action_id in range(ACTION_COUNT)
    ]
