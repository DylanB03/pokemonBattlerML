from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from pokemon_battler.actions import ACTION_COUNT
from pokemon_battler.interaction_features import PREPARED_SCHEMA_VERSION
from pokemon_battler.prepare import (
    _enrich_state,
    _history_event,
    _history_snapshot,
    _observe_move_history,
    _observe_opponent,
    _observe_rosters,
    _roster_snapshot,
)

_SPECIAL_FORCED_MOVES = {"fight", "recharge", "struggle"}


def _normalized_id(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if hasattr(value, "name"):
        value = value.name
    normalized = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return normalized or default


def _unknown_identity(value: Any, namespace: str) -> str:
    normalized = _normalized_id(value)
    if not normalized or normalized in {"unknown", f"unknown{namespace}"}:
        return f"unknown{namespace}"
    return normalized


def _move_to_state(move: Move | Any | None) -> dict[str, Any]:
    if move is None:
        return {
            "name": "nomove",
            "move_type": "nomove",
            "category": "nomove",
            "base_power": 0,
            "accuracy": 1.0,
            "priority": 0,
            "current_pp": 0,
            "max_pp": 0,
        }
    return {
        "name": _normalized_id(getattr(move, "id", None), "nomove"),
        "move_type": _normalized_id(getattr(move, "type", None), "notype"),
        "category": _normalized_id(getattr(move, "category", None), "status"),
        "base_power": int(getattr(move, "base_power", 0) or 0),
        "accuracy": float(getattr(move, "accuracy", 1.0) or 0.0),
        "priority": int(getattr(move, "priority", 0) or 0),
        "current_pp": int(getattr(move, "current_pp", 0) or 0),
        "max_pp": int(getattr(move, "max_pp", 0) or 0),
    }


def _effects_text(effects: Any) -> str:
    if not effects:
        return "noeffect"
    values = effects.keys() if isinstance(effects, dict) else effects
    names = sorted({_normalized_id(effect) for effect in values} - {""})
    return " ".join(names) or "noeffect"


def _pokemon_to_state(pokemon: Pokemon | Any) -> dict[str, Any]:
    moves = [_move_to_state(move) for move in getattr(pokemon, "moves", {}).values()]
    boosts = getattr(pokemon, "boosts", {}) or {}
    stats = getattr(pokemon, "base_stats", {}) or {}
    types = sorted(
        _normalized_id(type_, "notype")
        for type_ in (getattr(pokemon, "types", None) or [])
    )
    while len(types) < 2:
        types.append("notype")
    request = getattr(pokemon, "_last_request", None) or {}
    tera_type = _normalized_id(
        getattr(pokemon, "tera_type", None) or request.get("teraType"),
        "notype",
    )
    status = _normalized_id(getattr(pokemon, "status", None), "nostatus")
    return {
        "name": _normalized_id(getattr(pokemon, "species", None), "unknownpokemon"),
        "base_species": _normalized_id(
            getattr(pokemon, "base_species", None)
            or getattr(pokemon, "species", None),
            "unknownpokemon",
        ),
        "hp_pct": float(getattr(pokemon, "current_hp_fraction", 0.0) or 0.0),
        "types": " ".join(types[:2]),
        "tera_type": tera_type,
        "item": _unknown_identity(getattr(pokemon, "item", None), "item"),
        "ability": _unknown_identity(getattr(pokemon, "ability", None), "ability"),
        "lvl": int(getattr(pokemon, "level", 100) or 100),
        "status": status,
        "effect": _effects_text(getattr(pokemon, "effects", None)),
        "moves": moves[:4],
        **{
            f"{stat}_boost": int(boosts.get(stat, 0) or 0)
            for stat in ("atk", "spa", "def", "spd", "spe", "accuracy", "evasion")
        },
        **{
            f"base_{stat}": int(stats.get(stat, 0) or 0)
            for stat in ("atk", "spa", "def", "spd", "spe", "hp")
        },
    }


def _conditions_text(
    conditions: Any,
    default: str,
    *,
    include_hazard_layers: bool = False,
) -> str:
    if not conditions:
        return default
    items = conditions.items() if isinstance(conditions, dict) else (
        (condition, None) for condition in conditions
    )
    names: list[str] = []
    for condition, value in items:
        name = _normalized_id(condition)
        if not name:
            continue
        if include_hazard_layers and name in {"spikes", "toxicspikes"}:
            layers = int(value or 1)
            name = f"{name}{layers}"
        names.append(name)
    return " ".join(sorted(set(names))) or default


def _full_switch_options(battle: AbstractBattle) -> list[Pokemon | Any]:
    active = getattr(battle, "active_pokemon", None)
    reviving = bool(getattr(battle, "reviving", False))
    if reviving:
        options = [
            pokemon
            for pokemon in battle.team.values()
            if bool(getattr(pokemon, "fainted", False)) and pokemon is not active
        ]
    else:
        options = [
            pokemon
            for pokemon in battle.team.values()
            if not bool(getattr(pokemon, "fainted", False)) and pokemon is not active
        ]
    return sorted(options, key=lambda pokemon: _normalized_id(pokemon.species))


def _full_move_options(battle: AbstractBattle) -> list[Move | Any]:
    active = getattr(battle, "active_pokemon", None)
    if active is None:
        return []
    return sorted(
        list(getattr(active, "moves", {}).values())[:4],
        key=lambda move: _normalized_id(move.id),
    )


def _same_pokemon(left: Pokemon | Any, right: Pokemon | Any) -> bool:
    if left is right:
        return True
    left_name = _normalized_id(getattr(left, "name", None))
    right_name = _normalized_id(getattr(right, "name", None))
    if left_name and left_name == right_name:
        return True
    return _normalized_id(getattr(left, "species", None)) == _normalized_id(
        getattr(right, "species", None)
    )


def exact_live_legal_action_ids(battle: AbstractBattle) -> list[int]:
    """Map Showdown's exact request mask onto Metamon's stable A0-A12 space."""
    available_moves = list(getattr(battle, "available_moves", None) or [])
    available_switches = list(getattr(battle, "available_switches", None) or [])
    valid_move_ids = {_normalized_id(move.id) for move in available_moves}
    legal: set[int] = set()

    forced_switch = bool(getattr(battle, "force_switch", False))
    if not forced_switch and valid_move_ids:
        if valid_move_ids.issubset(_SPECIAL_FORCED_MOVES):
            legal.add(0)
        else:
            for index, move in enumerate(_full_move_options(battle)):
                if _normalized_id(move.id) in valid_move_ids:
                    legal.add(index)
                    if bool(getattr(battle, "can_tera", False)):
                        legal.add(9 + index)

    for index, pokemon in enumerate(_full_switch_options(battle)):
        if any(_same_pokemon(pokemon, available) for available in available_switches):
            legal.add(4 + index)

    return sorted(action_id for action_id in legal if 0 <= action_id < ACTION_COUNT)


def live_action_to_order(
    battle: AbstractBattle,
    action_id: int,
) -> BattleOrder | None:
    """Convert one legal universal action ID into a poke-env/Showdown order."""
    if action_id not in exact_live_legal_action_ids(battle):
        return None

    available_moves = list(getattr(battle, "available_moves", None) or [])
    valid_move_ids = {_normalized_id(move.id) for move in available_moves}
    if action_id == 0 and valid_move_ids and valid_move_ids.issubset(_SPECIAL_FORCED_MOVES):
        return Player.create_order(available_moves[0])

    wants_tera = action_id >= 9
    normalized_action = action_id - 9 if wants_tera else action_id
    if 0 <= normalized_action <= 3:
        moves = _full_move_options(battle)
        if normalized_action >= len(moves):
            return None
        move = moves[normalized_action]
        if _normalized_id(move.id) not in valid_move_ids:
            return None
        return Player.create_order(
            move,
            terastallize=wants_tera and bool(getattr(battle, "can_tera", False)),
        )

    switch_index = action_id - 4
    switches = _full_switch_options(battle)
    if not 0 <= switch_index < len(switches):
        return None
    pokemon = switches[switch_index]
    if not any(
        _same_pokemon(pokemon, available)
        for available in (getattr(battle, "available_switches", None) or [])
    ):
        return None
    return Player.create_order(pokemon)


def battle_to_metamon_state(battle: AbstractBattle) -> dict[str, Any]:
    """Convert a live poke-env singles battle to the replay schema used in training."""
    active = getattr(battle, "active_pokemon", None)
    opponent = getattr(battle, "opponent_active_pokemon", None)
    if active is None or opponent is None:
        raise ValueError("Live policy requires both active Pokémon to be known")

    opponent_team = list(getattr(battle, "opponent_team", {}).values())
    preview = list(getattr(battle, "teampreview_opponent_team", None) or [])
    preview_names = [
        _normalized_id(
            getattr(pokemon, "base_species", None)
            or getattr(pokemon, "species", None),
            "unknownpokemon",
        )
        for pokemon in preview
    ]
    if not preview_names:
        preview_names = [
            _normalized_id(
                getattr(pokemon, "base_species", None)
                or getattr(pokemon, "species", None),
                "unknownpokemon",
            )
            for pokemon in opponent_team
        ]
    team_size = max(len(preview_names), len(opponent_team), 6)
    opponents_remaining = team_size - sum(
        bool(getattr(pokemon, "fainted", False)) for pokemon in opponent_team
    )
    format_name = getattr(battle, "format", None)
    if not format_name:
        parts = str(getattr(battle, "battle_tag", "")).split("-")
        format_name = parts[1] if len(parts) > 1 else "gen9ou"

    return {
        "format": _normalized_id(format_name, "gen9ou"),
        "player_active_pokemon": _pokemon_to_state(active),
        "opponent_active_pokemon": _pokemon_to_state(opponent),
        # Deliberately include every surviving bench member. Exact switch legality
        # is represented separately so trapping does not shift action identities.
        "available_switches": [
            _pokemon_to_state(pokemon) for pokemon in _full_switch_options(battle)
        ],
        "player_prev_move": _move_to_state(getattr(active, "last_move", None)),
        "opponent_prev_move": _move_to_state(getattr(opponent, "last_move", None)),
        "opponents_remaining": opponents_remaining,
        "player_conditions": _conditions_text(
            getattr(battle, "side_conditions", None),
            "noconditions",
            include_hazard_layers=True,
        ),
        "opponent_conditions": _conditions_text(
            getattr(battle, "opponent_side_conditions", None),
            "noconditions",
            include_hazard_layers=True,
        ),
        "weather": _conditions_text(
            getattr(battle, "weather", None), "noweather"
        ),
        "battle_field": _conditions_text(
            getattr(battle, "fields", None), "nofield"
        ),
        "forced_switch": bool(getattr(battle, "force_switch", False)),
        "battle_won": bool(getattr(battle, "won", False)),
        "battle_lost": bool(getattr(battle, "lost", False)),
        "can_tera": bool(getattr(battle, "can_tera", False)),
        "opponent_teampreview": preview_names,
    }


def _request_key(battle: AbstractBattle) -> tuple[Any, ...]:
    request = getattr(battle, "last_request", None) or {}
    rqid = request.get("rqid")
    if rqid is not None:
        return ("rqid", rqid)
    active = getattr(battle, "active_pokemon", None)
    return (
        "state",
        int(getattr(battle, "turn", 0) or 0),
        _normalized_id(getattr(active, "species", None)),
        bool(getattr(battle, "force_switch", False)),
        tuple(sorted(_normalized_id(move.id) for move in battle.available_moves)),
        tuple(sorted(_normalized_id(pokemon.species) for pokemon in battle.available_switches)),
    )


@dataclass
class LiveBattleTracker:
    """Recreate schema-3 roster and recent-history context one live request at a time."""

    battle_id: str
    player_roster: dict[str, dict[str, Any]] = field(default_factory=dict)
    player_order: list[str] = field(default_factory=list)
    opponent_roster: dict[str, dict[str, Any]] = field(default_factory=dict)
    opponent_order: list[str] = field(default_factory=list)
    history_events: list[dict[str, Any]] = field(default_factory=list)
    revealed_opponents: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent_moves: list[dict[str, str]] = field(default_factory=list)
    previous_state: dict[str, Any] | None = None
    last_request_key: tuple[Any, ...] | None = None
    decision_count: int = 0

    def observe(self, battle: AbstractBattle) -> dict[str, Any]:
        legal = exact_live_legal_action_ids(battle)
        if not legal:
            raise ValueError("Showdown request contains no mappable legal A0-A12 action")
        state = battle_to_metamon_state(battle)
        state["prepared_legal_action_ids"] = legal
        request_key = _request_key(battle)
        is_new_request = request_key != self.last_request_key

        if is_new_request and self.previous_state is not None:
            self.history_events.append(_history_event(self.previous_state, state))
            del self.history_events[:-4]
        _observe_rosters(
            state,
            self.previous_state,
            self.player_roster,
            self.player_order,
            self.opponent_roster,
            self.opponent_order,
        )
        _observe_opponent(state, self.revealed_opponents)
        _observe_move_history(state, self.recent_moves)
        turn_index = self.decision_count if is_new_request else max(self.decision_count - 1, 0)
        enriched = _enrich_state(
            state,
            turn_index,
            self.revealed_opponents,
            self.recent_moves,
        )
        enriched["prepared_legal_action_ids"] = legal
        row = {
            "schema_version": PREPARED_SCHEMA_VERSION,
            "battle_id": self.battle_id,
            "turn_index": turn_index,
            "state": enriched,
            "player_roster": _roster_snapshot(
                self.player_roster,
                self.player_order,
                enriched.get("player_active_pokemon"),
                "player",
            ),
            "opponent_roster": _roster_snapshot(
                self.opponent_roster,
                self.opponent_order,
                enriched.get("opponent_active_pokemon"),
                "opponent",
            ),
            "history_events": _history_snapshot(self.history_events),
            "legal_action_ids": legal,
            "legal_mask_quality": "exact",
        }

        self.previous_state = copy.deepcopy(state)
        self.last_request_key = request_key
        if is_new_request:
            self.decision_count += 1
        return row
