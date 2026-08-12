from __future__ import annotations

"""Runtime-only bridge for collecting Foul Play MCTS teacher decisions.

This module is deliberately dependency-light: ``foul_play_worker.py`` imports it
inside Foul Play's isolated virtual environment.  It does not import the project,
torch, transformers, or poke-env.
"""

import copy
import json
import logging
import math
import re
import threading
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
ACTION_COUNT = 13
SCHEMA_VERSION = 3
TEACHER_SCHEMA = "foul-play-distillation-v1"


def _normalized(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if hasattr(value, "value"):
        value = value.value
    result = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return result or default


def _unknown(value: Any, namespace: str) -> str:
    result = _normalized(value)
    if not result or result in {"none", "unknown", f"unknown{namespace}"}:
        return f"unknown{namespace}"
    return result


def aggregate_mcts_policy(
    mcts_results: list[tuple[Any, float, int]],
) -> tuple[dict[str, float], int]:
    """Aggregate Foul Play's sampled-state MCTS visits before its 75% filter."""
    policy: dict[str, float] = {}
    visits = 0
    for result, sample_chance, _index in mcts_results:
        total_visits = int(getattr(result, "total_visits", 0) or 0)
        if total_visits <= 0:
            continue
        visits += total_visits
        for option in getattr(result, "side_one", ()):
            option_visits = int(getattr(option, "visits", 0) or 0)
            choice = str(getattr(option, "move_choice", ""))
            if not choice:
                continue
            # A zero-visit option is still legal and must remain in the exact
            # candidate mask even though its teacher probability is zero.
            policy.setdefault(choice, 0.0)
            if option_visits <= 0:
                continue
            weight = float(sample_chance) * option_visits / total_visits
            policy[choice] = policy.get(choice, 0.0) + weight
    total = sum(policy.values())
    if total > 0:
        policy = {choice: weight / total for choice, weight in policy.items()}
    return policy, visits


def _move_state(move: Any | None) -> dict[str, Any]:
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
    from fp import constants
    from fp.data import all_move_json

    name = _normalized(getattr(move, "name", None), "nomove")
    data = all_move_json.get(name, {})
    accuracy = data.get(constants.ACCURACY, 1.0)
    if accuracy is True:
        accuracy = 1.0
    elif isinstance(accuracy, (int, float)) and accuracy > 1:
        accuracy = accuracy / 100.0
    return {
        "name": name,
        "move_type": _normalized(data.get(constants.TYPE), "notype"),
        "category": _normalized(data.get(constants.CATEGORY), "status"),
        "base_power": int(data.get(constants.BASE_POWER, 0) or 0),
        "accuracy": float(accuracy or 0.0),
        "priority": int(data.get(constants.PRIORITY, 0) or 0),
        "current_pp": int(getattr(move, "current_pp", 0) or 0),
        "max_pp": int(getattr(move, "max_pp", 0) or 0),
    }


def _pokemon_state(pokemon: Any) -> dict[str, Any]:
    stats = getattr(pokemon, "base_stats", {}) or {}
    boosts = getattr(pokemon, "boosts", {}) or {}
    stat_keys = {
        "hp": "hp",
        "atk": "attack",
        "def": "defense",
        "spa": "special-attack",
        "spd": "special-defense",
        "spe": "speed",
        "accuracy": "accuracy",
        "evasion": "evasion",
    }
    hp = float(getattr(pokemon, "hp", 0) or 0)
    max_hp = float(getattr(pokemon, "max_hp", 0) or 0)
    types = list(getattr(pokemon, "types", ()) or ())
    if bool(getattr(pokemon, "terastallized", False)) and getattr(
        pokemon, "tera_type", None
    ):
        types = [getattr(pokemon, "tera_type")]
    normalized_types = [_normalized(value, "notype") for value in types[:2]]
    while len(normalized_types) < 2:
        normalized_types.append("notype")
    status = _normalized(getattr(pokemon, "status", None), "nostatus")
    if hp <= 0:
        status = "fnt"
    effects = sorted(
        {
            _normalized(value)
            for value in (getattr(pokemon, "volatile_statuses", ()) or ())
            if _normalized(value)
        }
    )
    result = {
        "name": _normalized(getattr(pokemon, "name", None), "unknownpokemon"),
        "base_species": _normalized(
            getattr(pokemon, "base_name", None)
            or getattr(pokemon, "name", None),
            "unknownpokemon",
        ),
        "hp_pct": hp / max_hp if max_hp > 0 else 0.0,
        "types": " ".join(normalized_types),
        "tera_type": _normalized(getattr(pokemon, "tera_type", None), "notype"),
        "terastallized": bool(getattr(pokemon, "terastallized", False)),
        "item": _unknown(getattr(pokemon, "item", None), "item"),
        "ability": _unknown(getattr(pokemon, "ability", None), "ability"),
        "lvl": int(getattr(pokemon, "level", 100) or 100),
        "status": status,
        "effect": " ".join(effects) or "noeffect",
        "moves": [_move_state(move) for move in getattr(pokemon, "moves", ())[:4]],
    }
    for short_name, source_name in stat_keys.items():
        if short_name in {"hp", "atk", "def", "spa", "spd", "spe"}:
            result[f"base_{short_name}"] = int(stats.get(source_name, 0) or 0)
        if short_name != "hp":
            result[f"{short_name}_boost"] = int(boosts.get(source_name, 0) or 0)
    return result


def _conditions_text(conditions: Any, default: str) -> str:
    if not conditions:
        return default
    items = conditions.items() if hasattr(conditions, "items") else (
        (condition, 1) for condition in conditions
    )
    names: list[str] = []
    for condition, value in items:
        name = _normalized(condition)
        if not name or not value:
            continue
        if name in {"spikes", "toxicspikes"}:
            name = f"{name}{int(value)}"
        names.append(name)
    return " ".join(sorted(set(names))) or default


def _move_from_last_used(last_used: Any) -> dict[str, Any]:
    name = _normalized(getattr(last_used, "move", None), "nomove")
    if name.startswith("switch") or name == "nomove":
        return _move_state(None)
    try:
        from fp.battle.state import Move

        return _move_state(Move(name))
    except (KeyError, ValueError):
        result = _move_state(None)
        result["name"] = name
        return result


def _alive(pokemon: Any) -> bool:
    return bool(pokemon is not None and float(getattr(pokemon, "hp", 0) or 0) > 0)


def _choice_action_id(state: dict[str, Any], choice: str) -> int | None:
    normalized_choice = choice.strip().lower()
    if normalized_choice.startswith("switch "):
        name = _normalized(normalized_choice.removeprefix("switch "))
        switches = sorted(
            state["available_switches"], key=lambda pokemon: _normalized(pokemon["name"])
        )
        for index, pokemon in enumerate(switches[:5]):
            if _normalized(pokemon["name"]) == name:
                return 4 + index
        return None

    tera = normalized_choice.endswith("-tera")
    name = _normalized(normalized_choice.removesuffix("-tera").removesuffix("-mega"))
    moves = sorted(
        state["player_active_pokemon"].get("moves", ())[:4],
        key=lambda move: _normalized(move["name"]),
    )
    for index, move in enumerate(moves):
        move_name = _normalized(move["name"])
        if move_name == name or (
            move_name.startswith("hiddenpower") and name.startswith("hiddenpower")
        ):
            return (9 if tera else 0) + index
    return None


def _remaining(side: Any) -> int:
    return sum(_alive(pokemon) for pokemon in [side.active, *side.reserve])


def _player_remaining(state: dict[str, Any]) -> int:
    active = state["player_active_pokemon"]
    return len(state.get("available_switches") or []) + int(
        float(active.get("hp_pct", 0) or 0) > 0 and active.get("status") != "fnt"
    )


def _history_event(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    player_before = previous["player_active_pokemon"]
    player_after = current["player_active_pokemon"]
    opponent_before = previous["opponent_active_pokemon"]
    opponent_after = current["opponent_active_pokemon"]
    player_switched = _normalized(player_before["name"]) != _normalized(player_after["name"])
    opponent_switched = _normalized(opponent_before["name"]) != _normalized(
        opponent_after["name"]
    )

    def hp_delta(before: dict[str, Any], after: dict[str, Any], switched: bool) -> float:
        if switched:
            return 0.0
        return max(min(float(after["hp_pct"]) - float(before["hp_pct"]), 1.0), -1.0)

    return {
        "decision_offset": -1,
        "player_move": current["player_prev_move"]["name"],
        "opponent_move": current["opponent_prev_move"]["name"],
        "player_species_before": player_before["name"],
        "player_species_after": player_after["name"],
        "opponent_species_before": opponent_before["name"],
        "opponent_species_after": opponent_after["name"],
        "player_hp_delta": hp_delta(player_before, player_after, player_switched),
        "opponent_hp_delta": hp_delta(opponent_before, opponent_after, opponent_switched),
        "player_switched": player_switched,
        "opponent_switched": opponent_switched,
        "player_fainted": _player_remaining(current) < _player_remaining(previous),
        "opponent_fainted": int(current["opponents_remaining"]) < int(
            previous["opponents_remaining"]
        ),
        "player_status_changed": player_before["status"] != player_after["status"],
        "opponent_status_changed": opponent_before["status"] != opponent_after["status"],
        "player_conditions_changed": previous["player_conditions"]
        != current["player_conditions"],
        "opponent_conditions_changed": previous["opponent_conditions"]
        != current["opponent_conditions"],
        "field_or_weather_changed": previous["battle_field"] != current["battle_field"]
        or previous["weather"] != current["weather"],
    }


class FoulPlayObservationTracker:
    def __init__(self, battle_id: str) -> None:
        self.battle_id = battle_id
        self.previous_state: dict[str, Any] | None = None
        self.history_events: list[dict[str, Any]] = []
        self.decision_count = 0
        self.opponent_slots: dict[str, int] = {}

    def _roster(self, side: Any, *, player: bool) -> list[dict[str, Any]]:
        pokemon = [side.active, *side.reserve]
        if player:
            pokemon.sort(
                key=lambda member: int(getattr(member, "index", 99) or 99)
                if member is not None
                else 99
            )
        rows: list[dict[str, Any]] = []
        for member in pokemon:
            if member is None or not _normalized(getattr(member, "name", None)):
                continue
            name = _normalized(member.name)
            if player:
                index = getattr(member, "index", None)
                slot = int(index) - 1 if isinstance(index, int) and index > 0 else len(rows)
            else:
                if name not in self.opponent_slots:
                    self.opponent_slots[name] = len(self.opponent_slots)
                slot = self.opponent_slots[name]
            row = _pokemon_state(member)
            row.update(
                {
                    "slot": slot,
                    "side": "player" if player else "opponent",
                    "active": member is side.active,
                    "present": True,
                    "revealed": player or bool(getattr(member, "revealed", False))
                    or member is side.active,
                    "fainted": not _alive(member),
                }
            )
            rows.append(row)
        return sorted(rows, key=lambda row: int(row["slot"]))[:6]

    def observe(
        self,
        battle: Any,
        raw_policy: dict[str, float],
        selected_choice: str,
        visit_count: int,
    ) -> dict[str, Any] | None:
        if battle.user.active is None or battle.opponent.active is None:
            return None
        available_switches = [
            _pokemon_state(pokemon)
            for pokemon in battle.user.reserve
            if _alive(pokemon)
        ]
        state = {
            "format": _normalized(getattr(battle, "pokemon_format", None), "gen9ou"),
            "player_active_pokemon": _pokemon_state(battle.user.active),
            "opponent_active_pokemon": _pokemon_state(battle.opponent.active),
            "available_switches": available_switches,
            "player_prev_move": _move_from_last_used(battle.user.last_used_move),
            "opponent_prev_move": _move_from_last_used(battle.opponent.last_used_move),
            "opponents_remaining": _remaining(battle.opponent),
            "player_remaining": _remaining(battle.user),
            "player_conditions": _conditions_text(
                battle.user.side_conditions, "noconditions"
            ),
            "opponent_conditions": _conditions_text(
                battle.opponent.side_conditions, "noconditions"
            ),
            "weather": _normalized(getattr(battle, "weather", None), "noweather"),
            "battle_field": " ".join(
                value
                for value in (
                    _normalized(getattr(battle, "field", None)),
                    "trickroom" if bool(getattr(battle, "trick_room", False)) else "",
                )
                if value
            )
            or "nofield",
            "forced_switch": bool(getattr(battle, "force_switch", False)),
            "battle_won": False,
            "battle_lost": False,
            "can_tera": any(choice.lower().endswith("-tera") for choice in raw_policy),
            "opponent_teampreview": [
                _normalized(getattr(pokemon, "base_name", None) or pokemon.name)
                for pokemon in [battle.opponent.active, *battle.opponent.reserve]
                if pokemon is not None and _normalized(getattr(pokemon, "name", None))
            ],
        }

        action_policy: dict[int, float] = {}
        unmapped: dict[str, float] = {}
        for choice, probability in raw_policy.items():
            action_id = _choice_action_id(state, choice)
            if action_id is None:
                unmapped[choice] = probability
                continue
            action_policy[action_id] = action_policy.get(action_id, 0.0) + probability
        selected_action = _choice_action_id(state, selected_choice)
        if selected_action is None or not action_policy:
            LOGGER.warning(
                "Skipping unmappable Foul Play teacher choice %r in %s; policy=%s",
                selected_choice,
                self.battle_id,
                sorted(raw_policy),
            )
            return None
        legal = sorted(action_policy)
        if selected_action not in legal:
            legal.append(selected_action)
            legal.sort()
            action_policy[selected_action] = 0.0
        total = sum(action_policy.values())
        if total <= 0:
            return None
        action_policy = {key: value / total for key, value in action_policy.items()}
        state["prepared_legal_action_ids"] = legal

        if self.previous_state is not None:
            self.history_events.append(_history_event(self.previous_state, state))
            self.history_events = self.history_events[-4:]
        history = copy.deepcopy(self.history_events)
        for index, event in enumerate(history):
            event["decision_offset"] = index - len(history)

        probabilities = [0.0] * ACTION_COUNT
        for action_id, probability in action_policy.items():
            probabilities[action_id] = probability
        positive = [value for value in probabilities if value > 0]
        entropy = -sum(value * math.log(value) for value in positive)
        confidence = max(probabilities)
        row = {
            "teacher_schema": TEACHER_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "battle_id": self.battle_id,
            "turn_index": self.decision_count,
            "state": state,
            "player_roster": self._roster(battle.user, player=True),
            "opponent_roster": self._roster(battle.opponent, player=False),
            "history_events": history,
            "legal_action_ids": legal,
            "legal_mask_quality": "exact",
            "action_id": selected_action,
            "target": f"A{selected_action}",
            "teacher": {
                "name": "foul-play",
                "policy": probabilities,
                "confidence": confidence,
                "entropy": entropy,
                "visit_count": visit_count,
                "selected_choice": selected_choice,
                "selected_action_id": selected_action,
                "raw_policy": raw_policy,
                "unmapped_policy_mass": sum(unmapped.values()),
            },
        }
        self.previous_state = copy.deepcopy(state)
        self.decision_count += 1
        return row


class _TraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, row: dict[str, Any]) -> None:
        payload = json.dumps(row, separators=(",", ":"), sort_keys=True)
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")


def install_foul_play_teacher_trace(path: str | Path) -> None:
    """Patch Foul Play's parent-process selection hook and save teacher rows."""
    from fp.modes import base as modes_base
    from fp.search import main as search_main

    writer = _TraceWriter(Path(path))
    trackers: dict[str, FoulPlayObservationTracker] = {}
    original_find = modes_base.find_best_move
    original_select = search_main.select_move_from_mcts_results
    search_lock = threading.Lock()

    def traced_find_best_move(battle: Any) -> str:
        if bool(getattr(battle, "team_preview", False)):
            return original_find(battle)
        captured_policy: dict[str, float] = {}
        captured_visits = 0

        def traced_select(results: list[tuple[Any, float, int]]) -> str:
            nonlocal captured_policy, captured_visits
            captured_policy, captured_visits = aggregate_mcts_policy(results)
            return original_select(results)

        # Foul Play normally searches one battle decision at a time. The lock also
        # makes the temporary module hook safe if upstream later overlaps games.
        with search_lock:
            search_main.select_move_from_mcts_results = traced_select
            try:
                choice = original_find(battle)
            finally:
                search_main.select_move_from_mcts_results = original_select
        battle_id = str(getattr(battle, "battle_tag", "unknown-battle"))
        tracker = trackers.setdefault(battle_id, FoulPlayObservationTracker(battle_id))
        row = tracker.observe(battle, captured_policy, choice, captured_visits)
        if row is not None:
            writer.write(row)
        return choice

    modes_base.find_best_move = traced_find_best_move
