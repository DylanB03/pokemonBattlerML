from __future__ import annotations

"""Canonical public-information observations shared by every policy path.

Teacher collection, replay preparation, live Showdown inference, and DAgger all
produce slightly different outer records.  This module deliberately depends
only on the standard library so the isolated Foul Play worker can use the same
contract as the Qwen process.
"""

import copy
from typing import Any

OBSERVATION_SCHEMA = "pokemon-battler-public-observation-v1"


def _move_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "nomove")
    return str(value or "nomove")


def _recent_moves(row: dict[str, Any], state: dict[str, Any]) -> list[dict[str, str]]:
    existing = state.get("recent_move_history")
    if isinstance(existing, list):
        return copy.deepcopy(existing[-4:])
    moves: list[dict[str, str]] = []
    for event in (row.get("history_events") or [])[-4:]:
        if not isinstance(event, dict):
            continue
        entry = {
            "player": _move_name(event.get("player_move")),
            "opponent": _move_name(event.get("opponent_move")),
        }
        if entry != {"player": "nomove", "opponent": "nomove"}:
            moves.append(entry)
    if not moves:
        entry = {
            "player": _move_name(state.get("player_prev_move")),
            "opponent": _move_name(state.get("opponent_prev_move")),
        }
        if entry != {"player": "nomove", "opponent": "nomove"}:
            moves.append(entry)
    return moves[-4:]


def _hide_private_opponent_fields(pokemon: dict[str, Any]) -> dict[str, Any]:
    """Retain preview-public species data while removing unrevealed battle data."""
    result = copy.deepcopy(pokemon)
    if result.get("side") == "opponent" and not bool(result.get("revealed")):
        result["hp_pct"] = None
        result["item"] = "unknownitem"
        result["ability"] = "unknownability"
        result["tera_type"] = "notype"
        result["terastallized"] = False
        result["status"] = "nostatus"
        result["effect"] = "noeffect"
        result["moves"] = []
        for stat in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"):
            result[f"{stat}_boost"] = 0
    return result


def canonicalize_observation(row: dict[str, Any]) -> dict[str, Any]:
    """Return one deep-copied, public-information, decision-time observation."""
    result = copy.deepcopy(row)
    state = result.get("state")
    if not isinstance(state, dict):
        raise TypeError("Observation is missing its state object")

    if state.get("turn_index") is None and result.get("turn_index") is not None:
        state["turn_index"] = int(result["turn_index"])
    if "player_remaining" not in state:
        active = state.get("player_active_pokemon") or {}
        active_alive = (
            float(active.get("hp_pct", 0) or 0) > 0
            and active.get("status") != "fnt"
        )
        state["player_remaining"] = len(state.get("available_switches") or []) + int(
            active_alive
        )

    state["recent_move_history"] = _recent_moves(result, state)
    opponent_roster = [
        _hide_private_opponent_fields(pokemon)
        for pokemon in (result.get("opponent_roster") or [])
        if isinstance(pokemon, dict)
    ]
    result["opponent_roster"] = opponent_roster
    result["player_roster"] = copy.deepcopy(result.get("player_roster") or [])
    state["opponent_revealed_pokemon"] = [
        copy.deepcopy(pokemon)
        for pokemon in opponent_roster
        if bool(pokemon.get("revealed")) and not bool(pokemon.get("active"))
    ]
    state["opponent_revealed_pokemon"].sort(
        key=lambda pokemon: (int(pokemon.get("slot", 99)), str(pokemon.get("name", "")))
    )
    if state.get("opponent_teampreview"):
        state["opponent_teampreview"] = sorted(
            {str(value) for value in state["opponent_teampreview"] if value}
        )
    result["observation_schema"] = OBSERVATION_SCHEMA
    return result


def canonical_state(row: dict[str, Any]) -> dict[str, Any]:
    return canonicalize_observation(row)["state"]
