from __future__ import annotations

import re
from typing import Any

ACTION_COUNT = 13
ACTION_IDS = tuple(range(ACTION_COUNT))


def _normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def sorted_moves(pokemon: dict[str, Any]) -> list[dict[str, Any]]:
    """Return moves in the alphabetical order used by Metamon action IDs."""
    moves = pokemon.get("moves") or []
    return sorted(moves[:4], key=lambda move: _normalized_name(move.get("name", "")))


def sorted_switches(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return available switches in the alphabetical order used by Metamon."""
    switches = state.get("available_switches") or []
    return sorted(switches[:5], key=lambda pokemon: _normalized_name(pokemon.get("name", "")))


def recoverable_legal_action_ids(state: dict[str, Any]) -> list[int]:
    """
    Reproduce ``UniversalAction.maybe_valid_actions`` from Metamon.

    These are the actions recoverable from a replay state. The live simulator can
    apply a stricter mask for disabled moves, trapping, choice locks, and similar
    mechanics.
    """
    moves = sorted_moves(state["player_active_pokemon"])
    switches = sorted_switches(state)
    legal: list[int] = []

    if not bool(state.get("forced_switch", False)):
        legal.extend(range(len(moves)))
        if bool(state.get("can_tera", False)):
            legal.extend(range(9, 9 + len(moves)))

    legal.extend(range(4, 4 + len(switches)))
    return sorted(legal)


def pp_aware_legal_action_ids(state: dict[str, Any]) -> list[int]:
    """Remove zero-PP moves while retaining replay-recoverable switch legality."""
    legal = recoverable_legal_action_ids(state)
    moves = sorted_moves(state["player_active_pokemon"])
    zero_pp_move_slots = {
        move_index
        for move_index, move in enumerate(moves)
        if isinstance(move.get("current_pp"), (int, float)) and move["current_pp"] <= 0
    }
    return [
        action_id
        for action_id in legal
        if not (
            action_id in zero_pp_move_slots
            or action_id - 9 in zero_pp_move_slots
        )
    ]


def legal_action_ids(state: dict[str, Any]) -> list[int]:
    """Return a prepared override when present, otherwise replay-recoverable actions."""
    prepared = state.get("prepared_legal_action_ids")
    if prepared is None:
        return recoverable_legal_action_ids(state)
    legal = sorted({int(value) for value in prepared})
    if not legal or any(action_id not in ACTION_IDS for action_id in legal):
        raise ValueError("prepared_legal_action_ids must contain actions from A0 through A12")
    recoverable = set(recoverable_legal_action_ids(state))
    if not set(legal).issubset(recoverable):
        raise ValueError("Prepared legal-action override contains unrecoverable actions")
    return legal


def action_label(action_id: int) -> str:
    if action_id not in ACTION_IDS:
        raise ValueError(f"action_id must be in [0, 12], got {action_id}")
    return f"A{action_id}"


def parse_action_label(value: str) -> int:
    match = re.fullmatch(r"\s*A(1[0-2]|[0-9])\s*", value)
    if not match:
        raise ValueError(f"Expected one action label from A0 through A12, got {value!r}")
    return int(match.group(1))


def describe_action(state: dict[str, Any], action_id: int) -> dict[str, Any]:
    if action_id not in legal_action_ids(state):
        raise ValueError(f"A{action_id} is not legal in this state")

    moves = sorted_moves(state["player_active_pokemon"])
    switches = sorted_switches(state)

    if 0 <= action_id <= 3:
        move = moves[action_id]
        return {
            "universal_action": action_id,
            "type": "move",
            "name": move["name"],
            "terastallize": False,
        }
    if 4 <= action_id <= 8:
        pokemon = switches[action_id - 4]
        return {
            "universal_action": action_id,
            "type": "switch",
            "species": pokemon["name"],
        }

    move = moves[action_id - 9]
    return {
        "universal_action": action_id,
        "type": "move",
        "name": move["name"],
        "terastallize": True,
    }
