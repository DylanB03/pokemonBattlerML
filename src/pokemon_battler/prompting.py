from __future__ import annotations

from typing import Any, Iterable

from pokemon_battler.actions import (
    action_label,
    describe_action,
    legal_action_ids,
    sorted_moves,
    sorted_switches,
)

TASK_HEADER = """TASK: Select the listed legal action that maximizes eventual battle win probability.
RULES:
- Use only the information in BATTLE_STATE.
- Treat unknownitem, unknownability, notype, and omitted opponent moves as unknown.
- Consider long-term strategy, not only immediate damage.
- Output exactly one listed action ID and nothing else."""

POKEMON_FIELDS = (
    "name",
    "base_species",
    "hp_pct",
    "types",
    "tera_type",
    "item",
    "ability",
    "lvl",
    "status",
    "effect",
    "base_hp",
    "base_atk",
    "base_def",
    "base_spa",
    "base_spd",
    "base_spe",
    "atk_boost",
    "def_boost",
    "spa_boost",
    "spd_boost",
    "spe_boost",
    "accuracy_boost",
    "evasion_boost",
)

MOVE_FIELDS = (
    "name",
    "move_type",
    "category",
    "base_power",
    "accuracy",
    "priority",
    "current_pp",
    "max_pp",
)


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple)):
        return ",".join(_scalar(item) for item in value)
    return str(value)


def _previous_move_name(value: Any) -> str:
    if isinstance(value, dict):
        return _scalar(value.get("name", "nomove"))
    return _scalar(value or "nomove")


def _render_move(move: dict[str, Any]) -> str:
    return " ".join(f"{field}={_scalar(move.get(field, 'unknown'))}" for field in MOVE_FIELDS)


def _render_pokemon(
    pokemon: dict[str, Any],
    tag: str,
    *,
    attributes: Iterable[str] = (),
) -> list[str]:
    attribute_text = "".join(f" {attribute}" for attribute in attributes)
    lines = [f"<{tag}{attribute_text}>"]
    lines.extend(f"{field}={_scalar(pokemon.get(field, 'unknown'))}" for field in POKEMON_FIELDS)
    lines.append("moves:")
    lines.extend(f"- {_render_move(move)}" for move in sorted_moves(pokemon))
    lines.append(f"</{tag}>")
    return lines


def _render_action(state: dict[str, Any], action_id: int) -> str:
    details = describe_action(state, action_id)
    fields = " ".join(f"{key}={_scalar(value)}" for key, value in details.items())
    return f"<{action_label(action_id)}> {fields}"


def validate_state(state: dict[str, Any]) -> None:
    required = {
        "format",
        "player_active_pokemon",
        "opponent_active_pokemon",
        "available_switches",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Metamon state is missing required fields: {', '.join(missing)}")
    if not isinstance(state["available_switches"], list):
        raise ValueError("available_switches must be a list")


def render_prompt(state: dict[str, Any]) -> str:
    """Serialize one Metamon ``UniversalState`` into the model's policy prompt."""
    validate_state(state)
    switches = sorted_switches(state)
    legal = legal_action_ids(state)
    if not legal:
        raise ValueError("State has no recoverable legal actions")

    lines = [
        TASK_HEADER,
        "",
        "<BATTLE_STATE>",
        f"format={_scalar(state['format'])}",
        f"forced_switch={_scalar(bool(state.get('forced_switch', False)))}",
        f"can_tera={_scalar(bool(state.get('can_tera', False)))}",
        f"weather={_scalar(state.get('weather', 'noweather'))}",
        f"battle_field={_scalar(state.get('battle_field', 'nofield'))}",
        f"player_conditions={_scalar(state.get('player_conditions', 'noconditions'))}",
        f"opponent_conditions={_scalar(state.get('opponent_conditions', 'noconditions'))}",
        f"opponents_remaining={_scalar(state.get('opponents_remaining', 'unknown'))}",
        f"opponent_teampreview={_scalar(state.get('opponent_teampreview', []))}",
        f"player_prev_move={_previous_move_name(state.get('player_prev_move'))}",
        f"opponent_prev_move={_previous_move_name(state.get('opponent_prev_move'))}",
        "",
    ]
    lines.extend(_render_pokemon(state["player_active_pokemon"], "PLAYER_ACTIVE_POKEMON"))
    lines.append("")
    lines.extend(_render_pokemon(state["opponent_active_pokemon"], "OPPONENT_ACTIVE_POKEMON"))

    for switch_index, pokemon in enumerate(switches):
        lines.append("")
        lines.extend(
            _render_pokemon(
                pokemon,
                "AVAILABLE_SWITCH",
                attributes=(f"action_id={action_label(4 + switch_index)}",),
            )
        )

    lines.extend(["</BATTLE_STATE>", "", "<LEGAL_ACTIONS>"])
    lines.extend(_render_action(state, action_id) for action_id in legal)
    lines.extend(["</LEGAL_ACTIONS>", "", "<ACTION>", ""])
    return "\n".join(lines)
