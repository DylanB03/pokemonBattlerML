from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pokemon_battler.core.actions import (
    action_label,
    describe_action,
    legal_action_ids,
    sorted_moves,
    sorted_switches,
)

TASK_HEADER = """TASK: Select the listed legal action that maximizes eventual battle win
probability.
RULES:
- Use only the information in BATTLE_STATE.
- Treat unknownitem, unknownability, notype, and omitted opponent moves as unknown.
- Consider long-term strategy, not only immediate damage.
- Output exactly one listed action ID and nothing else."""

COMPACT_TASK_HEADER = "Choose the legal action most likely to win. Output only its action ID."
MECHANICS_TASK_HEADER = "Encode this battle state for legal-action scoring."
PROMPT_FORMATS = ("verbose-v1", "compact-v1", "mechanics-v1", "mechanics-v2")

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


@dataclass(frozen=True)
class PromptSections:
    """Prompt text split around action lines so candidate token positions are recoverable."""

    prefix: str
    candidates: tuple[tuple[int, str], ...]
    suffix: str

    @property
    def text(self) -> str:
        return self.prefix + "\n".join(line for _, line in self.candidates) + self.suffix


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


def _move_history(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return ";".join(
        f"{_scalar(entry.get('player', 'nomove'))}>{_scalar(entry.get('opponent', 'nomove'))}"
        for entry in value
        if isinstance(entry, dict)
    ) or "none"


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


def _compact_move(move: dict[str, Any]) -> str:
    return ",".join(
        (
            _scalar(move.get("name", "unknown")),
            _scalar(move.get("move_type", "unknown")),
            _scalar(move.get("category", "unknown")),
            _scalar(move.get("base_power", "unknown")),
            _scalar(move.get("accuracy", "unknown")),
            _scalar(move.get("priority", "unknown")),
            (
                f"{_scalar(move.get('current_pp', 'unknown'))}/"
                f"{_scalar(move.get('max_pp', 'unknown'))}"
            ),
        )
    )


def _compact_pokemon(
    pokemon: dict[str, Any],
    label: str,
    *,
    include_moves: bool = True,
    include_move_names: bool = False,
) -> list[str]:
    name = _scalar(pokemon.get("name", "unknown"))
    fields = [
        label,
        name,
        f"hp={_scalar(pokemon.get('hp_pct', 'unknown'))}",
        f"type={_scalar(pokemon.get('types', 'unknown'))}",
        f"tera={_scalar(pokemon.get('tera_type', 'unknown'))}",
        f"item={_scalar(pokemon.get('item', 'unknown'))}",
        f"ability={_scalar(pokemon.get('ability', 'unknown'))}",
        "stats=" + ",".join(
            _scalar(pokemon.get(field, "unknown"))
            for field in ("base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe")
        ),
    ]
    base_species = _scalar(pokemon.get("base_species", name))
    if base_species != name:
        fields.append(f"base={base_species}")
    level = pokemon.get("lvl", 100)
    if level != 100:
        fields.append(f"lvl={_scalar(level)}")
    status = _scalar(pokemon.get("status", "nostatus"))
    if status != "nostatus":
        fields.append(f"status={status}")
    effect = _scalar(pokemon.get("effect", "noeffect"))
    if effect != "noeffect":
        fields.append(f"effect={effect}")
    boosts = [
        pokemon.get(field, 0)
        for field in (
            "atk_boost",
            "def_boost",
            "spa_boost",
            "spd_boost",
            "spe_boost",
            "accuracy_boost",
            "evasion_boost",
        )
    ]
    if any(value != 0 for value in boosts):
        fields.append("boosts=" + ",".join(_scalar(value) for value in boosts))
    lines = ["|".join(fields)]
    if include_moves:
        lines.extend(f"{label}M|{_compact_move(move)}" for move in sorted_moves(pokemon))
    elif include_move_names:
        names = [str(move.get("name", "unknown")) for move in sorted_moves(pokemon)]
        if names:
            if label == "PA":
                entries = [f"A{index}:{name}" for index, name in enumerate(names)]
            else:
                entries = names
            lines.append(f"{label}M|" + ",".join(entries))
    return lines


def _compact_action(state: dict[str, Any], action_id: int) -> str:
    details = describe_action(state, action_id)
    label = action_label(action_id)
    if details["type"] == "switch":
        return f"{label}|switch|{_scalar(details['species'])}"
    return (
        f"{label}|move|{_scalar(details['name'])}"
        f"|tera={int(bool(details['terastallize']))}"
    )


def _render_verbose_sections(state: dict[str, Any]) -> PromptSections:
    switches = sorted_switches(state)
    legal = legal_action_ids(state)
    lines = [
        TASK_HEADER,
        "",
        "<BATTLE_STATE>",
        f"format={_scalar(state['format'])}",
        f"turn_index={_scalar(state.get('turn_index', 'unknown'))}",
        f"forced_switch={_scalar(bool(state.get('forced_switch', False)))}",
        f"can_tera={_scalar(bool(state.get('can_tera', False)))}",
        f"weather={_scalar(state.get('weather', 'noweather'))}",
        f"battle_field={_scalar(state.get('battle_field', 'nofield'))}",
        f"player_conditions={_scalar(state.get('player_conditions', 'noconditions'))}",
        f"opponent_conditions={_scalar(state.get('opponent_conditions', 'noconditions'))}",
        f"player_remaining={_scalar(state.get('player_remaining', 'unknown'))}",
        f"opponents_remaining={_scalar(state.get('opponents_remaining', 'unknown'))}",
        f"opponent_teampreview={_scalar(state.get('opponent_teampreview', []))}",
        f"player_prev_move={_previous_move_name(state.get('player_prev_move'))}",
        f"opponent_prev_move={_previous_move_name(state.get('opponent_prev_move'))}",
        f"recent_move_history={_move_history(state.get('recent_move_history'))}",
        "",
    ]
    lines.extend(_render_pokemon(state["player_active_pokemon"], "PLAYER_ACTIVE_POKEMON"))
    lines.append("")
    lines.extend(_render_pokemon(state["opponent_active_pokemon"], "OPPONENT_ACTIVE_POKEMON"))
    for pokemon in state.get("opponent_revealed_pokemon", []):
        lines.append("")
        lines.extend(_render_pokemon(pokemon, "OPPONENT_REVEALED_POKEMON"))
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
    return PromptSections(
        prefix="\n".join(lines) + "\n",
        candidates=tuple((action_id, _render_action(state, action_id)) for action_id in legal),
        suffix="\n</LEGAL_ACTIONS>\n\n<ACTION>\n",
    )


def _render_compact_sections(state: dict[str, Any]) -> PromptSections:
    switches = sorted_switches(state)
    legal = legal_action_ids(state)
    globals_line = "|".join(
        (
            "S",
            f"fmt={_scalar(state['format'])}",
            f"turn={_scalar(state.get('turn_index', 'unknown'))}",
            f"forced={int(bool(state.get('forced_switch', False)))}",
            f"tera={int(bool(state.get('can_tera', False)))}",
            f"weather={_scalar(state.get('weather', 'noweather'))}",
            f"field={_scalar(state.get('battle_field', 'nofield'))}",
            f"pc={_scalar(state.get('player_conditions', 'noconditions'))}",
            f"oc={_scalar(state.get('opponent_conditions', 'noconditions'))}",
            f"prem={_scalar(state.get('player_remaining', 'unknown'))}",
            f"orem={_scalar(state.get('opponents_remaining', 'unknown'))}",
            f"pprev={_previous_move_name(state.get('player_prev_move'))}",
            f"oprev={_previous_move_name(state.get('opponent_prev_move'))}",
            f"hist={_move_history(state.get('recent_move_history'))}",
            f"opreview={_scalar(state.get('opponent_teampreview', []))}",
        )
    )
    lines = [COMPACT_TASK_HEADER, globals_line]
    lines.extend(_compact_pokemon(state["player_active_pokemon"], "PA"))
    lines.extend(_compact_pokemon(state["opponent_active_pokemon"], "OA"))
    active_name = str(state["opponent_active_pokemon"].get("name", "")).lower()
    for pokemon in state.get("opponent_revealed_pokemon", []):
        if str(pokemon.get("name", "")).lower() != active_name:
            lines.extend(_compact_pokemon(pokemon, "OR"))
    for switch_index, pokemon in enumerate(switches):
        lines.extend(_compact_pokemon(pokemon, f"SW{action_label(4 + switch_index)}"))
    lines.append("LEGAL")
    return PromptSections(
        prefix="\n".join(lines) + "\n",
        candidates=tuple((action_id, _compact_action(state, action_id)) for action_id in legal),
        suffix="\nANSWER\n",
    )


def _render_mechanics_sections(state: dict[str, Any]) -> PromptSections:
    """Render only strategic context; action mechanics travel in a numeric tensor."""
    switches = sorted_switches(state)
    globals_line = "|".join(
        (
            "S",
            f"fmt={_scalar(state['format'])}",
            f"turn={_scalar(state.get('turn_index', 'unknown'))}",
            f"forced={int(bool(state.get('forced_switch', False)))}",
            f"tera={int(bool(state.get('can_tera', False)))}",
            f"weather={_scalar(state.get('weather', 'noweather'))}",
            f"field={_scalar(state.get('battle_field', 'nofield'))}",
            f"pc={_scalar(state.get('player_conditions', 'noconditions'))}",
            f"oc={_scalar(state.get('opponent_conditions', 'noconditions'))}",
            f"prem={_scalar(state.get('player_remaining', 'unknown'))}",
            f"orem={_scalar(state.get('opponents_remaining', 'unknown'))}",
            f"opreview={_scalar(state.get('opponent_teampreview', []))}",
        )
    )
    lines = [MECHANICS_TASK_HEADER, globals_line]
    lines.extend(
        _compact_pokemon(state["player_active_pokemon"], "PA", include_moves=False)
    )
    lines.extend(
        _compact_pokemon(state["opponent_active_pokemon"], "OA", include_moves=False)
    )
    active_name = str(state["opponent_active_pokemon"].get("name", "")).lower()
    for pokemon in state.get("opponent_revealed_pokemon", []):
        if str(pokemon.get("name", "")).lower() != active_name:
            lines.extend(_compact_pokemon(pokemon, "OR", include_moves=False))
    for switch_index, pokemon in enumerate(switches):
        lines.extend(
            _compact_pokemon(
                pokemon,
                f"SW{action_label(4 + switch_index)}",
                include_moves=False,
            )
        )
    lines.append("STATE_END")
    return PromptSections(
        prefix="\n".join(lines) + "\n",
        candidates=(),
        suffix="",
    )


def _render_mechanics_v2_sections(state: dict[str, Any]) -> PromptSections:
    """Keep compact identity/history while action mechanics travel as tensors."""
    switches = sorted_switches(state)
    globals_line = "|".join(
        (
            "S",
            f"fmt={_scalar(state['format'])}",
            f"turn={_scalar(state.get('turn_index', 'unknown'))}",
            f"forced={int(bool(state.get('forced_switch', False)))}",
            f"tera={int(bool(state.get('can_tera', False)))}",
            f"weather={_scalar(state.get('weather', 'noweather'))}",
            f"field={_scalar(state.get('battle_field', 'nofield'))}",
            f"pc={_scalar(state.get('player_conditions', 'noconditions'))}",
            f"oc={_scalar(state.get('opponent_conditions', 'noconditions'))}",
            f"prem={_scalar(state.get('player_remaining', 'unknown'))}",
            f"orem={_scalar(state.get('opponents_remaining', 'unknown'))}",
            f"pprev={_previous_move_name(state.get('player_prev_move'))}",
            f"oprev={_previous_move_name(state.get('opponent_prev_move'))}",
            f"hist={_move_history(state.get('recent_move_history'))}",
            f"opreview={_scalar(state.get('opponent_teampreview', []))}",
        )
    )
    lines = [MECHANICS_TASK_HEADER, globals_line]
    lines.extend(
        _compact_pokemon(
            state["player_active_pokemon"],
            "PA",
            include_moves=False,
            include_move_names=True,
        )
    )
    lines.extend(
        _compact_pokemon(
            state["opponent_active_pokemon"],
            "OA",
            include_moves=False,
            include_move_names=True,
        )
    )
    active_name = str(state["opponent_active_pokemon"].get("name", "")).lower()
    for pokemon in state.get("opponent_revealed_pokemon", []):
        if str(pokemon.get("name", "")).lower() != active_name:
            lines.extend(
                _compact_pokemon(
                    pokemon,
                    "OR",
                    include_moves=False,
                    include_move_names=True,
                )
            )
    for switch_index, pokemon in enumerate(switches):
        lines.extend(
            _compact_pokemon(
                pokemon,
                f"SW{action_label(4 + switch_index)}",
                include_moves=False,
                include_move_names=True,
            )
        )
    lines.append("STATE_END")
    return PromptSections(prefix="\n".join(lines) + "\n", candidates=(), suffix="")


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


def render_prompt_sections(
    state: dict[str, Any],
    prompt_format: str = "verbose-v1",
) -> PromptSections:
    """Serialize a state while retaining the boundary of every legal candidate."""
    validate_state(state)
    legal = legal_action_ids(state)
    if not legal:
        raise ValueError("State has no recoverable legal actions")
    if prompt_format == "verbose-v1":
        return _render_verbose_sections(state)
    if prompt_format == "compact-v1":
        return _render_compact_sections(state)
    if prompt_format == "mechanics-v1":
        return _render_mechanics_sections(state)
    if prompt_format == "mechanics-v2":
        return _render_mechanics_v2_sections(state)
    raise ValueError(f"Unknown prompt format {prompt_format!r}; choose from {PROMPT_FORMATS}")


def render_prompt(state: dict[str, Any], prompt_format: str = "verbose-v1") -> str:
    """Serialize one Metamon ``UniversalState`` into the model's policy prompt."""
    return render_prompt_sections(state, prompt_format).text


def encode_candidate_prompt(
    tokenizer: Any,
    state: dict[str, Any],
    prompt_format: str,
    candidate_order: Iterable[int] | None = None,
) -> tuple[list[int], dict[int, int]]:
    """Tokenize once while recording each candidate line's final token position."""
    sections = render_prompt_sections(state, prompt_format)
    input_ids = tokenizer.encode(sections.prefix, add_special_tokens=True)
    candidate_positions: dict[int, int] = {}
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    candidates_by_action = dict(sections.candidates)
    if candidate_order is None:
        candidates = list(sections.candidates)
    else:
        order = list(candidate_order)
        if set(order) != set(candidates_by_action):
            raise ValueError("Candidate order must contain every legal action exactly once")
        candidates = [(action_id, candidates_by_action[action_id]) for action_id in order]
    for candidate_index, (action_id, line) in enumerate(candidates):
        line_ids = tokenizer.encode(line, add_special_tokens=False)
        if not line_ids:
            raise ValueError(f"Tokenizer produced no IDs for candidate A{action_id}")
        input_ids.extend(line_ids)
        candidate_positions[action_id] = len(input_ids) - 1
        if candidate_index + 1 < len(candidates):
            input_ids.extend(newline_ids)
    input_ids.extend(tokenizer.encode(sections.suffix, add_special_tokens=False))
    return input_ids, candidate_positions
