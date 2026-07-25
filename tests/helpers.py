from __future__ import annotations

from copy import deepcopy
from typing import Any


def move(name: str, move_type: str = "normal") -> dict[str, Any]:
    return {
        "name": name,
        "move_type": move_type,
        "category": "physical",
        "base_power": 80,
        "accuracy": 1.0,
        "priority": 0,
        "current_pp": 12,
        "max_pp": 16,
    }


def pokemon(name: str, moves: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "base_species": name,
        "hp_pct": 0.75,
        "types": "normal notype",
        "tera_type": "normal",
        "item": "leftovers",
        "ability": "pressure",
        "lvl": 100,
        "status": "nostatus",
        "effect": "noeffect",
        "moves": moves or [move("tackle")],
        "atk_boost": 0,
        "spa_boost": 0,
        "def_boost": 0,
        "spd_boost": 0,
        "spe_boost": 0,
        "accuracy_boost": 0,
        "evasion_boost": 0,
        "base_atk": 80,
        "base_spa": 80,
        "base_def": 80,
        "base_spd": 80,
        "base_spe": 80,
        "base_hp": 80,
    }


def state(*, forced_switch: bool = False, can_tera: bool = True) -> dict[str, Any]:
    return {
        "format": "gen9ou",
        "player_active_pokemon": pokemon(
            "zoroark",
            [
                move("thunderbolt", "electric"),
                move("protect"),
                move("quickattack"),
                move("ironhead", "steel"),
            ],
        ),
        "opponent_active_pokemon": pokemon("garganacl", [move("saltcure", "rock")]),
        # Deliberately not alphabetical. Metamon action IDs are alphabetical.
        "available_switches": [
            pokemon("charizard"),
            pokemon("alakazam"),
        ],
        "player_prev_move": move("protect"),
        "opponent_prev_move": move("saltcure", "rock"),
        "opponents_remaining": 4,
        "player_conditions": "stealthrock",
        "opponent_conditions": "spikes",
        "weather": "noweather",
        "battle_field": "nofield",
        "forced_switch": forced_switch,
        "battle_won": False,
        "battle_lost": False,
        "can_tera": can_tera,
        "opponent_teampreview": ["garganacl", "dragapult"],
    }


def terminal_state() -> dict[str, Any]:
    value = deepcopy(state())
    value["battle_won"] = True
    return value

