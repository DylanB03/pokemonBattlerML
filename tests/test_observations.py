from __future__ import annotations

import unittest

from pokemon_battler.core.observations import OBSERVATION_SCHEMA, canonicalize_observation
from tests.helpers import pokemon, state


class ObservationTests(unittest.TestCase):
    def test_canonical_observation_masks_hidden_information_and_adds_context(self) -> None:
        hidden = pokemon("dragapult") | {
            "side": "opponent",
            "slot": 1,
            "active": False,
            "revealed": False,
            "item": "choiceband",
            "ability": "infiltrator",
            "hp_pct": 1.0,
        }
        revealed = pokemon("garganacl") | {
            "side": "opponent",
            "slot": 0,
            "active": True,
            "revealed": True,
        }
        row = {
            "turn_index": 7,
            "state": state(),
            "player_roster": [],
            "opponent_roster": [revealed, hidden],
            "history_events": [
                {"player_move": "protect", "opponent_move": "saltcure"}
            ],
        }
        row["state"].pop("turn_index", None)
        result = canonicalize_observation(row)
        masked = result["opponent_roster"][1]
        self.assertEqual(result["observation_schema"], OBSERVATION_SCHEMA)
        self.assertEqual(result["state"]["turn_index"], 7)
        self.assertEqual(masked["item"], "unknownitem")
        self.assertEqual(masked["moves"], [])
        self.assertIsNone(masked["hp_pct"])
        self.assertEqual(
            result["state"]["recent_move_history"],
            [{"player": "protect", "opponent": "saltcure"}],
        )
        self.assertEqual(row["opponent_roster"][1]["item"], "choiceband")
