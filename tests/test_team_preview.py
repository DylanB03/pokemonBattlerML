from __future__ import annotations

import unittest

import torch

from pokemon_battler.models.team_preview import TeamPreviewCollator, TeamPreviewHead
from tests.helpers import pokemon
from tests.test_interaction import FakeTokenizer


class TeamPreviewTests(unittest.TestCase):
    def test_preview_head_scores_only_present_player_slots(self) -> None:
        player = [
            pokemon(name) | {"slot": index, "side": "player"}
            for index, name in enumerate(("dragapult", "kingambit", "corviknight"))
        ]
        opponent = [
            pokemon(name) | {
                "slot": index,
                "side": "opponent",
                "revealed": False,
            }
            for index, name in enumerate(("garchomp", "gholdengo", "primarina"))
        ]
        row = {
            "state": {"format": "gen9ou", "opponent_teampreview": []},
            "player_roster": player,
            "opponent_roster": opponent,
            "action_id": 1,
            "teacher": {"policy": [0.2, 0.7, 0.1]},
        }
        batch = TeamPreviewCollator(FakeTokenizer(), max_length=20_000)([row])
        head = TeamPreviewHead(16, d_model=32)
        logits = head(torch.zeros((1, 16)), batch)
        self.assertEqual(tuple(logits.shape), (1, 6))
        self.assertTrue(torch.isfinite(logits[0, :3]).all())
        self.assertTrue(torch.isneginf(logits[0, 3:]).all())
