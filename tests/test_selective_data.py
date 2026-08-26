from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pokemon_battler.training.distillation import TEACHER_SCHEMA
from pokemon_battler.data.selective_data import (
    select_disjoint_teacher_rows,
    select_teacher_rows,
)


def row(
    battle: str,
    action_id: int,
    *,
    behavior_action: int,
    confidence: float,
    team: str = "team-a",
) -> dict[str, object]:
    policy = [0.0] * 13
    policy[action_id] = confidence
    alternative = 0 if action_id != 0 else 1
    policy[alternative] += 1.0 - confidence
    return {
        "teacher_schema": TEACHER_SCHEMA,
        "battle_id": battle,
        "enemy_team_file": team,
        "state": {},
        "action_id": action_id,
        "legal_action_ids": sorted({action_id, alternative}),
        "behavior": {
            "source": "student",
            "action_id": behavior_action,
            "preferences": {str(behavior_action): 1.0},
        },
        "teacher": {
            "policy": policy,
            "selected_action_id": action_id,
            "confidence": confidence,
            "visit_count": 100,
        },
    }


class SelectiveDataTests(unittest.TestCase):
    def test_disjoint_selection_holds_out_whole_battles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "teacher.jsonl"
            rows = [
                row(
                    f"battle-{battle}",
                    battle % 9,
                    behavior_action=(battle + 1) % 9,
                    confidence=0.8,
                    team=f"team-{battle % 4}",
                )
                for battle in range(40)
                for _ in range(2)
            ]
            path.write_text(
                "".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8"
            )
            train, validation, report = select_disjoint_teacher_rows(
                [path],
                train_limit=40,
                validation_limit=12,
                seed=7,
                validation_fraction=0.25,
                battle_cap=2,
            )
            self.assertTrue(train)
            self.assertTrue(validation)
            self.assertTrue(
                {reference.battle_id for reference in train}.isdisjoint(
                    reference.battle_id for reference in validation
                )
            )
            self.assertEqual(report["strategy"], "whole-battle-hash")

    def test_student_disagreement_has_priority_within_action_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "teacher.jsonl"
            rows = [
                row("agreement", 0, behavior_action=0, confidence=0.99),
                row("disagreement", 1, behavior_action=0, confidence=0.60),
            ]
            path.write_text(
                "".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8"
            )
            selected, report = select_teacher_rows(
                [path], limit=1, seed=42, battle_cap=24
            )
            self.assertEqual(selected[0].battle_id, "disagreement")
            self.assertEqual(report["student_disagreements"], 1)

    def test_selection_caps_long_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "teacher.jsonl"
            rows = [
                row("long", 4, behavior_action=0, confidence=0.9)
                for _ in range(10)
            ] + [row("short", 4, behavior_action=0, confidence=0.8)]
            path.write_text(
                "".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8"
            )
            selected, _ = select_teacher_rows(
                [path], limit=5, seed=7, battle_cap=2
            )
            self.assertLessEqual(
                sum(reference.battle_id == "long" for reference in selected), 2
            )
            self.assertTrue(any(reference.battle_id == "short" for reference in selected))
