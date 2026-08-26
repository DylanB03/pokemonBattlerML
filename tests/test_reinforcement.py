from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from pokemon_battler.training.league import QwenLeague
from pokemon_battler.training.reinforcement import (
    ROLLOUT_SCHEMA,
    WinTrajectoryBuffer,
    expectile_loss,
    generalized_advantages,
    offline_outcome_loss,
    ppo_loss,
)
from pokemon_battler.pipelines.win_experiment import _default_team_rotations, build_parser


class ReinforcementTests(unittest.TestCase):
    def test_expectile_penalizes_underestimates_at_high_expectile(self) -> None:
        prediction = torch.tensor([0.0, 1.0])
        target = torch.tensor([1.0, 0.0])
        loss = expectile_loss(prediction, target, 0.8)
        self.assertAlmostEqual(float(loss), 0.5, places=6)

    def test_offline_loss_trains_policy_q_and_value(self) -> None:
        action_log_probs = torch.log_softmax(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True), dim=1
        )
        q_logits = torch.tensor(
            [[0.0, -1.0], [-1.0, 0.0]], requires_grad=True
        )
        value_logits = torch.zeros(2, requires_grad=True)
        outputs = {
            "action_log_probs": action_log_probs,
            "action_value_logits": q_logits,
            "value_logits": value_logits,
        }
        batch = {
            "action_ids": torch.tensor([0, 1]),
            "value_targets": torch.tensor([1.0, 0.0]),
        }
        loss, parts = offline_outcome_loss(outputs, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("q_loss", parts)
        loss.backward()
        self.assertIsNotNone(q_logits.grad)
        self.assertIsNotNone(value_logits.grad)

    def test_gae_propagates_terminal_win(self) -> None:
        advantages, returns = generalized_advantages(
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [False, False, True],
            gamma=1.0,
            gae_lambda=1.0,
        )
        self.assertEqual(advantages, [1.0, 1.0, 1.0])
        self.assertEqual(returns, [1.0, 1.0, 1.0])

    def test_ppo_uses_only_finite_legal_actions(self) -> None:
        log_probs = torch.tensor(
            [
                [torch.log(torch.tensor(0.75)), torch.log(torch.tensor(0.25)), -torch.inf],
                [torch.log(torch.tensor(0.25)), torch.log(torch.tensor(0.75)), -torch.inf],
            ],
            requires_grad=True,
        )
        outputs = {
            "action_log_probs": log_probs,
            "value_logits": torch.zeros(2, requires_grad=True),
        }
        batch = {
            "action_ids": torch.tensor([0, 1]),
            "old_log_probs": torch.log(torch.tensor([0.75, 0.75])),
            "old_values": torch.tensor([0.0, 0.0]),
            "advantages": torch.tensor([1.0, -1.0]),
            "returns": torch.tensor([1.0, -1.0]),
        }
        loss, parts = ppo_loss(outputs, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(parts["policy_entropy"]))
        loss.backward()
        self.assertIsNotNone(log_probs.grad)

    def test_live_buffer_writes_completed_training_rows(self) -> None:
        buffer = WinTrajectoryBuffer(gamma=1.0, gae_lambda=1.0)
        observation = {
            "schema_version": 3,
            "state": {},
            "legal_action_ids": [0, 1],
        }
        for action in (0, 1):
            buffer.record_decision(
                "battle-1",
                observation,
                action_id=action,
                old_log_probability=-0.5,
                value_probability=0.5,
            )
        buffer.finish_battle("battle-1", won=True, lost=False)
        rows = buffer.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["rollout_schema"], ROLLOUT_SCHEMA)
        self.assertEqual(rows[0]["return"], 1.0)
        self.assertTrue(rows[-1]["done"])
        self.assertEqual(buffer.pending_battles, 0)

    def test_league_keeps_rejected_checkpoints_and_promotes_winners(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / "initial"
            initial.mkdir()
            league = QwenLeague(root / "league.json")
            league.initialize(initial)
            league.add_reference(root / "behavior-cloning", entry_id="bc")
            self.assertEqual(len(league.data["entries"]), 2)
            rejected = league.record_candidate(
                candidate_id="bad",
                checkpoint=root / "bad",
                wins=4,
                losses=6,
                ties=0,
                promotion_threshold=0.55,
            )
            self.assertFalse(rejected["promoted"])
            self.assertEqual(league.champion["id"], "initial")
            promoted = league.record_candidate(
                candidate_id="good",
                checkpoint=root / "good",
                wins=6,
                losses=4,
                ties=0,
                promotion_threshold=0.55,
            )
            self.assertTrue(promoted["promoted"])
            self.assertEqual(league.champion["id"], "good")
            persisted = json.loads((root / "league.json").read_text(encoding="utf-8"))
            self.assertEqual(len(persisted["matches"]), 2)

    def test_win_command_defaults_to_qwen_offline_then_ppo(self) -> None:
        args = build_parser().parse_args([])
        self.assertFalse(args.skip_offline)
        self.assertEqual(args.iterations, 4)
        self.assertEqual(args.rollout_games, 64)
        self.assertTrue(args.load_in_4bit)

    def test_default_pool_rotates_every_member_into_the_lead(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "team.txt"
            source.write_text(
                "\n\n".join(f"Pokemon {index}\n- Tackle" for index in range(6)),
                encoding="utf-8",
            )
            paths = _default_team_rotations(source, Path(temporary) / "pool")
            leads = [
                path.read_text(encoding="utf-8").splitlines()[0] for path in paths
            ]
            self.assertEqual(leads, [f"Pokemon {index}" for index in range(6)])


if __name__ == "__main__":
    unittest.main()
