from __future__ import annotations

import math
import unittest

import torch

from pokemon_battler.training.distillation import (
    TEACHER_SCHEMA,
    teacher_distillation_loss,
    teacher_policy,
)


def teacher_row() -> dict[str, object]:
    policy = [0.0] * 13
    policy[0] = 0.2
    policy[4] = 0.8
    return {
        "teacher_schema": TEACHER_SCHEMA,
        "legal_action_ids": [0, 4],
        "action_id": 4,
        "teacher": {
            "policy": policy,
            "selected_action_id": 4,
            "confidence": 0.8,
            "visit_count": 1200,
        },
    }


class DistillationTests(unittest.TestCase):
    def test_teacher_policy_requires_a_normalized_exactly_masked_distribution(self) -> None:
        probabilities, action_id, confidence, visits = teacher_policy(teacher_row())
        self.assertEqual(action_id, 4)
        self.assertEqual(confidence, 0.8)
        self.assertEqual(visits, 1200)
        self.assertAlmostEqual(sum(probabilities), 1.0)

        invalid = teacher_row()
        invalid["teacher"]["policy"][1] = 0.1  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "illegal action"):
            teacher_policy(invalid)

    def test_soft_policy_loss_backpropagates_and_reports_disagreement(self) -> None:
        logits = torch.full((2, 13), float("-inf"))
        logits[0, 0] = math.log(0.7)
        logits[0, 4] = math.log(0.3)
        logits[1, 0] = math.log(0.1)
        logits[1, 4] = math.log(0.9)
        logits.requires_grad_()
        teacher = torch.zeros((2, 13))
        teacher[:, 0] = 0.2
        teacher[:, 4] = 0.8
        legal = torch.zeros((2, 13), dtype=torch.bool)
        legal[:, [0, 4]] = True
        loss, metrics = teacher_distillation_loss(
            {"action_log_probs": logits},
            {
                "teacher_probabilities": teacher,
                "teacher_action_ids": torch.tensor([4, 4]),
                "legal_action_mask": legal,
            },
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(metrics["teacher_top1_agreement"]), 0.5)
        self.assertGreater(float(metrics["teacher_student_kl"]), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_q_root_and_outcome_targets_train_separate_value_heads(self) -> None:
        log_probs = torch.full((1, 13), float("-inf"))
        log_probs[0, 0] = math.log(0.2)
        log_probs[0, 4] = math.log(0.8)
        log_probs.requires_grad_()
        action_values = torch.zeros((1, 13))
        action_values[0, 0] = 0.25
        action_values[0, 4] = 0.75
        value_logits = torch.zeros(1, requires_grad=True)
        q_logits = torch.zeros((1, 13), requires_grad=True)
        teacher = torch.zeros((1, 13))
        teacher[0, 0] = 0.2
        teacher[0, 4] = 0.8
        legal = torch.zeros((1, 13), dtype=torch.bool)
        legal[0, [0, 4]] = True
        q_mask = legal.clone()
        loss, metrics = teacher_distillation_loss(
            {
                "action_log_probs": log_probs,
                "action_value_logits": q_logits,
                "value_logits": value_logits,
            },
            {
                "teacher_probabilities": teacher,
                "teacher_action_ids": torch.tensor([4]),
                "legal_action_mask": legal,
                "teacher_action_values": action_values,
                "teacher_action_value_mask": q_mask,
                "teacher_root_values": torch.tensor([0.65]),
                "teacher_root_value_mask": torch.tensor([True]),
                "teacher_outcome_values": torch.tensor([1.0]),
                "teacher_outcome_mask": torch.tensor([True]),
                "teacher_row_weights": torch.ones(1),
            },
        )
        loss.backward()
        self.assertGreater(float(metrics["action_value_loss"]), 0)
        self.assertGreater(float(metrics["root_value_loss"]), 0)
        self.assertGreater(float(metrics["outcome_value_loss"]), 0)
        self.assertIsNotNone(q_logits.grad)
        self.assertIsNotNone(value_logits.grad)

    def test_relative_q_ranking_trains_order_without_absolute_calibration(self) -> None:
        log_probs = torch.full((1, 13), float("-inf"))
        log_probs[0, 0] = math.log(0.5)
        log_probs[0, 4] = math.log(0.5)
        log_probs.requires_grad_()
        q_logits = torch.zeros((1, 13), requires_grad=True)
        teacher = torch.zeros((1, 13))
        teacher[0, 0] = 0.5
        teacher[0, 4] = 0.5
        q_targets = torch.zeros((1, 13))
        q_targets[0, 0] = 0.2
        q_targets[0, 4] = 0.8
        legal = torch.zeros((1, 13), dtype=torch.bool)
        legal[0, [0, 4]] = True
        loss, metrics = teacher_distillation_loss(
            {
                "action_log_probs": log_probs,
                "action_value_logits": q_logits,
            },
            {
                "teacher_probabilities": teacher,
                "teacher_action_ids": torch.tensor([0]),
                "legal_action_mask": legal,
                "teacher_action_values": q_targets,
                "teacher_action_value_mask": legal,
            },
            family_aux_weight=0.0,
            action_value_weight=1.0,
            action_value_loss_type="ranking",
            root_value_weight=0.0,
            outcome_value_weight=0.0,
        )
        loss.backward()
        self.assertAlmostEqual(float(metrics["action_value_loss"]), math.log(2), places=5)
        self.assertGreater(float(q_logits.grad[0, 0]), 0.0)
        self.assertLess(float(q_logits.grad[0, 4]), 0.0)


if __name__ == "__main__":
    unittest.main()
