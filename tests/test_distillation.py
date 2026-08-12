from __future__ import annotations

import math
import unittest

import torch

from pokemon_battler.distillation import (
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


if __name__ == "__main__":
    unittest.main()
