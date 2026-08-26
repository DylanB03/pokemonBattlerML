from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, Subset


def action_kind(action_id: int) -> str:
    """Return the coarse decision kind used by the move-versus-switch metric."""
    return "switch" if 4 <= action_id <= 8 else "move"


def action_family(action_id: int) -> str:
    """Keep ordinary moves, switches, and Terastallized moves separate."""
    if 4 <= action_id <= 8:
        return "switch"
    if 9 <= action_id <= 12:
        return "tera_move"
    return "move"


def _row_hash(index: int, seed: int) -> int:
    # Prepared JSONL files are immutable experiment artifacts. Hashing their row
    # positions selects across the entire file without decoding every unselected
    # multi-kilobyte state first; the selected-index digest records the exact sample.
    identity = f"{seed}:{index}"
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big")


def select_evaluation_dataset(
    dataset: Dataset[dict[str, Any]],
    *,
    max_examples: int | None,
    mode: str,
    seed: int,
) -> tuple[Dataset[dict[str, Any]], dict[str, Any]]:
    """Select a reproducible head or hash sample and describe the selected rows."""
    if mode not in {"head", "hash"}:
        raise ValueError("sample mode must be 'head' or 'hash'")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")

    wanted = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    if mode == "head" or wanted == len(dataset):
        indices = list(range(wanted))
    else:
        # Retain the rows with the smallest stable hashes without holding every
        # decoded row in memory. Negative values turn heapq into a max heap.
        selected: list[tuple[int, int]] = []
        for index in range(len(dataset)):
            row_hash = _row_hash(index, seed)
            candidate = (-row_hash, index)
            if len(selected) < wanted:
                heapq.heappush(selected, candidate)
            elif candidate > selected[0]:
                heapq.heapreplace(selected, candidate)
        indices = sorted(index for _, index in selected)

    digest = hashlib.sha256(
        ",".join(str(index) for index in indices).encode("utf-8")
    ).hexdigest()
    metadata = {
        "mode": mode,
        "seed": seed,
        "requested_examples": max_examples,
        "available_examples": len(dataset),
        "selected_examples": len(indices),
        "selected_indices_sha256": digest,
    }
    return Subset(dataset, indices), metadata


def load_action_counts(path: str | Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                counts[int(row["action_id"])] += 1
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed action_id at {path}:{line_number}") from exc
    if not counts:
        raise ValueError(f"No action labels found in baseline training file: {path}")
    return counts


class ActionMetrics:
    """Accumulate comparable policy, ranking, and simple-baseline metrics."""

    def __init__(
        self,
        *,
        train_action_counts: Counter[int] | None = None,
        max_saved_errors: int = 100,
    ) -> None:
        self.train_action_counts = train_action_counts
        self.max_saved_errors = max_saved_errors
        self.count = 0
        self.correct = 0
        self.type_correct = 0
        self.oracle_type_correct = 0
        self.top_k_correct: Counter[int] = Counter()
        self.reciprocal_rank_sum = 0.0
        self.uniform_legal_sum = 0.0
        self.uniform_oracle_type_sum = 0.0
        self.prior_correct = 0
        self.prior_oracle_type_correct = 0
        self.nll_sum = 0.0
        self.entropy_sum = 0.0
        self.margins: list[float] = []
        self.target_kind: Counter[str] = Counter()
        self.correct_kind: Counter[str] = Counter()
        self.target_family: Counter[str] = Counter()
        self.correct_family: Counter[str] = Counter()
        self.target_counts: Counter[str] = Counter()
        self.prediction_counts: Counter[str] = Counter()
        self.battle_ids: set[str] = set()
        self.dates: list[str] = []
        self.errors: list[dict[str, Any]] = []

    def add(self, index: int, row: dict[str, Any], scores: dict[int, float]) -> None:
        target = int(row["action_id"])
        legal = [int(value) for value in row.get("legal_action_ids", scores)]
        if set(legal) != set(scores):
            raise ValueError("Scored actions do not match the row's legal-action set")
        if target not in scores:
            raise ValueError(f"Target A{target} was not scored")

        ranked = sorted(scores, key=lambda action_id: (-scores[action_id], action_id))
        prediction = ranked[0]
        rank = ranked.index(target) + 1
        kind = action_kind(target)
        family = action_family(target)
        same_kind = [action_id for action_id in ranked if action_kind(action_id) == kind]

        self.count += 1
        self.correct += int(prediction == target)
        self.type_correct += int(action_kind(prediction) == kind)
        self.oracle_type_correct += int(same_kind[0] == target)
        for k in (1, 2, 3):
            self.top_k_correct[k] += int(rank <= k)
        self.reciprocal_rank_sum += 1.0 / rank
        self.uniform_legal_sum += 1.0 / len(legal)
        self.uniform_oracle_type_sum += 1.0 / len(same_kind)

        self.target_kind[kind] += 1
        self.correct_kind[kind] += int(prediction == target)
        self.target_family[family] += 1
        self.correct_family[family] += int(prediction == target)
        self.target_counts[f"A{target}"] += 1
        self.prediction_counts[f"A{prediction}"] += 1

        score_values = list(scores.values())
        maximum = max(score_values)
        exponentials = [math.exp(value - maximum) for value in score_values]
        normalizer = sum(exponentials)
        log_normalizer = maximum + math.log(normalizer)
        self.nll_sum += log_normalizer - scores[target]
        probabilities = [value / normalizer for value in exponentials]
        self.entropy_sum += -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0
        )
        if len(ranked) > 1:
            self.margins.append(scores[ranked[0]] - scores[ranked[1]])

        if self.train_action_counts is not None:
            prior_prediction = max(
                legal,
                key=lambda action_id: (self.train_action_counts[action_id], -action_id),
            )
            prior_kind_prediction = max(
                same_kind,
                key=lambda action_id: (self.train_action_counts[action_id], -action_id),
            )
            self.prior_correct += int(prior_prediction == target)
            self.prior_oracle_type_correct += int(prior_kind_prediction == target)

        battle_id = row.get("battle_id")
        if battle_id is not None:
            self.battle_ids.add(str(battle_id))
        battle_date = row.get("battle_date")
        if battle_date:
            self.dates.append(str(battle_date))

        if prediction != target and len(self.errors) < self.max_saved_errors:
            self.errors.append(
                {
                    "index": index,
                    "battle_id": battle_id,
                    "turn_index": row.get("turn_index"),
                    "target": f"A{target}",
                    "prediction": f"A{prediction}",
                    "target_rank": rank,
                    "legal_action_ids": legal,
                    "scores": {f"A{key}": value for key, value in sorted(scores.items())},
                }
            )

    def report(self) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("No examples were evaluated")
        report: dict[str, Any] = {
            "examples": self.count,
            "accuracy": self.correct / self.count,
            "correct": self.correct,
            "top_k_accuracy": {
                f"top_{k}": self.top_k_correct[k] / self.count for k in (1, 2, 3)
            },
            "mean_reciprocal_rank": self.reciprocal_rank_sum / self.count,
            "action_type_accuracy": self.type_correct / self.count,
            "oracle_type_accuracy": self.oracle_type_correct / self.count,
            "accuracy_by_target_kind": {
                kind: self.correct_kind[kind] / total
                for kind, total in sorted(self.target_kind.items())
            },
            "accuracy_by_target_family": {
                family: self.correct_family[family] / total
                for family, total in sorted(self.target_family.items())
            },
            "baselines": {
                "uniform_legal_expected_accuracy": self.uniform_legal_sum / self.count,
                "uniform_oracle_type_expected_accuracy": (
                    self.uniform_oracle_type_sum / self.count
                ),
            },
            "average_candidate_nll": self.nll_sum / self.count,
            "average_policy_entropy": self.entropy_sum / self.count,
            "average_top1_margin": (
                sum(self.margins) / len(self.margins) if self.margins else None
            ),
            "target_counts": dict(sorted(self.target_counts.items())),
            "prediction_counts": dict(sorted(self.prediction_counts.items())),
            "sample_coverage": {
                "battles": len(self.battle_ids),
                "first_date": min(self.dates) if self.dates else None,
                "last_date": max(self.dates) if self.dates else None,
            },
            "constraint": (
                "Predictions are restricted to the replay-recoverable candidate set. "
                "This is not a measured legality rate and is weaker than a live simulator mask."
            ),
            "errors": self.errors,
        }
        if self.train_action_counts is not None:
            report["baselines"].update(
                {
                    "train_frequency_legal_accuracy": self.prior_correct / self.count,
                    "train_frequency_oracle_type_accuracy": (
                        self.prior_oracle_type_correct / self.count
                    ),
                    "train_action_counts": {
                        f"A{key}": value
                        for key, value in sorted(self.train_action_counts.items())
                    },
                }
            )
        return report
