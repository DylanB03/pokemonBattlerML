from __future__ import annotations

import json
import math
import random
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pokemon_battler.frozen_cache import checkpoint_signature
from pokemon_battler.modeling import load_training_metadata
from pokemon_battler.residual_cache import ResidualTeacherCache
from pokemon_battler.residual_modeling import (
    RESIDUAL_CONFIG_FILENAME,
    RESIDUAL_HEAD_FILENAME,
    ResidualPolicyHead,
    load_champion_scorer,
    save_residual_head,
)
from pokemon_battler.train import set_seed
from pokemon_battler.trajectory_cache import EncodedTrajectoryCache
from pokemon_battler.trajectory_modeling import TRAJECTORY_HEAD_FILENAME


class ReplayEmbeddingDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, cache: EncodedTrajectoryCache, indices: Sequence[int]) -> None:
        self.cache = cache
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = self.indices[index]
        return {
            "global": torch.from_numpy(
                np.array(self.cache.arrays["global"][source_index], copy=True)
            ),
            "candidates": torch.from_numpy(
                np.array(self.cache.arrays["candidates"][source_index], copy=True)
            ),
            "legal": torch.from_numpy(
                np.array(self.cache.arrays["legal"][source_index], copy=True)
            ),
        }


def _sample_indices(length: int, limit: int, seed: int) -> list[int]:
    count = min(length, limit)
    return sorted(random.Random(seed).sample(range(length), count))


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _policy_kl(
    target_probabilities: torch.Tensor,
    candidate_log_probs: torch.Tensor,
    legal: torch.Tensor,
) -> torch.Tensor:
    safe_logs = torch.where(legal, candidate_log_probs, torch.zeros_like(candidate_log_probs))
    entropy_terms = torch.where(
        target_probabilities > 0,
        target_probabilities * torch.log(target_probabilities.clamp_min(1e-12)),
        torch.zeros_like(target_probabilities),
    )
    return (entropy_terms - target_probabilities * safe_logs).sum(dim=1)


def _evaluate_teacher(
    head: ResidualPolicyHead,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    head.eval()
    totals = {"kl": 0.0, "cross_entropy": 0.0, "correct": 0.0, "delta": 0.0}
    examples = 0
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = _move(cpu_batch, device)
            legal = batch["legal"].bool()
            output = head(
                batch["global"],
                batch["candidates"],
                legal,
                batch["champion_log_probs"],
            )
            log_probs = output["action_log_probs"]
            teacher = batch["teacher_probabilities"].float()
            size = int(teacher.shape[0])
            safe_logs = torch.where(legal, log_probs, torch.zeros_like(log_probs))
            totals["cross_entropy"] += float((-(teacher * safe_logs).sum(1)).sum().item())
            totals["kl"] += float(_policy_kl(teacher, log_probs, legal).sum().item())
            totals["correct"] += float(
                (log_probs.argmax(1) == batch["teacher_actions"].long()).sum().item()
            )
            totals["delta"] += float(
                output["logit_deltas"].abs().sum().item()
                / legal.sum().clamp_min(1).item()
                * size
            )
            examples += size
    return {
        "examples": float(examples),
        "kl": totals["kl"] / max(examples, 1),
        "cross_entropy": totals["cross_entropy"] / max(examples, 1),
        "top1_agreement": totals["correct"] / max(examples, 1),
        "mean_absolute_logit_delta": totals["delta"] / max(examples, 1),
    }


def _evaluate_replay(
    head: ResidualPolicyHead,
    scorer: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    head.eval()
    kl_total = 0.0
    changed = 0
    delta_total = 0.0
    legal_total = 0
    examples = 0
    maximum_log_probability_difference = 0.0
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = _move(cpu_batch, device)
            legal = batch["legal"].bool()
            base = scorer(batch["global"], batch["candidates"], legal)
            output = head(batch["global"], batch["candidates"], legal, base)
            candidate = output["action_log_probs"]
            maximum_log_probability_difference = max(
                maximum_log_probability_difference,
                float((candidate[legal] - base[legal]).abs().max().item()),
            )
            base_probabilities = base.exp()
            kl_total += float(_policy_kl(base_probabilities, candidate, legal).sum().item())
            changed += int((candidate.argmax(1) != base.argmax(1)).sum().item())
            delta_total += float(output["logit_deltas"].abs().sum().item())
            legal_total += int(legal.sum().item())
            examples += int(legal.shape[0])
    return {
        "examples": float(examples),
        "kl_from_champion": kl_total / max(examples, 1),
        "top_action_change_rate": changed / max(examples, 1),
        "mean_absolute_logit_delta": delta_total / max(legal_total, 1),
        "maximum_legal_log_probability_difference": (
            maximum_log_probability_difference
        ),
    }


def _copy_champion(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    excluded = {
        RESIDUAL_HEAD_FILENAME,
        RESIDUAL_CONFIG_FILENAME,
        TRAJECTORY_HEAD_FILENAME,
    }
    for path in source.iterdir():
        if path.is_file() and path.name not in excluded:
            shutil.copy2(path, destination / path.name)


def train_residual_policy(
    *,
    checkpoint: Path,
    teacher_train_cache: Path,
    teacher_validation_cache: Path,
    replay_train_cache: Path,
    replay_validation_cache: Path,
    output_dir: Path,
    epochs: int = 12,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    maximum_gradient_norm: float = 1.0,
    hidden_size: int = 256,
    dropout: float = 0.1,
    maximum_logit_delta: float = 1.5,
    rehearsal_weight: float = 0.5,
    residual_penalty_weight: float = 0.01,
    replay_train_rows: int = 32_000,
    replay_validation_rows: int = 8_000,
    early_stopping_patience: int = 3,
    minimum_teacher_kl_gain: float = 0.02,
    minimum_teacher_agreement_gain: float = 0.02,
    maximum_replay_kl: float = 0.05,
    maximum_replay_action_change: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("Residual training settings must be positive")
    if replay_train_rows <= 0 or replay_validation_rows <= 0:
        raise ValueError("Replay sample sizes must be positive")
    if rehearsal_weight < 0 or residual_penalty_weight < 0:
        raise ValueError("Residual loss weights cannot be negative")
    set_seed(seed)
    train_source = ResidualTeacherCache(teacher_train_cache)
    validation_source = ResidualTeacherCache(teacher_validation_cache)
    signature = checkpoint_signature(checkpoint)
    if any(
        source.metadata.get("checkpoint_signature") != signature
        for source in (train_source, validation_source)
    ):
        raise ValueError("Teacher embeddings were built from a different champion")
    replay_train_source = EncodedTrajectoryCache(replay_train_cache)
    replay_validation_source = EncodedTrajectoryCache(replay_validation_cache)
    if any(
        source.metadata.get("checkpoint_signature") != signature
        for source in (replay_train_source, replay_validation_source)
    ):
        raise ValueError("Replay embeddings were built from a different champion")
    d_model = int(train_source.metadata["d_model"])
    if int(replay_train_source.metadata["d_model"]) != d_model:
        raise ValueError("Teacher and replay embedding sizes disagree")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = ResidualPolicyHead(
        d_model,
        hidden_size=hidden_size,
        dropout=dropout,
        maximum_logit_delta=maximum_logit_delta,
    ).to(device)
    scorer = load_champion_scorer(checkpoint, d_model=d_model, device=device)
    train_loader = DataLoader(train_source, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_source, batch_size=batch_size, shuffle=False)
    replay_train = ReplayEmbeddingDataset(
        replay_train_source,
        _sample_indices(
            int(replay_train_source.metadata["rows"]), replay_train_rows, seed + 1
        ),
    )
    replay_validation = ReplayEmbeddingDataset(
        replay_validation_source,
        _sample_indices(
            int(replay_validation_source.metadata["rows"]),
            replay_validation_rows,
            seed + 2,
        ),
    )
    replay_loader = DataLoader(replay_train, batch_size=batch_size, shuffle=True)
    replay_validation_loader = DataLoader(
        replay_validation, batch_size=batch_size, shuffle=False
    )
    teacher_before = _evaluate_teacher(head, validation_loader, device)
    replay_before = _evaluate_replay(head, scorer, replay_validation_loader, device)
    identity_error = replay_before["maximum_legal_log_probability_difference"]
    if identity_error > 2e-6 or replay_before["top_action_change_rate"] != 0:
        raise RuntimeError(
            f"Zero-initialized residual does not reproduce champion: {replay_before}"
        )
    print(
        json.dumps(
            {
                "phase": "residual-before",
                "identity_max_log_probability_difference": identity_error,
                "teacher": teacher_before,
                "replay": replay_before,
            }
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_score = math.inf
    best_epoch = 0
    stale = 0
    updates = 0
    started = time.monotonic()
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        replay_iterator = iter(replay_loader)
        totals = {"teacher": 0.0, "replay": 0.0, "penalty": 0.0, "examples": 0}
        for teacher_cpu in train_loader:
            try:
                replay_cpu = next(replay_iterator)
            except StopIteration:
                replay_iterator = iter(replay_loader)
                replay_cpu = next(replay_iterator)
            teacher_batch = _move(teacher_cpu, device)
            replay_batch = _move(replay_cpu, device)
            teacher_legal = teacher_batch["legal"].bool()
            teacher_output = head(
                teacher_batch["global"],
                teacher_batch["candidates"],
                teacher_legal,
                teacher_batch["champion_log_probs"],
            )
            teacher_probabilities = teacher_batch["teacher_probabilities"].float()
            safe_teacher_logs = torch.where(
                teacher_legal,
                teacher_output["action_log_probs"],
                torch.zeros_like(teacher_output["action_log_probs"]),
            )
            per_example = -(teacher_probabilities * safe_teacher_logs).sum(1)
            confidence_weights = 0.5 + teacher_batch["teacher_confidence"].float()
            confidence_weights = confidence_weights / confidence_weights.mean().clamp_min(1e-6)
            teacher_loss = (per_example * confidence_weights).mean()

            replay_legal = replay_batch["legal"].bool()
            with torch.no_grad():
                replay_base = scorer(
                    replay_batch["global"], replay_batch["candidates"], replay_legal
                )
            replay_output = head(
                replay_batch["global"],
                replay_batch["candidates"],
                replay_legal,
                replay_base,
            )
            replay_loss = _policy_kl(
                replay_base.exp(), replay_output["action_log_probs"], replay_legal
            ).mean()
            legal_deltas = teacher_output["logit_deltas"].masked_select(teacher_legal)
            penalty = legal_deltas.square().mean()
            loss = (
                teacher_loss
                + rehearsal_weight * replay_loss
                + residual_penalty_weight * penalty
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), maximum_gradient_norm)
            optimizer.step()
            size = int(teacher_legal.shape[0])
            totals["teacher"] += float(teacher_loss.item()) * size
            totals["replay"] += float(replay_loss.item()) * size
            totals["penalty"] += float(penalty.item()) * size
            totals["examples"] += size
            updates += 1
        teacher_metrics = _evaluate_teacher(head, validation_loader, device)
        replay_metrics = _evaluate_replay(head, scorer, replay_validation_loader, device)
        score = teacher_metrics["kl"] + 10.0 * max(
            0.0, replay_metrics["kl_from_champion"] - maximum_replay_kl
        )
        row = {
            "epoch": epoch,
            "train_teacher_cross_entropy": totals["teacher"] / totals["examples"],
            "train_replay_kl": totals["replay"] / totals["examples"],
            "train_residual_penalty": totals["penalty"] / totals["examples"],
            "teacher_validation": teacher_metrics,
            "replay_validation": replay_metrics,
            "selection_score": score,
        }
        history.append(row)
        print(json.dumps({"phase": "residual-epoch", **row}), flush=True)
        if score < best_score - 1e-4:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Residual training produced no checkpoint")
    head.load_state_dict(best_state)
    teacher_after = _evaluate_teacher(head, validation_loader, device)
    replay_after = _evaluate_replay(head, scorer, replay_validation_loader, device)
    teacher_kl_gain = teacher_before["kl"] - teacher_after["kl"]
    agreement_gain = teacher_after["top1_agreement"] - teacher_before["top1_agreement"]
    teacher_gate = (
        teacher_kl_gain >= minimum_teacher_kl_gain
        or agreement_gain >= minimum_teacher_agreement_gain
    )
    replay_gate = (
        replay_after["kl_from_champion"] <= maximum_replay_kl
        and replay_after["top_action_change_rate"] <= maximum_replay_action_change
    )
    offline_gate_passed = teacher_gate and replay_gate
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_champion(checkpoint, output_dir)
    save_residual_head(head, output_dir)
    source_metadata = load_training_metadata(checkpoint)
    training_config = source_metadata | {
        "schema": "champion-residual-training-v1",
        "source_checkpoint": str(checkpoint),
        "teacher_train_cache": str(teacher_train_cache),
        "teacher_validation_cache": str(teacher_validation_cache),
        "replay_train_cache": str(replay_train_cache),
        "replay_validation_cache": str(replay_validation_cache),
        "residual_policy": head.config(),
        "deployment_action_value_weight": 0.0,
        "offline_gate_passed": offline_gate_passed,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "schema": "champion-residual-training-v1",
        "source_checkpoint": str(checkpoint),
        "checkpoint_signature": signature,
        "updates": updates,
        "best_epoch": best_epoch,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "identity_max_log_probability_difference": identity_error,
        "teacher_before": teacher_before,
        "teacher_after": teacher_after,
        "teacher_kl_gain": teacher_kl_gain,
        "teacher_agreement_gain": agreement_gain,
        "replay_before": replay_before,
        "replay_after": replay_after,
        "offline_gate": {
            "teacher_passed": teacher_gate,
            "replay_passed": replay_gate,
            "passed": offline_gate_passed,
            "minimum_teacher_kl_gain": minimum_teacher_kl_gain,
            "minimum_teacher_agreement_gain": minimum_teacher_agreement_gain,
            "maximum_replay_kl": maximum_replay_kl,
            "maximum_replay_action_change": maximum_replay_action_change,
        },
        "history": history,
    }
    (output_dir / "residual_training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report
