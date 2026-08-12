from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from pokemon_battler.actions import ACTION_COUNT
from pokemon_battler.modeling import indexed_logits_parameter, interaction_outputs
from pokemon_battler.rl_training import (
    _autocast,
    _load_trainable_policy,
    _mean_metrics,
    _optimizer,
)
from pokemon_battler.train import _save_checkpoint, set_seed
from pokemon_battler.training_data import (
    InteractionInferenceCollator,
    JsonlOffsetDataset,
)

TEACHER_SCHEMA = "foul-play-distillation-v1"


def teacher_policy(row: dict[str, Any]) -> tuple[list[float], int, float, int]:
    """Validate and return one legal 13-candidate teacher distribution."""
    if row.get("teacher_schema") != TEACHER_SCHEMA:
        raise ValueError(
            f"Teacher row uses {row.get('teacher_schema')!r}; expected {TEACHER_SCHEMA!r}"
        )
    teacher = row.get("teacher")
    if not isinstance(teacher, dict):
        raise ValueError("Teacher row is missing its teacher metadata")
    raw_policy = teacher.get("policy")
    if not isinstance(raw_policy, list) or len(raw_policy) != ACTION_COUNT:
        raise ValueError("teacher.policy must contain exactly 13 probabilities")
    probabilities = [float(value) for value in raw_policy]
    if any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("teacher.policy contains an invalid probability")
    legal = {int(value) for value in row["legal_action_ids"]}
    if any(
        probability > 1e-8 and action_id not in legal
        for action_id, probability in enumerate(probabilities)
    ):
        raise ValueError("teacher.policy assigns probability to an illegal action")
    total = sum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError(f"teacher.policy sums to {total}, not 1")
    selected_action = int(teacher.get("selected_action_id", row.get("action_id", -1)))
    if selected_action not in legal:
        raise ValueError("Teacher selected action is absent from the exact legal mask")
    if int(row.get("action_id", selected_action)) != selected_action:
        raise ValueError("Teacher selected action disagrees with row.action_id")
    confidence = max(probabilities)
    recorded_confidence = float(teacher.get("confidence", confidence))
    if not math.isclose(confidence, recorded_confidence, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Teacher confidence disagrees with teacher.policy")
    visits = int(teacher.get("visit_count", 0) or 0)
    return probabilities, selected_action, confidence, visits


class TeacherDistillationCollator(InteractionInferenceCollator):
    """Collate public observations plus soft Foul Play policy targets."""

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = super().__call__(rows)
        targets = [teacher_policy(row) for row in rows]
        batch.update(
            {
                "teacher_probabilities": torch.tensor(
                    [target[0] for target in targets], dtype=torch.float32
                ),
                "teacher_action_ids": torch.tensor(
                    [target[1] for target in targets], dtype=torch.long
                ),
                "teacher_confidence": torch.tensor(
                    [target[2] for target in targets], dtype=torch.float32
                ),
                "teacher_visits": torch.tensor(
                    [target[3] for target in targets], dtype=torch.long
                ),
            }
        )
        return batch


def teacher_distillation_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    hard_target_weight: float = 0.1,
    confident_disagreement_weight: float = 1.0,
    confidence_power: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked soft-policy cross entropy with confident-disagreement weighting."""
    if not 0 <= hard_target_weight <= 1:
        raise ValueError("hard_target_weight must be between zero and one")
    if confident_disagreement_weight < 0 or confidence_power < 0:
        raise ValueError("Distillation weights cannot be negative")
    log_probs = outputs["action_log_probs"].float()
    teacher = batch["teacher_probabilities"].to(log_probs.device).float()
    legal = batch["legal_action_mask"].to(log_probs.device).bool()
    if teacher.shape != log_probs.shape:
        raise ValueError("Teacher and student policies must both use 13 candidate slots")
    if bool((teacher.masked_select(~legal) > 1e-8).any()):
        raise ValueError("Teacher assigns probability to an illegal student candidate")
    safe_log_probs = torch.where(legal, log_probs, torch.zeros_like(log_probs))
    soft_cross_entropy = -(teacher * safe_log_probs).sum(dim=1)
    teacher_entropy = -(
        teacher * torch.where(teacher > 0, torch.log(teacher), torch.zeros_like(teacher))
    ).sum(dim=1)
    kl_divergence = soft_cross_entropy - teacher_entropy

    selected = batch["teacher_action_ids"].to(log_probs.device).long()
    hard_cross_entropy = -log_probs.gather(1, selected[:, None]).squeeze(1)
    per_example = (
        (1.0 - hard_target_weight) * soft_cross_entropy
        + hard_target_weight * hard_cross_entropy
    )
    teacher_confidence, teacher_top = teacher.max(dim=1)
    student_top = log_probs.argmax(dim=1)
    disagreement = (student_top != teacher_top).float()
    weights = teacher_confidence.pow(confidence_power) * (
        1.0
        + confident_disagreement_weight * teacher_confidence * disagreement
    )
    # Preserve the learning-rate scale while changing examples' relative weight.
    weights = weights / weights.mean().clamp_min(1e-6)
    loss = (per_example * weights).mean()
    student_probabilities = torch.where(legal, log_probs.exp(), torch.zeros_like(log_probs))
    student_entropy = -(
        student_probabilities
        * torch.where(
            student_probabilities > 0,
            log_probs,
            torch.zeros_like(log_probs),
        )
    ).sum(dim=1)
    return loss, {
        "soft_cross_entropy": soft_cross_entropy.mean().detach(),
        "hard_cross_entropy": hard_cross_entropy.mean().detach(),
        "teacher_student_kl": kl_divergence.mean().detach(),
        "teacher_top1_agreement": (1.0 - disagreement).mean().detach(),
        "teacher_confidence": teacher_confidence.mean().detach(),
        "student_entropy": student_entropy.mean().detach(),
        "example_weight": weights.mean().detach(),
    }


def _evaluate_teacher(
    model: Any,
    head: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    logits_parameter: str | None,
    hard_target_weight: float,
    confident_disagreement_weight: float,
    confidence_power: float,
) -> dict[str, float]:
    model.eval()
    head.eval()
    totals: dict[str, float] = {}
    examples = 0
    with torch.inference_mode():
        for batch in loader:
            batch_size = int(batch["input_ids"].shape[0])
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast(device, dtype):
                outputs = interaction_outputs(
                    model, head, batch, logits_parameter=logits_parameter
                )
                loss, parts = teacher_distillation_loss(
                    outputs,
                    batch,
                    hard_target_weight=hard_target_weight,
                    confident_disagreement_weight=confident_disagreement_weight,
                    confidence_power=confidence_power,
                )
            for key, value in {"distillation_loss": loss.detach(), **parts}.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
            examples += batch_size
    return {"examples": float(examples), **_mean_metrics(totals, examples)}


def train_teacher_distillation(
    *,
    checkpoint: Path,
    teacher_file: Path,
    output_dir: Path,
    validation_file: Path | None = None,
    model_name: str | None = None,
    epochs: int = 3,
    max_steps: int | None = None,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    qwen_learning_rate: float = 5e-6,
    head_learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    hard_target_weight: float = 0.1,
    confident_disagreement_weight: float = 1.0,
    confidence_power: float = 1.0,
    max_length: int | None = None,
    dtype_name: str = "auto",
    load_in_4bit: bool = True,
    local_files_only: bool = True,
    attn_implementation: str = "sdpa",
    seed: int = 42,
    log_steps: int = 20,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Distillation output directory is not empty: {output_dir}")
    if epochs <= 0 or batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("Distillation epoch and batch settings must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")
    set_seed(seed)
    model, tokenizer, head, device, dtype, source_metadata = _load_trainable_policy(
        checkpoint,
        model_name=model_name,
        dtype_name=dtype_name,
        load_in_4bit=load_in_4bit,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    train_dataset = JsonlOffsetDataset(teacher_file)
    validation_dataset = (
        JsonlOffsetDataset(validation_file) if validation_file is not None else None
    )
    resolved_max_length = max_length or int(source_metadata.get("max_length", 4096))
    collator = TeacherDistillationCollator(
        tokenizer,
        max_length=resolved_max_length,
        truncation="error",
        prompt_format=str(source_metadata.get("prompt_format", "mechanics-v2")),
    )
    evaluation_dataset = validation_dataset or train_dataset
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    logits_parameter = indexed_logits_parameter(model)
    before = _evaluate_teacher(
        model,
        head,
        evaluation_loader,
        device=device,
        dtype=dtype,
        logits_parameter=logits_parameter,
        hard_target_weight=hard_target_weight,
        confident_disagreement_weight=confident_disagreement_weight,
        confidence_power=confidence_power,
    )
    print(json.dumps({"phase": "teacher-before", **before}), flush=True)

    optimizer, parameters = _optimizer(
        model,
        head,
        qwen_learning_rate=qwen_learning_rate,
        head_learning_rate=head_learning_rate,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and dtype == torch.float16
    )
    model.train()
    head.train()
    optimizer.zero_grad(set_to_none=True)
    updates = 0
    examples = 0
    totals: dict[str, float] = {}
    started = time.monotonic()
    stop = False
    for epoch in range(epochs):
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        microsteps = 0
        for batch_index, batch in enumerate(loader):
            batch_size_now = int(batch["input_ids"].shape[0])
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast(device, dtype):
                outputs = interaction_outputs(
                    model, head, batch, logits_parameter=logits_parameter
                )
                loss, parts = teacher_distillation_loss(
                    outputs,
                    batch,
                    hard_target_weight=hard_target_weight,
                    confident_disagreement_weight=confident_disagreement_weight,
                    confidence_power=confidence_power,
                )
                scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            examples += batch_size_now
            microsteps += 1
            for key, value in {"distillation_loss": loss.detach(), **parts}.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size_now
            should_update = (
                microsteps == gradient_accumulation_steps
                or batch_index + 1 == len(loader)
            )
            if not should_update:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            microsteps = 0
            updates += 1
            if updates == 1 or updates % log_steps == 0:
                print(
                    json.dumps(
                        {
                            "phase": "teacher-distillation",
                            "epoch": epoch + 1,
                            "step": updates,
                            "examples": examples,
                            **_mean_metrics(totals, examples),
                        }
                    ),
                    flush=True,
                )
            if max_steps is not None and updates >= max_steps:
                stop = True
                break
        if stop:
            break

    after = _evaluate_teacher(
        model,
        head,
        evaluation_loader,
        device=device,
        dtype=dtype,
        logits_parameter=logits_parameter,
        hard_target_weight=hard_target_weight,
        confident_disagreement_weight=confident_disagreement_weight,
        confidence_power=confidence_power,
    )
    print(json.dumps({"phase": "teacher-after", **after}), flush=True)
    report: dict[str, Any] = {
        "schema": "qwen-foul-play-distillation-v1",
        "source_checkpoint": str(checkpoint),
        "teacher_file": str(teacher_file),
        "validation_file": str(validation_file) if validation_file is not None else None,
        "teacher_rows": len(train_dataset),
        "updates": updates,
        "examples": examples,
        "elapsed_seconds": time.monotonic() - started,
        "epochs": epochs,
        "hard_target_weight": hard_target_weight,
        "confident_disagreement_weight": confident_disagreement_weight,
        "confidence_power": confidence_power,
        "qwen_learning_rate": qwen_learning_rate,
        "head_learning_rate": head_learning_rate,
        "before": before,
        "after": after,
        **_mean_metrics(totals, examples),
    }
    config = source_metadata | report | {
        "model": model_name or source_metadata.get("model", "Qwen/Qwen2.5-0.5B"),
        "max_length": resolved_max_length,
        "load_in_4bit": load_in_4bit,
        "local_files_only": local_files_only,
        "dtype": dtype_name,
        "attn_implementation": attn_implementation,
        "training_objective": "foul-play-policy-distillation",
    }
    _save_checkpoint(model, tokenizer, output_dir, config, head, "interaction-head")
    (output_dir / "distillation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
