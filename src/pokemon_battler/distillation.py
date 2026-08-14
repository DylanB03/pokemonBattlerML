from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

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
    InteractionCollator,
    InteractionInferenceCollator,
    JsonlOffsetDataset,
)

TEACHER_SCHEMA = "foul-play-distillation-v2"
SUPPORTED_TEACHER_SCHEMAS = {"foul-play-distillation-v1", TEACHER_SCHEMA}


def teacher_policy(row: dict[str, Any]) -> tuple[list[float], int, float, int]:
    """Validate and return one legal 13-candidate teacher distribution."""
    if row.get("teacher_schema") not in SUPPORTED_TEACHER_SCHEMAS:
        raise ValueError(
            f"Teacher row uses {row.get('teacher_schema')!r}; expected one of "
            f"{sorted(SUPPORTED_TEACHER_SCHEMAS)!r}"
        )
    teacher = row.get("teacher")
    if not isinstance(teacher, dict):
        raise TypeError("Teacher row is missing its teacher metadata")
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


class TeacherTurnDataset(Dataset[dict[str, Any]]):
    """A grouped subset of turn decisions with trajectory-balanced row weights."""

    def __init__(
        self,
        source: JsonlOffsetDataset,
        indices: Sequence[int],
        *,
        trajectory_cap: int = 48,
    ) -> None:
        if not indices:
            raise ValueError("Teacher turn split contains no examples")
        self.source = source
        self.indices = list(indices)
        battle_counts = Counter(
            str(source[index].get("battle_id") or f"row-{index}") for index in self.indices
        )
        raw_weights = {
            battle_id: min(count, trajectory_cap) / count
            for battle_id, count in battle_counts.items()
        }
        mean = sum(
            raw_weights[str(source[index].get("battle_id") or f"row-{index}")]
            for index in self.indices
        ) / len(self.indices)
        self.weights = {
            battle_id: weight / max(mean, 1e-8)
            for battle_id, weight in raw_weights.items()
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self.indices[index]
        row = self.source[source_index]
        battle_id = str(row.get("battle_id") or f"row-{source_index}")
        return row | {"_teacher_row_weight": self.weights[battle_id]}


def grouped_teacher_split(
    source: JsonlOffsetDataset,
    *,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Hold out whole enemy teams when possible, otherwise whole battles."""
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between zero and 0.5")
    turn_indices: list[int] = []
    groups: dict[str, list[int]] = {}
    team_groups: set[str] = set()
    for index in range(len(source)):
        row = source[index]
        if row.get("decision_phase") == "team_preview":
            continue
        teacher_policy(row)
        turn_indices.append(index)
        team = str(row.get("enemy_team_file") or "")
        battle = str(row.get("battle_id") or f"row-{index}")
        group = f"team:{team}" if team else f"battle:{battle}"
        if team:
            team_groups.add(group)
        groups.setdefault(group, []).append(index)
    if len(turn_indices) < 2:
        raise ValueError("Teacher data needs at least two turn rows for grouped validation")
    candidates = sorted(team_groups if len(team_groups) >= 3 else groups)
    random.Random(seed).shuffle(candidates)
    target = max(1, round(len(turn_indices) * validation_fraction))
    validation_groups: set[str] = set()
    examples = 0
    for group in candidates:
        if examples >= target and validation_groups:
            break
        validation_groups.add(group)
        examples += len(groups[group])
    validation = [
        index
        for group in validation_groups
        for index in groups[group]
    ]
    validation_set = set(validation)
    train = [index for index in turn_indices if index not in validation_set]
    if not train or not validation:
        raise ValueError("Grouped split could not produce non-empty train and validation sets")
    return train, validation, {
        "strategy": "enemy-team" if len(team_groups) >= 3 else "battle",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "validation_groups": sorted(validation_groups),
    }


class TeacherDistillationCollator(InteractionInferenceCollator):
    """Collate public observations plus soft Foul Play policy targets."""

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = super().__call__(rows)
        targets = [teacher_policy(row) for row in rows]
        action_values: list[list[float]] = []
        action_value_masks: list[list[bool]] = []
        root_values: list[float] = []
        root_masks: list[bool] = []
        outcome_values: list[float] = []
        outcome_masks: list[bool] = []
        for row in rows:
            teacher = row.get("teacher") or {}
            values = teacher.get("action_values")
            if not isinstance(values, list) or len(values) != ACTION_COUNT:
                values = [None] * ACTION_COUNT
            action_values.append(
                [float(value) if value is not None else 0.0 for value in values]
            )
            action_value_masks.append([value is not None for value in values])
            root = teacher.get("root_value")
            root_values.append(float(root) if root is not None else 0.0)
            root_masks.append(root is not None)
            outcome = str(row.get("outcome") or "").upper()
            outcome_values.append(
                1.0 if outcome == "WIN" else 0.0 if outcome == "LOSS" else 0.5
            )
            outcome_masks.append(outcome in {"WIN", "LOSS", "TIE"})
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
                "teacher_action_values": torch.tensor(action_values, dtype=torch.float32),
                "teacher_action_value_mask": torch.tensor(
                    action_value_masks, dtype=torch.bool
                ),
                "teacher_root_values": torch.tensor(root_values, dtype=torch.float32),
                "teacher_root_value_mask": torch.tensor(root_masks, dtype=torch.bool),
                "teacher_outcome_values": torch.tensor(outcome_values, dtype=torch.float32),
                "teacher_outcome_mask": torch.tensor(outcome_masks, dtype=torch.bool),
                "teacher_row_weights": torch.tensor(
                    [float(row.get("_teacher_row_weight", 1.0)) for row in rows],
                    dtype=torch.float32,
                ),
            }
        )
        return batch


def teacher_distillation_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    hard_target_weight: float = 0.0,
    confident_disagreement_weight: float = 0.0,
    confidence_power: float = 0.0,
    family_aux_weight: float = 0.25,
    action_value_weight: float = 0.25,
    action_value_loss_type: str = "bce",
    root_value_weight: float = 0.1,
    outcome_value_weight: float = 0.05,
    tera_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill the full search distribution and optional MCTS Q/V targets."""
    if not 0 <= hard_target_weight <= 1:
        raise ValueError("hard_target_weight must be between zero and one")
    if confident_disagreement_weight < 0 or confidence_power < 0:
        raise ValueError("Distillation weights cannot be negative")
    if action_value_loss_type not in {"bce", "ranking"}:
        raise ValueError("action_value_loss_type must be 'bce' or 'ranking'")
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
    policy_per_example = (
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
    weights = weights * batch.get(
        "teacher_row_weights", torch.ones_like(weights)
    ).to(log_probs.device)
    tera_mass = teacher[:, 9:13].sum(dim=1)
    weights = weights * (1.0 + (tera_weight - 1.0) * tera_mass)
    # Preserve the learning-rate scale while changing examples' relative weight.
    weights = weights / weights.mean().clamp_min(1e-6)
    policy_loss = (policy_per_example * weights).mean()

    family_loss = policy_loss.new_zeros(())
    if family_aux_weight > 0 and "family_logits" in outputs:
        teacher_family = torch.stack(
            (teacher[:, :4].sum(1), teacher[:, 4:9].sum(1), teacher[:, 9:13].sum(1)),
            dim=1,
        )
        family_log_probs = torch.log_softmax(outputs["family_logits"].float(), dim=1)
        family_per_example = -(
            teacher_family
            * torch.where(
                teacher_family > 0,
                family_log_probs,
                torch.zeros_like(family_log_probs),
            )
        ).sum(dim=1)
        family_loss = (family_per_example * weights).mean()

    action_value_loss = policy_loss.new_zeros(())
    action_value_examples = policy_loss.new_zeros(())
    q_mask = batch.get("teacher_action_value_mask")
    if action_value_weight > 0 and q_mask is not None and "action_value_logits" in outputs:
        q_mask = q_mask.to(log_probs.device).bool() & legal
        if bool(q_mask.any()):
            q_targets = batch["teacher_action_values"].to(log_probs.device).float()
            q_logits = outputs["action_value_logits"].float()
            if action_value_loss_type == "bce":
                action_value_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    q_logits[q_mask],
                    q_targets[q_mask],
                    reduction="mean",
                )
                action_value_examples = q_mask.sum().float()
            else:
                target_difference = q_targets[:, :, None] - q_targets[:, None, :]
                logit_difference = q_logits[:, :, None] - q_logits[:, None, :]
                pair_mask = q_mask[:, :, None] & q_mask[:, None, :]
                pair_mask = pair_mask & (
                    torch.triu(
                        torch.ones_like(pair_mask[0], dtype=torch.bool), diagonal=1
                    )[None, :, :]
                )
                pair_mask = pair_mask & (target_difference.abs() >= 0.02)
                if bool(pair_mask.any()):
                    signs = target_difference[pair_mask].sign()
                    action_value_loss = torch.nn.functional.softplus(
                        -signs * logit_difference[pair_mask]
                    ).mean()
                    action_value_examples = pair_mask.sum().float()

    root_value_loss = policy_loss.new_zeros(())
    root_mask = batch.get("teacher_root_value_mask")
    if root_value_weight > 0 and root_mask is not None and "value_logits" in outputs:
        root_mask = root_mask.to(log_probs.device).bool()
        if bool(root_mask.any()):
            root_targets = batch["teacher_root_values"].to(log_probs.device).float()
            root_value_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs["value_logits"].float()[root_mask], root_targets[root_mask]
            )
    outcome_value_loss = policy_loss.new_zeros(())
    outcome_mask = batch.get("teacher_outcome_mask")
    if outcome_value_weight > 0 and outcome_mask is not None and "value_logits" in outputs:
        outcome_mask = outcome_mask.to(log_probs.device).bool()
        if bool(outcome_mask.any()):
            outcome_targets = batch["teacher_outcome_values"].to(log_probs.device).float()
            outcome_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs["value_logits"].float()[outcome_mask],
                outcome_targets[outcome_mask],
                reduction="none",
            )
            outcome_weights = batch.get(
                "teacher_row_weights", torch.ones_like(outcome_targets)
            ).to(log_probs.device)[outcome_mask]
            outcome_value_loss = (outcome_losses * outcome_weights).mean()
    loss = (
        policy_loss
        + family_aux_weight * family_loss
        + action_value_weight * action_value_loss
        + root_value_weight * root_value_loss
        + outcome_value_weight * outcome_value_loss
    )
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
        "policy_distillation_loss": policy_loss.detach(),
        "family_distillation_loss": family_loss.detach(),
        "action_value_loss": action_value_loss.detach(),
        "root_value_loss": root_value_loss.detach(),
        "outcome_value_loss": outcome_value_loss.detach(),
        "action_value_targets": action_value_examples.detach(),
        "teacher_tera_mass": tera_mass.mean().detach(),
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
    family_aux_weight: float,
    action_value_weight: float,
    root_value_weight: float,
    outcome_value_weight: float,
    tera_weight: float,
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
                    family_aux_weight=family_aux_weight,
                    action_value_weight=action_value_weight,
                    root_value_weight=root_value_weight,
                    outcome_value_weight=outcome_value_weight,
                    tera_weight=tera_weight,
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
    hard_target_weight: float = 0.0,
    confident_disagreement_weight: float = 0.0,
    confidence_power: float = 0.0,
    family_aux_weight: float = 0.25,
    action_value_weight: float = 0.25,
    root_value_weight: float = 0.1,
    outcome_value_weight: float = 0.05,
    tera_weight: float = 2.0,
    validation_fraction: float = 0.15,
    trajectory_cap: int = 48,
    freeze_qwen: bool = True,
    rehearsal_file: Path | None = None,
    rehearsal_weight: float = 0.1,
    early_stopping_patience: int = 2,
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
    source_dataset = JsonlOffsetDataset(teacher_file)
    if validation_file is None:
        train_indices, validation_indices, split_report = grouped_teacher_split(
            source_dataset,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        train_dataset = TeacherTurnDataset(
            source_dataset, train_indices, trajectory_cap=trajectory_cap
        )
        validation_dataset: Dataset[dict[str, Any]] = TeacherTurnDataset(
            source_dataset, validation_indices, trajectory_cap=trajectory_cap
        )
    else:
        train_indices = [
            index
            for index in range(len(source_dataset))
            if source_dataset[index].get("decision_phase") != "team_preview"
        ]
        validation_source = JsonlOffsetDataset(validation_file)
        validation_indices = [
            index
            for index in range(len(validation_source))
            if validation_source[index].get("decision_phase") != "team_preview"
        ]
        train_dataset = TeacherTurnDataset(
            source_dataset, train_indices, trajectory_cap=trajectory_cap
        )
        validation_dataset = TeacherTurnDataset(
            validation_source, validation_indices, trajectory_cap=trajectory_cap
        )
        split_report = {
            "strategy": "explicit-file",
            "train_rows": len(train_dataset),
            "validation_rows": len(validation_dataset),
        }
    resolved_max_length = max_length or int(source_metadata.get("max_length", 4096))
    collator = TeacherDistillationCollator(
        tokenizer,
        max_length=resolved_max_length,
        truncation="error",
        prompt_format=str(source_metadata.get("prompt_format", "mechanics-v2")),
    )
    evaluation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    logits_parameter = indexed_logits_parameter(model)
    if freeze_qwen:
        model.requires_grad_(False)
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
        family_aux_weight=family_aux_weight,
        action_value_weight=action_value_weight,
        root_value_weight=root_value_weight,
        outcome_value_weight=outcome_value_weight,
        tera_weight=tera_weight,
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
    optimizer.zero_grad(set_to_none=True)
    updates = 0
    examples = 0
    totals: dict[str, float] = {}
    started = time.monotonic()
    rehearsal_loader = None
    rehearsal_iterator: Any = None
    rehearsal_collator = None
    if rehearsal_file is not None and rehearsal_weight > 0:
        rehearsal_dataset = JsonlOffsetDataset(rehearsal_file)
        rehearsal_collator = InteractionCollator(
            tokenizer,
            max_length=resolved_max_length,
            truncation="error",
            prompt_format=str(source_metadata.get("prompt_format", "mechanics-v2")),
        )
        rehearsal_loader = DataLoader(
            rehearsal_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=rehearsal_collator,
        )
        rehearsal_iterator = iter(rehearsal_loader)
    best_validation = float("inf")
    best_head_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    stop = False
    for epoch in range(epochs):
        if freeze_qwen:
            model.eval()
        else:
            model.train()
        head.train()
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
                    family_aux_weight=family_aux_weight,
                    action_value_weight=action_value_weight,
                    root_value_weight=root_value_weight,
                    outcome_value_weight=outcome_value_weight,
                    tera_weight=tera_weight,
                )
                if rehearsal_loader is not None:
                    try:
                        rehearsal_batch = next(rehearsal_iterator)
                    except StopIteration:
                        rehearsal_iterator = iter(rehearsal_loader)
                        rehearsal_batch = next(rehearsal_iterator)
                    rehearsal_batch = {
                        key: value.to(device) for key, value in rehearsal_batch.items()
                    }
                    rehearsal_outputs = interaction_outputs(
                        model,
                        head,
                        rehearsal_batch,
                        logits_parameter=logits_parameter,
                    )
                    from pokemon_battler.interaction_modeling import interaction_policy_loss

                    rehearsal_loss, _ = interaction_policy_loss(
                        rehearsal_outputs,
                        rehearsal_batch,
                        value_loss_weight=0.0,
                    )
                    loss = loss + rehearsal_weight * rehearsal_loss
                    parts["rehearsal_loss"] = rehearsal_loss.detach()
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

        epoch_validation = _evaluate_teacher(
            model,
            head,
            evaluation_loader,
            device=device,
            dtype=dtype,
            logits_parameter=logits_parameter,
            hard_target_weight=hard_target_weight,
            confident_disagreement_weight=confident_disagreement_weight,
            confidence_power=confidence_power,
            family_aux_weight=family_aux_weight,
            action_value_weight=action_value_weight,
            root_value_weight=root_value_weight,
            outcome_value_weight=outcome_value_weight,
            tera_weight=tera_weight,
        )
        print(
            json.dumps({"phase": "teacher-validation", "epoch": epoch + 1, **epoch_validation}),
            flush=True,
        )
        validation_loss = float(epoch_validation["distillation_loss"])
        if validation_loss < best_validation - 1e-5:
            best_validation = validation_loss
            stale_epochs = 0
            best_head_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                print(
                    json.dumps(
                        {
                            "phase": "teacher-early-stop",
                            "epoch": epoch + 1,
                            "best_validation_loss": best_validation,
                        }
                    ),
                    flush=True,
                )
                break

    if best_head_state is not None:
        head.load_state_dict(best_head_state)
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
        family_aux_weight=family_aux_weight,
        action_value_weight=action_value_weight,
        root_value_weight=root_value_weight,
        outcome_value_weight=outcome_value_weight,
        tera_weight=tera_weight,
    )
    print(json.dumps({"phase": "teacher-after", **after}), flush=True)
    report: dict[str, Any] = {
        "schema": "qwen-foul-play-distillation-v2",
        "source_checkpoint": str(checkpoint),
        "teacher_file": str(teacher_file),
        "validation_file": str(validation_file) if validation_file is not None else None,
        "teacher_rows": len(train_dataset),
        "split": split_report,
        "updates": updates,
        "examples": examples,
        "elapsed_seconds": time.monotonic() - started,
        "epochs": epochs,
        "hard_target_weight": hard_target_weight,
        "confident_disagreement_weight": confident_disagreement_weight,
        "confidence_power": confidence_power,
        "family_aux_weight": family_aux_weight,
        "action_value_weight": action_value_weight,
        "root_value_weight": root_value_weight,
        "outcome_value_weight": outcome_value_weight,
        "tera_weight": tera_weight,
        "trajectory_cap": trajectory_cap,
        "freeze_qwen": freeze_qwen,
        "rehearsal_file": str(rehearsal_file) if rehearsal_file else None,
        "rehearsal_weight": rehearsal_weight,
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
        "deployment_action_value_weight": (
            0.35 if float(after.get("action_value_targets", 0.0)) > 0 else 0.0
        ),
    }
    _save_checkpoint(model, tokenizer, output_dir, config, head, "interaction-head")
    (output_dir / "distillation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
