from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from pokemon_battler.models.modeling import _last_hidden_states, indexed_logits_parameter
from pokemon_battler.training.rl_training import _autocast, _load_trainable_policy
from pokemon_battler.models.team_preview import (
    PREVIEW_SLOTS,
    TeamPreviewCollator,
    TeamPreviewHead,
    save_team_preview_head,
)
from pokemon_battler.training.train import _save_checkpoint, set_seed
from pokemon_battler.data.training_data import JsonlOffsetDataset


class PreviewDataset(Dataset[dict[str, Any]]):
    def __init__(self, source: JsonlOffsetDataset, indices: list[int]) -> None:
        if not indices:
            raise ValueError("Teacher trace contains no team-preview examples")
        self.source = source
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.source[self.indices[index]]


def _forward(
    model: Any,
    head: TeamPreviewHead,
    batch: dict[str, torch.Tensor],
    logits_parameter: str | None,
) -> torch.Tensor:
    hidden = _last_hidden_states(model, batch, logits_parameter=logits_parameter)
    positions = batch["attention_mask"].sum(1) - 1
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return head(hidden[rows, positions], batch)


def _loss(logits: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float(), dim=1)
    teacher = teacher.to(logits.device).float()
    teacher = teacher / teacher.sum(1, keepdim=True).clamp_min(1e-8)
    return -(teacher * torch.where(teacher > 0, log_probs, torch.zeros_like(log_probs))).sum(1).mean()


def _evaluate(
    model: Any,
    head: TeamPreviewHead,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    logits_parameter: str | None,
) -> dict[str, float]:
    head.eval()
    total_loss = 0.0
    agreements = 0
    examples = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast(device, dtype):
                logits = _forward(model, head, batch, logits_parameter)
                loss = _loss(logits, batch["preview_teacher_policy"])
            size = logits.shape[0]
            total_loss += float(loss.item()) * size
            agreements += int(
                (logits.argmax(1) == batch["preview_teacher_policy"].argmax(1)).sum().item()
            )
            examples += size
    return {
        "examples": float(examples),
        "loss": total_loss / max(examples, 1),
        "teacher_top1_agreement": agreements / max(examples, 1),
    }


def train_team_preview(
    *,
    checkpoint: Path,
    teacher_file: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    validation_fraction: float = 0.2,
    seed: int = 42,
    dtype_name: str = "auto",
    load_in_4bit: bool = True,
    local_files_only: bool = True,
    attn_implementation: str = "sdpa",
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Preview output directory is not empty: {output_dir}")
    set_seed(seed)
    model, tokenizer, interaction_head, device, dtype, metadata = _load_trainable_policy(
        checkpoint,
        model_name=None,
        dtype_name=dtype_name,
        load_in_4bit=load_in_4bit,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    model.requires_grad_(False)
    model.eval()
    source = JsonlOffsetDataset(teacher_file)
    by_battle: dict[str, list[int]] = {}
    for index in range(len(source)):
        row = source[index]
        if row.get("decision_phase") == "team_preview":
            by_battle.setdefault(str(row.get("battle_id") or index), []).append(index)
    groups = sorted(by_battle)
    random.Random(seed).shuffle(groups)
    validation_count = max(1, round(len(groups) * validation_fraction))
    validation_groups = set(groups[:validation_count])
    validation_indices = [index for group in validation_groups for index in by_battle[group]]
    train_indices = [index for group in groups if group not in validation_groups for index in by_battle[group]]
    if not train_indices:
        raise ValueError("Need at least two collected preview battles for train/validation")
    collator = TeamPreviewCollator(tokenizer, max_length=int(metadata.get("max_length", 4096)))
    train_loader = DataLoader(
        PreviewDataset(source, train_indices), batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    validation_loader = DataLoader(
        PreviewDataset(source, validation_indices), batch_size=batch_size, collate_fn=collator
    )
    head = TeamPreviewHead(int(model.config.hidden_size)).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.01)
    logits_parameter = indexed_logits_parameter(model)
    before = _evaluate(
        model, head, validation_loader, device=device, dtype=dtype, logits_parameter=logits_parameter
    )
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        head.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, dtype):
                loss = _loss(
                    _forward(model, head, batch, logits_parameter),
                    batch["preview_teacher_policy"],
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        validation = _evaluate(
            model,
            head,
            validation_loader,
            device=device,
            dtype=dtype,
            logits_parameter=logits_parameter,
        )
        print(json.dumps({"phase": "preview-validation", "epoch": epoch + 1, **validation}), flush=True)
        if validation["loss"] < best:
            best = validation["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    after = _evaluate(
        model, head, validation_loader, device=device, dtype=dtype, logits_parameter=logits_parameter
    )
    report = {
        "schema": "team-preview-training-v1",
        "source_checkpoint": str(checkpoint),
        "teacher_file": str(teacher_file),
        "train_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        "before": before,
        "after": after,
    }
    config = metadata | report | {
        "team_preview_d_model": 256,
        "team_preview_slots": PREVIEW_SLOTS,
        "training_objective": "foul-play-team-preview-distillation",
    }
    _save_checkpoint(model, tokenizer, output_dir, config, interaction_head, "interaction-head")
    save_team_preview_head(head, output_dir)
    (output_dir / "preview_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
