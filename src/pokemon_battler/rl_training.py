from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from pokemon_battler.interaction_cache import InteractionCacheDataset
from pokemon_battler.modeling import (
    indexed_logits_parameter,
    interaction_outputs,
    load_interaction_head,
    load_policy_model,
    load_training_metadata,
    resolve_dtype,
)
from pokemon_battler.reinforcement import (
    PPORolloutCollator,
    offline_outcome_loss,
    ppo_loss,
)
from pokemon_battler.train import _save_checkpoint, set_seed
from pokemon_battler.training_data import (
    InteractionCollator,
    JsonlOffsetDataset,
)
from pokemon_battler.trajectory_modeling import has_trajectory_head


def _autocast(device: torch.device, dtype: torch.dtype) -> Any:
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _load_trainable_policy(
    checkpoint: Path,
    *,
    model_name: str | None,
    dtype_name: str,
    load_in_4bit: bool,
    local_files_only: bool,
    attn_implementation: str,
) -> tuple[Any, Any, torch.nn.Module, torch.device, torch.dtype, dict[str, Any]]:
    metadata = load_training_metadata(checkpoint)
    resolved_model = model_name or str(metadata.get("model", "Qwen/Qwen2.5-0.5B"))
    adapter_path = str(checkpoint) if (checkpoint / "adapter_config.json").is_file() else None
    model, tokenizer, device = load_policy_model(
        resolved_model,
        adapter_path=adapter_path,
        dtype=dtype_name,
        load_in_4bit=load_in_4bit,
        for_training=True,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    head = load_interaction_head(model, checkpoint, device)
    return model, tokenizer, head, device, resolve_dtype(dtype_name, device), metadata


def _optimizer(
    model: Any,
    head: torch.nn.Module,
    *,
    qwen_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, list[torch.nn.Parameter]]:
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    head_parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    if not model_parameters and not head_parameters:
        raise RuntimeError("The loaded Qwen policy has no trainable parameters")
    groups: list[dict[str, Any]] = []
    if model_parameters:
        groups.append({"params": model_parameters, "lr": qwen_learning_rate})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": head_learning_rate})
    return (
        torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=weight_decay),
        [*model_parameters, *head_parameters],
    )


def _mean_metrics(totals: dict[str, float], examples: int) -> dict[str, float]:
    return {key: value / max(examples, 1) for key, value in totals.items()}


def train_offline_outcomes(
    *,
    checkpoint: Path,
    train_file: Path,
    interaction_cache: Path,
    output_dir: Path,
    model_name: str | None = None,
    epochs: int = 1,
    max_steps: int | None = None,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    qwen_learning_rate: float = 1e-5,
    head_learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    expectile: float = 0.7,
    advantage_temperature: float = 0.1,
    max_advantage_weight: float = 20.0,
    behavior_clone_weight: float = 0.1,
    max_length: int | None = None,
    dtype_name: str = "auto",
    load_in_4bit: bool = True,
    local_files_only: bool = True,
    attn_implementation: str = "sdpa",
    seed: int = 42,
    log_steps: int = 20,
) -> dict[str, Any]:
    """Fit outcome-aware Q, V, and policy heads from human replay results."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Offline output directory is not empty: {output_dir}")
    if epochs <= 0 or batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("Offline epoch and batch settings must be positive")
    set_seed(seed)
    model, tokenizer, head, device, dtype, source_metadata = _load_trainable_policy(
        checkpoint,
        model_name=model_name,
        dtype_name=dtype_name,
        load_in_4bit=load_in_4bit,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    dataset = InteractionCacheDataset(
        JsonlOffsetDataset(train_file), interaction_cache
    )
    resolved_max_length = max_length or int(source_metadata.get("max_length", 4096))
    collator = InteractionCollator(
        tokenizer,
        max_length=resolved_max_length,
        truncation="error",
        prompt_format=str(source_metadata.get("prompt_format", "mechanics-v2")),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
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
    logits_parameter = indexed_logits_parameter(model)
    model.train()
    head.train()
    optimizer.zero_grad(set_to_none=True)
    updates = 0
    examples = 0
    microsteps = 0
    started = time.monotonic()
    totals: dict[str, float] = {}
    stop = False
    for _epoch in range(epochs):
        for batch_index, batch in enumerate(loader):
            batch_size_now = int(batch["input_ids"].shape[0])
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast(device, dtype):
                outputs = interaction_outputs(
                    model, head, batch, logits_parameter=logits_parameter
                )
                loss, parts = offline_outcome_loss(
                    outputs,
                    batch,
                    expectile=expectile,
                    advantage_temperature=advantage_temperature,
                    max_advantage_weight=max_advantage_weight,
                    behavior_clone_weight=behavior_clone_weight,
                )
                scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            examples += batch_size_now
            microsteps += 1
            for key, value in {"offline_total_loss": loss.detach(), **parts}.items():
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
                            "phase": "offline-outcome",
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
    report = {
        "schema": "qwen-offline-outcome-v1",
        "source_checkpoint": str(checkpoint),
        "train_file": str(train_file),
        "updates": updates,
        "examples": examples,
        "elapsed_seconds": time.monotonic() - started,
        "expectile": expectile,
        "advantage_temperature": advantage_temperature,
        "max_advantage_weight": max_advantage_weight,
        "behavior_clone_weight": behavior_clone_weight,
        "qwen_learning_rate": qwen_learning_rate,
        "head_learning_rate": head_learning_rate,
        **_mean_metrics(totals, examples),
    }
    config = source_metadata | report | {
        "model": model_name or source_metadata.get("model", "Qwen/Qwen2.5-0.5B"),
        "max_length": resolved_max_length,
        "load_in_4bit": load_in_4bit,
        "local_files_only": local_files_only,
        "dtype": dtype_name,
        "attn_implementation": attn_implementation,
        "training_objective": "offline-outcome-iql",
    }
    _save_checkpoint(model, tokenizer, output_dir, config, head, "interaction-head")
    (output_dir / "offline_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def train_ppo_rollouts(
    *,
    checkpoint: Path,
    rollout_file: Path,
    output_dir: Path,
    model_name: str | None = None,
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    qwen_learning_rate: float = 1e-5,
    head_learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    clip_ratio: float = 0.2,
    value_clip: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    target_kl: float = 0.02,
    max_length: int | None = None,
    dtype_name: str = "auto",
    load_in_4bit: bool = True,
    local_files_only: bool = True,
    attn_implementation: str = "sdpa",
    seed: int = 42,
    rollout_source: str = "local-self-play",
) -> dict[str, Any]:
    """Run clipped PPO updates from completed on-policy Showdown trajectories."""
    if has_trajectory_head(checkpoint):
        raise ValueError(
            "The statewise PPO updater cannot train a trajectory checkpoint. "
            "Run it frozen; temporal PPO must preserve ordered battle sequences."
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"PPO output directory is not empty: {output_dir}")
    if epochs <= 0 or batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("PPO epoch and batch settings must be positive")
    set_seed(seed)
    model, tokenizer, head, device, dtype, source_metadata = _load_trainable_policy(
        checkpoint,
        model_name=model_name,
        dtype_name=dtype_name,
        load_in_4bit=load_in_4bit,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    dataset = JsonlOffsetDataset(rollout_file)
    if not len(dataset):
        raise ValueError("PPO rollout file contains no decisions")
    resolved_max_length = max_length or int(source_metadata.get("max_length", 4096))
    collator = PPORolloutCollator(
        tokenizer,
        max_length=resolved_max_length,
        truncation="error",
        prompt_format=str(source_metadata.get("prompt_format", "mechanics-v2")),
    )
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
    logits_parameter = indexed_logits_parameter(model)
    model.train()
    head.train()
    optimizer.zero_grad(set_to_none=True)
    updates = 0
    examples = 0
    totals: dict[str, float] = {}
    stopped_for_kl = False
    started = time.monotonic()
    completed_epochs = 0
    for _epoch in range(epochs):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
        microsteps = 0
        epoch_kl_total = 0.0
        epoch_examples = 0
        for batch_index, batch in enumerate(loader):
            batch_size_now = int(batch["input_ids"].shape[0])
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast(device, dtype):
                outputs = interaction_outputs(
                    model, head, batch, logits_parameter=logits_parameter
                )
                loss, parts = ppo_loss(
                    outputs,
                    batch,
                    clip_ratio=clip_ratio,
                    value_clip=value_clip,
                    value_coefficient=value_coefficient,
                    entropy_coefficient=entropy_coefficient,
                )
                scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            examples += batch_size_now
            epoch_examples += batch_size_now
            microsteps += 1
            epoch_kl_total += float(parts["approximate_kl"].item()) * batch_size_now
            for key, value in {"ppo_total_loss": loss.detach(), **parts}.items():
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
        completed_epochs += 1
        mean_epoch_kl = epoch_kl_total / max(epoch_examples, 1)
        print(
            json.dumps(
                {
                    "phase": "ppo",
                    "epoch": completed_epochs,
                    "updates": updates,
                    "examples": examples,
                    "approximate_kl": mean_epoch_kl,
                }
            ),
            flush=True,
        )
        if target_kl > 0 and mean_epoch_kl > 1.5 * target_kl:
            stopped_for_kl = True
            break
    report = {
        "schema": "qwen-ppo-update-v1",
        "source_checkpoint": str(checkpoint),
        "rollout_file": str(rollout_file),
        "rollout_source": rollout_source,
        "rollout_decisions": len(dataset),
        "epochs": completed_epochs,
        "updates": updates,
        "examples": examples,
        "stopped_for_target_kl": stopped_for_kl,
        "elapsed_seconds": time.monotonic() - started,
        "clip_ratio": clip_ratio,
        "value_clip": value_clip,
        "value_coefficient": value_coefficient,
        "entropy_coefficient": entropy_coefficient,
        "target_kl": target_kl,
        **_mean_metrics(totals, examples),
    }
    config = source_metadata | report | {
        "model": model_name or source_metadata.get("model", "Qwen/Qwen2.5-0.5B"),
        "max_length": resolved_max_length,
        "load_in_4bit": load_in_4bit,
        "local_files_only": local_files_only,
        "dtype": dtype_name,
        "attn_implementation": attn_implementation,
        "training_objective": f"{rollout_source}-ppo",
    }
    _save_checkpoint(model, tokenizer, output_dir, config, head, "interaction-head")
    (output_dir / "ppo_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
