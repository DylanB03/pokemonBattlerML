from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from pokemon_battler.training.distillation import teacher_distillation_loss
from pokemon_battler.data.frozen_cache import FrozenCacheDataset, checkpoint_signature
from pokemon_battler.models.interaction_modeling import interaction_policy_loss
from pokemon_battler.training.rl_training import _autocast, _load_trainable_policy
from pokemon_battler.training.train import _save_checkpoint, set_seed


def _limited(dataset: Dataset[Any], limit: int | None) -> Dataset[Any]:
    if limit is None or limit >= len(dataset):
        return dataset
    if limit <= 0:
        raise ValueError("Cache row limits must be positive")
    return Subset(dataset, range(limit))


def _outputs(head: Any, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_hidden = batch.pop("qwen_state_hidden")
    return head(state_hidden, batch)


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _evaluate_teacher(
    head: Any,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    family_aux_weight: float,
    action_value_weight: float,
    action_value_loss_type: str,
    root_value_weight: float,
    outcome_value_weight: float,
    tera_weight: float,
) -> dict[str, float]:
    head.eval()
    totals: dict[str, float] = {}
    examples = 0
    with torch.inference_mode():
        for batch in loader:
            batch = _move(batch, device)
            size = int(batch["qwen_state_hidden"].shape[0])
            with _autocast(device, dtype):
                outputs = _outputs(head, batch)
                loss, parts = teacher_distillation_loss(
                    outputs,
                    batch,
                    hard_target_weight=0.0,
                    confident_disagreement_weight=0.0,
                    confidence_power=0.0,
                    family_aux_weight=family_aux_weight,
                    action_value_weight=action_value_weight,
                    action_value_loss_type=action_value_loss_type,
                    root_value_weight=root_value_weight,
                    outcome_value_weight=outcome_value_weight,
                    tera_weight=tera_weight,
                )
            for key, value in {"distillation_loss": loss.detach(), **parts}.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * size
            examples += size
    return {
        "examples": float(examples),
        **{key: value / max(examples, 1) for key, value in totals.items()},
    }


def _evaluate_replay(
    head: Any,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    family_aux_weight: float,
) -> dict[str, float]:
    head.eval()
    losses = 0.0
    correct = 0
    family_correct = 0
    examples = 0
    with torch.inference_mode():
        for batch in loader:
            batch = _move(batch, device)
            size = int(batch["qwen_state_hidden"].shape[0])
            with _autocast(device, dtype):
                outputs = _outputs(head, batch)
                loss, _ = interaction_policy_loss(
                    outputs,
                    batch,
                    family_aux_weight=family_aux_weight,
                    value_loss_weight=0.0,
                )
            correct += int(
                (outputs["action_log_probs"].argmax(1) == batch["action_ids"]).sum().item()
            )
            family_correct += int(
                (outputs["family_logits"].argmax(1) == batch["action_family_ids"])
                .sum()
                .item()
            )
            losses += float(loss.item()) * size
            examples += size
    return {
        "examples": float(examples),
        "loss": losses / max(examples, 1),
        "action_accuracy": correct / max(examples, 1),
        "family_accuracy": family_correct / max(examples, 1),
    }


def train_cached_distillation(
    *,
    checkpoint: Path,
    teacher_train_cache: Path,
    teacher_validation_cache: Path,
    replay_train_cache: Path | None,
    replay_validation_cache: Path | None,
    output_dir: Path,
    variant: str,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    family_aux_weight: float = 0.25,
    action_value_weight: float = 0.0,
    action_value_loss_type: str = "ranking",
    root_value_weight: float = 0.0,
    outcome_value_weight: float = 0.0,
    tera_weight: float = 1.0,
    rehearsal_weight: float = 0.15,
    early_stopping_patience: int = 3,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    minimum_teacher_agreement_gain: float = 0.03,
    maximum_replay_accuracy_drop: float = 0.02,
    dtype_name: str = "auto",
    load_in_4bit: bool = True,
    local_files_only: bool = True,
    attn_implementation: str = "sdpa",
    seed: int = 42,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("Cached training epochs, batch size, and learning rate must be positive")
    set_seed(seed)
    teacher_train_source = FrozenCacheDataset(teacher_train_cache, expected_kind="teacher")
    teacher_validation_source = FrozenCacheDataset(
        teacher_validation_cache, expected_kind="teacher"
    )
    replay_train_source = (
        FrozenCacheDataset(replay_train_cache, expected_kind="replay")
        if replay_train_cache is not None and rehearsal_weight > 0
        else None
    )
    replay_validation_source = (
        FrozenCacheDataset(replay_validation_cache, expected_kind="replay")
        if replay_validation_cache is not None
        else None
    )
    expected_signature = checkpoint_signature(checkpoint)
    cache_sources = [teacher_train_source, teacher_validation_source]
    cache_sources.extend(
        source
        for source in (replay_train_source, replay_validation_source)
        if source is not None
    )
    if any(
        source.metadata.get("checkpoint_signature") != expected_signature
        for source in cache_sources
    ):
        raise ValueError("Frozen Qwen cache was built from a different checkpoint")
    teacher_train = _limited(teacher_train_source, train_limit)
    teacher_validation = _limited(
        teacher_validation_source,
        validation_limit,
    )
    replay_train = (
        replay_train_source if replay_train_source is not None else None
    )
    replay_validation = (
        _limited(replay_validation_source, validation_limit)
        if replay_validation_source is not None
        else None
    )
    model, tokenizer, head, device, dtype, source_metadata = _load_trainable_policy(
        checkpoint,
        model_name=None,
        dtype_name=dtype_name,
        load_in_4bit=load_in_4bit,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    model.requires_grad_(False)
    model.eval()
    head.requires_grad_(True)
    teacher_validation_loader = DataLoader(
        teacher_validation, batch_size=batch_size, shuffle=False
    )
    replay_validation_loader = (
        DataLoader(replay_validation, batch_size=batch_size, shuffle=False)
        if replay_validation is not None
        else None
    )
    teacher_before = _evaluate_teacher(
        head,
        teacher_validation_loader,
        device=device,
        dtype=dtype,
        family_aux_weight=family_aux_weight,
        action_value_weight=action_value_weight,
        action_value_loss_type=action_value_loss_type,
        root_value_weight=root_value_weight,
        outcome_value_weight=outcome_value_weight,
        tera_weight=tera_weight,
    )
    replay_before = (
        _evaluate_replay(
            head,
            replay_validation_loader,
            device=device,
            dtype=dtype,
            family_aux_weight=family_aux_weight,
        )
        if replay_validation_loader is not None
        else None
    )
    print(
        json.dumps(
            {
                "phase": "cached-before",
                "variant": variant,
                "teacher": teacher_before,
                "replay": replay_before,
            }
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and dtype == torch.float16
    )
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    updates = 0
    started = time.monotonic()
    history = []
    for epoch in range(epochs):
        head.train()
        teacher_loader = DataLoader(
            teacher_train, batch_size=batch_size, shuffle=True
        )
        replay_loader = (
            DataLoader(replay_train, batch_size=batch_size, shuffle=True)
            if replay_train is not None
            else None
        )
        replay_iterator = iter(replay_loader) if replay_loader is not None else None
        epoch_loss = 0.0
        epoch_examples = 0
        for teacher_batch in teacher_loader:
            teacher_batch = _move(teacher_batch, device)
            size = int(teacher_batch["qwen_state_hidden"].shape[0])
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, dtype):
                outputs = _outputs(head, teacher_batch)
                loss, _ = teacher_distillation_loss(
                    outputs,
                    teacher_batch,
                    hard_target_weight=0.0,
                    confident_disagreement_weight=0.0,
                    confidence_power=0.0,
                    family_aux_weight=family_aux_weight,
                    action_value_weight=action_value_weight,
                    action_value_loss_type=action_value_loss_type,
                    root_value_weight=root_value_weight,
                    outcome_value_weight=outcome_value_weight,
                    tera_weight=tera_weight,
                )
                if replay_iterator is not None:
                    try:
                        replay_batch = next(replay_iterator)
                    except StopIteration:
                        assert replay_loader is not None
                        replay_iterator = iter(replay_loader)
                        replay_batch = next(replay_iterator)
                    replay_batch = _move(replay_batch, device)
                    replay_outputs = _outputs(head, replay_batch)
                    replay_loss, _ = interaction_policy_loss(
                        replay_outputs,
                        replay_batch,
                        family_aux_weight=family_aux_weight,
                        value_loss_weight=0.0,
                    )
                    loss = loss + rehearsal_weight * replay_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            updates += 1
            epoch_loss += float(loss.detach().item()) * size
            epoch_examples += size
        teacher_validation_metrics = _evaluate_teacher(
            head,
            teacher_validation_loader,
            device=device,
            dtype=dtype,
            family_aux_weight=family_aux_weight,
            action_value_weight=action_value_weight,
            action_value_loss_type=action_value_loss_type,
            root_value_weight=root_value_weight,
            outcome_value_weight=outcome_value_weight,
            tera_weight=tera_weight,
        )
        replay_validation_metrics = (
            _evaluate_replay(
                head,
                replay_validation_loader,
                device=device,
                dtype=dtype,
                family_aux_weight=family_aux_weight,
            )
            if replay_validation_loader is not None
            else None
        )
        replay_penalty = 0.0
        if replay_before is not None and replay_validation_metrics is not None:
            replay_penalty = max(
                0.0, replay_validation_metrics["loss"] - replay_before["loss"]
            )
        score = float(teacher_validation_metrics["teacher_student_kl"]) + 0.5 * replay_penalty
        row = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(epoch_examples, 1),
            "teacher": teacher_validation_metrics,
            "replay": replay_validation_metrics,
            "selection_score": score,
        }
        history.append(row)
        print(
            json.dumps({"phase": "cached-validation", "variant": variant, **row}),
            flush=True,
        )
        if score < best_score - 1e-5:
            best_score = score
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    teacher_after = _evaluate_teacher(
        head,
        teacher_validation_loader,
        device=device,
        dtype=dtype,
        family_aux_weight=family_aux_weight,
        action_value_weight=action_value_weight,
        action_value_loss_type=action_value_loss_type,
        root_value_weight=root_value_weight,
        outcome_value_weight=outcome_value_weight,
        tera_weight=tera_weight,
    )
    replay_after = (
        _evaluate_replay(
            head,
            replay_validation_loader,
            device=device,
            dtype=dtype,
            family_aux_weight=family_aux_weight,
        )
        if replay_validation_loader is not None
        else None
    )
    agreement_gain = (
        teacher_after["teacher_top1_agreement"]
        - teacher_before["teacher_top1_agreement"]
    )
    kl_improvement = (
        teacher_before["teacher_student_kl"] - teacher_after["teacher_student_kl"]
    )
    replay_accuracy_drop = (
        max(0.0, replay_before["action_accuracy"] - replay_after["action_accuracy"])
        if replay_before is not None and replay_after is not None
        else 0.0
    )
    eligible = (
        agreement_gain >= minimum_teacher_agreement_gain
        and kl_improvement > 0
        and replay_accuracy_drop <= maximum_replay_accuracy_drop
    )
    report = {
        "schema": "cached-foul-play-distillation-v1",
        "variant": variant,
        "source_checkpoint": str(checkpoint),
        "teacher_train_cache": str(teacher_train_cache),
        "teacher_validation_cache": str(teacher_validation_cache),
        "replay_train_cache": str(replay_train_cache) if replay_train_cache else None,
        "replay_validation_cache": (
            str(replay_validation_cache) if replay_validation_cache else None
        ),
        "teacher_train_rows": len(teacher_train),
        "teacher_validation_rows": len(teacher_validation),
        "replay_train_rows": len(replay_train) if replay_train is not None else 0,
        "replay_validation_rows": (
            len(replay_validation) if replay_validation is not None else 0
        ),
        "epochs_completed": len(history),
        "updates": updates,
        "elapsed_seconds": time.monotonic() - started,
        "family_aux_weight": family_aux_weight,
        "action_value_weight": action_value_weight,
        "action_value_loss_type": action_value_loss_type,
        "rehearsal_weight": rehearsal_weight,
        "teacher_before": teacher_before,
        "teacher_after": teacher_after,
        "replay_before": replay_before,
        "replay_after": replay_after,
        "teacher_agreement_gain": agreement_gain,
        "teacher_kl_improvement": kl_improvement,
        "replay_accuracy_drop": replay_accuracy_drop,
        "minimum_teacher_agreement_gain": minimum_teacher_agreement_gain,
        "maximum_replay_accuracy_drop": maximum_replay_accuracy_drop,
        "eligible_for_battle_evaluation": eligible,
        "history": history,
    }
    config = source_metadata | report | {
        "training_objective": "cached-selective-foul-play-distillation",
        "deployment_action_value_weight": 0.0,
        "team_preview_enabled": False,
        "dtype": dtype_name,
        "load_in_4bit": load_in_4bit,
        "local_files_only": local_files_only,
        "attn_implementation": attn_implementation,
    }
    _save_checkpoint(model, tokenizer, output_dir, config, head, "interaction-head")
    (output_dir / "cached_distillation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train only the interaction head from reusable frozen-Qwen caches."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-train-cache", type=Path, required=True)
    parser.add_argument("--teacher-validation-cache", type=Path, required=True)
    parser.add_argument("--replay-train-cache", type=Path)
    parser.add_argument("--replay-validation-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--family-aux-weight", type=float, default=0.25)
    parser.add_argument("--action-value-weight", type=float, default=0.0)
    parser.add_argument(
        "--action-value-loss-type", choices=("bce", "ranking"), default="ranking"
    )
    parser.add_argument("--root-value-weight", type=float, default=0.0)
    parser.add_argument("--outcome-value-weight", type=float, default=0.0)
    parser.add_argument("--tera-weight", type=float, default=1.0)
    parser.add_argument("--rehearsal-weight", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--minimum-teacher-agreement-gain", type=float, default=0.03)
    parser.add_argument("--maximum-replay-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_cached_distillation(
        checkpoint=args.checkpoint,
        teacher_train_cache=args.teacher_train_cache,
        teacher_validation_cache=args.teacher_validation_cache,
        replay_train_cache=args.replay_train_cache,
        replay_validation_cache=args.replay_validation_cache,
        output_dir=args.output_dir,
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        family_aux_weight=args.family_aux_weight,
        action_value_weight=args.action_value_weight,
        action_value_loss_type=args.action_value_loss_type,
        root_value_weight=args.root_value_weight,
        outcome_value_weight=args.outcome_value_weight,
        tera_weight=args.tera_weight,
        rehearsal_weight=args.rehearsal_weight,
        early_stopping_patience=args.early_stopping_patience,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        minimum_teacher_agreement_gain=args.minimum_teacher_agreement_gain,
        maximum_replay_accuracy_drop=args.maximum_replay_accuracy_drop,
        dtype_name=args.dtype,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
