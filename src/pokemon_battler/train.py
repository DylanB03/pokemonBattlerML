from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from pokemon_battler.evaluation_utils import (
    action_family,
    action_kind,
    load_action_counts,
    select_evaluation_dataset,
)
from pokemon_battler.modeling import (
    assistant_only_loss,
    attach_lora,
    candidate_head_loss,
    create_candidate_head,
    create_mechanics_head,
    create_policy_head,
    indexed_logits_parameter,
    load_policy_model,
    masked_candidate_logits,
    masked_mechanics_logits,
    masked_policy_logits,
    mechanics_head_loss,
    policy_head_loss,
    resolve_dtype,
    save_candidate_head,
    save_mechanics_head,
    save_policy_head,
)
from pokemon_battler.mechanics import MECHANICS_FEATURE_COUNT, MECHANICS_SCHEMA
from pokemon_battler.prompting import PROMPT_FORMATS
from pokemon_battler.training_data import (
    CandidateCollator,
    JsonlOffsetDataset,
    MechanicsCacheDataset,
    MechanicsCollator,
    PolicyCollator,
    SFTCollator,
)

DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
ACTION_OBJECTIVES = {"policy-head", "candidate-head", "mechanics-head"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_context(device: torch.device, dtype: torch.dtype) -> Any:
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def learning_rate_multiplier(
    step: int,
    *,
    warmup_updates: int,
    scheduler_updates: int,
    scheduler_name: str,
    min_lr_ratio: float,
) -> float:
    if warmup_updates and step < warmup_updates:
        return max(step, 1) / warmup_updates
    if scheduler_name == "constant-with-warmup":
        return 1.0
    progress = (step - warmup_updates) / max(scheduler_updates - warmup_updates, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def meaningful_validation_improvement(
    current: float,
    best: float,
    *,
    minimum_delta: float,
    higher_is_better: bool,
) -> bool:
    """Return whether a metric moved far enough to reset early-stop patience."""
    if not math.isfinite(best):
        return math.isfinite(current)
    if not math.isfinite(current):
        return False
    if higher_is_better:
        return current > best + minimum_delta
    return current < best - minimum_delta


def _action_logits(
    model: Any,
    action_head: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    objective: str,
    logits_parameter: str | None,
) -> torch.Tensor:
    if objective == "candidate-head":
        return masked_candidate_logits(
            model,
            action_head,
            batch,
            logits_parameter=logits_parameter,
        )
    if objective == "mechanics-head":
        return masked_mechanics_logits(
            model,
            action_head,
            batch,
            logits_parameter=logits_parameter,
        )
    return masked_policy_logits(
        model,
        action_head,
        batch,
        logits_parameter=logits_parameter,
    )


def _action_loss(
    model: Any,
    action_head: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    objective: str,
    logits_parameter: str | None,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if class_weights is not None:
        logits = _action_logits(
            model,
            action_head,
            batch,
            objective,
            logits_parameter,
        )
        targets = batch["action_ids"].to(logits.device)
        return torch.nn.functional.cross_entropy(logits, targets, weight=class_weights)
    if objective == "candidate-head":
        return candidate_head_loss(
            model,
            action_head,
            batch,
            logits_parameter=logits_parameter,
        )
    if objective == "mechanics-head":
        return mechanics_head_loss(
            model,
            action_head,
            batch,
            logits_parameter=logits_parameter,
        )
    return policy_head_loss(
        model,
        action_head,
        batch,
        logits_parameter=logits_parameter,
    )


def _evaluate_model(
    model: Any,
    action_head: torch.nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int | None,
    logits_parameter: str | None,
    objective: str,
) -> dict[str, Any]:
    model.eval()
    if action_head is not None:
        action_head.eval()
    losses: list[float] = []
    total_loss = 0.0
    total_examples = 0
    correct = 0
    type_correct = 0
    top_k_correct: Counter[int] = Counter()
    reciprocal_rank_sum = 0.0
    entropy_sum = 0.0
    target_kind: Counter[str] = Counter()
    correct_kind: Counter[str] = Counter()
    target_family: Counter[str] = Counter()
    correct_family: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast_context(device, dtype):
                if objective in ACTION_OBJECTIVES:
                    assert action_head is not None
                    logits = _action_logits(
                        model,
                        action_head,
                        batch,
                        objective,
                        logits_parameter,
                    )
                    targets = batch["action_ids"].to(logits.device)
                    loss = torch.nn.functional.cross_entropy(
                        logits,
                        targets,
                        reduction="sum",
                    )
                else:
                    loss = assistant_only_loss(
                        model,
                        batch,
                        logits_parameter=logits_parameter,
                    )
            if objective == "sft":
                losses.append(float(loss.item()))
                continue

            batch_size = int(targets.numel())
            total_loss += float(loss.item())
            total_examples += batch_size
            predictions = logits.argmax(dim=1)
            ranked = logits.argsort(dim=1, descending=True)
            ranks = (ranked == targets[:, None]).nonzero(as_tuple=False)[:, 1] + 1
            correct += int((predictions == targets).sum().item())
            for k in (1, 2, 3):
                top_k_correct[k] += int((ranks <= k).sum().item())
            reciprocal_rank_sum += float((1.0 / ranks.float()).sum().item())
            log_probabilities = torch.log_softmax(logits.float(), dim=1)
            probabilities = log_probabilities.exp()
            entropy_terms = torch.where(
                torch.isfinite(log_probabilities),
                probabilities * log_probabilities,
                torch.zeros_like(probabilities),
            )
            entropy_sum += float((-entropy_terms.sum(dim=1)).sum().item())
            for target, prediction in zip(
                targets.detach().cpu().tolist(),
                predictions.detach().cpu().tolist(),
            ):
                kind = action_kind(target)
                family = action_family(target)
                target_kind[kind] += 1
                target_family[family] += 1
                target_counts[f"A{target}"] += 1
                prediction_counts[f"A{prediction}"] += 1
                type_correct += int(action_kind(prediction) == kind)
                correct_kind[kind] += int(prediction == target)
                correct_family[family] += int(prediction == target)
    model.train()
    if action_head is not None:
        action_head.train()
    if objective == "sft":
        return {"validation_loss": sum(losses) / len(losses) if losses else math.nan}
    if total_examples == 0:
        return {"validation_loss": math.nan}
    return {
        "validation_loss": total_loss / total_examples,
        "validation_accuracy": correct / total_examples,
        "validation_top_k_accuracy": {
            f"top_{k}": top_k_correct[k] / total_examples for k in (1, 2, 3)
        },
        "validation_mrr": reciprocal_rank_sum / total_examples,
        "validation_action_type_accuracy": type_correct / total_examples,
        "validation_accuracy_by_target_kind": {
            kind: correct_kind[kind] / count
            for kind, count in sorted(target_kind.items())
        },
        "validation_accuracy_by_target_family": {
            family: correct_family[family] / count
            for family, count in sorted(target_family.items())
        },
        "validation_policy_entropy": entropy_sum / total_examples,
        "validation_target_counts": dict(sorted(target_counts.items())),
        "validation_prediction_counts": dict(sorted(prediction_counts.items())),
        "validation_examples_evaluated": total_examples,
    }


def _save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    training_config: dict[str, Any],
    action_head: torch.nn.Module | None = None,
    objective: str = "sft",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    if action_head is not None:
        if objective == "candidate-head":
            save_candidate_head(action_head, output_dir)
        elif objective == "mechanics-head":
            save_mechanics_head(action_head, output_dir)
        else:
            save_policy_head(action_head, output_dir)
    (output_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _gradient_l2_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = parameter.grad.detach().float().norm(2)
            squared_norm += float(value.item()) ** 2
    return math.sqrt(squared_norm)


def _nonfinite_gradient_count(parameters: Sequence[torch.nn.Parameter]) -> int:
    return sum(
        int((~torch.isfinite(parameter.grad)).sum().item())
        for parameter in parameters
        if parameter.grad is not None
    )


def _training_class_weights(
    train_file: str,
    mode: str,
    maximum_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor | None, list[float] | None]:
    if mode == "none":
        return None, None
    counts = load_action_counts(train_file)
    values: list[float] = []
    if mode == "sqrt-inverse":
        largest = max(counts.values())
        values = [math.sqrt(largest / max(counts[action_id], 1)) for action_id in range(13)]
    else:
        family_counts: Counter[str] = Counter()
        for action_id, count in counts.items():
            family_counts[action_family(action_id)] += count
        largest_family = max(family_counts.values())
        values = [
            largest_family / max(family_counts[action_family(action_id)], 1)
            for action_id in range(13)
        ]
    values = [min(value, maximum_weight) for value in values]
    weighted_mean = sum(values[action_id] * counts[action_id] for action_id in range(13)) / sum(
        counts.values()
    )
    values = [value / weighted_mean for value in values]
    return torch.tensor(values, dtype=torch.float32, device=device), values


def train(args: argparse.Namespace) -> dict[str, Any]:
    positive_values = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "log_steps": args.log_steps,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.eval_batches is not None and args.eval_batches <= 0:
        raise ValueError("eval_batches must be positive")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if args.scheduler_steps is not None and args.scheduler_steps <= 0:
        raise ValueError("scheduler_steps must be positive")
    if args.max_class_weight < 1:
        raise ValueError("max_class_weight must be at least 1")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative")
    if args.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta cannot be negative")
    if args.objective == "candidate-head" and args.prompt_format == "mechanics-v1":
        raise ValueError("candidate-head requires prompt candidate lines; use compact-v1")
    if args.objective in ACTION_OBJECTIVES and args.loss_projection == "full":
        raise ValueError(
            "--loss-projection full benchmarks the causal LM head and is not applicable "
            "to an action-head objective"
        )
    if args.objective == "mechanics-head":
        args.mechanics_schema = MECHANICS_SCHEMA
        args.mechanics_feature_count = MECHANICS_FEATURE_COUNT

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Pass --overwrite-output to reuse it."
        )

    model, tokenizer, device = load_policy_model(
        args.model,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        for_training=True,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    dtype = resolve_dtype(args.dtype, device)
    if args.method == "lora":
        model = attach_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            is_4bit=args.load_in_4bit,
        )
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False

    available_logits_parameter = indexed_logits_parameter(model)
    if args.loss_projection == "full":
        logits_parameter = None
    elif args.loss_projection == "selective":
        if available_logits_parameter is None:
            raise RuntimeError(
                "The selected model does not support selective LM-head projection"
            )
        logits_parameter = available_logits_parameter
    else:
        logits_parameter = available_logits_parameter
    if args.objective == "policy-head":
        action_head: torch.nn.Module | None = create_policy_head(model, device)
    elif args.objective == "candidate-head":
        action_head = create_candidate_head(model, device)
    elif args.objective == "mechanics-head":
        action_head = create_mechanics_head(model, device)
    else:
        action_head = None
    print(
        json.dumps(
            {
                "training_objective": args.objective,
                "loss_projection": (
                    "causal_lm_head_bypassed"
                    if args.objective in ACTION_OBJECTIVES
                    else (
                        "supervised_positions_only"
                        if logits_parameter is not None
                        else "all_sequence_positions"
                    )
                ),
                "logits_parameter": (
                    None
                    if args.objective in ACTION_OBJECTIVES
                    else logits_parameter
                ),
                "prompt_format": args.prompt_format,
            }
        ),
        flush=True,
    )

    dataset_limit = args.overfit_examples
    train_dataset: Any = JsonlOffsetDataset(args.train_file, limit=dataset_limit)
    if args.objective == "mechanics-head" and args.train_mechanics_cache:
        train_dataset = MechanicsCacheDataset(train_dataset, args.train_mechanics_cache)
    class_weights, serialized_class_weights = _training_class_weights(
        args.train_file,
        args.class_weighting,
        args.max_class_weight,
        device,
    )
    print(
        json.dumps(
            {
                "class_weighting": args.class_weighting,
                "class_weights": serialized_class_weights,
            }
        ),
        flush=True,
    )
    validation_sample_metadata: dict[str, Any] | None = None
    if args.validation_file:
        complete_validation_dataset: Any = JsonlOffsetDataset(args.validation_file)
        if args.objective == "mechanics-head" and args.validation_mechanics_cache:
            complete_validation_dataset = MechanicsCacheDataset(
                complete_validation_dataset,
                args.validation_mechanics_cache,
            )
        validation_dataset, validation_sample_metadata = select_evaluation_dataset(
            complete_validation_dataset,
            max_examples=args.validation_examples,
            mode=args.validation_sample_mode,
            seed=args.validation_sample_seed,
        )
    else:
        validation_dataset = None
    collator_classes = {
        "sft": SFTCollator,
        "policy-head": PolicyCollator,
        "candidate-head": CandidateCollator,
        "mechanics-head": MechanicsCollator,
    }
    collator_kwargs: dict[str, Any] = {
        "max_length": args.max_length,
        "truncation": args.truncation,
        "prompt_format": args.prompt_format,
    }
    if args.objective == "candidate-head":
        collator_kwargs["shuffle_candidates"] = True
    train_collator = collator_classes[args.objective](
        tokenizer,
        **collator_kwargs,
    )
    if args.objective == "candidate-head":
        collator_kwargs["shuffle_candidates"] = False
    validation_collator = collator_classes[args.objective](tokenizer, **collator_kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=validation_collator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if validation_dataset is not None
        else None
    )

    model_trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    head_parameters = list(action_head.parameters()) if action_head is not None else []
    trainable_parameters = [*model_trainable_parameters, *head_parameters]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters were found")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    planned_updates = updates_per_epoch * args.epochs
    training_updates = (
        min(planned_updates, args.max_steps) if args.max_steps else planned_updates
    )
    scheduler_updates = args.scheduler_steps or planned_updates
    warmup_updates = int(scheduler_updates * args.warmup_ratio)

    def lr_multiplier(step: int) -> float:
        return learning_rate_multiplier(
            step,
            warmup_updates=warmup_updates,
            scheduler_updates=scheduler_updates,
            scheduler_name=args.lr_scheduler,
            min_lr_ratio=args.min_lr_ratio,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    if action_head is not None:
        action_head.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    started = time.monotonic()
    loss_since_log: torch.Tensor | None = None
    microbatches_since_log = 0
    accumulated_micro_steps = 0
    best_validation_loss = math.inf
    best_validation_accuracy = -math.inf
    early_stopping_best = math.inf if args.objective == "sft" else -math.inf
    validations_without_improvement = 0
    early_stop_requested = False
    last_validation_step = 0
    history: list[dict[str, Any]] = []
    examples_seen = 0
    tokens_seen = 0
    interval_examples = 0
    interval_tokens = 0
    interval_started = started

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    expected_passes = training_updates * effective_batch_size / len(train_dataset)
    run_shape = {
        "train_examples": len(train_dataset),
        "effective_batch_size": effective_batch_size,
        "updates_per_epoch": updates_per_epoch,
        "planned_updates": planned_updates,
        "training_updates": training_updates,
        "scheduler_updates": scheduler_updates,
        "expected_dataset_passes": round(expected_passes, 4),
    }
    print(json.dumps(run_shape), flush=True)
    if expected_passes < 1 and not args.overfit_examples:
        print(
            json.dumps(
                {
                    "warning": "Training stops before one complete dataset pass.",
                    "expected_dataset_passes": round(expected_passes, 4),
                }
            ),
            flush=True,
        )

    def run_validation(step: int) -> dict[str, Any] | None:
        nonlocal best_validation_accuracy, best_validation_loss
        nonlocal early_stopping_best, early_stop_requested
        nonlocal last_validation_step, validations_without_improvement
        if validation_loader is None:
            return None
        metrics = _evaluate_model(
            model,
            action_head,
            validation_loader,
            device,
            dtype,
            args.eval_batches,
            logits_parameter,
            args.objective,
        )
        last_validation_step = step
        validation_loss = float(metrics["validation_loss"])
        validation_accuracy = float(metrics.get("validation_accuracy", -math.inf))
        if args.objective == "sft":
            is_best = validation_loss < best_validation_loss
        else:
            is_best = validation_accuracy > best_validation_accuracy or (
                validation_accuracy == best_validation_accuracy
                and validation_loss < best_validation_loss
            )
        if is_best:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            _save_checkpoint(
                model,
                tokenizer,
                output_dir / "best",
                vars(args)
                | {
                    "best_validation_loss": best_validation_loss,
                    "best_validation_accuracy": (
                        best_validation_accuracy
                        if math.isfinite(best_validation_accuracy)
                        else None
                    ),
                    "best_global_step": step,
                    "validation_sample": validation_sample_metadata,
                },
                action_head,
                args.objective,
            )
        if args.early_stopping_patience > 0:
            if args.objective == "sft":
                meaningful_improvement = meaningful_validation_improvement(
                    validation_loss,
                    early_stopping_best,
                    minimum_delta=args.early_stopping_min_delta,
                    higher_is_better=False,
                )
                if meaningful_improvement:
                    early_stopping_best = validation_loss
            else:
                meaningful_improvement = meaningful_validation_improvement(
                    validation_accuracy,
                    early_stopping_best,
                    minimum_delta=args.early_stopping_min_delta,
                    higher_is_better=True,
                )
                if meaningful_improvement:
                    early_stopping_best = validation_accuracy
            validations_without_improvement = (
                0
                if meaningful_improvement
                else validations_without_improvement + 1
            )
            early_stop_requested = (
                validations_without_improvement >= args.early_stopping_patience
            )
        record = {
            "step": step,
            **metrics,
            "validations_without_improvement": validations_without_improvement,
            "early_stop_requested": early_stop_requested,
        }
        print(json.dumps(record), flush=True)
        history.append(record)
        return metrics

    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(train_loader):
            current_batch_size = int(batch["input_ids"].shape[0])
            current_tokens = int(batch["attention_mask"].sum().item())
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            examples_seen += current_batch_size
            tokens_seen += current_tokens
            interval_examples += current_batch_size
            interval_tokens += current_tokens
            with _autocast_context(device, dtype):
                if args.objective in ACTION_OBJECTIVES:
                    assert action_head is not None
                    batch_loss = _action_loss(
                        model,
                        action_head,
                        batch,
                        args.objective,
                        logits_parameter,
                        class_weights,
                    )
                else:
                    batch_loss = assistant_only_loss(
                        model,
                        batch,
                        logits_parameter=logits_parameter,
                    )
                loss = batch_loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            detached_batch_loss = batch_loss.detach()
            loss_since_log = (
                detached_batch_loss
                if loss_since_log is None
                else loss_since_log + detached_batch_loss
            )
            microbatches_since_log += 1
            accumulated_micro_steps += 1

            should_update = (
                accumulated_micro_steps == args.gradient_accumulation_steps
                or batch_index + 1 == len(train_loader)
            )
            if not should_update:
                continue

            scaler.unscale_(optimizer)
            next_global_step = global_step + 1
            should_log = next_global_step % args.log_steps == 0 or next_global_step == 1
            head_gradient_norm = (
                _gradient_l2_norm(head_parameters) if should_log and head_parameters else None
            )
            lora_gradient_norm = (
                _gradient_l2_norm(model_trainable_parameters)
                if should_log and model_trainable_parameters
                else None
            )
            nonfinite_gradients = (
                _nonfinite_gradient_count(trainable_parameters) if should_log else None
            )
            if args.max_grad_norm > 0:
                unclipped_gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    args.max_grad_norm,
                )
                unclipped_gradient_norm = (
                    float(unclipped_gradient_norm_tensor.item()) if should_log else math.nan
                )
            else:
                unclipped_gradient_norm = (
                    _gradient_l2_norm(trainable_parameters) if should_log else math.nan
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            accumulated_micro_steps = 0

            if global_step % args.log_steps == 0 or global_step == 1:
                assert loss_since_log is not None
                average_loss = float(
                    (loss_since_log / max(microbatches_since_log, 1)).item()
                )
                now = time.monotonic()
                interval_seconds = max(now - interval_started, 1e-9)
                record: dict[str, Any] = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "epoch_fraction": round(examples_seen / len(train_dataset), 6),
                    "examples_seen": examples_seen,
                    "tokens_seen": tokens_seen,
                    "train_loss": average_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "examples_per_second": round(interval_examples / interval_seconds, 3),
                    "tokens_per_second": round(interval_tokens / interval_seconds, 1),
                    "gradient_norm": unclipped_gradient_norm,
                    "gradient_clipped": (
                        args.max_grad_norm > 0
                        and unclipped_gradient_norm > args.max_grad_norm
                    ),
                    "head_gradient_norm": head_gradient_norm,
                    "lora_gradient_norm": lora_gradient_norm,
                    "nonfinite_gradient_values": nonfinite_gradients,
                    "elapsed_seconds": round(now - started, 1),
                }
                if device.type == "cuda":
                    gibibyte = 1024**3
                    record.update(
                        {
                            "vram_allocated_gib": round(
                                torch.cuda.memory_allocated(device) / gibibyte,
                                2,
                            ),
                            "vram_reserved_gib": round(
                                torch.cuda.memory_reserved(device) / gibibyte,
                                2,
                            ),
                            "vram_peak_allocated_gib": round(
                                torch.cuda.max_memory_allocated(device) / gibibyte,
                                2,
                            ),
                        }
                    )
                print(json.dumps(record), flush=True)
                history.append(record)
                loss_since_log = None
                microbatches_since_log = 0
                interval_examples = 0
                interval_tokens = 0
                interval_started = now

            if (
                validation_loader is not None
                and args.eval_steps > 0
                and global_step % args.eval_steps == 0
            ):
                run_validation(global_step)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                _save_checkpoint(
                    model,
                    tokenizer,
                    output_dir / f"checkpoint-{global_step}",
                    vars(args)
                    | {
                        "global_step": global_step,
                        "validation_sample": validation_sample_metadata,
                    },
                    action_head,
                    args.objective,
                )

            if global_step >= training_updates or early_stop_requested:
                break
        if global_step >= training_updates or early_stop_requested:
            break

    if validation_loader is not None and last_validation_step != global_step:
        run_validation(global_step)

    final_config = vars(args) | {
        "global_step": global_step,
        "early_stopped": early_stop_requested,
        "validations_without_improvement": validations_without_improvement,
        "train_examples": len(train_dataset),
        "validation_examples_loaded": (
            len(validation_dataset) if validation_dataset is not None else 0
        ),
        "best_validation_loss": (
            best_validation_loss if math.isfinite(best_validation_loss) else None
        ),
        "best_validation_accuracy": (
            best_validation_accuracy if math.isfinite(best_validation_accuracy) else None
        ),
        **run_shape,
        "validation_sample": validation_sample_metadata,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    if device.type == "cuda":
        final_config["peak_allocated_vram_gib"] = round(
            torch.cuda.max_memory_allocated(device) / 1024**3,
            2,
        )
    _save_checkpoint(
        model,
        tokenizer,
        output_dir / "final",
        final_config,
        action_head,
        args.objective,
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    return final_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a generative, fixed policy-head, or shared candidate-head model."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-output", action="store_true")

    parser.add_argument("--method", choices=("lora", "full"), default="lora")
    parser.add_argument(
        "--objective",
        choices=("sft", "policy-head", "candidate-head", "mechanics-head"),
        default="mechanics-head",
        help=(
            "Use legacy generative SFT, a fixed 13-way head, a shared text-candidate "
            "scorer, or the zero-token numeric mechanics scorer."
        ),
    )
    parser.add_argument(
        "--loss-projection",
        choices=("auto", "selective", "full"),
        default="auto",
        help=(
            "For SFT, use supported selective LM-head logits, require them, or force "
            "full-sequence logits for controlled memory/throughput benchmarks."
        ),
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not make Hugging Face network requests; require a cached/local model.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=list(DEFAULT_LORA_TARGETS),
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--scheduler-steps",
        type=int,
        help="Optional LR-schedule horizon; max_steps no longer shortens it implicitly.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("cosine", "constant-with-warmup"),
        default="cosine",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--class-weighting",
        choices=("none", "sqrt-inverse", "family-balanced"),
        default="none",
        help="Optional capped action weighting; validation metrics remain unweighted.",
    )
    parser.add_argument("--max-class-weight", type=float, default=3.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--truncation", choices=("error", "left"), default="error")
    parser.add_argument(
        "--prompt-format",
        choices=PROMPT_FORMATS,
        default="mechanics-v1",
    )
    parser.add_argument(
        "--train-mechanics-cache",
        help="Optional mechanics-v1 .npy cache aligned with the training JSONL.",
    )
    parser.add_argument(
        "--validation-mechanics-cache",
        help="Optional mechanics-v1 .npy cache aligned with the validation JSONL.",
    )
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--overfit-examples",
        type=int,
        help="Restrict training to the first N rows for the pipeline smoke test.",
    )
    parser.add_argument("--validation-examples", type=int, default=512)
    parser.add_argument(
        "--validation-sample-mode",
        choices=("head", "hash"),
        default="hash",
        help="Select the earliest validation rows or a deterministic full-file sample.",
    )
    parser.add_argument("--validation-sample-seed", type=int, default=42)
    parser.add_argument(
        "--eval-batches",
        type=int,
        help="Optional cap; by default evaluate every selected validation example.",
    )
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "Stop after this many validations without a meaningful improvement; "
            "zero disables early stopping."
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation accuracy (or SFT loss) change that resets patience.",
    )
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.overfit_examples and args.validation_file:
        print(
            "Overfit mode is a pipeline test; validation loss is not expected to improve.",
            flush=True,
        )
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
