from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from pokemon_battler.modeling import (
    assistant_only_loss,
    attach_lora,
    indexed_logits_parameter,
    load_policy_model,
    resolve_dtype,
)
from pokemon_battler.training_data import JsonlOffsetDataset, SFTCollator

DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_context(device: torch.device, dtype: torch.dtype) -> Any:
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _evaluate_loss(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int,
    logits_parameter: str | None,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast_context(device, dtype):
                loss = assistant_only_loss(
                    model,
                    batch,
                    logits_parameter=logits_parameter,
                )
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses) if losses else math.nan


def _save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    training_config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")

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

    logits_parameter = indexed_logits_parameter(model)
    print(
        json.dumps(
            {
                "loss_projection": (
                    "supervised_positions_only"
                    if logits_parameter is not None
                    else "all_sequence_positions"
                ),
                "logits_parameter": logits_parameter,
            }
        ),
        flush=True,
    )

    dataset_limit = args.overfit_examples
    train_dataset = JsonlOffsetDataset(args.train_file, limit=dataset_limit)
    validation_dataset = (
        JsonlOffsetDataset(args.validation_file, limit=args.validation_examples)
        if args.validation_file
        else None
    )
    collator = SFTCollator(
        tokenizer,
        max_length=args.max_length,
        truncation=args.truncation,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if validation_dataset is not None
        else None
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
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
    total_updates = min(planned_updates, args.max_steps) if args.max_steps else planned_updates
    warmup_updates = int(total_updates * args.warmup_ratio)

    def lr_multiplier(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return max(step, 1) / warmup_updates
        progress = (step - warmup_updates) / max(total_updates - warmup_updates, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    started = time.monotonic()
    loss_since_log = 0.0
    microbatches_since_log = 0
    accumulated_micro_steps = 0
    best_validation_loss = math.inf
    history: list[dict[str, Any]] = []

    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with _autocast_context(device, dtype):
                loss = (
                    assistant_only_loss(
                        model,
                        batch,
                        logits_parameter=logits_parameter,
                    )
                    / args.gradient_accumulation_steps
                )
            scaler.scale(loss).backward()
            loss_since_log += float(loss.item()) * args.gradient_accumulation_steps
            microbatches_since_log += 1
            accumulated_micro_steps += 1

            should_update = (
                accumulated_micro_steps == args.gradient_accumulation_steps
                or batch_index + 1 == len(train_loader)
            )
            if not should_update:
                continue

            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            accumulated_micro_steps = 0

            if global_step % args.log_steps == 0 or global_step == 1:
                average_loss = loss_since_log / max(microbatches_since_log, 1)
                record = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": average_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": round(time.monotonic() - started, 1),
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
                loss_since_log = 0.0
                microbatches_since_log = 0

            if (
                validation_loader is not None
                and args.eval_steps > 0
                and global_step % args.eval_steps == 0
            ):
                validation_loss = _evaluate_loss(
                    model,
                    validation_loader,
                    device,
                    dtype,
                    args.eval_batches,
                    logits_parameter,
                )
                record = {"step": global_step, "validation_loss": validation_loss}
                print(json.dumps(record), flush=True)
                history.append(record)
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    _save_checkpoint(
                        model,
                        tokenizer,
                        output_dir / "best",
                        vars(args) | {"best_validation_loss": best_validation_loss},
                    )

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                _save_checkpoint(
                    model,
                    tokenizer,
                    output_dir / f"checkpoint-{global_step}",
                    vars(args) | {"global_step": global_step},
                )

            if global_step >= total_updates:
                break
        if global_step >= total_updates:
            break

    final_config = vars(args) | {
        "global_step": global_step,
        "train_examples": len(train_dataset),
        "validation_examples_loaded": (
            len(validation_dataset) if validation_dataset is not None else 0
        ),
        "best_validation_loss": (
            best_validation_loss if math.isfinite(best_validation_loss) else None
        ),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    if device.type == "cuda":
        final_config["peak_allocated_vram_gib"] = round(
            torch.cuda.max_memory_allocated(device) / 1024**3,
            2,
        )
    _save_checkpoint(model, tokenizer, output_dir / "final", final_config)
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    return final_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT a causal LM on prepared Pokémon turns.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-output", action="store_true")

    parser.add_argument("--method", choices=("lora", "full"), default="lora")
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--truncation", choices=("error", "left"), default="error")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--overfit-examples",
        type=int,
        help="Restrict training to the first N rows for the pipeline smoke test.",
    )
    parser.add_argument("--validation-examples", type=int, default=512)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
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
