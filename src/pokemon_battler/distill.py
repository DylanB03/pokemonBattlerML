from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pokemon_battler.distillation import train_teacher_distillation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill Foul Play MCTS policies into a Qwen interaction checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--qwen-learning-rate", type=float, default=5e-6)
    parser.add_argument("--head-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hard-target-weight", type=float, default=0.0)
    parser.add_argument("--confident-disagreement-weight", type=float, default=0.0)
    parser.add_argument("--confidence-power", type=float, default=0.0)
    parser.add_argument("--family-aux-weight", type=float, default=0.25)
    parser.add_argument("--action-value-weight", type=float, default=0.25)
    parser.add_argument("--root-value-weight", type=float, default=0.1)
    parser.add_argument("--outcome-value-weight", type=float, default=0.05)
    parser.add_argument("--tera-weight", type=float, default=2.0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--trajectory-cap", type=int, default=48)
    parser.add_argument(
        "--freeze-qwen", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--rehearsal-data", type=Path)
    parser.add_argument("--rehearsal-weight", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--max-length", type=int)
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument(
        "--load-in-4bit", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-steps", type=int, default=20)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    report = train_teacher_distillation(
        checkpoint=args.checkpoint,
        teacher_file=args.teacher_data,
        validation_file=args.validation_data,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        qwen_learning_rate=args.qwen_learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        hard_target_weight=args.hard_target_weight,
        confident_disagreement_weight=args.confident_disagreement_weight,
        confidence_power=args.confidence_power,
        family_aux_weight=args.family_aux_weight,
        action_value_weight=args.action_value_weight,
        root_value_weight=args.root_value_weight,
        outcome_value_weight=args.outcome_value_weight,
        tera_weight=args.tera_weight,
        validation_fraction=args.validation_fraction,
        trajectory_cap=args.trajectory_cap,
        freeze_qwen=args.freeze_qwen,
        rehearsal_file=args.rehearsal_data,
        rehearsal_weight=args.rehearsal_weight,
        early_stopping_patience=args.early_stopping_patience,
        max_length=args.max_length,
        dtype_name=args.dtype,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
        seed=args.seed,
        log_steps=args.log_steps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved distilled checkpoint to {args.output_dir}")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
