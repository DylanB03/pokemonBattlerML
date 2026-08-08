from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from pokemon_battler.evaluate import build_parser as build_evaluate_parser
from pokemon_battler.evaluate import evaluate
from pokemon_battler.mechanics_v2 import MECHANICS_SCHEMA
from pokemon_battler.mechanics_cache import build_feature_cache, default_cache_path
from pokemon_battler.interaction_cache import (
    build_interaction_cache,
    default_interaction_cache_path,
)
from pokemon_battler.train import build_parser as build_train_parser
from pokemon_battler.train import train


def _release_model_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _train_arguments(args: argparse.Namespace) -> argparse.Namespace:
    values = [
        "--model",
        args.model,
        "--train-file",
        str(args.train_file),
        "--validation-file",
        str(args.validation_file),
        "--output-dir",
        str(args.output_dir),
        "--objective",
        args.objective,
        "--prompt-format",
        args.prompt_format,
        "--method",
        "lora",
        "--dtype",
        args.dtype,
        "--attn-implementation",
        args.attn_implementation,
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--class-weighting",
        args.class_weighting,
        "--max-class-weight",
        str(args.max_class_weight),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--epochs",
        str(args.epochs),
        "--lr-scheduler",
        args.lr_scheduler,
        "--min-lr-ratio",
        str(args.min_lr_ratio),
        "--max-length",
        str(args.max_length),
        "--validation-examples",
        str(args.validation_examples),
        "--validation-sample-mode",
        "hash",
        "--validation-sample-seed",
        str(args.seed),
        "--eval-steps",
        str(args.eval_steps),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
        "--log-steps",
        str(args.log_steps),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--qwen-mode",
        args.qwen_mode,
        "--interaction-d-model",
        str(args.interaction_d_model),
        "--interaction-attention-heads",
        str(args.interaction_attention_heads),
        "--interaction-layers",
        str(args.interaction_layers),
        "--interaction-feedforward-size",
        str(args.interaction_feedforward_size),
        "--interaction-dropout",
        str(args.interaction_dropout),
        "--interaction-identity-embedding-size",
        str(args.interaction_identity_embedding_size),
        "--family-aux-weight",
        str(args.family_aux_weight),
        "--value-loss-weight",
        str(args.value_loss_weight),
    ]
    if args.load_in_4bit:
        values.append("--load-in-4bit")
    if args.local_files_only:
        values.append("--local-files-only")
    if args.gradient_checkpointing:
        values.append("--gradient-checkpointing")
    if args.objective == "mechanics-head":
        values.extend(
            [
                "--train-mechanics-cache",
                str(
                    args.train_mechanics_cache
                    or default_cache_path(args.train_file, MECHANICS_SCHEMA)
                ),
                "--validation-mechanics-cache",
                str(
                    args.validation_mechanics_cache
                    or default_cache_path(args.validation_file, MECHANICS_SCHEMA)
                ),
            ]
        )
    if args.objective == "interaction-head":
        values.extend(
            [
                "--train-interaction-cache",
                str(
                    args.train_interaction_cache
                    or default_interaction_cache_path(args.train_file)
                ),
                "--validation-interaction-cache",
                str(
                    args.validation_interaction_cache
                    or default_interaction_cache_path(args.validation_file)
                ),
            ]
        )
    return build_train_parser().parse_args(values)


def _evaluate_checkpoint(
    args: argparse.Namespace,
    checkpoint: str,
) -> dict[str, Any]:
    adapter = args.output_dir / checkpoint
    report_path = args.output_dir / "reports" / f"{checkpoint}-validation.json"
    values = [
        "--model",
        args.model,
        "--adapter",
        str(adapter),
        "--data-file",
        str(args.validation_file),
        "--max-examples",
        str(args.evaluation_examples),
        "--batch-size",
        str(args.eval_batch_size),
        "--sample-mode",
        "hash",
        "--sample-seed",
        str(args.seed),
        "--scoring",
        "auto",
        "--prompt-format",
        "auto",
        "--max-length",
        str(args.max_length),
        "--attn-implementation",
        args.attn_implementation,
        "--output",
        str(report_path),
        "--log-every",
        str(args.evaluation_log_every),
    ]
    if args.load_in_4bit:
        values.append("--load-in-4bit")
    if args.local_files_only:
        values.append("--local-files-only")
    if args.objective == "mechanics-head":
        values.extend(
            [
                "--mechanics-cache",
                str(
                    args.validation_mechanics_cache
                    or default_cache_path(args.validation_file, MECHANICS_SCHEMA)
                ),
            ]
        )
    if args.objective == "interaction-head":
        values.extend(
            [
                "--interaction-cache",
                str(
                    args.validation_interaction_cache
                    or default_interaction_cache_path(args.validation_file)
                ),
            ]
        )
    report = evaluate(build_evaluate_parser().parse_args(values))
    _release_model_memory()
    return report


def _summary_metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": report["accuracy"],
        "top_k_accuracy": report["top_k_accuracy"],
        "mean_reciprocal_rank": report["mean_reciprocal_rank"],
        "average_candidate_nll": report["average_candidate_nll"],
        "action_type_accuracy": report["action_type_accuracy"],
        "accuracy_by_target_kind": report["accuracy_by_target_kind"],
        "accuracy_by_target_family": report["accuracy_by_target_family"],
        "prediction_counts": report["prediction_counts"],
        "target_counts": report["target_counts"],
        "value": report.get("value"),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if not args.train_file.is_file():
        raise FileNotFoundError(args.train_file)
    if not args.validation_file.is_file():
        raise FileNotFoundError(args.validation_file)
    if args.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("The default 4-bit experiment requires CUDA; pass --no-4bit for CPU")

    if args.objective == "mechanics-head":
        for data_file, configured_cache in (
            (args.train_file, args.train_mechanics_cache),
            (args.validation_file, args.validation_mechanics_cache),
        ):
            cache_path = configured_cache or default_cache_path(
                data_file,
                MECHANICS_SCHEMA,
            )
            print(
                json.dumps(
                    {
                        "phase": "prepare-mechanics-cache",
                        "data_file": str(data_file),
                        "cache_file": str(cache_path),
                    }
                ),
                flush=True,
            )
            build_feature_cache(
                data_file,
                cache_path,
                overwrite=args.rebuild_mechanics_cache,
                progress_every=args.mechanics_cache_progress_every,
                schema=MECHANICS_SCHEMA,
            )

    if args.objective == "interaction-head":
        for data_file, configured_cache in (
            (args.train_file, args.train_interaction_cache),
            (args.validation_file, args.validation_interaction_cache),
        ):
            cache_path = configured_cache or default_interaction_cache_path(data_file)
            print(
                json.dumps(
                    {
                        "phase": "prepare-interaction-cache",
                        "data_file": str(data_file),
                        "cache_dir": str(cache_path),
                    }
                ),
                flush=True,
            )
            build_interaction_cache(
                data_file,
                cache_path,
                overwrite=args.rebuild_interaction_cache,
                progress_every=args.interaction_cache_progress_every,
            )

    print(
        json.dumps(
            {
                "phase": "train",
                "objective": args.objective,
                "prompt_format": args.prompt_format,
                "epochs": args.epochs,
                "output_dir": str(args.output_dir),
            }
        ),
        flush=True,
    )
    training = train(_train_arguments(args))
    _release_model_memory()

    reports: dict[str, dict[str, Any]] = {}
    for checkpoint in ("best", "final"):
        if not (args.output_dir / checkpoint).is_dir():
            continue
        print(json.dumps({"phase": "evaluate", "checkpoint": checkpoint}), flush=True)
        reports[checkpoint] = _evaluate_checkpoint(args, checkpoint)
    if not reports:
        raise RuntimeError("Training produced no evaluable checkpoint")

    selected = max(
        reports,
        key=lambda name: (
            reports[name]["accuracy"],
            -reports[name]["average_candidate_nll"],
        ),
    )
    summary = {
        "selected_checkpoint": selected,
        "training": training,
        "checkpoints": {
            name: _summary_metrics(report) for name, report in reports.items()
        },
    }
    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train up to one complete policy experiment, evaluate, and summarize."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, default=Path("data/gen9ou-dev/train.jsonl"))
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/gen9ou-dev/validation.jsonl"),
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument(
        "--objective",
        choices=("interaction-head", "mechanics-head", "candidate-head", "policy-head"),
        default="mechanics-head",
    )
    parser.add_argument(
        "--prompt-format",
        choices=("mechanics-v2", "mechanics-v1", "compact-v1", "verbose-v1"),
        default="mechanics-v2",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--class-weighting",
        choices=("none", "sqrt-inverse", "family-balanced"),
        default="none",
    )
    parser.add_argument("--max-class-weight", type=float, default=3.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lr-scheduler",
        choices=("cosine", "constant-with-warmup"),
        default="cosine",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--validation-examples", type=int, default=1024)
    parser.add_argument("--evaluation-examples", type=int, default=5000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=4,
        help="Stop after this many validation checks without a meaningful gain.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.002,
        help="Accuracy gain required to reset early-stopping patience.",
    )
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--evaluation-log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-mechanics-cache", type=Path)
    parser.add_argument("--validation-mechanics-cache", type=Path)
    parser.add_argument("--rebuild-mechanics-cache", action="store_true")
    parser.add_argument("--mechanics-cache-progress-every", type=int, default=10_000)
    parser.add_argument("--train-interaction-cache", type=Path)
    parser.add_argument("--validation-interaction-cache", type=Path)
    parser.add_argument("--rebuild-interaction-cache", action="store_true")
    parser.add_argument("--interaction-cache-progress-every", type=int, default=10_000)
    parser.add_argument("--interaction-d-model", type=int, default=384)
    parser.add_argument("--interaction-attention-heads", type=int, default=8)
    parser.add_argument("--interaction-layers", type=int, default=4)
    parser.add_argument("--interaction-feedforward-size", type=int, default=1536)
    parser.add_argument("--interaction-dropout", type=float, default=0.1)
    parser.add_argument("--interaction-identity-embedding-size", type=int, default=16)
    parser.add_argument("--qwen-mode", choices=("lora", "frozen", "none"), default="lora")
    parser.add_argument("--family-aux-weight", type=float, default=0.25)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only")
    parser.set_defaults(load_in_4bit=True, local_files_only=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run_experiment(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
