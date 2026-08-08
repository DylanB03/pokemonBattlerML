from __future__ import annotations

import argparse
import gc
import json
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import torch

from pokemon_battler.evaluate import build_parser as build_evaluate_parser
from pokemon_battler.evaluate import evaluate
from pokemon_battler.experiment import build_parser as build_experiment_parser
from pokemon_battler.experiment import run_experiment
from pokemon_battler.interaction_cache import (
    build_interaction_cache,
    default_interaction_cache_path,
)
from pokemon_battler.interaction_features import PREPARED_SCHEMA_VERSION
from pokemon_battler.prepare import SplitConfig, prepare_dataset
from pokemon_battler.train import build_parser as build_train_parser
from pokemon_battler.train import train

DEFAULT_RAW_INPUT = Path("data/raw/metamon/gen9ou.tar.gz")
DEFAULT_DATA_DIR = Path("data/gen9ou-interaction-v1")
DEFAULT_OUTPUT_DIR = Path("outputs/interaction-v1-1epoch")


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _dataset_matches(args: argparse.Namespace) -> bool:
    report_path = args.data_dir / "prepare_report.json"
    split_paths = [args.data_dir / f"{split}.jsonl" for split in ("train", "validation", "test")]
    if not report_path.is_file() or not all(path.is_file() for path in split_paths):
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("schema_version", -1)) != PREPARED_SCHEMA_VERSION:
            return False
        expected = {
            "inputs": [str(args.input)],
            "battle_format": args.format,
            "min_rating": args.min_rating,
            "outcome": "both",
            "sample_rate": args.sample_rate,
        }
        if any(report.get(key) != value for key, value in expected.items()):
            return False
        split = report.get("split_config") or {}
        if split.get("mode") != "chronological":
            return False
        if split.get("validation_start") != args.validation_start.isoformat():
            return False
        if split.get("test_start") != args.test_start.isoformat():
            return False
        if int(split.get("seed", -1)) != args.seed:
            return False
        for path in split_paths:
            with path.open(encoding="utf-8") as stream:
                first = json.loads(next(line for line in stream if line.strip()))
            if int(first.get("schema_version", -1)) != PREPARED_SCHEMA_VERSION:
                return False
        return True
    except (OSError, StopIteration, TypeError, ValueError, json.JSONDecodeError):
        return False


def _prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    if _dataset_matches(args) and not args.rebuild_data:
        return json.loads((args.data_dir / "prepare_report.json").read_text(encoding="utf-8"))
    if args.data_dir.exists() and any(args.data_dir.iterdir()) and not args.rebuild_data:
        raise FileExistsError(
            f"Prepared data directory exists but is not the required schema/configuration: "
            f"{args.data_dir}. Use a new --data-dir or pass --rebuild-data."
        )
    print(
        json.dumps(
            {
                "phase": "prepare-schema-3-dataset",
                "input": str(args.input),
                "data_dir": str(args.data_dir),
                "sample_rate": args.sample_rate,
            }
        ),
        flush=True,
    )
    if args.rebuild_data:
        # Do not leave an old success marker beside newly truncated JSONL files
        # if an explicit rebuild is interrupted.
        (args.data_dir / "prepare_report.json").unlink(missing_ok=True)
    return prepare_dataset(
        [args.input],
        args.data_dir,
        split_config=SplitConfig(
            mode="chronological",
            seed=args.seed,
            validation_start=args.validation_start,
            test_start=args.test_start,
        ),
        battle_format=args.format,
        min_rating=args.min_rating,
        outcome="both",
        sample_rate=args.sample_rate,
        progress_every=args.prepare_progress_every,
        overwrite=args.rebuild_data,
    )


def _prepare_caches(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        data_file = args.data_dir / f"{split}.jsonl"
        cache_dir = default_interaction_cache_path(data_file)
        print(
            json.dumps(
                {
                    "phase": "prepare-interaction-cache",
                    "split": split,
                    "data_file": str(data_file),
                    "cache_dir": str(cache_dir),
                }
            ),
            flush=True,
        )
        reports[split] = build_interaction_cache(
            data_file,
            cache_dir,
            overwrite=args.rebuild_cache or args.rebuild_data,
            progress_every=args.cache_progress_every,
        )
    return reports


def _overfit_train_arguments(args: argparse.Namespace) -> argparse.Namespace:
    train_file = args.data_dir / "train.jsonl"
    values = [
        "--model",
        args.model,
        "--train-file",
        str(train_file),
        "--train-interaction-cache",
        str(default_interaction_cache_path(train_file)),
        "--output-dir",
        str(args.output_dir / "overfit-128"),
        "--objective",
        "interaction-head",
        "--prompt-format",
        "mechanics-v2",
        "--method",
        "lora",
        "--dtype",
        args.dtype,
        "--attn-implementation",
        args.attn_implementation,
        "--qwen-mode",
        args.qwen_mode,
        "--overfit-examples",
        "128",
        "--batch-size",
        "1",
        "--gradient-accumulation-steps",
        "8",
        "--learning-rate",
        str(args.overfit_learning_rate),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        "0",
        "--interaction-dropout",
        "0",
        "--interaction-d-model",
        str(args.interaction_d_model),
        "--interaction-attention-heads",
        str(args.interaction_attention_heads),
        "--interaction-layers",
        str(args.interaction_layers),
        "--interaction-feedforward-size",
        str(args.interaction_feedforward_size),
        "--interaction-identity-embedding-size",
        str(args.interaction_identity_embedding_size),
        "--family-aux-weight",
        "0",
        "--value-loss-weight",
        "0",
        "--epochs",
        "100",
        "--max-steps",
        str(args.overfit_steps),
        "--max-length",
        str(args.max_length),
        "--eval-steps",
        "0",
        "--log-steps",
        "10",
        "--seed",
        str(args.seed),
    ]
    if args.load_in_4bit:
        values.append("--load-in-4bit")
    if args.local_files_only:
        values.append("--local-files-only")
    return build_train_parser().parse_args(values)


def _run_overfit_gate(args: argparse.Namespace) -> dict[str, Any]:
    print(json.dumps({"phase": "interaction-overfit-gate", "examples": 128}), flush=True)
    training = train(_overfit_train_arguments(args))
    _release_memory()
    train_file = args.data_dir / "train.jsonl"
    report_path = args.output_dir / "reports" / "overfit-128.json"
    values = [
        "--model",
        args.model,
        "--adapter",
        str(args.output_dir / "overfit-128" / "final"),
        "--data-file",
        str(train_file),
        "--interaction-cache",
        str(default_interaction_cache_path(train_file)),
        "--max-examples",
        "128",
        "--sample-mode",
        "head",
        "--batch-size",
        str(args.eval_batch_size),
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
        "0",
    ]
    if args.load_in_4bit:
        values.append("--load-in-4bit")
    if args.local_files_only:
        values.append("--local-files-only")
    report = evaluate(build_evaluate_parser().parse_args(values))
    _release_memory()
    if float(report["accuracy"]) < args.overfit_required_accuracy:
        raise RuntimeError(
            f"Interaction memorization gate failed: {report['accuracy']:.2%} < "
            f"{args.overfit_required_accuracy:.2%}. Full training was not started."
        )
    return {"training": training, "evaluation": report}


def _full_experiment_arguments(args: argparse.Namespace) -> argparse.Namespace:
    train_file = args.data_dir / "train.jsonl"
    validation_file = args.data_dir / "validation.jsonl"
    values = [
        "--output-dir",
        str(args.output_dir / "policy"),
        "--train-file",
        str(train_file),
        "--validation-file",
        str(validation_file),
        "--train-interaction-cache",
        str(default_interaction_cache_path(train_file)),
        "--validation-interaction-cache",
        str(default_interaction_cache_path(validation_file)),
        "--objective",
        "interaction-head",
        "--prompt-format",
        "mechanics-v2",
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
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
        "--qwen-mode",
        args.qwen_mode,
        "--family-aux-weight",
        str(args.family_aux_weight),
        "--value-loss-weight",
        str(args.value_loss_weight),
        "--max-length",
        str(args.max_length),
        "--validation-examples",
        str(args.validation_examples),
        "--evaluation-examples",
        str(args.evaluation_examples),
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
        "--dtype",
        args.dtype,
        "--attn-implementation",
        args.attn_implementation,
        "--seed",
        str(args.seed),
    ]
    if not args.load_in_4bit:
        values.append("--no-4bit")
    if not args.local_files_only:
        values.append("--allow-download")
    return build_experiment_parser().parse_args(values)


def run_interaction_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("The default interaction run requires CUDA; pass --no-4bit for CPU")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Choose a new --output-dir."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_report = _prepare_data(args)
    cache_reports = _prepare_caches(args)
    overfit = None if args.skip_overfit_gate else _run_overfit_gate(args)
    policy = run_experiment(_full_experiment_arguments(args))
    summary = {
        "dataset": dataset_report,
        "caches": cache_reports,
        "overfit_gate": overfit,
        "policy": policy,
    }
    (args.output_dir / "end_to_end_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare schema-3 replay data, build interaction caches, pass the "
            "memorization gate, train, evaluate, and summarize in one command."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_INPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--format", default="gen9ou")
    parser.add_argument("--min-rating", type=int, default=1600)
    parser.add_argument("--sample-rate", type=float, default=0.02)
    parser.add_argument("--validation-start", type=date.fromisoformat, default=date(2026, 1, 1))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2026, 4, 1))
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--prepare-progress-every", type=int, default=10_000)
    parser.add_argument("--cache-progress-every", type=int, default=10_000)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--interaction-d-model", type=int, default=384)
    parser.add_argument("--interaction-attention-heads", type=int, default=8)
    parser.add_argument("--interaction-layers", type=int, default=4)
    parser.add_argument("--interaction-feedforward-size", type=int, default=1536)
    parser.add_argument("--interaction-dropout", type=float, default=0.1)
    parser.add_argument("--interaction-identity-embedding-size", type=int, default=16)
    parser.add_argument("--qwen-mode", choices=("lora", "frozen", "none"), default="lora")
    parser.add_argument("--family-aux-weight", type=float, default=0.25)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--overfit-steps", type=int, default=400)
    parser.add_argument("--overfit-learning-rate", type=float, default=1e-3)
    parser.add_argument("--overfit-required-accuracy", type=float, default=0.95)
    parser.add_argument("--skip-overfit-gate", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--validation-examples", type=int, default=1024)
    parser.add_argument("--evaluation-examples", type=int, default=5000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.002)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--no-4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only")
    parser.set_defaults(load_in_4bit=True, local_files_only=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    result = run_interaction_experiment(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
