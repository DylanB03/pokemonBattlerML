from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from pokemon_battler.evaluation_utils import (
    ActionMetrics,
    load_action_counts,
    select_evaluation_dataset,
)
from pokemon_battler.modeling import (
    has_candidate_head,
    has_mechanics_head,
    has_policy_head,
    indexed_logits_parameter,
    load_candidate_head,
    load_mechanics_head,
    load_policy_head,
    load_policy_model,
    load_training_metadata,
    masked_candidate_logits,
    masked_mechanics_logits,
    masked_policy_logits,
    score_legal_actions,
)
from pokemon_battler.prompting import PROMPT_FORMATS
from pokemon_battler.training_data import (
    CandidateCollator,
    JsonlOffsetDataset,
    MechanicsCacheDataset,
    MechanicsCollator,
    PolicyCollator,
    state_with_row_context,
)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    training_metadata = load_training_metadata(args.adapter) if args.adapter else {}
    prompt_format = args.prompt_format
    if prompt_format == "auto":
        prompt_format = str(training_metadata.get("prompt_format", "verbose-v1"))
    model, tokenizer, device = load_policy_model(
        args.model,
        adapter_path=args.adapter,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    scoring = args.scoring
    if scoring == "auto":
        if has_mechanics_head(args.adapter):
            scoring = "mechanics-head"
        elif has_candidate_head(args.adapter):
            scoring = "candidate-head"
        elif has_policy_head(args.adapter):
            scoring = "policy-head"
        else:
            scoring = "generative"
    if scoring == "policy-head":
        if not args.adapter:
            raise ValueError("Policy-head scoring requires --adapter")
        policy_head = load_policy_head(model, args.adapter, device)
        policy_head.eval()
        candidate_head = None
        mechanics_head = None
    elif scoring == "candidate-head":
        if not args.adapter:
            raise ValueError("Candidate-head scoring requires --adapter")
        candidate_head = load_candidate_head(model, args.adapter, device)
        candidate_head.eval()
        policy_head = None
        mechanics_head = None
    elif scoring == "mechanics-head":
        if not args.adapter:
            raise ValueError("Mechanics-head scoring requires --adapter")
        mechanics_head = load_mechanics_head(model, args.adapter, device)
        mechanics_head.eval()
        policy_head = None
        candidate_head = None
    else:
        policy_head = None
        candidate_head = None
        mechanics_head = None

    complete_dataset: Any = JsonlOffsetDataset(args.data_file)
    if mechanics_head is not None and args.mechanics_cache:
        complete_dataset = MechanicsCacheDataset(
            complete_dataset,
            args.mechanics_cache,
            mechanics_schema=mechanics_head.schema,
        )
    dataset, sample_metadata = select_evaluation_dataset(
        complete_dataset,
        max_examples=args.max_examples,
        mode=args.sample_mode,
        seed=args.sample_seed,
    )
    train_counts = (
        load_action_counts(args.baseline_train_file) if args.baseline_train_file else None
    )
    metrics = ActionMetrics(
        train_action_counts=train_counts,
        max_saved_errors=args.max_saved_errors,
    )

    if candidate_head is not None or policy_head is not None or mechanics_head is not None:
        if mechanics_head is not None:
            collator_class = MechanicsCollator
        elif candidate_head is not None:
            collator_class = CandidateCollator
        else:
            collator_class = PolicyCollator
        collator_kwargs: dict[str, Any] = {
            "max_length": args.max_length,
            "truncation": "error",
            "prompt_format": prompt_format,
        }
        if mechanics_head is not None:
            collator_kwargs["mechanics_schema"] = mechanics_head.schema
        collator = collator_class(
            tokenizer,
            **collator_kwargs,
        )
        evaluated = 0
        logits_parameter = indexed_logits_parameter(model)
        with torch.inference_mode():
            for start in range(0, len(dataset), args.batch_size):
                stop = min(start + args.batch_size, len(dataset))
                rows = [dataset[index] for index in range(start, stop)]
                batch = {
                    key: value.to(device)
                    for key, value in collator(rows).items()
                }
                if candidate_head is not None:
                    logits = masked_candidate_logits(
                        model,
                        candidate_head,
                        batch,
                        logits_parameter=logits_parameter,
                    )
                elif mechanics_head is not None:
                    logits = masked_mechanics_logits(
                        model,
                        mechanics_head,
                        batch,
                        logits_parameter=logits_parameter,
                    )
                else:
                    assert policy_head is not None
                    logits = masked_policy_logits(
                        model,
                        policy_head,
                        batch,
                        logits_parameter=logits_parameter,
                    )
                for row_index, row in enumerate(rows):
                    legal = [int(value) for value in row["legal_action_ids"]]
                    scores = {
                        action_id: float(logits[row_index, action_id].item())
                        for action_id in legal
                    }
                    metrics.add(start + row_index, row, scores)
                evaluated += len(rows)
                if args.log_every and evaluated % args.log_every < len(rows):
                    print(
                        json.dumps(
                            {
                                "evaluated": evaluated,
                                "accuracy": metrics.correct / evaluated,
                            }
                        ),
                        flush=True,
                    )
    else:
        for index, row in enumerate(dataset):
            state = state_with_row_context(row)
            scores = score_legal_actions(
                model,
                tokenizer,
                state,
                device,
                max_length=args.max_length,
                prompt_format=prompt_format,
            )
            metrics.add(index, row, scores)

            if args.log_every and (index + 1) % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "evaluated": index + 1,
                            "accuracy": metrics.correct / (index + 1),
                        }
                    ),
                    flush=True,
                )

    report = metrics.report()
    report["evaluation"] = {
        "model": args.model,
        "adapter": args.adapter,
        "scoring": scoring,
        "data_file": args.data_file,
        "baseline_train_file": args.baseline_train_file,
        "mechanics_cache": args.mechanics_cache,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "prompt_format": prompt_format,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "attn_implementation": args.attn_implementation,
        "sample": sample_metadata,
        "training_metadata": training_metadata,
    }
    report["note"] = (
        "Accuracy measures imitation of the recorded human action, not battle win rate. "
        "Oracle-type metrics are diagnostics that reveal the target move-versus-switch kind."
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate action imitation over replay-recoverable candidates."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter", help="Optional PEFT adapter directory.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--sample-mode",
        choices=("head", "hash"),
        default="hash",
        help="Use the first rows or a deterministic sample spanning the complete file.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--baseline-train-file",
        help="Optional prepared training JSONL used for a legally masked frequency baseline.",
    )
    parser.add_argument(
        "--scoring",
        choices=("auto", "generative", "policy-head", "candidate-head", "mechanics-head"),
        default="auto",
        help="Auto-detect a saved action head or use generative candidate scoring.",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--prompt-format",
        choices=("auto", *PROMPT_FORMATS),
        default="auto",
        help="Use checkpoint metadata automatically or select a prompt serializer.",
    )
    parser.add_argument("--max-saved-errors", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output")
    parser.add_argument("--mechanics-cache")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = evaluate(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
