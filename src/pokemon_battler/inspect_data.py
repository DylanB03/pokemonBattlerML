from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from pokemon_battler.actions import sorted_switches
from pokemon_battler.modeling import load_tokenizer
from pokemon_battler.prompting import PROMPT_FORMATS, render_prompt
from pokemon_battler.training_data import JsonlOffsetDataset, state_with_row_context


def _percentile(sorted_values: list[int], quantile: float) -> int:
    if not sorted_values:
        return 0
    index = math.ceil(quantile * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = load_tokenizer(args.model, local_files_only=args.local_files_only)
    dataset = JsonlOffsetDataset(args.data_file, limit=args.max_examples)
    token_lengths: list[int] = []
    character_lengths: list[int] = []
    action_counts: Counter[str] = Counter()
    legal_action_counts: Counter[str] = Counter()
    switch_counts: Counter[str] = Counter()
    forced_switches = 0
    example_prompt: str | None = None

    for row in dataset:
        prompt = render_prompt(state_with_row_context(row), args.prompt_format)
        if example_prompt is None:
            example_prompt = prompt
        token_lengths.append(len(tokenizer.encode(prompt, add_special_tokens=True)))
        character_lengths.append(len(prompt))
        action_counts[f"A{row['action_id']}"] += 1
        legal_action_counts[str(len(row["legal_action_ids"]))] += 1
        switch_counts[str(len(sorted_switches(row["state"])))] += 1
        forced_switches += int(bool(row["state"].get("forced_switch", False)))

    token_lengths.sort()
    character_lengths.sort()
    length_summary = {
        "min": token_lengths[0],
        "p50": _percentile(token_lengths, 0.50),
        "p90": _percentile(token_lengths, 0.90),
        "p95": _percentile(token_lengths, 0.95),
        "p99": _percentile(token_lengths, 0.99),
        "max": token_lengths[-1],
    }
    report = {
        "examples": len(dataset),
        "model_tokenizer": args.model,
        "prompt_format": args.prompt_format,
        "prompt_tokens": length_summary,
        "prompt_characters": {
            "min": character_lengths[0],
            "p50": _percentile(character_lengths, 0.50),
            "p95": _percentile(character_lengths, 0.95),
            "max": character_lengths[-1],
        },
        "prompts_over_max_length": sum(
            length_value > args.max_length for length_value in token_lengths
        ),
        "configured_max_length": args.max_length,
        "forced_switch_fraction": forced_switches / len(dataset),
        "target_action_counts": dict(sorted(action_counts.items())),
        "legal_action_count_distribution": dict(sorted(legal_action_counts.items())),
        "available_switch_count_distribution": dict(sorted(switch_counts.items())),
    }
    if args.example_output and example_prompt is not None:
        example_path = Path(args.example_output)
        example_path.parent.mkdir(parents=True, exist_ok=True)
        example_path.write_text(example_prompt, encoding="utf-8")
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
        description="Inspect rendered prompt lengths and action distributions."
    )
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-examples", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--prompt-format",
        choices=PROMPT_FORMATS,
        default="compact-v1",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--example-output")
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = inspect(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
