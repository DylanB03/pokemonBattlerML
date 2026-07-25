from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from pokemon_battler.actions import describe_action
from pokemon_battler.modeling import load_policy_model, score_legal_actions
from pokemon_battler.training_data import JsonlOffsetDataset


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_policy_model(
        args.model,
        adapter_path=args.adapter,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
    )
    model.eval()
    dataset = JsonlOffsetDataset(args.data_file, limit=args.max_examples)

    correct = 0
    target_kind: Counter[str] = Counter()
    correct_kind: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    margins: list[float] = []
    errors: list[dict[str, Any]] = []

    for index, row in enumerate(dataset):
        state = row["state"]
        target = int(row["action_id"])
        scores = score_legal_actions(
            model,
            tokenizer,
            state,
            device,
            max_length=args.max_length,
        )
        ranked = sorted(scores, key=scores.get, reverse=True)
        prediction = ranked[0]
        target_action = describe_action(state, target)
        kind = str(target_action["type"])
        target_kind[kind] += 1
        target_counts[f"A{target}"] += 1
        prediction_counts[f"A{prediction}"] += 1
        if prediction == target:
            correct += 1
            correct_kind[kind] += 1
        elif len(errors) < args.max_saved_errors:
            errors.append(
                {
                    "index": index,
                    "battle_id": row.get("battle_id"),
                    "turn_index": row.get("turn_index"),
                    "target": f"A{target}",
                    "prediction": f"A{prediction}",
                    "legal_action_ids": row.get("legal_action_ids"),
                    "scores": {f"A{key}": value for key, value in sorted(scores.items())},
                }
            )
        if len(ranked) > 1:
            margins.append(scores[ranked[0]] - scores[ranked[1]])

        if args.log_every and (index + 1) % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "evaluated": index + 1,
                        "accuracy": correct / (index + 1),
                    }
                ),
                flush=True,
            )

    count = len(dataset)
    report = {
        "examples": count,
        "accuracy": correct / count,
        "correct": correct,
        "accuracy_by_target_kind": {
            kind: correct_kind[kind] / total for kind, total in target_kind.items()
        },
        "target_counts": dict(target_counts),
        "prediction_counts": dict(prediction_counts),
        "average_top1_margin": sum(margins) / len(margins) if margins else None,
        "legality_rate": 1.0,
        "note": (
            "Legality is guaranteed because evaluation ranks only the state's legal actions. "
            "Accuracy measures imitation of the recorded human action, not battle win rate."
        ),
        "errors": errors,
    }
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
        description="Evaluate action imitation with a hard legal-action mask."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter", help="Optional PEFT adapter directory.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-saved-errors", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = evaluate(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
