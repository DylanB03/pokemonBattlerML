from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pokemon_battler.actions import action_label, describe_action
from pokemon_battler.modeling import load_policy_model, score_legal_actions


def _load_state(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "state" in data and isinstance(data["state"], dict):
        return data["state"]
    if not isinstance(data, dict):
        raise ValueError("State file must contain a JSON object")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Choose the highest-scoring legal action for one Metamon state."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter")
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
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
    state = _load_state(args.state_file)
    model, tokenizer, device = load_policy_model(
        args.model,
        adapter_path=args.adapter,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
    )
    model.eval()
    scores = score_legal_actions(
        model,
        tokenizer,
        state,
        device,
        max_length=args.max_length,
    )
    prediction = max(scores, key=scores.get)
    print(
        json.dumps(
            {
                "action_id": prediction,
                "action_label": action_label(prediction),
                "action": describe_action(state, prediction),
                "scores": {action_label(key): value for key, value in sorted(scores.items())},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
