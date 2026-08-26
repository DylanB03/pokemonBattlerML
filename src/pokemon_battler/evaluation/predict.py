from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pokemon_battler.core.actions import action_label, describe_action
from pokemon_battler.models.modeling import (
    has_candidate_head,
    has_mechanics_head,
    has_policy_head,
    load_candidate_head,
    load_mechanics_head,
    load_policy_head,
    load_policy_model,
    load_training_metadata,
    score_candidate_head_actions,
    score_legal_actions,
    score_mechanics_head_actions,
    score_policy_head_actions,
)
from pokemon_battler.core.prompting import PROMPT_FORMATS
from pokemon_battler.data.training_data import state_with_row_context


def _load_state(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "state" in data and isinstance(data["state"], dict):
        return state_with_row_context(data)
    if not isinstance(data, dict):
        raise ValueError("State file must contain a JSON object")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Choose the highest-scoring legal action for one Metamon state."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter")
    parser.add_argument(
        "--scoring",
        choices=("auto", "generative", "policy-head", "candidate-head", "mechanics-head"),
        default="auto",
    )
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--prompt-format",
        choices=("auto", *PROMPT_FORMATS),
        default="auto",
    )
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
    state = _load_state(args.state_file)
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
    if scoring == "mechanics-head":
        if not args.adapter:
            raise ValueError("Mechanics-head scoring requires --adapter")
        mechanics_head = load_mechanics_head(model, args.adapter, device)
        mechanics_head.eval()
        scores = score_mechanics_head_actions(
            model,
            tokenizer,
            mechanics_head,
            state,
            device,
            max_length=args.max_length,
            prompt_format=prompt_format,
        )
    elif scoring == "candidate-head":
        if not args.adapter:
            raise ValueError("Candidate-head scoring requires --adapter")
        candidate_head = load_candidate_head(model, args.adapter, device)
        candidate_head.eval()
        scores = score_candidate_head_actions(
            model,
            tokenizer,
            candidate_head,
            state,
            device,
            max_length=args.max_length,
            prompt_format=prompt_format,
        )
    elif scoring == "policy-head":
        if not args.adapter:
            raise ValueError("Policy-head scoring requires --adapter")
        policy_head = load_policy_head(model, args.adapter, device)
        policy_head.eval()
        scores = score_policy_head_actions(
            model,
            tokenizer,
            policy_head,
            state,
            device,
            max_length=args.max_length,
            prompt_format=prompt_format,
        )
    else:
        scores = score_legal_actions(
            model,
            tokenizer,
            state,
            device,
            max_length=args.max_length,
            prompt_format=prompt_format,
        )
    prediction = max(scores, key=scores.get)
    print(
        json.dumps(
            {
                "action_id": prediction,
                "action_label": action_label(prediction),
                "action": describe_action(state, prediction),
                "scoring": scoring,
                "prompt_format": prompt_format,
                "scores": {action_label(key): value for key, value in sorted(scores.items())},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
