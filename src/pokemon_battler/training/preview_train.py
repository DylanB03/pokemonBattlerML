from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pokemon_battler.training.preview_training import train_team_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the learned team-preview lead head.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = train_team_preview(
        checkpoint=args.checkpoint,
        teacher_file=args.teacher_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        dtype_name=args.dtype,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
