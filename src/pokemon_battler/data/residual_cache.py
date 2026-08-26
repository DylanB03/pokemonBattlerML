from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pokemon_battler.core.actions import ACTION_COUNT
from pokemon_battler.training.distillation import TeacherDistillationCollator
from pokemon_battler.data.frozen_cache import checkpoint_signature
from pokemon_battler.models.modeling import indexed_logits_parameter, interaction_outputs
from pokemon_battler.training.rl_training import _autocast, _load_trainable_policy
from pokemon_battler.data.training_data import JsonlOffsetDataset

RESIDUAL_CACHE_SCHEMA = "champion-residual-embeddings-v1"
RESIDUAL_ARRAY_SPECS = {
    "global": ("global.npy", np.float16),
    "candidates": ("candidates.npy", np.float16),
    "legal": ("legal.npy", np.uint8),
    "champion_log_probs": ("champion_log_probs.npy", np.float32),
    "teacher_probabilities": ("teacher_probabilities.npy", np.float32),
    "teacher_actions": ("teacher_actions.npy", np.int8),
    "teacher_confidence": ("teacher_confidence.npy", np.float32),
}


def _open_arrays(output_dir: Path, rows: int, d_model: int) -> dict[str, np.memmap]:
    shapes = {
        "global": (rows, d_model),
        "candidates": (rows, ACTION_COUNT, d_model),
        "legal": (rows, ACTION_COUNT),
        "champion_log_probs": (rows, ACTION_COUNT),
        "teacher_probabilities": (rows, ACTION_COUNT),
        "teacher_actions": (rows,),
        "teacher_confidence": (rows,),
    }
    return {
        name: np.lib.format.open_memmap(
            output_dir / filename,
            mode="w+",
            dtype=dtype,
            shape=shapes[name],
        )
        for name, (filename, dtype) in RESIDUAL_ARRAY_SPECS.items()
    }


def build_residual_teacher_cache(
    *,
    checkpoint: Path,
    data_file: Path,
    output_dir: Path,
    batch_size: int = 8,
    dtype_name: str = "auto",
    load_in_4bit: bool = True,
    local_files_only: bool = True,
    attn_implementation: str = "sdpa",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = JsonlOffsetDataset(data_file)
    if not len(source):
        raise ValueError("Teacher cache source is empty")
    model, tokenizer, head, device, dtype, metadata = _load_trainable_policy(
        checkpoint,
        model_name=None,
        dtype_name=dtype_name,
        load_in_4bit=load_in_4bit,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    model.requires_grad_(False)
    head.requires_grad_(False)
    model.eval()
    head.eval()
    collator = TeacherDistillationCollator(
        tokenizer,
        max_length=int(metadata.get("max_length", 4096)),
        truncation="error",
        prompt_format=str(metadata.get("prompt_format", "mechanics-v2")),
    )
    loader = DataLoader(source, batch_size=batch_size, shuffle=False, collate_fn=collator)
    output_dir.mkdir(parents=True)
    arrays = _open_arrays(output_dir, len(source), int(head.d_model))
    logits_parameter = indexed_logits_parameter(model)
    cursor = 0
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader):
            size = int(cpu_batch["input_ids"].shape[0])
            device_batch = {key: value.to(device) for key, value in cpu_batch.items()}
            with _autocast(device, dtype):
                outputs = interaction_outputs(
                    model, head, device_batch, logits_parameter=logits_parameter
                )
            selection = slice(cursor, cursor + size)
            arrays["global"][selection] = outputs["global_embedding"].float().cpu()
            arrays["candidates"][selection] = outputs[
                "candidate_embeddings"
            ].float().cpu()
            arrays["legal"][selection] = cpu_batch["legal_action_mask"].numpy()
            arrays["champion_log_probs"][selection] = outputs[
                "action_log_probs"
            ].float().cpu()
            arrays["teacher_probabilities"][selection] = cpu_batch[
                "teacher_probabilities"
            ].numpy()
            arrays["teacher_actions"][selection] = cpu_batch[
                "teacher_action_ids"
            ].numpy()
            arrays["teacher_confidence"][selection] = cpu_batch[
                "teacher_confidence"
            ].numpy()
            cursor += size
            if batch_index == 0 or cursor == len(source) or cursor % 1000 < size:
                print(
                    json.dumps(
                        {
                            "phase": "residual-teacher-cache",
                            "rows": cursor,
                            "total_rows": len(source),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    for array in arrays.values():
        array.flush()
    report = {
        "schema": RESIDUAL_CACHE_SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_signature": checkpoint_signature(checkpoint),
        "source": str(data_file.resolve()),
        "rows": len(source),
        "d_model": int(head.d_model),
        "batch_size": batch_size,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "arrays": {
            name: {
                "file": RESIDUAL_ARRAY_SPECS[name][0],
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            for name, array in arrays.items()
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


class ResidualTeacherCache(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.metadata = json.loads(
            (self.cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("schema") != RESIDUAL_CACHE_SCHEMA:
            raise ValueError("Residual teacher cache schema does not match this code")
        self.arrays = {
            name: np.load(self.cache_dir / filename, mmap_mode="r")
            for name, (filename, _dtype) in RESIDUAL_ARRAY_SPECS.items()
        }
        self.rows = int(self.metadata["rows"])
        if any(int(array.shape[0]) != self.rows for array in self.arrays.values()):
            raise ValueError("Residual cache arrays have inconsistent row counts")

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            name: torch.from_numpy(np.array(array[index], copy=True))
            for name, array in self.arrays.items()
        }


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Cache frozen champion embeddings for residual-policy training."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_residual_teacher_cache(
        checkpoint=args.checkpoint,
        data_file=args.data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        dtype_name=args.dtype,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )


if __name__ == "__main__":
    main()
