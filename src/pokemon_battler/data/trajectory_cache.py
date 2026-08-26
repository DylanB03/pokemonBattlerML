from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from pokemon_battler.core.actions import ACTION_COUNT
from pokemon_battler.data.frozen_cache import checkpoint_signature
from pokemon_battler.models.modeling import indexed_logits_parameter, interaction_outputs
from pokemon_battler.training.rl_training import _autocast, _load_trainable_policy
from pokemon_battler.data.training_data import InteractionCollator, JsonlOffsetDataset
from pokemon_battler.data.trajectory_prepare import (
    PREVIOUS_ACTION_SENTINEL,
    TRAJECTORY_SCHEMA_VERSION,
)

TRAJECTORY_CACHE_SCHEMA = "encoded-trajectory-interaction-v1"

ARRAY_SPECS = {
    "global": ("global.npy", np.float16),
    "candidates": ("candidates.npy", np.float16),
    "legal": ("legal.npy", np.uint8),
    "actions": ("actions.npy", np.int8),
    "rewards": ("rewards.npy", np.float32),
    "dones": ("dones.npy", np.uint8),
    "transition_steps": ("transition_steps.npy", np.int16),
    "previous_actions": ("previous_actions.npy", np.int8),
    "previous_rewards": ("previous_rewards.npy", np.float32),
    "outcomes": ("outcomes.npy", np.int8),
}


class TrajectoryCacheCollator:
    def __init__(self, base: InteractionCollator) -> None:
        self.base = base

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.base(rows)
        batch.update(
            {
                "trajectory_rewards": torch.tensor(
                    [float(row["reward"]) for row in rows], dtype=torch.float32
                ),
                "trajectory_dones": torch.tensor(
                    [bool(row["done"]) for row in rows], dtype=torch.bool
                ),
                "trajectory_steps": torch.tensor(
                    [int(row["transition_steps"]) for row in rows], dtype=torch.long
                ),
                "trajectory_previous_actions": torch.tensor(
                    [
                        int(row.get("previous_action_id", PREVIOUS_ACTION_SENTINEL))
                        for row in rows
                    ],
                    dtype=torch.long,
                ),
                "trajectory_previous_rewards": torch.tensor(
                    [float(row.get("previous_reward", 0.0)) for row in rows],
                    dtype=torch.float32,
                ),
                "trajectory_outcomes": torch.tensor(
                    [
                        1
                        if str(row.get("outcome") or "").upper() == "WIN"
                        else -1
                        if str(row.get("outcome") or "").upper() == "LOSS"
                        else 0
                        for row in rows
                    ],
                    dtype=torch.int8,
                ),
            }
        )
        return batch


def _trajectory_spans(dataset: JsonlOffsetDataset) -> tuple[list[dict[str, Any]], float]:
    spans: list[dict[str, Any]] = []
    current_id: str | None = None
    current_start = 0
    previous_position = -1
    seen: set[str] = set()
    reward_gamma: float | None = None
    for index in range(len(dataset)):
        row = dataset[index]
        if int(row.get("schema_version", -1)) != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError("Trajectory cache requires schema-version 4 prepared rows")
        row_gamma = float(row.get("reward_gamma", -1.0))
        if not 0 < row_gamma <= 1:
            raise ValueError(f"Row {index} has no valid reward_gamma")
        if reward_gamma is None:
            reward_gamma = row_gamma
        elif not np.isclose(reward_gamma, row_gamma):
            raise ValueError("Trajectory rows use inconsistent reward discount factors")
        identifier = str(row.get("trajectory_id") or "")
        if not identifier:
            raise ValueError(f"Row {index} has no trajectory_id")
        position = int(row.get("trajectory_position", -1))
        if identifier != current_id:
            if current_id is not None:
                spans.append(
                    {"trajectory_id": current_id, "start": current_start, "end": index}
                )
            if identifier in seen:
                raise ValueError(f"Trajectory {identifier!r} is not contiguous")
            seen.add(identifier)
            current_id = identifier
            current_start = index
            previous_position = -1
        if position != previous_position + 1:
            raise ValueError(
                f"Trajectory {identifier!r} positions are not consecutive at row {index}"
            )
        previous_position = position
    assert current_id is not None
    spans.append({"trajectory_id": current_id, "start": current_start, "end": len(dataset)})
    assert reward_gamma is not None
    return spans, reward_gamma


def _open_arrays(output_dir: Path, rows: int, d_model: int) -> dict[str, np.memmap]:
    shapes = {
        "global": (rows, d_model),
        "candidates": (rows, ACTION_COUNT, d_model),
        "legal": (rows, ACTION_COUNT),
        "actions": (rows,),
        "rewards": (rows,),
        "dones": (rows,),
        "transition_steps": (rows,),
        "previous_actions": (rows,),
        "previous_rewards": (rows,),
        "outcomes": (rows,),
    }
    return {
        name: np.lib.format.open_memmap(
            output_dir / filename,
            mode="w+",
            dtype=dtype,
            shape=shapes[name],
        )
        for name, (filename, dtype) in ARRAY_SPECS.items()
    }


def build_trajectory_cache(
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
    spans, reward_gamma = _trajectory_spans(source)
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
    collator = TrajectoryCacheCollator(
        InteractionCollator(
            tokenizer,
            max_length=int(metadata.get("max_length", 4096)),
            truncation="error",
            prompt_format=str(metadata.get("prompt_format", "mechanics-v2")),
        )
    )
    loader = DataLoader(source, batch_size=batch_size, shuffle=False, collate_fn=collator)
    output_dir.mkdir(parents=True)
    arrays = _open_arrays(output_dir, len(source), int(head.d_model))
    logits_parameter = indexed_logits_parameter(model)
    started = time.monotonic()
    cursor = 0
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader):
            size = int(cpu_batch["input_ids"].shape[0])
            model_batch = {key: value.to(device) for key, value in cpu_batch.items()}
            with _autocast(device, dtype):
                outputs = interaction_outputs(
                    model, head, model_batch, logits_parameter=logits_parameter
                )
            selection = slice(cursor, cursor + size)
            arrays["global"][selection] = (
                outputs["global_embedding"].detach().float().cpu().numpy()
            )
            arrays["candidates"][selection] = (
                outputs["candidate_embeddings"].detach().float().cpu().numpy()
            )
            arrays["legal"][selection] = cpu_batch["legal_action_mask"].numpy()
            arrays["actions"][selection] = cpu_batch["action_ids"].numpy()
            arrays["rewards"][selection] = cpu_batch["trajectory_rewards"].numpy()
            arrays["dones"][selection] = cpu_batch["trajectory_dones"].numpy()
            arrays["transition_steps"][selection] = cpu_batch[
                "trajectory_steps"
            ].numpy()
            arrays["previous_actions"][selection] = cpu_batch[
                "trajectory_previous_actions"
            ].numpy()
            arrays["previous_rewards"][selection] = cpu_batch[
                "trajectory_previous_rewards"
            ].numpy()
            arrays["outcomes"][selection] = cpu_batch["trajectory_outcomes"].numpy()
            cursor += size
            if batch_index == 0 or cursor == len(source) or cursor % 1000 < size:
                print(
                    json.dumps(
                        {
                            "phase": "trajectory-cache",
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
        "schema": TRAJECTORY_CACHE_SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_signature": checkpoint_signature(checkpoint),
        "source": str(data_file.resolve()),
        "rows": len(source),
        "trajectories": len(spans),
        "d_model": int(head.d_model),
        "reward_gamma": reward_gamma,
        "batch_size": batch_size,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "arrays": {
            name: {
                "file": ARRAY_SPECS[name][0],
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            for name, array in arrays.items()
        },
    }
    (output_dir / "spans.json").write_text(
        json.dumps(spans, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


class EncodedTrajectoryCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        metadata_path = self.cache_dir / "metadata.json"
        spans_path = self.cache_dir / "spans.json"
        if not metadata_path.is_file() or not spans_path.is_file():
            raise FileNotFoundError(f"Incomplete trajectory cache: {self.cache_dir}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema") != TRAJECTORY_CACHE_SCHEMA:
            raise ValueError("Trajectory cache schema does not match this code")
        self.spans: list[dict[str, Any]] = json.loads(spans_path.read_text(encoding="utf-8"))
        self.arrays = {
            name: np.load(self.cache_dir / filename, mmap_mode="r")
            for name, (filename, _dtype) in ARRAY_SPECS.items()
        }
        rows = int(self.metadata["rows"])
        if any(int(array.shape[0]) != rows for array in self.arrays.values()):
            raise ValueError("Trajectory cache arrays have inconsistent row counts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze Qwen plus the interaction encoder into per-turn trajectory vectors."
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
    build_trajectory_cache(
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
