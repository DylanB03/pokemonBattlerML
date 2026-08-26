from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from pokemon_battler.training.distillation import TeacherDistillationCollator, TeacherTurnDataset
from pokemon_battler.models.modeling import _last_hidden_states, indexed_logits_parameter
from pokemon_battler.training.rl_training import _autocast, _load_trainable_policy
from pokemon_battler.data.training_data import InteractionCollator, JsonlOffsetDataset

FROZEN_CACHE_SCHEMA = "frozen-qwen-interaction-v1"
DATA_FILENAME = "tensors.pt"


class FrozenCacheDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, cache_dir: str | Path, *, expected_kind: str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        metadata_path = self.cache_dir / "metadata.json"
        data_path = self.cache_dir / DATA_FILENAME
        if not metadata_path.is_file() or not data_path.is_file():
            raise FileNotFoundError(f"Frozen Qwen cache is incomplete: {self.cache_dir}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema") != FROZEN_CACHE_SCHEMA:
            raise ValueError("Frozen Qwen cache schema does not match this code")
        if expected_kind is not None and self.metadata.get("kind") != expected_kind:
            raise ValueError(
                f"Expected a {expected_kind!r} cache, got {self.metadata.get('kind')!r}"
            )
        self.tensors: dict[str, torch.Tensor] = torch.load(
            data_path, map_location="cpu", weights_only=True
        )
        rows = int(self.metadata["rows"])
        if not self.tensors or any(tensor.shape[0] != rows for tensor in self.tensors.values()):
            raise ValueError("Frozen Qwen cache tensors have inconsistent row counts")

    def __len__(self) -> int:
        return int(self.metadata["rows"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


def _source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def checkpoint_signature(path: Path) -> str:
    digest = hashlib.sha256()
    for name in (
        "training_config.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "interaction_head.safetensors",
    ):
        candidate = path / name
        if not candidate.is_file():
            continue
        stat = candidate.stat()
        digest.update(name.encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _cache_one(
    *,
    kind: str,
    data_file: Path,
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    head: Any,
    device: torch.device,
    dtype: torch.dtype,
    metadata: dict[str, Any],
    logits_parameter: str | None,
    batch_size: int,
    source_checkpoint_signature: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source = JsonlOffsetDataset(data_file)
    if kind == "teacher":
        indices = [
            index
            for index in range(len(source))
            if source[index].get("decision_phase") != "team_preview"
        ]
        dataset: Dataset[dict[str, Any]] = TeacherTurnDataset(
            source, indices, trajectory_cap=24
        )
        collator = TeacherDistillationCollator(
            tokenizer,
            max_length=int(metadata.get("max_length", 4096)),
            truncation="error",
            prompt_format=str(metadata.get("prompt_format", "mechanics-v2")),
        )
    elif kind == "replay":
        dataset = source
        collator = InteractionCollator(
            tokenizer,
            max_length=int(metadata.get("max_length", 4096)),
            truncation="error",
            prompt_format=str(metadata.get("prompt_format", "mechanics-v2")),
        )
    else:
        raise ValueError("Frozen cache kind must be 'teacher' or 'replay'")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    pieces: dict[str, list[torch.Tensor]] = {}
    started = time.monotonic()
    head.eval()
    model.eval()
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader):
            model_batch = {
                key: value.to(device)
                for key, value in cpu_batch.items()
                if key in {"input_ids", "attention_mask"}
            }
            if head.qwen_mode == "none":
                state_hidden = torch.zeros(
                    (int(model_batch["input_ids"].shape[0]), head.qwen_hidden_size),
                    device=device,
                )
            else:
                with _autocast(device, dtype):
                    hidden = _last_hidden_states(
                        model, model_batch, logits_parameter=logits_parameter
                    )
                    positions = model_batch["attention_mask"].sum(1) - 1
                    rows = torch.arange(hidden.shape[0], device=device)
                    state_hidden = hidden[rows, positions]
            cached_batch = {
                key: value
                for key, value in cpu_batch.items()
                if key not in {"input_ids", "attention_mask"}
            }
            cached_batch["qwen_state_hidden"] = state_hidden.detach().cpu()
            for key, value in cached_batch.items():
                value = value.contiguous()
                if value.is_floating_point() and (
                    key.startswith("interaction_") or key == "qwen_state_hidden"
                ):
                    value = value.to(torch.float16)
                pieces.setdefault(key, []).append(value)
            completed = min((batch_index + 1) * batch_size, len(dataset))
            if batch_index == 0 or completed == len(dataset) or completed % 1000 < batch_size:
                print(
                    json.dumps(
                        {
                            "phase": "frozen-qwen-cache",
                            "kind": kind,
                            "rows": completed,
                            "total_rows": len(dataset),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    tensors = {key: torch.cat(values, dim=0) for key, values in pieces.items()}
    output_dir.mkdir(parents=True)
    torch.save(tensors, output_dir / DATA_FILENAME)
    report = {
        "schema": FROZEN_CACHE_SCHEMA,
        "kind": kind,
        "checkpoint_signature": source_checkpoint_signature,
        "source": _source_signature(data_file),
        "rows": len(dataset),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "batch_size": batch_size,
        "tensor_shapes": {key: list(value.shape) for key, value in tensors.items()},
        "tensor_dtypes": {key: str(value.dtype) for key, value in tensors.items()},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_frozen_caches(
    *,
    checkpoint: Path,
    teacher_train: Path,
    teacher_validation: Path,
    replay_train: Path,
    replay_validation: Path,
    output_dir: Path,
    batch_size: int = 4,
    dtype_name: str = "auto",
    load_in_4bit: bool = True,
    local_files_only: bool = True,
    attn_implementation: str = "sdpa",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if batch_size <= 0:
        raise ValueError("Cache batch size must be positive")
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
    logits_parameter = indexed_logits_parameter(model)
    output_dir.mkdir(parents=True)
    source_checkpoint_signature = checkpoint_signature(checkpoint)
    specs = (
        ("teacher-train", "teacher", teacher_train),
        ("teacher-validation", "teacher", teacher_validation),
        ("replay-train", "replay", replay_train),
        ("replay-validation", "replay", replay_validation),
    )
    caches = {}
    for name, kind, data_file in specs:
        caches[name] = _cache_one(
            kind=kind,
            data_file=data_file,
            output_dir=output_dir / name,
            model=model,
            tokenizer=tokenizer,
            head=head,
            device=device,
            dtype=dtype,
            metadata=metadata,
            logits_parameter=logits_parameter,
            batch_size=batch_size,
            source_checkpoint_signature=source_checkpoint_signature,
        )
    report = {
        "schema": FROZEN_CACHE_SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_signature": source_checkpoint_signature,
        "caches": caches,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen Qwen once and cache hidden states for fast head training."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-train", type=Path, required=True)
    parser.add_argument("--teacher-validation", type=Path, required=True)
    parser.add_argument("--replay-train", type=Path, required=True)
    parser.add_argument("--replay-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_frozen_caches(
        checkpoint=args.checkpoint,
        teacher_train=args.teacher_train,
        teacher_validation=args.teacher_validation,
        replay_train=args.replay_train,
        replay_validation=args.replay_validation,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        dtype_name=args.dtype,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )


if __name__ == "__main__":
    main()
