from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from pokemon_battler.models.interaction_modeling import InteractionPolicyHead

STRUCTURED_HEAD_FILENAME = "structured_policy_head.safetensors"
STRUCTURED_POLICY_SCHEMA = "qwen-structured-sidecar-v1"


def has_structured_head(checkpoint: str | Path | None) -> bool:
    return bool(checkpoint) and (Path(checkpoint) / STRUCTURED_HEAD_FILENAME).is_file()


def set_structured_blend_weight(checkpoint: str | Path, weight: float) -> None:
    if weight < 0:
        raise ValueError("Structured blend weight must be non-negative")
    path = Path(checkpoint) / "training_config.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["structured_policy_blend_weight"] = float(weight)
    metadata["selected_structured_blend_weight"] = float(weight)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, path)


def _head_from_metadata(metadata: dict[str, Any], qwen_hidden_size: int) -> InteractionPolicyHead:
    return InteractionPolicyHead(
        qwen_hidden_size,
        d_model=int(metadata.get("interaction_d_model", 384)),
        attention_heads=int(metadata.get("interaction_attention_heads", 8)),
        layers=int(metadata.get("interaction_layers", 4)),
        feedforward_size=int(metadata.get("interaction_feedforward_size", 1536)),
        dropout=float(metadata.get("interaction_dropout", 0.1)),
        identity_embedding_size=int(metadata.get("interaction_identity_embedding_size", 16)),
        qwen_mode="none",
    )


def initialize_structured_head(
    source_checkpoint: str | Path,
    metadata: dict[str, Any],
    device: torch.device,
) -> InteractionPolicyHead:
    if has_structured_head(source_checkpoint):
        return load_structured_head(source_checkpoint, device)
    source = Path(source_checkpoint) / "interaction_head.safetensors"
    if not source.is_file():
        raise FileNotFoundError(source)
    state = load_safetensors(source, device=str(device))
    projection = state.get("qwen_projection.weight")
    if projection is None or projection.ndim != 2:
        raise ValueError("Interaction checkpoint has no valid Qwen projection")
    head = _head_from_metadata(metadata, int(projection.shape[1])).to(device)
    head.load_state_dict(state, strict=True)
    head.qwen_mode = "none"
    head.qwen_norm.requires_grad_(False)
    head.qwen_projection.requires_grad_(False)
    return head


def save_structured_head(head: InteractionPolicyHead, output_dir: str | Path) -> None:
    state = {
        key: value.detach().cpu().float().contiguous() for key, value in head.state_dict().items()
    }
    save_safetensors(state, Path(output_dir) / STRUCTURED_HEAD_FILENAME)


def load_structured_head(checkpoint: str | Path, device: torch.device) -> InteractionPolicyHead:
    root = Path(checkpoint)
    metadata = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
    if metadata.get("structured_policy_schema") != STRUCTURED_POLICY_SCHEMA:
        raise ValueError("Structured policy schema does not match this code")
    state = load_safetensors(root / STRUCTURED_HEAD_FILENAME, device=str(device))
    projection = state.get("qwen_projection.weight")
    if projection is None:
        raise ValueError("Structured policy has no Qwen projection shape metadata")
    head = _head_from_metadata(metadata, int(projection.shape[1])).to(device)
    head.load_state_dict(state, strict=True)
    head.eval()
    return head
