from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from pokemon_battler.actions import action_label
from pokemon_battler.prompting import render_prompt


class JsonlOffsetDataset(Dataset[dict[str, Any]]):
    """Random-access JSONL without holding the prepared states in memory."""

    def __init__(self, path: str | Path, limit: int | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {self.path}")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")

        self.offsets: list[int] = []
        with self.path.open("rb") as stream:
            while limit is None or len(self.offsets) < limit:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f"Dataset contains no examples: {self.path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            line = stream.readline()
        row = json.loads(line)
        if "state" not in row or "action_id" not in row:
            raise ValueError(f"Malformed prepared example at line index {index}")
        return row


class SFTCollator:
    """Build causal-LM batches with loss only on the action and EOS tokens."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int = 4096,
        truncation: str = "error",
    ) -> None:
        if truncation not in {"error", "left"}:
            raise ValueError("truncation must be 'error' or 'left'")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer needs a pad token or EOS token")

    def _encode(self, row: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
        prompt = render_prompt(row["state"])
        expected_target = action_label(int(row["action_id"]))
        target = row.get("target", expected_target)
        if target != expected_target:
            raise ValueError(
                f"Prepared target {target!r} disagrees with action_id {row['action_id']}"
            )

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            raise ValueError(f"Tokenizer produced no IDs for target {target!r}")
        if self.tokenizer.eos_token_id is not None:
            target_ids = [*target_ids, self.tokenizer.eos_token_id]

        allowed_prompt_length = self.max_length - len(target_ids)
        if allowed_prompt_length <= 0:
            raise ValueError("max_length is too short to contain the action target")
        if len(prompt_ids) > allowed_prompt_length:
            if self.truncation == "error":
                raise ValueError(
                    f"Prompt has {len(prompt_ids)} tokens but only "
                    f"{allowed_prompt_length} fit. Increase --max-length or explicitly "
                    "select --truncation left."
                )
            prompt_ids = prompt_ids[-allowed_prompt_length:]

        input_ids = [*prompt_ids, *target_ids]
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask, labels

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(row) for row in rows]
        max_batch_length = max(len(input_ids) for input_ids, _, _ in encoded)

        batch_input_ids: list[list[int]] = []
        batch_attention: list[list[int]] = []
        batch_labels: list[list[int]] = []
        for input_ids, attention_mask, labels in encoded:
            padding = max_batch_length - len(input_ids)
            batch_input_ids.append(input_ids + [self.pad_token_id] * padding)
            batch_attention.append(attention_mask + [0] * padding)
            batch_labels.append(labels + [-100] * padding)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

