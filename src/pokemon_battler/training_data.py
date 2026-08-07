from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import torch
from torch.utils.data import Dataset

from pokemon_battler.actions import ACTION_COUNT, action_label, legal_action_ids
from pokemon_battler.prompting import encode_candidate_prompt, render_prompt


def state_with_row_context(row: dict[str, Any]) -> dict[str, Any]:
    """Add decision-time metadata available in legacy prepared rows."""
    state = row["state"]
    additions: dict[str, Any] = {}
    if "turn_index" not in state and row.get("turn_index") is not None:
        additions["turn_index"] = int(row["turn_index"])
    if "player_remaining" not in state:
        active = state.get("player_active_pokemon") or {}
        active_alive = (
            float(active.get("hp_pct", 0) or 0) > 0 and active.get("status") != "fnt"
        )
        additions["player_remaining"] = len(state.get("available_switches") or []) + int(
            active_alive
        )
    if not additions:
        return state
    return state | additions


class JsonlOffsetDataset(Dataset[dict[str, Any]]):
    """Random-access JSONL without holding the prepared states in memory."""

    def __init__(self, path: str | Path, limit: int | None = None) -> None:
        self.path = Path(path)
        self._stream: BinaryIO | None = None
        self._stream_process_id: int | None = None
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

    def _reader(self) -> BinaryIO:
        process_id = os.getpid()
        if self._stream is None or self._stream_process_id != process_id:
            if self._stream is not None:
                self._stream.close()
            self._stream = self.path.open("rb")
            self._stream_process_id = process_id
        return self._stream

    def __getitem__(self, index: int) -> dict[str, Any]:
        stream = self._reader()
        stream.seek(self.offsets[index])
        line = stream.readline()
        row = json.loads(line)
        if "state" not in row or "action_id" not in row:
            raise ValueError(f"Malformed prepared example at line index {index}")
        return row

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_stream"] = None
        state["_stream_process_id"] = None
        return state

    def __del__(self) -> None:
        stream = getattr(self, "_stream", None)
        if stream is not None:
            stream.close()


class SFTCollator:
    """Build causal-LM batches with loss only on the action and EOS tokens."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int = 4096,
        truncation: str = "error",
        prompt_format: str = "verbose-v1",
    ) -> None:
        if truncation not in {"error", "left"}:
            raise ValueError("truncation must be 'error' or 'left'")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.prompt_format = prompt_format
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer needs a pad token or EOS token")

    def _encode(self, row: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
        prompt = render_prompt(state_with_row_context(row), self.prompt_format)
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


class PolicyCollator:
    """Build prompt-only batches for masked 13-way action classification."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int = 4096,
        truncation: str = "error",
        prompt_format: str = "verbose-v1",
    ) -> None:
        if truncation not in {"error", "left"}:
            raise ValueError("truncation must be 'error' or 'left'")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.prompt_format = prompt_format
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer needs a pad token or EOS token")

    def _encode(self, row: dict[str, Any]) -> tuple[list[int], int, list[bool]]:
        state = state_with_row_context(row)
        action_id = int(row["action_id"])
        prompt_ids = self.tokenizer.encode(
            render_prompt(state, self.prompt_format),
            add_special_tokens=True,
        )
        if len(prompt_ids) > self.max_length:
            if self.truncation == "error":
                raise ValueError(
                    f"Prompt has {len(prompt_ids)} tokens and exceeds "
                    f"max_length={self.max_length}. Increase --max-length or explicitly "
                    "select --truncation left."
                )
            prompt_ids = prompt_ids[-self.max_length :]

        legal = legal_action_ids(state)
        prepared_legal = row.get("legal_action_ids")
        if prepared_legal is not None and [int(value) for value in prepared_legal] != legal:
            raise ValueError("Prepared legal_action_ids disagree with the rendered state")
        if action_id not in legal:
            raise ValueError(f"Target A{action_id} is absent from the legal-action mask")
        legal_mask = [action_index in legal for action_index in range(ACTION_COUNT)]
        return prompt_ids, action_id, legal_mask

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(row) for row in rows]
        max_batch_length = max(len(prompt_ids) for prompt_ids, _, _ in encoded)
        input_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        action_ids: list[int] = []
        legal_masks: list[list[bool]] = []
        for prompt_ids, action_id, legal_mask in encoded:
            padding = max_batch_length - len(prompt_ids)
            input_rows.append(prompt_ids + [self.pad_token_id] * padding)
            attention_rows.append([1] * len(prompt_ids) + [0] * padding)
            action_ids.append(action_id)
            legal_masks.append(legal_mask)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "action_ids": torch.tensor(action_ids, dtype=torch.long),
            "legal_action_mask": torch.tensor(legal_masks, dtype=torch.bool),
        }


class CandidateCollator:
    """Build batches with one hidden-state marker position per legal candidate."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int = 4096,
        truncation: str = "error",
        prompt_format: str = "compact-v1",
        shuffle_candidates: bool = False,
    ) -> None:
        if truncation not in {"error", "left"}:
            raise ValueError("truncation must be 'error' or 'left'")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.prompt_format = prompt_format
        self.shuffle_candidates = shuffle_candidates
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer needs a pad token or EOS token")

    def _encode(self, row: dict[str, Any]) -> tuple[list[int], int, list[bool], list[int]]:
        state = state_with_row_context(row)
        action_id = int(row["action_id"])
        legal = legal_action_ids(state)
        candidate_order = list(legal)
        if self.shuffle_candidates:
            random.shuffle(candidate_order)
        input_ids, positions_by_action = encode_candidate_prompt(
            self.tokenizer,
            state,
            self.prompt_format,
            candidate_order,
        )
        prepared_legal = row.get("legal_action_ids")
        if prepared_legal is not None and [int(value) for value in prepared_legal] != legal:
            raise ValueError("Prepared legal_action_ids disagree with the rendered state")
        if action_id not in legal:
            raise ValueError(f"Target A{action_id} is absent from the legal-action mask")

        removed = max(len(input_ids) - self.max_length, 0)
        if removed:
            if self.truncation == "error":
                raise ValueError(
                    f"Prompt has {len(input_ids)} tokens and exceeds "
                    f"max_length={self.max_length}. Increase --max-length or explicitly "
                    "select --truncation left."
                )
            input_ids = input_ids[removed:]

        candidate_positions = [-1] * ACTION_COUNT
        for legal_action in legal:
            shifted_position = positions_by_action[legal_action] - removed
            if shifted_position < 0:
                raise ValueError(
                    "Left truncation removed a legal candidate marker; use a compact "
                    "prompt or increase --max-length"
                )
            candidate_positions[legal_action] = shifted_position
        legal_mask = [action_index in legal for action_index in range(ACTION_COUNT)]
        return input_ids, action_id, legal_mask, candidate_positions

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(row) for row in rows]
        max_batch_length = max(len(input_ids) for input_ids, _, _, _ in encoded)
        input_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        action_ids: list[int] = []
        legal_masks: list[list[bool]] = []
        candidate_positions: list[list[int]] = []
        for input_ids, action_id, legal_mask, positions in encoded:
            padding = max_batch_length - len(input_ids)
            input_rows.append(input_ids + [self.pad_token_id] * padding)
            attention_rows.append([1] * len(input_ids) + [0] * padding)
            action_ids.append(action_id)
            legal_masks.append(legal_mask)
            candidate_positions.append(positions)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "action_ids": torch.tensor(action_ids, dtype=torch.long),
            "legal_action_mask": torch.tensor(legal_masks, dtype=torch.bool),
            "candidate_positions": torch.tensor(candidate_positions, dtype=torch.long),
        }
