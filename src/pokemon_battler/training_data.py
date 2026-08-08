from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from pokemon_battler.actions import ACTION_COUNT, action_label, legal_action_ids
from pokemon_battler.mechanics import MECHANICS_SCHEMA as LEGACY_MECHANICS_SCHEMA
from pokemon_battler.mechanics_v2 import MECHANICS_SCHEMA as DEFAULT_MECHANICS_SCHEMA
from pokemon_battler.prompting import encode_candidate_prompt, render_prompt


def _mechanics_spec(schema: str) -> tuple[int, int, Any]:
    if schema == LEGACY_MECHANICS_SCHEMA:
        from pokemon_battler.mechanics import (
            MECHANICS_FEATURE_COUNT,
            candidate_feature_matrix,
        )

        return MECHANICS_FEATURE_COUNT, 0, candidate_feature_matrix
    if schema == DEFAULT_MECHANICS_SCHEMA:
        from pokemon_battler.mechanics_v2 import (
            MECHANICS_FEATURE_COUNT,
            MECHANICS_IDENTITY_COUNT,
            candidate_feature_matrix,
        )

        return MECHANICS_FEATURE_COUNT, MECHANICS_IDENTITY_COUNT, candidate_feature_matrix
    raise ValueError(f"Unsupported mechanics schema: {schema!r}")


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


class MechanicsCacheDataset(Dataset[dict[str, Any]]):
    """Attach a memory-mapped mechanics matrix to rows without adding prompt tokens."""

    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        cache_path: str | Path,
        *,
        mechanics_schema: str = DEFAULT_MECHANICS_SCHEMA,
    ) -> None:
        self.dataset = dataset
        self.cache_path = Path(cache_path)
        self.mechanics_schema = mechanics_schema
        self.feature_count, _, _ = _mechanics_spec(mechanics_schema)
        self.metadata_path = self.cache_path.with_suffix(self.cache_path.suffix + ".json")
        if not self.cache_path.is_file() or not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Mechanics cache or metadata is missing: {self.cache_path}"
            )
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != mechanics_schema:
            raise ValueError(
                f"Mechanics cache uses {metadata.get('schema')!r}; "
                f"expected {mechanics_schema!r}"
            )
        if int(metadata.get("feature_count", -1)) != self.feature_count:
            raise ValueError("Mechanics cache feature count does not match this code")
        if int(metadata.get("rows", -1)) < len(dataset):
            raise ValueError("Mechanics cache has fewer rows than its JSONL dataset")
        self.metadata = metadata
        self._features: np.ndarray[Any, Any] | None = None

    def __len__(self) -> int:
        return len(self.dataset)

    def _matrix(self) -> np.ndarray[Any, Any]:
        if self._features is None:
            matrix = np.load(self.cache_path, mmap_mode="r")
            expected_tail = (ACTION_COUNT, self.feature_count)
            if matrix.ndim != 3 or tuple(matrix.shape[1:]) != expected_tail:
                raise ValueError(
                    f"Invalid mechanics cache shape {matrix.shape}; expected [rows, 13, "
                    f"{self.feature_count}]"
                )
            self._features = matrix
        return self._features

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[index]
        return row | {"_mechanics_features": self._matrix()[index]}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_features"] = None
        return state


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


class MechanicsCollator:
    """Build compact state batches plus numeric and categorical candidate mechanics."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int = 4096,
        truncation: str = "error",
        prompt_format: str = "mechanics-v2",
        mechanics_schema: str = DEFAULT_MECHANICS_SCHEMA,
    ) -> None:
        if truncation not in {"error", "left"}:
            raise ValueError("truncation must be 'error' or 'left'")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.prompt_format = prompt_format
        self.mechanics_schema = mechanics_schema
        self.feature_count, self.identity_count, self.feature_builder = _mechanics_spec(
            mechanics_schema
        )
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer needs a pad token or EOS token")

    def _encode(
        self,
        row: dict[str, Any],
    ) -> tuple[list[int], int, list[bool], torch.Tensor, torch.Tensor | None]:
        state = state_with_row_context(row)
        action_id = int(row["action_id"])
        legal = legal_action_ids(state)
        prepared_legal = row.get("legal_action_ids")
        if prepared_legal is not None and [int(value) for value in prepared_legal] != legal:
            raise ValueError("Prepared legal_action_ids disagree with the rendered state")
        if action_id not in legal:
            raise ValueError(f"Target A{action_id} is absent from the legal-action mask")
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
        raw_features = row.get("_mechanics_features")
        if raw_features is None:
            raw_features = self.feature_builder(state)
        feature_tensor = torch.as_tensor(raw_features, dtype=torch.float32)
        if tuple(feature_tensor.shape) != (ACTION_COUNT, self.feature_count):
            raise ValueError(
                f"Mechanics features have shape {tuple(feature_tensor.shape)}; expected "
                f"({ACTION_COUNT}, {self.feature_count})"
            )
        identity_tensor: torch.Tensor | None = None
        if self.mechanics_schema == DEFAULT_MECHANICS_SCHEMA:
            from pokemon_battler.mechanics_v2 import candidate_identity_matrix

            identity_tensor = torch.tensor(
                candidate_identity_matrix(state),
                dtype=torch.long,
            )
            if tuple(identity_tensor.shape) != (ACTION_COUNT, self.identity_count):
                raise ValueError(
                    f"Mechanics identities have shape {tuple(identity_tensor.shape)}; "
                    f"expected ({ACTION_COUNT}, {self.identity_count})"
                )
        legal_mask = [candidate in legal for candidate in range(ACTION_COUNT)]
        return prompt_ids, action_id, legal_mask, feature_tensor, identity_tensor

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(row) for row in rows]
        max_batch_length = max(len(prompt_ids) for prompt_ids, _, _, _, _ in encoded)
        input_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        action_ids: list[int] = []
        legal_masks: list[list[bool]] = []
        feature_rows: list[torch.Tensor] = []
        identity_rows: list[torch.Tensor] = []
        for prompt_ids, action_id, legal_mask, features, identities in encoded:
            padding = max_batch_length - len(prompt_ids)
            input_rows.append(prompt_ids + [self.pad_token_id] * padding)
            attention_rows.append([1] * len(prompt_ids) + [0] * padding)
            action_ids.append(action_id)
            legal_masks.append(legal_mask)
            feature_rows.append(features)
            if identities is not None:
                identity_rows.append(identities)
        batch = {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "action_ids": torch.tensor(action_ids, dtype=torch.long),
            "legal_action_mask": torch.tensor(legal_masks, dtype=torch.bool),
            "mechanics_features": torch.stack(feature_rows),
        }
        if identity_rows:
            batch["mechanics_identity_ids"] = torch.stack(identity_rows)
        return batch


class InteractionCollator:
    """Build Qwen state inputs plus cached structured interaction tensors."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int = 4096,
        truncation: str = "error",
        prompt_format: str = "mechanics-v2",
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

    def _encode(self, row: dict[str, Any]) -> dict[str, Any]:
        from pokemon_battler.interaction_features import (
            build_interaction_features,
            validate_interaction_row,
        )

        validate_interaction_row(row)
        state = row["state"]
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
        features = row.get("_interaction_features")
        if features is None:
            features = build_interaction_features(row)
        action_id = int(row["action_id"])
        outcome = str(row.get("outcome") or "").upper()
        battle_decisions = max(int(row.get("battle_decision_count", 1)), 1)
        return {
            "prompt_ids": prompt_ids,
            "action_id": action_id,
            "family_id": 0 if action_id < 4 else 1 if action_id < 9 else 2,
            "value_target": 1.0 if outcome == "WIN" else 0.0 if outcome == "LOSS" else -1.0,
            # Keep the auxiliary loss near unit scale while preventing long
            # battles from contributing one full value target per decision.
            "value_weight": min(max(32.0 / battle_decisions, 0.25), 4.0),
            "features": features,
        }

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(row) for row in rows]
        max_batch_length = max(len(item["prompt_ids"]) for item in encoded)
        input_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        for item in encoded:
            prompt_ids = item["prompt_ids"]
            padding = max_batch_length - len(prompt_ids)
            input_rows.append(prompt_ids + [self.pad_token_id] * padding)
            attention_rows.append([1] * len(prompt_ids) + [0] * padding)

        def stack_feature(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.stack(
                [torch.tensor(item["features"][name], dtype=dtype) for item in encoded]
            )

        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "action_ids": torch.tensor(
                [item["action_id"] for item in encoded], dtype=torch.long
            ),
            "action_family_ids": torch.tensor(
                [item["family_id"] for item in encoded], dtype=torch.long
            ),
            "value_targets": torch.tensor(
                [item["value_target"] for item in encoded], dtype=torch.float32
            ),
            "value_weights": torch.tensor(
                [item["value_weight"] for item in encoded], dtype=torch.float32
            ),
            "interaction_global_numeric": stack_feature("global_numeric", torch.float32),
            "interaction_global_ids": stack_feature("global_ids", torch.long),
            "interaction_pokemon_numeric": stack_feature("pokemon_numeric", torch.float32),
            "interaction_pokemon_ids": stack_feature("pokemon_ids", torch.long),
            "interaction_pokemon_mask": stack_feature("pokemon_mask", torch.bool),
            "interaction_candidate_numeric": stack_feature(
                "candidate_numeric", torch.float32
            ),
            "interaction_candidate_ids": stack_feature("candidate_ids", torch.long),
            "legal_action_mask": stack_feature("candidate_mask", torch.bool),
            "interaction_candidate_actor_slot": stack_feature(
                "candidate_actor_slot", torch.long
            ),
            "interaction_history_numeric": stack_feature("history_numeric", torch.float32),
            "interaction_history_ids": stack_feature("history_ids", torch.long),
            "interaction_history_mask": stack_feature("history_mask", torch.bool),
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
