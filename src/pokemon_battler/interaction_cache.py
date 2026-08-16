from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from pokemon_battler.actions import ACTION_COUNT
from pokemon_battler.interaction_features import (
    GLOBAL_ID_FIELDS,
    GLOBAL_NUMERIC_NAMES,
    HISTORY_ID_FIELDS,
    HISTORY_NUMERIC_NAMES,
    INTERACTION_CACHE_SCHEMA,
    INTERACTION_VOCAB_SIZES,
    POKEMON_ID_FIELDS,
    POKEMON_NUMERIC_NAMES,
    SUPPORTED_PREPARED_SCHEMA_VERSIONS,
    build_interaction_features,
)
from pokemon_battler.mechanics_v2 import (
    MECHANICS_FEATURE_COUNT,
    MECHANICS_FEATURE_NAMES,
    MECHANICS_IDENTITY_COUNT,
    MECHANICS_IDENTITY_NAMES,
)
from pokemon_battler.training_data import JsonlOffsetDataset

ARRAY_SPECS: dict[str, tuple[Any, tuple[int, ...]]] = {
    "global_numeric": (np.float16, (len(GLOBAL_NUMERIC_NAMES),)),
    "global_ids": (np.uint32, (len(GLOBAL_ID_FIELDS),)),
    "pokemon_numeric": (np.float16, (12, len(POKEMON_NUMERIC_NAMES))),
    "pokemon_ids": (np.uint32, (12, len(POKEMON_ID_FIELDS))),
    "pokemon_mask": (np.uint8, (12,)),
    "candidate_numeric": (np.float16, (ACTION_COUNT, MECHANICS_FEATURE_COUNT)),
    "candidate_ids": (np.uint32, (ACTION_COUNT, MECHANICS_IDENTITY_COUNT)),
    "candidate_mask": (np.uint8, (ACTION_COUNT,)),
    "candidate_actor_slot": (np.int8, (ACTION_COUNT,)),
    "history_numeric": (np.float16, (4, len(HISTORY_NUMERIC_NAMES))),
    "history_ids": (np.uint32, (4, len(HISTORY_ID_FIELDS))),
    "history_mask": (np.uint8, (4,)),
}


def _schema_descriptor() -> dict[str, Any]:
    return {
        "arrays": {
            name: {"dtype": np.dtype(dtype).name, "tail_shape": list(tail)}
            for name, (dtype, tail) in ARRAY_SPECS.items()
        },
        "feature_names": {
            "global_numeric": list(GLOBAL_NUMERIC_NAMES),
            "global_ids": [name for name, _ in GLOBAL_ID_FIELDS],
            "pokemon_numeric": list(POKEMON_NUMERIC_NAMES),
            "pokemon_ids": [name for name, _ in POKEMON_ID_FIELDS],
            "candidate_numeric": list(MECHANICS_FEATURE_NAMES),
            "candidate_ids": list(MECHANICS_IDENTITY_NAMES),
            "history_numeric": list(HISTORY_NUMERIC_NAMES),
            "history_ids": [name for name, _ in HISTORY_ID_FIELDS],
        },
        "vocabulary_sizes": dict(sorted(INTERACTION_VOCAB_SIZES.items())),
    }


def _schema_fingerprint() -> str:
    payload = json.dumps(
        _schema_descriptor(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def default_interaction_cache_path(data_file: str | Path) -> Path:
    path = Path(data_file)
    stem = path.name.removesuffix(".jsonl")
    return path.with_name(f"{stem}.{INTERACTION_CACHE_SCHEMA}")


def _source_signature(data_file: Path, rows: int) -> dict[str, Any]:
    stat = data_file.stat()
    return {
        "path": str(data_file),
        "bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "rows": rows,
    }


def _array_path(cache_dir: Path, name: str) -> Path:
    dtype = np.dtype(ARRAY_SPECS[name][0]).name
    return cache_dir / f"{name}.{dtype}.npy"


def interaction_cache_is_current(data_file: str | Path, cache_dir: str | Path) -> bool:
    source = Path(data_file)
    cache = Path(cache_dir)
    metadata_path = cache / "metadata.json"
    if not source.is_file() or not cache.is_dir() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = int(metadata["rows"])
        if metadata.get("cache_schema") != INTERACTION_CACHE_SCHEMA:
            return False
        if int(metadata.get("prepared_schema_version", -1)) not in (
            SUPPORTED_PREPARED_SCHEMA_VERSIONS
        ):
            return False
        if metadata.get("schema_fingerprint") != _schema_fingerprint():
            return False
        if metadata.get("source") != _source_signature(source, rows):
            return False
        for name, (dtype, tail) in ARRAY_SPECS.items():
            array = np.load(_array_path(cache, name), mmap_mode="r")
            if array.dtype != np.dtype(dtype) or tuple(array.shape) != (rows, *tail):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_interaction_cache(
    data_file: str | Path,
    cache_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
    progress_every: int = 10_000,
) -> dict[str, Any]:
    source = Path(data_file)
    output = Path(cache_dir) if cache_dir is not None else default_interaction_cache_path(source)
    if interaction_cache_is_current(source, output) and not overwrite:
        return json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    partial = output.with_name(output.name + ".partial")
    if (output.exists() or partial.exists()) and not overwrite:
        raise FileExistsError(
            f"Interaction cache exists but is stale, invalid, or incomplete: {output}. "
            "Pass --overwrite to rebuild it."
        )
    if overwrite:
        if output.is_dir():
            shutil.rmtree(output)
        elif output.exists():
            output.unlink()
        if partial.is_dir():
            shutil.rmtree(partial)
        elif partial.exists():
            partial.unlink()

    dataset = JsonlOffsetDataset(source)
    prepared_schema_version = int(dataset[0].get("schema_version", -1))
    if prepared_schema_version not in SUPPORTED_PREPARED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported prepared schema version: {prepared_schema_version}"
        )
    partial.mkdir(parents=True)
    arrays: dict[str, np.ndarray[Any, Any]] = {
        name: np.lib.format.open_memmap(
            _array_path(partial, name),
            mode="w+",
            dtype=dtype,
            shape=(len(dataset), *tail),
        )
        for name, (dtype, tail) in ARRAY_SPECS.items()
    }
    started = time.monotonic()
    try:
        for index in range(len(dataset)):
            features = build_interaction_features(dataset[index])
            for name, array in arrays.items():
                array[index] = np.asarray(features[name], dtype=array.dtype)
            if progress_every and (index + 1) % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "phase": "interaction-cache",
                            "rows": index + 1,
                            "total_rows": len(dataset),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
        for array in arrays.values():
            array.flush()
        arrays.clear()
        metadata = {
            "cache_schema": INTERACTION_CACHE_SCHEMA,
            "prepared_schema_version": prepared_schema_version,
            "schema_fingerprint": _schema_fingerprint(),
            "schema_descriptor": _schema_descriptor(),
            "rows": len(dataset),
            "source": _source_signature(source, len(dataset)),
            "arrays": {
                name: {
                    "file": _array_path(partial, name).name,
                    "dtype": np.dtype(dtype).name,
                    "shape": [len(dataset), *tail],
                }
                for name, (dtype, tail) in ARRAY_SPECS.items()
            },
            "feature_names": {
                "global_numeric": list(GLOBAL_NUMERIC_NAMES),
                "global_ids": [name for name, _ in GLOBAL_ID_FIELDS],
                "pokemon_numeric": list(POKEMON_NUMERIC_NAMES),
                "pokemon_ids": [name for name, _ in POKEMON_ID_FIELDS],
                "candidate_numeric_count": MECHANICS_FEATURE_COUNT,
                "candidate_ids": list(MECHANICS_IDENTITY_NAMES),
                "history_numeric": list(HISTORY_NUMERIC_NAMES),
                "history_ids": [name for name, _ in HISTORY_ID_FIELDS],
            },
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }
        (partial / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, output)
        return metadata
    except BaseException:
        arrays.clear()
        if partial.is_dir():
            shutil.rmtree(partial)
        raise


class InteractionCacheDataset(Dataset[dict[str, Any]]):
    """Attach validated interaction tensors to prepared schema-3 rows."""

    def __init__(self, dataset: Dataset[dict[str, Any]], cache_dir: str | Path) -> None:
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Interaction cache metadata is missing: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("cache_schema") != INTERACTION_CACHE_SCHEMA:
            raise ValueError("Interaction cache schema does not match this code")
        if int(self.metadata.get("prepared_schema_version", -1)) not in (
            SUPPORTED_PREPARED_SCHEMA_VERSIONS
        ):
            raise ValueError("Interaction cache prepared-row schema does not match this code")
        if self.metadata.get("schema_fingerprint") != _schema_fingerprint():
            raise ValueError("Interaction cache feature schema does not match this code")
        rows = int(self.metadata.get("rows", -1))
        if rows < len(dataset):
            raise ValueError("Interaction cache has fewer rows than its JSONL dataset")
        data_path = getattr(dataset, "path", None)
        if data_path is not None:
            recorded = dict(self.metadata.get("source") or {})
            current = _source_signature(Path(data_path), rows)
            if recorded != current:
                raise ValueError("Interaction cache source signature is stale")
        self._arrays: dict[str, np.ndarray[Any, Any]] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def _array(self, name: str) -> np.ndarray[Any, Any]:
        if name not in self._arrays:
            self._arrays[name] = np.load(_array_path(self.cache_dir, name), mmap_mode="r")
        return self._arrays[name]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[index]
        features = {name: self._array(name)[index] for name in ARRAY_SPECS}
        return row | {"_interaction_features": features}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state
