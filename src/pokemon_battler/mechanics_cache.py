from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pokemon_battler.actions import ACTION_COUNT
from pokemon_battler.mechanics import MECHANICS_SCHEMA as LEGACY_MECHANICS_SCHEMA
from pokemon_battler.mechanics_v2 import MECHANICS_SCHEMA as DEFAULT_MECHANICS_SCHEMA
from pokemon_battler.training_data import JsonlOffsetDataset, state_with_row_context


def mechanics_spec(schema: str) -> tuple[int, tuple[str, ...], Any]:
    if schema == LEGACY_MECHANICS_SCHEMA:
        from pokemon_battler.mechanics import (
            MECHANICS_FEATURE_COUNT,
            MECHANICS_FEATURE_NAMES,
            candidate_feature_matrix,
        )

        return MECHANICS_FEATURE_COUNT, MECHANICS_FEATURE_NAMES, candidate_feature_matrix
    if schema == DEFAULT_MECHANICS_SCHEMA:
        from pokemon_battler.mechanics_v2 import (
            MECHANICS_FEATURE_COUNT,
            MECHANICS_FEATURE_NAMES,
            candidate_feature_matrix,
        )

        return MECHANICS_FEATURE_COUNT, MECHANICS_FEATURE_NAMES, candidate_feature_matrix
    raise ValueError(
        f"Unknown mechanics schema {schema!r}; choose {LEGACY_MECHANICS_SCHEMA!r} "
        f"or {DEFAULT_MECHANICS_SCHEMA!r}"
    )


def default_cache_path(
    data_file: str | Path,
    schema: str = DEFAULT_MECHANICS_SCHEMA,
) -> Path:
    path = Path(data_file)
    stem = path.name.removesuffix(".jsonl")
    return path.with_name(f"{stem}.{schema}.npy")


def _source_signature(data_file: Path, rows: int) -> dict[str, Any]:
    stat = data_file.stat()
    return {
        "path": str(data_file),
        "bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "rows": rows,
    }


def cache_metadata_path(cache_path: str | Path) -> Path:
    path = Path(cache_path)
    return path.with_suffix(path.suffix + ".json")


def cache_is_current(
    data_file: str | Path,
    cache_path: str | Path,
    *,
    schema: str = DEFAULT_MECHANICS_SCHEMA,
) -> bool:
    feature_count, _, _ = mechanics_spec(schema)
    source = Path(data_file)
    cache = Path(cache_path)
    metadata_path = cache_metadata_path(cache)
    if not source.is_file() or not cache.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = int(metadata["rows"])
        if metadata.get("schema") != schema:
            return False
        if int(metadata.get("feature_count", -1)) != feature_count:
            return False
        if metadata.get("source") != _source_signature(source, rows):
            return False
        matrix = np.load(cache, mmap_mode="r")
        return tuple(matrix.shape) == (rows, ACTION_COUNT, feature_count)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_feature_cache(
    data_file: str | Path,
    cache_path: str | Path | None = None,
    *,
    overwrite: bool = False,
    progress_every: int = 10_000,
    schema: str = DEFAULT_MECHANICS_SCHEMA,
) -> dict[str, Any]:
    feature_count, feature_names, feature_builder = mechanics_spec(schema)
    source = Path(data_file)
    output = (
        Path(cache_path)
        if cache_path is not None
        else default_cache_path(source, schema)
    )
    metadata_path = cache_metadata_path(output)
    if cache_is_current(source, output, schema=schema) and not overwrite:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    if (output.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Mechanics cache exists but is stale or invalid: {output}. "
            "Pass --overwrite to rebuild it."
        )

    dataset = JsonlOffsetDataset(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        partial.unlink()
    matrix = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float16,
        shape=(len(dataset), ACTION_COUNT, feature_count),
    )
    started = time.monotonic()
    try:
        for index in range(len(dataset)):
            row = dataset[index]
            state = state_with_row_context(row)
            matrix[index] = np.asarray(feature_builder(state), dtype=np.float16)
            if progress_every and (index + 1) % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "phase": "mechanics-cache",
                            "rows": index + 1,
                            "total_rows": len(dataset),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
        matrix.flush()
        os.replace(partial, output)
        del matrix
    except BaseException:
        del matrix
        if partial.exists():
            partial.unlink()
        raise

    metadata = {
        "schema": schema,
        "feature_count": feature_count,
        "feature_names": list(feature_names),
        "rows": len(dataset),
        "dtype": "float16",
        "shape": [len(dataset), ACTION_COUNT, feature_count],
        "source": _source_signature(source, len(dataset)),
        "cache_file": str(output),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "notes": (
            "Features are decision-time only. Damage fractions use neutral-stat estimates "
            "because replay states omit EVs, IVs, and natures."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute zero-token candidate mechanics vectors for a JSONL split."
    )
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument(
        "--schema",
        choices=(LEGACY_MECHANICS_SCHEMA, DEFAULT_MECHANICS_SCHEMA),
        default=DEFAULT_MECHANICS_SCHEMA,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = build_feature_cache(
        args.data_file,
        args.output,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
        schema=args.schema,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
