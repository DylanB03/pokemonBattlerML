from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from pokemon_battler.interaction_cache import (
    INTERACTION_CACHE_SCHEMA,
    build_interaction_cache,
    interaction_cache_is_current,
)
from pokemon_battler.training_data import JsonlOffsetDataset

PARALLEL_CACHE_SCHEMA = "parallel-interaction-cache-v1"
STRUCTURED_TARGET_SCHEMA = "structured-policy-targets-v1"


def _target_signature(source: Path) -> dict[str, int]:
    stat = source.stat()
    return {"bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}


def _targets_are_current(source: Path, destination: Path) -> bool:
    metadata_path = destination / "structured_targets.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != STRUCTURED_TARGET_SCHEMA:
            return False
        if metadata.get("source") != _target_signature(source):
            return False
        rows = int(metadata["rows"])
        return all(
            np.load(destination / filename, mmap_mode="r").shape == (rows,)
            for filename in (
                "structured_actions.npy",
                "structured_outcomes.npy",
                "structured_decision_counts.npy",
            )
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _build_targets(source: Path, destination: Path) -> dict[str, Any]:
    if _targets_are_current(source, destination):
        return json.loads((destination / "structured_targets.json").read_text(encoding="utf-8"))
    dataset = JsonlOffsetDataset(source, cache_offsets=True)
    partials = {
        "actions": destination / "structured_actions.partial.npy",
        "outcomes": destination / "structured_outcomes.partial.npy",
        "decision_counts": destination / "structured_decision_counts.partial.npy",
    }
    arrays = {
        "actions": np.lib.format.open_memmap(
            partials["actions"], mode="w+", dtype=np.int8, shape=(len(dataset),)
        ),
        "outcomes": np.lib.format.open_memmap(
            partials["outcomes"], mode="w+", dtype=np.int8, shape=(len(dataset),)
        ),
        "decision_counts": np.lib.format.open_memmap(
            partials["decision_counts"], mode="w+", dtype=np.int16, shape=(len(dataset),)
        ),
    }
    for index in range(len(dataset)):
        row = dataset[index]
        outcome = str(row.get("outcome") or "").upper()
        arrays["actions"][index] = int(row["action_id"])
        arrays["outcomes"][index] = 1 if outcome == "WIN" else -1 if outcome == "LOSS" else 0
        arrays["decision_counts"][index] = min(
            max(int(row.get("battle_decision_count", 1)), 1),
            np.iinfo(np.int16).max,
        )
    for array in arrays.values():
        array.flush()
    arrays.clear()
    destinations = {
        "actions": destination / "structured_actions.npy",
        "outcomes": destination / "structured_outcomes.npy",
        "decision_counts": destination / "structured_decision_counts.npy",
    }
    for name, partial in partials.items():
        os.replace(partial, destinations[name])
    metadata = {
        "schema": STRUCTURED_TARGET_SCHEMA,
        "source": _target_signature(source),
        "rows": len(dataset),
        "files": {name: path.name for name, path in destinations.items()},
    }
    (destination / "structured_targets.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _cache_one(task: tuple[Path, Path, int]) -> dict[str, Any]:
    source, destination, progress_every = task
    if interaction_cache_is_current(source, destination):
        metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
        targets = _build_targets(source, destination)
        return {
            **metadata,
            "source_file": str(source),
            "cache_dir": str(destination),
            "reused": True,
            "structured_targets": targets,
        }
    metadata = build_interaction_cache(source, destination, progress_every=progress_every)
    targets = _build_targets(source, destination)
    return {
        **metadata,
        "source_file": str(source),
        "cache_dir": str(destination),
        "reused": False,
        "structured_targets": targets,
    }


def build_parallel_interaction_caches(
    prepared_dir: Path,
    output_dir: Path,
    *,
    workers: int = 4,
    splits: Sequence[str] = ("train", "validation", "test"),
    progress_every: int = 10_000,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Parallel cache input needs a prepared manifest: {manifest_path}")
    prepared_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if prepared_manifest.get("status") != "complete":
        raise ValueError("Prepared trajectory manifest is not complete")
    tasks: list[tuple[Path, Path, int]] = []
    for split in splits:
        source_dir = prepared_dir / split
        if not source_dir.is_dir():
            continue
        for source in sorted(source_dir.glob("*.jsonl")):
            destination = output_dir / split / (source.stem + f".{INTERACTION_CACHE_SCHEMA}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((source, destination, progress_every))
    if not tasks:
        raise ValueError(f"No prepared JSONL shards found under {prepared_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reports: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {executor.submit(_cache_one, task): task for task in tasks}
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(
                json.dumps(
                    {
                        "phase": "parallel-interaction-cache",
                        "complete": len(reports),
                        "total": len(tasks),
                        "source": report["source_file"],
                        "rows": report["rows"],
                        "reused": report["reused"],
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    reports.sort(key=lambda item: str(item["source_file"]))
    manifest = {
        "schema": PARALLEL_CACHE_SCHEMA,
        "prepared_manifest": str(manifest_path.resolve()),
        "workers": workers,
        "shards": reports,
        "rows": sum(int(report["rows"]) for report in reports),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "status": "complete",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build structured feature caches for prepared JSONL shards in parallel."
    )
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--split", dest="splits", action="append", choices=("train", "validation", "test")
    )
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = build_parallel_interaction_caches(
        args.prepared_dir,
        args.output_dir,
        workers=args.workers,
        splits=args.splits or ("train", "validation", "test"),
        progress_every=args.progress_every,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
