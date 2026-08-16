from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import tarfile
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from pokemon_battler.prepare import (
    SUPPORTED_FILE_SUFFIXES,
    SUPPORTED_TAR_SUFFIXES,
    ReplayMetadata,
    SplitConfig,
    _format_matches,
    choose_split,
    parse_replay_metadata,
)
from pokemon_battler.trajectory_prepare import (
    DEFAULT_REWARD_GAMMA,
    TRAJECTORY_SCHEMA_VERSION,
    _trajectory_is_selected,
    trajectory_rows,
)

PARALLEL_PREPARE_SCHEMA = "parallel-trajectory-prepare-v1"
ACTION_CONTRACT = {
    "count": 13,
    "moves": [0, 1, 2, 3],
    "switches": [4, 5, 6, 7, 8],
    "tera_moves": [9, 10, 11, 12],
    "ordering": "Metamon consistent alphabetical move and switch order",
}


@dataclass(frozen=True)
class EncodedTrajectorySource:
    name: str
    payload: bytes


@dataclass(frozen=True)
class WorkerSettings:
    split_config: SplitConfig
    battle_format: str | None
    min_rating: int | None
    outcome: str
    trajectory_sample_rate: float
    reward_gamma: float
    require_outcome: bool


@dataclass(frozen=True)
class PreparedResult:
    sequence: int
    split: str | None
    battle_id: str | None
    rows_jsonl: bytes
    rows: int
    counters: dict[str, int]


def _has_suffix(name: str, suffixes: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in suffixes)


def _iter_tar_encoded(path: Path) -> Iterator[EncodedTrajectorySource]:
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile() or not _has_suffix(member.name, SUPPORTED_FILE_SUFFIXES):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with extracted:
                yield EncodedTrajectorySource(
                    name=f"{path.name}:{member.name}", payload=extracted.read()
                )


def iter_encoded_trajectory_sources(
    paths: Sequence[Path],
) -> Iterator[EncodedTrajectorySource]:
    """Read archive members once; LZ4/JSON decoding happens in worker processes."""
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
                if _has_suffix(child.name, SUPPORTED_TAR_SUFFIXES):
                    yield from _iter_tar_encoded(child)
                elif _has_suffix(child.name, SUPPORTED_FILE_SUFFIXES):
                    yield EncodedTrajectorySource(str(child), child.read_bytes())
            continue
        if _has_suffix(path.name, SUPPORTED_TAR_SUFFIXES):
            yield from _iter_tar_encoded(path)
        elif _has_suffix(path.name, SUPPORTED_FILE_SUFFIXES):
            yield EncodedTrajectorySource(str(path), path.read_bytes())
        else:
            raise ValueError(f"Unsupported trajectory input: {path}")


def _decode_trajectory(source: EncodedTrajectorySource) -> dict[str, Any]:
    payload = source.payload
    if source.name.lower().endswith(".lz4"):
        import lz4.frame

        payload = lz4.frame.decompress(payload)
    trajectory = json.loads(payload)
    if not isinstance(trajectory, dict):
        raise TypeError(f"Trajectory must be a JSON object: {source.name}")
    return trajectory


def _prepare_one(
    sequence: int,
    source: EncodedTrajectorySource,
    settings: WorkerSettings,
) -> PreparedResult:
    counters: Counter[str] = Counter(trajectories_seen=1)
    try:
        trajectory = _decode_trajectory(source)
    except (UnicodeDecodeError, json.JSONDecodeError, RuntimeError, TypeError, ValueError):
        counters["decode_errors"] += 1
        return PreparedResult(sequence, None, None, b"", 0, dict(counters))
    metadata: ReplayMetadata = parse_replay_metadata(source.name)
    if settings.require_outcome and metadata.outcome not in {"WIN", "LOSS"}:
        counters["missing_outcome_filtered"] += 1
        return PreparedResult(sequence, None, metadata.battle_id, b"", 0, dict(counters))
    if settings.min_rating is not None and metadata.rating < settings.min_rating:
        counters["rating_filtered"] += 1
        return PreparedResult(sequence, None, metadata.battle_id, b"", 0, dict(counters))
    if settings.outcome == "wins" and metadata.outcome != "WIN":
        counters["outcome_filtered"] += 1
        return PreparedResult(sequence, None, metadata.battle_id, b"", 0, dict(counters))
    if settings.outcome == "losses" and metadata.outcome != "LOSS":
        counters["outcome_filtered"] += 1
        return PreparedResult(sequence, None, metadata.battle_id, b"", 0, dict(counters))
    if not _format_matches(trajectory, settings.battle_format):
        counters["format_filtered"] += 1
        return PreparedResult(sequence, None, metadata.battle_id, b"", 0, dict(counters))
    if not _trajectory_is_selected(
        metadata.battle_id,
        seed=settings.split_config.seed,
        sample_rate=settings.trajectory_sample_rate,
    ):
        counters["trajectories_sampled_out"] += 1
        return PreparedResult(sequence, None, metadata.battle_id, b"", 0, dict(counters))
    try:
        split = choose_split(metadata.battle_id, metadata.battle_date, settings.split_config)
    except ValueError:
        counters["missing_split_date"] += 1
        return PreparedResult(sequence, None, metadata.battle_id, b"", 0, dict(counters))
    rows = trajectory_rows(
        source.name,
        trajectory,
        metadata,
        split,
        counters,
        reward_gamma=settings.reward_gamma,
    )
    if not rows:
        return PreparedResult(sequence, split, metadata.battle_id, b"", 0, dict(counters))
    counters["trajectories_written"] += 1
    payload = b"".join(
        json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n" for row in rows
    )
    return PreparedResult(sequence, split, metadata.battle_id, payload, len(rows), dict(counters))


def _path_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "is_dir": path.is_dir(),
    }


def _serializable_split(config: SplitConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("validation_start", "test_start"):
        value = payload[key]
        payload[key] = value.isoformat() if value else None
    return payload


def _configuration(
    inputs: Sequence[Path],
    settings: WorkerSettings,
    shard_trajectories: int,
    maximum_unmapped_fraction: float | None,
) -> dict[str, Any]:
    return {
        "inputs": [_path_signature(path) for path in inputs],
        "split_config": _serializable_split(settings.split_config),
        "battle_format": settings.battle_format,
        "min_rating": settings.min_rating,
        "outcome": settings.outcome,
        "trajectory_sample_rate": settings.trajectory_sample_rate,
        "reward_gamma": settings.reward_gamma,
        "require_outcome": settings.require_outcome,
        "shard_trajectories": shard_trajectories,
        "maximum_unmapped_fraction": maximum_unmapped_fraction,
        "action_contract": ACTION_CONTRACT,
    }


def _configuration_sha256(configuration: dict[str, Any]) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_offsets(path: Path, offsets: list[int]) -> None:
    offsets_path = path.with_suffix(path.suffix + ".offsets.npy")
    metadata_path = path.with_suffix(path.suffix + ".offsets.json")
    with offsets_path.open("wb") as stream:
        np.save(stream, np.asarray(offsets, dtype=np.uint64))
    stat = path.stat()
    metadata_path.write_text(
        json.dumps(
            {
                "source": {"bytes": stat.st_size, "modified_ns": stat.st_mtime_ns},
                "rows": len(offsets),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _commit_shard(
    output_dir: Path,
    shard_index: int,
    results: Sequence[PreparedResult],
) -> dict[str, Any]:
    split_payloads: dict[str, list[bytes]] = {name: [] for name in ("train", "validation", "test")}
    counters: Counter[str] = Counter()
    battle_counts: Counter[str] = Counter()
    row_counts: Counter[str] = Counter()
    for result in results:
        counters.update(result.counters)
        if result.split is not None and result.rows:
            split_payloads[result.split].append(result.rows_jsonl)
            row_counts[result.split] += result.rows
            battle_counts[result.split] += 1
    files: dict[str, str] = {}
    for split, payloads in split_payloads.items():
        if not payloads:
            continue
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        destination = split_dir / f"part-{shard_index:06d}.jsonl"
        partial = destination.with_suffix(destination.suffix + ".partial")
        offsets: list[int] = []
        cursor = 0
        with partial.open("wb") as stream:
            for payload in payloads:
                for line in payload.splitlines(keepends=True):
                    if line.strip():
                        offsets.append(cursor)
                    stream.write(line)
                    cursor += len(line)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, destination)
        _write_offsets(destination, offsets)
        files[split] = str(destination.relative_to(output_dir))
    return {
        "shard": shard_index,
        "source_start": results[0].sequence,
        "source_end": results[-1].sequence + 1,
        "source_items": len(results),
        "files": files,
        "rows": dict(row_counts),
        "trajectories": dict(battle_counts),
        "counters": dict(counters),
    }


def _aggregate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    rows: Counter[str] = Counter()
    trajectories: Counter[str] = Counter()
    for shard in manifest["shards"]:
        counters.update(shard["counters"])
        rows.update(shard["rows"])
        trajectories.update(shard["trajectories"])
    checked_actions = counters["transitions_written"] + counters["actions_not_recoverably_legal"]
    manifest["summary"] = {
        "source_items": sum(int(shard["source_items"]) for shard in manifest["shards"]),
        "transitions_per_split": dict(rows),
        "trajectories_per_split": dict(trajectories),
        "counters": dict(counters),
        "action_parity": {
            "checked_actions": checked_actions,
            "unmapped_actions": counters["actions_not_recoverably_legal"],
            "unmapped_fraction": (
                counters["actions_not_recoverably_legal"] / max(checked_actions, 1)
            ),
            "contract": ACTION_CONTRACT,
        },
    }
    return manifest


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(_aggregate_manifest(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def prepare_trajectory_dataset_parallel(
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    split_config: SplitConfig,
    battle_format: str | None = "gen9ou",
    min_rating: int | None = None,
    outcome: str = "both",
    trajectory_sample_rate: float = 1.0,
    reward_gamma: float = DEFAULT_REWARD_GAMMA,
    workers: int = 4,
    shard_trajectories: int = 2_000,
    progress_every: int = 2_000,
    resume: bool = True,
    require_outcome: bool = False,
    maximum_unmapped_fraction: float | None = 0.001,
) -> dict[str, Any]:
    if workers <= 0 or shard_trajectories <= 0:
        raise ValueError("workers and shard_trajectories must be positive")
    if not 0 < trajectory_sample_rate <= 1:
        raise ValueError("trajectory_sample_rate must be in (0, 1]")
    if outcome not in {"both", "wins", "losses"}:
        raise ValueError("outcome must be both, wins, or losses")
    if maximum_unmapped_fraction is not None and not 0 <= maximum_unmapped_fraction <= 1:
        raise ValueError("maximum_unmapped_fraction must be between zero and one")
    settings = WorkerSettings(
        split_config=split_config,
        battle_format=battle_format,
        min_rating=min_rating,
        outcome=outcome,
        trajectory_sample_rate=trajectory_sample_rate,
        reward_gamma=reward_gamma,
        require_outcome=require_outcome,
    )
    configuration = _configuration(inputs, settings, shard_trajectories, maximum_unmapped_fraction)
    configuration_hash = _configuration_sha256(configuration)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("configuration_sha256") != configuration_hash:
            raise ValueError(
                "Existing prepared shards were built with different inputs or settings"
            )
        if manifest.get("status") == "complete":
            return manifest
    else:
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is non-empty but has no resumable manifest: {output_dir}"
            )
        manifest = {
            "schema": PARALLEL_PREPARE_SCHEMA,
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "configuration": configuration,
            "configuration_sha256": configuration_hash,
            "status": "running",
            "workers": workers,
            "shards": [],
        }
        _write_manifest(manifest_path, manifest)

    completed_sources = sum(int(shard["source_items"]) for shard in manifest["shards"])
    next_shard = len(manifest["shards"])
    started = time.monotonic()
    ready: dict[int, PreparedResult] = {}
    pending: dict[Future[PreparedResult], int] = {}
    shard_results: list[PreparedResult] = []
    next_sequence = completed_sources

    def accept_ready() -> None:
        nonlocal next_sequence, next_shard, shard_results
        while next_sequence in ready:
            shard_results.append(ready.pop(next_sequence))
            next_sequence += 1
            if len(shard_results) == shard_trajectories:
                shard = _commit_shard(output_dir, next_shard, shard_results)
                manifest["shards"].append(shard)
                manifest["status"] = "running"
                manifest["elapsed_seconds"] = round(time.monotonic() - started, 1)
                _write_manifest(manifest_path, manifest)
                next_shard += 1
                shard_results = []
                print(
                    json.dumps(
                        {
                            "phase": "parallel-trajectory-prepare",
                            "source_items": next_sequence,
                            "shards": next_shard,
                            "workers": workers,
                            "rows": manifest["summary"]["transitions_per_split"],
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            for sequence, source in enumerate(iter_encoded_trajectory_sources(inputs)):
                if sequence < completed_sources:
                    continue
                future = executor.submit(_prepare_one, sequence, source, settings)
                pending[future] = sequence
                if len(pending) < workers * 2:
                    continue
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for completed in done:
                    pending.pop(completed)
                    result = completed.result()
                    ready[result.sequence] = result
                accept_ready()
                if progress_every and next_sequence % progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "parallel-trajectory-prepare-progress",
                                "source_items": next_sequence,
                                "in_flight": len(pending),
                            }
                        ),
                        flush=True,
                    )
            for completed in pending:
                result = completed.result()
                ready[result.sequence] = result
            accept_ready()
        if ready:
            raise AssertionError("Parallel preparation did not drain results in order")
        if shard_results:
            shard = _commit_shard(output_dir, next_shard, shard_results)
            manifest["shards"].append(shard)
        _aggregate_manifest(manifest)
        unmapped_fraction = float(manifest["summary"]["action_parity"]["unmapped_fraction"])
        if maximum_unmapped_fraction is not None and unmapped_fraction > maximum_unmapped_fraction:
            raise ValueError(
                "Metamon action parity failed: "
                f"{unmapped_fraction:.4%} of checked actions were unmapped; "
                f"maximum is {maximum_unmapped_fraction:.4%}"
            )
        manifest["status"] = "complete"
        manifest["workers"] = workers
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 1)
        _write_manifest(manifest_path, manifest)
        print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
        return manifest
    except BaseException as exc:
        manifest["status"] = "interrupted"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 1)
        _write_manifest(manifest_path, manifest)
        raise


def _iso_date(value: str) -> date:
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Metamon trajectories into resumable JSONL shards in parallel."
    )
    parser.add_argument("--input", dest="inputs", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", dest="battle_format", default="gen9ou")
    parser.add_argument("--min-rating", type=int)
    parser.add_argument("--outcome", choices=("both", "wins", "losses"), default="both")
    parser.add_argument("--trajectory-sample-rate", type=float, default=1.0)
    parser.add_argument("--reward-gamma", type=float, default=DEFAULT_REWARD_GAMMA)
    parser.add_argument("--split-mode", choices=("hash", "chronological"), default="hash")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-start", type=_iso_date)
    parser.add_argument("--test-start", type=_iso_date)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-trajectories", type=int, default=2_000)
    parser.add_argument("--progress-every", type=int, default=2_000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-outcome", action="store_true")
    parser.add_argument("--maximum-unmapped-fraction", type=float, default=0.001)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_trajectory_dataset_parallel(
        args.inputs,
        args.output_dir,
        split_config=SplitConfig(
            mode=args.split_mode,
            seed=args.seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            validation_start=args.validation_start,
            test_start=args.test_start,
        ),
        battle_format=args.battle_format,
        min_rating=args.min_rating,
        outcome=args.outcome,
        trajectory_sample_rate=args.trajectory_sample_rate,
        reward_gamma=args.reward_gamma,
        workers=args.workers,
        shard_trajectories=args.shard_trajectories,
        progress_every=args.progress_every,
        resume=args.resume,
        require_outcome=args.require_outcome,
        maximum_unmapped_fraction=args.maximum_unmapped_fraction,
    )


if __name__ == "__main__":
    main()
