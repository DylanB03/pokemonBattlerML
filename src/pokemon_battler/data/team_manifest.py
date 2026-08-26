from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from pokemon_battler.data.team_pool import team_composition


def _inspect_team(path: Path) -> tuple[Path, str, tuple[str, ...]] | None:
    try:
        contents = path.read_text(encoding="utf-8").strip()
        if not contents:
            return None
        return (
            path.resolve(),
            hashlib.sha256(contents.encode()).hexdigest(),
            team_composition(contents),
        )
    except (OSError, UnicodeDecodeError, TypeError, ValueError):
        return None


def build_team_manifests(
    team_directories: Sequence[Path],
    output_dir: Path,
    *,
    workers: int = 4,
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    maximum_per_split: int | None = 512,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions cannot be negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    paths = sorted(
        {
            path
            for directory in team_directories
            for path in directory.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        }
    )
    if not paths:
        raise ValueError("No team files were found")
    if workers == 1 or len(paths) < workers * 64:
        inspected = [_inspect_team(path) for path in paths]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            inspected = list(executor.map(_inspect_team, paths, chunksize=64))
    unique: dict[tuple[str, ...], tuple[Path, str]] = {}
    invalid = 0
    seen_contents: set[str] = set()
    for result in inspected:
        if result is None:
            invalid += 1
            continue
        path, content_hash, composition = result
        if content_hash in seen_contents or composition in unique:
            continue
        seen_contents.add(content_hash)
        unique[composition] = (path, content_hash)
    splits: dict[str, list[Path]] = {"train": [], "validation": [], "test": []}
    for composition, (path, _content_hash) in sorted(unique.items()):
        key = ",".join(composition)
        digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64)
        if unit < test_fraction:
            split = "test"
        elif unit < test_fraction + validation_fraction:
            split = "validation"
        else:
            split = "train"
        if maximum_per_split is None or len(splits[split]) < maximum_per_split:
            splits[split].append(path)
    if any(len(paths_for_split) < 2 for paths_for_split in splits.values()):
        raise ValueError("Team corpus did not produce at least two teams in every split")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, str] = {}
    for split, split_paths in splits.items():
        manifest = output_dir / f"{split}.txt"
        manifest.write_text("\n".join(str(path) for path in split_paths) + "\n", encoding="utf-8")
        manifests[split] = str(manifest)
    report = {
        "schema": "team-corpus-manifest-v1",
        "source_directories": [str(path.resolve()) for path in team_directories],
        "files_seen": len(paths),
        "invalid_files": invalid,
        "duplicate_files_or_compositions": len(paths) - invalid - len(unique),
        "distinct_compositions": len(unique),
        "teams_per_split": {name: len(values) for name, values in splits.items()},
        "maximum_per_split": maximum_per_split,
        "seed": seed,
        "manifests": manifests,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, deduplicate, and split a large Showdown team corpus."
    )
    parser.add_argument(
        "--team-dir", dest="team_directories", action="append", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--maximum-per-split", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = build_team_manifests(
        args.team_directories,
        args.output_dir,
        workers=args.workers,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        maximum_per_split=args.maximum_per_split,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
