from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from pokemon_battler.distillation import teacher_policy
from pokemon_battler.training_data import JsonlOffsetDataset


@dataclass(frozen=True)
class RowReference:
    path: Path
    offset: int
    length: int
    battle_id: str
    team: str
    family: str
    behavior_source: str
    disagreement: bool
    priority: float
    stable: int


def _stable_integer(*values: object) -> int:
    payload = "\0".join(str(value) for value in values).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _family(action_id: int) -> str:
    if action_id < 4:
        return "attack"
    if action_id < 9:
        return "switch"
    return "tera"


def _teacher_priority(row: dict[str, Any], *, seed: int) -> tuple[float, bool]:
    probabilities, selected, confidence, _ = teacher_policy(row)
    behavior = row.get("behavior") or {}
    behavior_source = str(behavior.get("source") or "unknown")
    behavior_action = behavior.get("action_id")
    disagreement = behavior_action is not None and int(behavior_action) != selected
    priority = confidence
    if behavior_source == "student":
        priority += 1.0
        if disagreement:
            priority += 4.0
        preferences = behavior.get("preferences")
        if isinstance(preferences, dict):
            student = [float(preferences.get(str(index), 0.0)) for index in range(13)]
            total = sum(student)
            if total > 0:
                student = [value / total for value in student]
                priority += 2.0 * 0.5 * sum(
                    abs(left - right)
                    for left, right in zip(probabilities, student, strict=True)
                )
    if str(row.get("outcome") or "").upper() == "LOSS":
        priority += 0.25
    # A deterministic fractional tie breaker makes repeated preparation exact.
    stable = _stable_integer(seed, row.get("battle_id"), row.get("request_id"), selected)
    priority += stable / 2**64 * 1e-3
    return priority, disagreement


def _iter_teacher_references(
    paths: Iterable[Path], *, seed: int, representative: bool
) -> Iterable[RowReference]:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("decision_phase") == "team_preview":
                    continue
                probabilities, selected, _, _ = teacher_policy(row)
                del probabilities
                battle_id = str(row.get("battle_id") or f"{path}:{offset}")
                team = str(row.get("enemy_team_file") or "unknown-team")
                behavior_source = str((row.get("behavior") or {}).get("source") or "unknown")
                if representative:
                    stable = _stable_integer(seed, path, offset, battle_id)
                    priority = stable / 2**64
                    disagreement = False
                else:
                    priority, disagreement = _teacher_priority(row, seed=seed)
                    stable = _stable_integer(seed, path, offset)
                yield RowReference(
                    path=path,
                    offset=offset,
                    length=len(line),
                    battle_id=battle_id,
                    team=team,
                    family=_family(selected),
                    behavior_source=behavior_source,
                    disagreement=disagreement,
                    priority=priority,
                    stable=stable,
                )


def _bounded_candidates(
    paths: Iterable[Path],
    *,
    limit: int,
    seed: int,
    representative: bool,
) -> tuple[list[RowReference], int]:
    heaps: dict[str, list[tuple[float, int, str, int, RowReference]]] = defaultdict(list)
    seen = 0
    # Keep overflow candidates so the later per-battle cap can replace many
    # high-priority turns from one unusually long trajectory.
    capacity = max(limit * (2 if representative else 3), 1)
    for reference in _iter_teacher_references(
        paths, seed=seed, representative=representative
    ):
        seen += 1
        key = "all" if representative else reference.family
        heap = heaps[key]
        item = (
            reference.priority,
            reference.stable,
            str(reference.path),
            reference.offset,
            reference,
        )
        if len(heap) < capacity:
            heapq.heappush(heap, item)
        elif item[:4] > heap[0][:4]:
            heapq.heapreplace(heap, item)
    candidates = [item[-1] for heap in heaps.values() for item in heap]
    return candidates, seen


def _balanced_take(
    candidates: Iterable[RowReference],
    count: int,
    *,
    battle_cap: int,
    existing: set[tuple[str, int]] | None = None,
    battle_counts: Counter[str] | None = None,
) -> list[RowReference]:
    selected_keys = existing if existing is not None else set()
    counts = battle_counts if battle_counts is not None else Counter()
    by_team: dict[str, list[RowReference]] = defaultdict(list)
    for row in candidates:
        if (str(row.path), row.offset) not in selected_keys:
            by_team[row.team].append(row)
    for rows in by_team.values():
        rows.sort(key=lambda row: (row.priority, row.stable), reverse=True)
    positions = {team: 0 for team in by_team}
    result: list[RowReference] = []
    teams = sorted(by_team)
    while len(result) < count:
        progressed = False
        for team in teams:
            rows = by_team[team]
            position = positions[team]
            while position < len(rows) and counts[rows[position].battle_id] >= battle_cap:
                position += 1
            positions[team] = position
            if position >= len(rows):
                continue
            row = rows[position]
            positions[team] += 1
            key = (str(row.path), row.offset)
            if key in selected_keys:
                continue
            selected_keys.add(key)
            counts[row.battle_id] += 1
            result.append(row)
            progressed = True
            if len(result) >= count:
                break
        if not progressed:
            break
    return result


def select_teacher_rows(
    paths: Sequence[Path],
    *,
    limit: int,
    seed: int,
    battle_cap: int = 24,
    representative: bool = False,
) -> tuple[list[RowReference], dict[str, Any]]:
    if limit <= 0 or battle_cap <= 0:
        raise ValueError("Teacher selection limits must be positive")
    candidates, seen = _bounded_candidates(
        paths, limit=limit, seed=seed, representative=representative
    )
    if representative:
        selected = _balanced_take(candidates, limit, battle_cap=battle_cap)
    else:
        quotas = {
            "attack": round(limit * 0.60),
            "switch": round(limit * 0.35),
        }
        quotas["tera"] = max(0, limit - quotas["attack"] - quotas["switch"])
        selected = []
        selected_keys: set[tuple[str, int]] = set()
        battle_counts: Counter[str] = Counter()
        for family in ("tera", "switch", "attack"):
            selected.extend(
                _balanced_take(
                    (row for row in candidates if row.family == family),
                    quotas[family],
                    battle_cap=battle_cap,
                    existing=selected_keys,
                    battle_counts=battle_counts,
                )
            )
        if len(selected) < limit:
            selected.extend(
                _balanced_take(
                    sorted(candidates, key=lambda row: row.priority, reverse=True),
                    limit - len(selected),
                    battle_cap=battle_cap,
                    existing=selected_keys,
                    battle_counts=battle_counts,
                )
            )
    if not selected:
        raise ValueError("Teacher sources contain no turn decisions")
    report = {
        "source_rows_seen": seen,
        "selected_rows": len(selected),
        "family_counts": dict(Counter(row.family for row in selected)),
        "team_counts": dict(Counter(row.team for row in selected)),
        "behavior_source_counts": dict(Counter(row.behavior_source for row in selected)),
        "student_disagreements": sum(row.disagreement for row in selected),
        "battle_count": len({row.battle_id for row in selected}),
        "battle_cap": battle_cap,
        "representative": representative,
    }
    return selected, report


def _copy_references(rows: Sequence[RowReference], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    streams: dict[Path, BinaryIO] = {}
    try:
        with output.open("wb") as destination:
            for row in rows:
                stream = streams.get(row.path)
                if stream is None:
                    stream = row.path.open("rb")
                    streams[row.path] = stream
                stream.seek(row.offset)
                line = stream.read(row.length)
                if not line.endswith(b"\n"):
                    line += b"\n"
                destination.write(line)
    finally:
        for stream in streams.values():
            stream.close()


def sample_replay(source_path: Path, output: Path, *, rows: int, seed: int) -> dict[str, Any]:
    source = JsonlOffsetDataset(source_path)
    count = min(rows, len(source))
    indices = sorted(random.Random(seed).sample(range(len(source)), count))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for index in indices:
            destination.write(json.dumps(source[index], separators=(",", ":"), sort_keys=True))
            destination.write("\n")
    digest = hashlib.sha256(
        ",".join(str(index) for index in indices).encode()
    ).hexdigest()
    return {
        "source": str(source_path),
        "available_rows": len(source),
        "selected_rows": count,
        "selected_indices_sha256": digest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select bounded, diverse expert corrections and replay rehearsal rows."
    )
    parser.add_argument("--teacher-data", type=Path, action="append", required=True)
    parser.add_argument("--teacher-validation-data", type=Path, required=True)
    parser.add_argument("--replay-train-data", type=Path, required=True)
    parser.add_argument("--replay-validation-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-train-rows", type=int, default=8_000)
    parser.add_argument("--teacher-validation-rows", type=int, default=2_000)
    parser.add_argument("--replay-train-rows", type=int, default=4_000)
    parser.add_argument("--replay-validation-rows", type=int, default=2_000)
    parser.add_argument("--trajectory-cap", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    numeric = (
        args.teacher_train_rows,
        args.teacher_validation_rows,
        args.replay_train_rows,
        args.replay_validation_rows,
        args.trajectory_cap,
    )
    if any(value <= 0 or not math.isfinite(value) for value in numeric):
        raise ValueError("All row counts and caps must be positive")
    args.output_dir.mkdir(parents=True)
    teacher_train, teacher_train_report = select_teacher_rows(
        args.teacher_data,
        limit=args.teacher_train_rows,
        seed=args.seed,
        battle_cap=args.trajectory_cap,
    )
    teacher_validation, teacher_validation_report = select_teacher_rows(
        [args.teacher_validation_data],
        limit=args.teacher_validation_rows,
        seed=args.seed + 1,
        battle_cap=args.trajectory_cap,
        representative=True,
    )
    _copy_references(teacher_train, args.output_dir / "teacher-train.jsonl")
    _copy_references(teacher_validation, args.output_dir / "teacher-validation.jsonl")
    report = {
        "schema": "selective-distillation-data-v1",
        "teacher_train": teacher_train_report,
        "teacher_validation": teacher_validation_report,
        "replay_train": sample_replay(
            args.replay_train_data,
            args.output_dir / "replay-train.jsonl",
            rows=args.replay_train_rows,
            seed=args.seed + 2,
        ),
        "replay_validation": sample_replay(
            args.replay_validation_data,
            args.output_dir / "replay-validation.jsonl",
            rows=args.replay_validation_rows,
            seed=args.seed + 3,
        ),
    }
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
