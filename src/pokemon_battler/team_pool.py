from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from poke_env.teambuilder.teambuilder import Teambuilder


def team_composition(team: str) -> tuple[str, ...]:
    """Return an order-independent species fingerprint for a Showdown export."""
    members = Teambuilder.parse_showdown_team(team)
    composition = tuple(
        sorted(
            "".join(
                character
                for character in str(member.species or member.nickname).lower()
                if character.isalnum()
            )
            for member in members
        )
    )
    if not composition or any(not species for species in composition):
        raise ValueError("Showdown team does not contain recognizable species")
    return composition


def resolve_team_pool(
    team_files: Sequence[Path],
    team_directory: Path | None = None,
    *,
    minimum_teams: int = 2,
) -> list[Path]:
    """Resolve and validate a pool of distinct Showdown export files."""
    paths = [Path(path) for path in team_files]
    if team_directory is not None:
        if not team_directory.is_dir():
            raise FileNotFoundError(f"Enemy team directory does not exist: {team_directory}")
        paths.extend(
            path
            for path in sorted(team_directory.iterdir())
            if path.is_file() and not path.name.startswith(".")
        )
    unique_paths: list[Path] = []
    seen_paths: set[Path] = set()
    seen_contents: set[str] = set()
    seen_compositions: set[tuple[str, ...]] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        if not resolved.is_file():
            raise FileNotFoundError(f"Showdown team file does not exist: {path}")
        team = resolved.read_text(encoding="utf-8").strip()
        if not team:
            raise ValueError(f"Showdown team file is empty: {path}")
        if team in seen_contents:
            continue
        composition = team_composition(team)
        if composition in seen_compositions:
            continue
        seen_paths.add(resolved)
        seen_contents.add(team)
        seen_compositions.add(composition)
        unique_paths.append(resolved)
    if len(unique_paths) < minimum_teams:
        raise ValueError(
            f"Enemy randomization requires at least {minimum_teams} distinct team "
            f"compositions; found {len(unique_paths)}"
        )
    return unique_paths


class ShuffledTeamPool(Teambuilder):
    """Choose one team per battle from shuffled, non-repeating pool cycles."""

    def __init__(self, team_files: Sequence[Path], *, seed: int = 42) -> None:
        if len(team_files) < 2:
            raise ValueError("ShuffledTeamPool requires at least two teams")
        self.team_files = [Path(path).resolve() for path in team_files]
        self._packed_teams = [
            self.join_team(
                self.parse_showdown_team(path.read_text(encoding="utf-8").strip())
            )
            for path in self.team_files
        ]
        self._random = random.Random(seed)
        self._remaining: list[int] = []
        self._last_index: int | None = None
        self.selections: list[dict[str, Any]] = []

    def _refill(self) -> None:
        self._remaining = list(range(len(self.team_files)))
        self._random.shuffle(self._remaining)
        if (
            self._last_index is not None
            and self._remaining[-1] == self._last_index
            and len(self._remaining) > 1
        ):
            self._remaining[-1], self._remaining[-2] = (
                self._remaining[-2],
                self._remaining[-1],
            )

    def yield_team(self) -> str:
        if not self._remaining:
            self._refill()
        index = self._remaining.pop()
        self._last_index = index
        self.selections.append(
            {
                "selection_index": len(self.selections),
                "team_index": index,
                "team_file": str(self.team_files[index]),
            }
        )
        return self._packed_teams[index]

    def report(self) -> dict[str, Any]:
        counts = Counter(selection["team_file"] for selection in self.selections)
        return {
            "pool_size": len(self.team_files),
            "team_files": [str(path) for path in self.team_files],
            "selections": list(self.selections),
            "selection_counts": dict(sorted(counts.items())),
        }
