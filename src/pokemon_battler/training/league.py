from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEAGUE_SCHEMA = "qwen-self-play-league-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class QwenLeague:
    """Persistent population of frozen Qwen checkpoints used for self-play."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("schema") != LEAGUE_SCHEMA:
                raise ValueError(f"Unsupported league manifest: {self.path}")
        else:
            self.data: dict[str, Any] = {
                "schema": LEAGUE_SCHEMA,
                "created_at": _now(),
                "champion": None,
                "entries": [],
                "matches": [],
            }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @property
    def champion(self) -> dict[str, Any]:
        champion_id = self.data.get("champion")
        for entry in self.data["entries"]:
            if entry["id"] == champion_id:
                return entry
        raise RuntimeError("The league has no champion")

    def initialize(self, checkpoint: str | Path, *, entry_id: str = "initial") -> None:
        if self.data["entries"]:
            return
        resolved = str(Path(checkpoint).resolve())
        self.data["entries"].append(
            {
                "id": entry_id,
                "checkpoint": resolved,
                "rating": 1000.0,
                "promoted_at": _now(),
                "parent": None,
            }
        )
        self.data["champion"] = entry_id
        self.save()

    def add_reference(
        self,
        checkpoint: str | Path,
        *,
        entry_id: str,
        rating: float = 1000.0,
    ) -> None:
        """Add a frozen regression opponent without making it champion."""
        if any(entry["id"] == entry_id for entry in self.data["entries"]):
            return
        resolved = str(Path(checkpoint).resolve())
        if any(entry["checkpoint"] == resolved for entry in self.data["entries"]):
            return
        self.data["entries"].append(
            {
                "id": entry_id,
                "checkpoint": resolved,
                "rating": float(rating),
                "promoted_at": _now(),
                "parent": None,
                "reference_only": True,
            }
        )
        self.save()

    def sample_opponent(self, *, seed: int) -> dict[str, Any]:
        """Mix the champion, similarly rated policies, and old snapshots."""
        entries = self.data["entries"]
        if not entries:
            raise RuntimeError("Initialize the league before sampling")
        champion = self.champion
        rng = random.Random(seed)
        weights = []
        for entry in entries:
            rating_gap = abs(float(entry["rating"]) - float(champion["rating"]))
            closeness = math.exp(-rating_gap / 200.0)
            champion_bonus = 2.0 if entry["id"] == champion["id"] else 1.0
            weights.append(0.25 + closeness * champion_bonus)
        return rng.choices(entries, weights=weights, k=1)[0]

    def record_candidate(
        self,
        *,
        candidate_id: str,
        checkpoint: str | Path,
        wins: int,
        losses: int,
        ties: int,
        promotion_threshold: float,
    ) -> dict[str, Any]:
        games = wins + losses + ties
        if games <= 0:
            raise ValueError("A promotion decision requires completed games")
        champion = self.champion
        score = (wins + 0.5 * ties) / games
        expected = 1.0 / (
            1.0 + 10.0 ** ((float(champion["rating"]) - 1000.0) / 400.0)
        )
        candidate_rating = 1000.0 + 32.0 * games**0.5 * (score - expected)
        champion["rating"] = float(champion["rating"]) + 32.0 * games**0.5 * (
            (1.0 - score) - (1.0 - expected)
        )
        promoted = score >= promotion_threshold
        entry = {
            "id": candidate_id,
            "checkpoint": str(Path(checkpoint).resolve()),
            "rating": candidate_rating,
            "promoted_at": _now() if promoted else None,
            "parent": champion["id"],
            "promotion_score": score,
            "promotion_games": games,
            "promoted": promoted,
        }
        # Rejected checkpoints stay on disk and in match history, but do not
        # enter the opponent population.
        if promoted:
            self.data["entries"].append(entry)
            self.data["champion"] = candidate_id
        self.data["matches"].append(
            {
                "created_at": _now(),
                "candidate": entry,
                "champion": champion["id"],
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "score": score,
                "threshold": promotion_threshold,
                "promoted": promoted,
            }
        )
        self.save()
        return entry
