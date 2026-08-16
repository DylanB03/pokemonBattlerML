from __future__ import annotations

import argparse
import json
from argparse import Namespace
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pokemon_battler.live_eval import DEFAULT_TEAM
from pokemon_battler.policy_ablation import _comparison, _release_policy_memory
from pokemon_battler.policy_suite import (
    _checkpoint_preview_enabled,
    _run_policy,
    _schedule,
)
from pokemon_battler.showdown_server import LocalShowdownServer
from pokemon_battler.structured_modeling import has_structured_head
from pokemon_battler.team_pool import resolve_team_pool

DEFAULT_BLEND_WEIGHTS = (0.25, 0.5, 0.75, 1.0)


def _normalized_weights(values: Sequence[float] | None) -> list[float]:
    weights = list(DEFAULT_BLEND_WEIGHTS if not values else values)
    if any(value < 0 for value in weights):
        raise ValueError("Structured blend weights must be non-negative")
    return sorted(set(float(value) for value in weights))


def _weight_slug(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.games <= 0 or args.concurrent_games <= 0:
        raise ValueError("--games and --concurrent-games must be positive")
    if not has_structured_head(args.candidate):
        raise FileNotFoundError(
            f"Blend sweep candidate has no structured sidecar: {args.candidate}"
        )
    weights = _normalized_weights(getattr(args, "blend_weights", None))
    teams = resolve_team_pool(args.enemy_team_file, args.enemy_team_dir, minimum_teams=3)
    schedule = _schedule(teams, args.games, args.seed)
    candidate_preview = _checkpoint_preview_enabled(args.candidate, True)
    champion_preview = _checkpoint_preview_enabled(args.champion, True)
    args.output_dir.mkdir(parents=True)
    with LocalShowdownServer(
        args.showdown_dir,
        port=args.server_port,
        bootstrap=not args.no_bootstrap_server,
        log_path=args.output_dir / "showdown.log",
    ):
        champion = _run_policy(
            args,
            args.champion,
            schedule,
            args.output_dir / "champion",
            load_preview_head=champion_preview,
            team_preview_policy="learned" if champion_preview else "first",
        )
        _release_policy_memory()
        results = []
        for index, weight in enumerate(weights):
            candidate = _run_policy(
                args,
                args.candidate,
                schedule,
                args.output_dir / f"candidate-blend-{_weight_slug(weight)}",
                structured_blend_weight=weight,
                load_preview_head=candidate_preview,
                team_preview_policy="learned" if candidate_preview else "first",
            )
            _release_policy_memory()
            comparison = _comparison(candidate, champion, seed=args.seed + index)
            row = {
                "structured_blend_weight": weight,
                **comparison,
                "candidate": candidate,
            }
            results.append(row)
            print(json.dumps({"phase": "structured-blend-sweep", **row}), flush=True)
    results.sort(
        key=lambda row: (
            float(row["win_rate_delta"]),
            float(row["paired_bootstrap_delta_95"][0]),
            float(row["candidate_win_rate"]),
        ),
        reverse=True,
    )
    report = {
        "schema": "structured-blend-sweep-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_checkpoint": str(args.candidate),
        "champion_checkpoint": str(args.champion),
        "enemy_teams": [str(path) for path in teams],
        "schedule": [str(path) for path in schedule],
        "champion": champion,
        "configurations": results,
        "best_configuration": results[0],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a structured-sidecar blend on a paired Foul Play suite."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--enemy-team-file", type=Path, action="append", default=[])
    parser.add_argument("--enemy-team-dir", type=Path)
    parser.add_argument("--blend-weight", dest="blend_weights", action="append", type=float)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--showdown-dir", type=Path, default=Path("data/pokemon-showdown"))
    parser.add_argument("--opponents-dir", type=Path, default=Path("data/opponents"))
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--foul-play-search-time-ms", type=int, default=250)
    parser.add_argument("--foul-play-parallelism", type=int, default=1)
    parser.add_argument("--foul-play-search-threads", type=int, default=1)
    parser.add_argument("--concurrent-games", type=int, default=4)
    parser.add_argument("--no-bootstrap-server", action="store_true")
    parser.add_argument("--no-bootstrap-opponents", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
