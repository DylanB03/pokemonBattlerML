from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from pokemon_battler.live_eval import DEFAULT_TEAM
from pokemon_battler.policy_suite import (
    _difference_interval,
    _indexed_results,
    _run_policy,
    _schedule,
)
from pokemon_battler.showdown_server import LocalShowdownServer
from pokemon_battler.team_pool import resolve_team_pool
from pokemon_battler.team_preview import has_team_preview_head


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate policy-only, Q-blended, preview, and preview-plus-Q inference "
            "without retraining either checkpoint."
        )
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--enemy-team-file", type=Path, action="append", default=[])
    parser.add_argument("--enemy-team-dir", type=Path)
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
    parser.add_argument("--q-weight", type=float, default=0.35)
    parser.add_argument("--no-bootstrap-server", action="store_true")
    parser.add_argument("--no-bootstrap-opponents", action="store_true")
    return parser


def _configuration_rows(candidate: Path, q_weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "name": "policy-only",
            "action_value_weight": 0.0,
            "load_preview_head": False,
            "team_preview_policy": "first",
        },
        {
            "name": "q-blend",
            "action_value_weight": q_weight,
            "load_preview_head": False,
            "team_preview_policy": "first",
        },
    ]
    if has_team_preview_head(candidate):
        rows.extend(
            [
                {
                    "name": "preview-policy",
                    "action_value_weight": 0.0,
                    "load_preview_head": True,
                    "team_preview_policy": "learned",
                },
                {
                    "name": "preview-q-blend",
                    "action_value_weight": q_weight,
                    "load_preview_head": True,
                    "team_preview_policy": "learned",
                },
            ]
        )
    return rows


def _comparison(
    candidate: dict[str, Any], champion: dict[str, Any], *, seed: int
) -> dict[str, Any]:
    candidate_index = _indexed_results(candidate)
    champion_index = _indexed_results(champion)
    if candidate_index.keys() != champion_index.keys():
        raise RuntimeError("Candidate and champion ablation schedules did not align")
    keys = sorted(candidate_index)
    candidate_results = [candidate_index[key] for key in keys]
    champion_results = [champion_index[key] for key in keys]
    candidate_rate = sum(candidate_results) / max(len(candidate_results), 1)
    champion_rate = sum(champion_results) / max(len(champion_results), 1)
    return {
        "candidate_win_rate": candidate_rate,
        "champion_win_rate": champion_rate,
        "win_rate_delta": candidate_rate - champion_rate,
        "paired_bootstrap_delta_95": _difference_interval(
            candidate_results, champion_results, seed=seed
        ),
        "candidate_only_wins": sum(
            left == 1 and right == 0
            for left, right in zip(candidate_results, champion_results, strict=True)
        ),
        "champion_only_wins": sum(
            left == 0 and right == 1
            for left, right in zip(candidate_results, champion_results, strict=True)
        ),
    }


def _release_policy_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.games <= 0 or args.concurrent_games <= 0:
        raise ValueError("--games and --concurrent-games must be positive")
    teams = resolve_team_pool(args.enemy_team_file, args.enemy_team_dir, minimum_teams=2)
    schedule = _schedule(teams, args.games, args.seed)
    configurations = _configuration_rows(args.candidate, args.q_weight)
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
            args.output_dir / "champion-policy-only",
            action_value_weight=0.0,
            load_preview_head=False,
            team_preview_policy="first",
        )
        _release_policy_memory()
        results = []
        for index, configuration in enumerate(configurations):
            candidate = _run_policy(
                args,
                args.candidate,
                schedule,
                args.output_dir / str(configuration["name"]),
                action_value_weight=float(configuration["action_value_weight"]),
                load_preview_head=bool(configuration["load_preview_head"]),
                team_preview_policy=str(configuration["team_preview_policy"]),
            )
            _release_policy_memory()
            comparison = _comparison(candidate, champion, seed=args.seed + index)
            row = {**configuration, **comparison, "candidate": candidate}
            results.append(row)
            print(json.dumps({"phase": "inference-ablation", **row}), flush=True)
    results.sort(
        key=lambda row: (
            float(row["win_rate_delta"]),
            float(row["paired_bootstrap_delta_95"][0]),
        ),
        reverse=True,
    )
    report = {
        "schema": "inference-ablation-v1",
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


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
