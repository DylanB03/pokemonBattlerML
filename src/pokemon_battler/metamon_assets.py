from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

METAMON_SELFPLAY_REPO = "jakegrigsby/metamon-parsed-pile"
METAMON_TEAMS_REPO = "jakegrigsby/metamon-teams"
DEFAULT_SELFPLAY_SUBSETS = ("pac-base", "pac-exploratory")
DEFAULT_TEAM_SETS = ("gl_05_26", "hl_05_26")


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    members = archive.getmembers()
    for member in members:
        if member.issym() or member.islnk():
            raise ValueError(f"Links are not allowed in downloaded archives: {member.name}")
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Unsafe path in downloaded archive: {member.name}")
    archive.extractall(destination, members=members)


def _download_file(*, repo_id: str, filename: str, root: Path, revision: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Downloading Metamon data requires huggingface-hub. Reinstall the project "
            "dependencies before running this command."
        ) from exc
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=root,
        revision=revision,
        repo_type="dataset",
    )
    return Path(downloaded)


def _decompress_tar_lz4(source: Path, destination: Path) -> None:
    import lz4.frame

    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    started = time.monotonic()
    written = 0
    with lz4.frame.open(source, "rb") as compressed, partial.open("wb") as output:
        while True:
            chunk = compressed.read(64 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            written += len(chunk)
            print(
                json.dumps(
                    {
                        "phase": "metamon-decompress",
                        "source": str(source),
                        "written_gib": round(written / 1024**3, 2),
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, destination)


def download_selfplay(
    root: Path,
    *,
    subset: str,
    battle_format: str = "gen9ou",
    revision: str = "main",
    keep_compressed: bool = False,
) -> Path:
    subset_dir = root / "self-play" / subset
    destination = subset_dir / f"{battle_format}.tar"
    if destination.is_file():
        return destination
    compressed = _download_file(
        repo_id=METAMON_SELFPLAY_REPO,
        filename=f"{subset}/{battle_format}.tar.lz4",
        root=root / "self-play",
        revision=revision,
    )
    subset_dir.mkdir(parents=True, exist_ok=True)
    _decompress_tar_lz4(compressed, destination)
    if not keep_compressed:
        compressed.unlink(missing_ok=True)
    return destination


def download_team_set(
    root: Path,
    *,
    set_name: str,
    battle_format: str = "gen9ou",
    revision: str = "v5",
    keep_compressed: bool = False,
) -> Path:
    destination = root / "teams" / set_name / battle_format
    if destination.is_dir() and any(destination.rglob("*")):
        return destination
    compressed = _download_file(
        repo_id=METAMON_TEAMS_REPO,
        filename=f"{set_name}/{battle_format}.tar.gz",
        root=root / "teams",
        revision=revision,
    )
    compressed.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(compressed, mode="r:gz") as archive:
        _safe_extract(archive, compressed.parent)
    if not destination.is_dir():
        candidates = [path for path in compressed.parent.rglob(battle_format) if path.is_dir()]
        if len(candidates) != 1:
            raise FileNotFoundError(f"Downloaded team archive did not create {destination}")
        destination = candidates[0]
    if not keep_compressed:
        compressed.unlink(missing_ok=True)
    return destination


def download_metamon_assets(
    root: Path,
    *,
    selfplay_subsets: Sequence[str] = DEFAULT_SELFPLAY_SUBSETS,
    team_sets: Sequence[str] = DEFAULT_TEAM_SETS,
    battle_format: str = "gen9ou",
    selfplay_revision: str = "main",
    teams_revision: str = "v5",
    keep_compressed: bool = False,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    selfplay = [
        download_selfplay(
            root,
            subset=subset,
            battle_format=battle_format,
            revision=selfplay_revision,
            keep_compressed=keep_compressed,
        )
        for subset in selfplay_subsets
    ]
    teams = {
        set_name: download_team_set(
            root,
            set_name=set_name,
            battle_format=battle_format,
            revision=teams_revision,
            keep_compressed=keep_compressed,
        )
        for set_name in team_sets
    }
    report = {
        "schema": "metamon-assets-v1",
        "battle_format": battle_format,
        "selfplay_revision": selfplay_revision,
        "teams_revision": teams_revision,
        "selfplay": [str(path) for path in selfplay],
        "teams": {name: str(path) for name, path in teams.items()},
    }
    (root / "assets.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download official Metamon self-play trajectories and Gen 9 teams."
    )
    parser.add_argument("--root", type=Path, default=Path("data/metamon-large"))
    parser.add_argument("--selfplay-subset", dest="selfplay_subsets", action="append")
    parser.add_argument("--team-set", dest="team_sets", action="append")
    parser.add_argument("--format", dest="battle_format", default="gen9ou")
    parser.add_argument("--selfplay-revision", default="main")
    parser.add_argument("--teams-revision", default="v5")
    parser.add_argument("--keep-compressed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = download_metamon_assets(
        args.root,
        selfplay_subsets=args.selfplay_subsets or DEFAULT_SELFPLAY_SUBSETS,
        team_sets=args.team_sets or DEFAULT_TEAM_SETS,
        battle_format=args.battle_format,
        selfplay_revision=args.selfplay_revision,
        teams_revision=args.teams_revision,
        keep_compressed=args.keep_compressed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
