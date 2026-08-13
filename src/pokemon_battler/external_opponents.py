from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO


@dataclass(frozen=True)
class ExternalOpponentSpec:
    name: str
    repository: str
    revision: str
    checkout_name: str
    username: str
    worker: str
    license: str


EXTERNAL_OPPONENTS = {
    "pokechamp-one-step": ExternalOpponentSpec(
        name="pokechamp-one-step",
        repository="https://github.com/DanielLi10720/PokeChamp.git",
        revision="0f84c460319ebe733f8c3028e58a2a5452c60d85",
        checkout_name="pokechamp",
        username="PBOneStep",
        worker="pokechamp",
        license="MIT",
    ),
    "pokechamp-abyssal": ExternalOpponentSpec(
        name="pokechamp-abyssal",
        repository="https://github.com/DanielLi10720/PokeChamp.git",
        revision="0f84c460319ebe733f8c3028e58a2a5452c60d85",
        checkout_name="pokechamp",
        username="PBAbyssal",
        worker="pokechamp",
        license="MIT",
    ),
    "foul-play": ExternalOpponentSpec(
        name="foul-play",
        repository="https://github.com/pmariglia/foul-play.git",
        revision="25c976f05cbf2880eaa579afd6db1dcb2c3b57c6",
        checkout_name="foul-play",
        username="PBFoulPlay",
        worker="foul-play",
        license="GPL-3.0",
    ),
}

_FOUL_PLAY_DEPENDENCY_MARKER = ".pokemon-battler-dependencies-v1"
_FOUL_PLAY_DEPENDENCIES = (
    "requests==2.33.0",
    "websockets==14.1",
    "python-dateutil==2.8.0",
)


def _run_checked(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _git_revision(checkout: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _ensure_checkout(
    spec: ExternalOpponentSpec,
    opponents_dir: Path,
    *,
    bootstrap: bool,
) -> Path:
    checkout = opponents_dir / spec.checkout_name
    if not (checkout / ".git").is_dir():
        if not bootstrap:
            raise FileNotFoundError(
                f"{spec.name} is not installed at {checkout}; remove "
                "--no-bootstrap-opponents to install it"
            )
        if checkout.exists() and any(checkout.iterdir()):
            raise FileNotFoundError(
                f"Opponent checkout exists but is incomplete: {checkout}"
            )
        git = shutil.which("git")
        if git is None:
            raise RuntimeError("Automatic opponent setup requires git")
        opponents_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {spec.name} into {checkout}...", file=sys.stderr)
        _run_checked([git, "clone", "--depth", "1", spec.repository, str(checkout)])

    actual_revision = _git_revision(checkout)
    if actual_revision != spec.revision:
        if not bootstrap:
            raise RuntimeError(
                f"{spec.name} checkout is at {actual_revision}, expected {spec.revision}"
            )
        git = shutil.which("git")
        if git is None:
            raise RuntimeError("Pinning opponent revisions requires git")
        print(f"Pinning {spec.name} to {spec.revision[:12]}...", file=sys.stderr)
        fetch = subprocess.run(
            [git, "fetch", "--depth", "1", "origin", spec.revision],
            cwd=checkout,
            check=False,
        )
        if fetch.returncode != 0:
            raise RuntimeError(
                f"Could not fetch pinned {spec.name} revision {spec.revision}"
            )
        _run_checked([git, "checkout", "--detach", spec.revision], cwd=checkout)

    return checkout


def _find_uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Foul Play setup requires uv. Install uv or prepare "
            "data/opponents/foul-play/.venv manually."
        )
    return uv


def _ensure_foul_play_environment(checkout: Path, *, bootstrap: bool) -> Path:
    environment = checkout / ".venv"
    python = environment / "bin" / "python"
    marker = environment / _FOUL_PLAY_DEPENDENCY_MARKER
    if python.is_file() and marker.is_file():
        return Path(os.path.abspath(python))
    if not bootstrap:
        raise FileNotFoundError(
            f"Foul Play's isolated environment is incomplete at {environment}; "
            "remove --no-bootstrap-opponents to prepare it"
        )

    uv = _find_uv()
    if not python.is_file():
        print("Creating Foul Play's isolated Python environment...", file=sys.stderr)
        _run_checked([uv, "venv", str(environment), "--python", sys.executable])
    print("Installing Foul Play's pinned dependencies...", file=sys.stderr)
    _run_checked(
        [uv, "pip", "install", "--python", str(python), *_FOUL_PLAY_DEPENDENCIES]
    )
    # Upstream's requirements.txt quotes this value in a way uv cannot parse.
    # Passing it directly preserves the published feature selection.
    _run_checked(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "poke-engine==0.0.48",
            "--config-settings",
            "build-args=--features poke-engine/terastallization --no-default-features",
        ]
    )
    marker.write_text("poke-engine==0.0.48\n", encoding="utf-8")
    return Path(os.path.abspath(python))


class ExternalOpponentProcess:
    """Bootstrap and manage a pinned third-party Showdown opponent."""

    def __init__(
        self,
        opponent: str,
        *,
        opponents_dir: Path,
        output_dir: Path,
        team_file: Path,
        battle_format: str,
        games: int,
        server_port: int,
        bootstrap: bool = True,
        startup_timeout: float = 90.0,
        challenger: str = "PBPolicy",
        foul_play_search_time_ms: int = 100,
        foul_play_parallelism: int = 1,
        foul_play_search_threads: int = 1,
        username: str | None = None,
        foul_play_mode: str = "challenge_user",
        foul_play_team_files: Sequence[Path] | None = None,
        capture_teacher_trace: bool = True,
        student_advisor_url: str | None = None,
        student_action_probability: float = 0.0,
        dagger_seed: int = 42,
    ) -> None:
        self.spec = EXTERNAL_OPPONENTS[opponent]
        self.opponents_dir = opponents_dir
        self.output_dir = output_dir
        self.team_file = team_file
        self.battle_format = battle_format
        self.games = games
        self.server_port = server_port
        self.bootstrap = bootstrap
        self.startup_timeout = startup_timeout
        self.challenger = challenger
        self.foul_play_search_time_ms = foul_play_search_time_ms
        self.foul_play_parallelism = foul_play_parallelism
        self.foul_play_search_threads = foul_play_search_threads
        self._username = username or self.spec.username
        if foul_play_mode not in {"challenge_user", "accept_challenge"}:
            raise ValueError(
                "foul_play_mode must be 'challenge_user' or 'accept_challenge'"
            )
        self.foul_play_mode = foul_play_mode
        self.foul_play_team_files = (
            [Path(path).resolve() for path in foul_play_team_files]
            if foul_play_team_files is not None
            else None
        )
        self.capture_teacher_trace = capture_teacher_trace
        self.student_advisor_url = student_advisor_url
        self.student_action_probability = student_action_probability
        self.dagger_seed = dagger_seed
        self.checkout: Path | None = None
        self.process: subprocess.Popen[str] | None = None
        self.log_path = output_dir / "opponent.log"
        self.ready_path = output_dir / "opponent.ready"
        self.start_path = output_dir / "opponent.start"
        self.teacher_trace_path = output_dir / "foul_play_teacher.jsonl"
        self._log_stream: IO[str] | None = None

    @property
    def username(self) -> str:
        return self._username

    def prepare(self) -> None:
        self.checkout = _ensure_checkout(
            self.spec,
            self.opponents_dir,
            bootstrap=self.bootstrap,
        )
        if self.spec.worker == "foul-play":
            _ensure_foul_play_environment(self.checkout, bootstrap=self.bootstrap)

    def _pokechamp_command(self) -> tuple[list[str], dict[str, str]]:
        assert self.checkout is not None
        worker = Path(__file__).with_name("pokechamp_worker.py")
        kind = "one-step" if self.spec.name.endswith("one-step") else "abyssal"
        command = [
            sys.executable,
            str(worker),
            "--kind",
            kind,
            "--username",
            self.username,
            "--challenger",
            self.challenger,
            "--games",
            str(self.games),
            "--battle-format",
            self.battle_format,
            "--team-file",
            str(self.team_file.resolve()),
            "--server-port",
            str(self.server_port),
            "--ready-file",
            str(self.ready_path.resolve()),
        ]
        environment = os.environ.copy()
        existing = environment.get("PYTHONPATH")
        checkout_path = str(self.checkout.resolve())
        environment["PYTHONPATH"] = (
            checkout_path
            if not existing
            else os.pathsep.join((checkout_path, existing))
        )
        return command, environment

    def _foul_play_command(self) -> tuple[list[str], dict[str, str]]:
        assert self.checkout is not None
        python = _ensure_foul_play_environment(
            self.checkout,
            bootstrap=self.bootstrap,
        )
        team_directory = self.checkout / "fp" / "teams" / "teams"
        safe_username = "".join(
            character for character in self.username.lower() if character.isalnum()
        )
        team_prefix = f"pokemon-battler-{safe_username}"
        team_arguments: list[str]
        if self.foul_play_team_files is None:
            team_target = team_directory / f"{team_prefix}.txt"
            shutil.copyfile(self.team_file, team_target)
            team_arguments = ["--team-name", team_target.name]
        else:
            if not self.foul_play_team_files:
                raise ValueError("A Foul Play team schedule cannot be empty")
            copied_names: dict[Path, str] = {}
            scheduled_names: list[str] = []
            for source in self.foul_play_team_files:
                if source not in copied_names:
                    target_name = f"{team_prefix}-{len(copied_names):03d}.txt"
                    shutil.copyfile(source, team_directory / target_name)
                    copied_names[source] = target_name
                scheduled_names.append(copied_names[source])
            team_list = team_directory / f"{team_prefix}-list.txt"
            team_list.write_text("\n".join(scheduled_names) + "\n", encoding="utf-8")
            team_arguments = ["--team-list", team_list.name]
        trace_arguments = (
            ["--teacher-trace", str(self.teacher_trace_path.resolve())]
            if self.capture_teacher_trace
            else []
        )
        advisor_arguments = (
            [
                "--student-advisor-url",
                self.student_advisor_url,
                "--student-action-probability",
                str(self.student_action_probability),
                "--dagger-seed",
                str(self.dagger_seed),
            ]
            if self.student_advisor_url is not None
            else []
        )
        worker = Path(__file__).with_name("foul_play_worker.py")
        command = [
            str(python),
            str(worker),
            "--checkout",
            str(self.checkout.resolve()),
            "--ready-file",
            str(self.ready_path.resolve()),
            "--start-file",
            str(self.start_path.resolve()),
            *trace_arguments,
            *advisor_arguments,
            "--websocket-uri",
            f"ws://localhost:{self.server_port}/showdown/websocket",
            "--ps-username",
            self.username,
            "--bot-mode",
            self.foul_play_mode,
            "--pokemon-format",
            self.battle_format,
            "--run-count",
            str(self.games),
            *team_arguments,
            "--search-time-ms",
            str(self.foul_play_search_time_ms),
            "--search-parallelism",
            str(self.foul_play_parallelism),
            "--search-threads",
            str(self.foul_play_search_threads),
            "--log-level",
            "INFO",
        ]
        if self.foul_play_mode == "challenge_user":
            command.extend(["--user-to-challenge", self.challenger])
        return command, os.environ.copy()

    def __enter__(self) -> ExternalOpponentProcess:  # noqa: PYI034
        self.prepare()
        self.ready_path.unlink(missing_ok=True)
        self.start_path.unlink(missing_ok=True)
        command, environment = (
            self._pokechamp_command()
            if self.spec.worker == "pokechamp"
            else self._foul_play_command()
        )
        self._log_stream = self.log_path.open("w", encoding="utf-8")
        print(
            f"Starting {self.spec.name} as Showdown user {self.username}...",
            file=sys.stderr,
        )
        self.process = subprocess.Popen(
            command,
            cwd=self.checkout,
            env=environment,
            stdout=self._log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.ready_path.is_file():
                return self
            if self.process.poll() is not None:
                self._close_log()
                tail = self._log_tail()
                raise RuntimeError(
                    f"{self.spec.name} exited with status {self.process.returncode} "
                    f"during startup. Log tail:\n{tail}"
                )
            time.sleep(0.1)
        self.close()
        raise TimeoutError(
            f"{self.spec.name} did not become ready within {self.startup_timeout:g} "
            f"seconds; inspect {self.log_path}"
        )

    def _close_log(self) -> None:
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None

    def _log_tail(self, lines: int = 40) -> str:
        if not self.log_path.is_file():
            return "(no opponent log)"
        return "\n".join(self.log_path.read_text(encoding="utf-8").splitlines()[-lines:])

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        self._close_log()

    def ensure_success(self) -> None:
        if self.process is None:
            return
        try:
            return_code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                f"{self.spec.name} did not exit after its {self.games} battles; "
                f"inspect {self.log_path}"
            ) from error
        finally:
            self._close_log()
        if return_code != 0:
            raise RuntimeError(
                f"{self.spec.name} exited with status {return_code}. Log tail:\n"
                f"{self._log_tail()}"
            )

    @property
    def challenges_player(self) -> bool:
        return (
            self.spec.worker == "foul-play"
            and self.foul_play_mode == "challenge_user"
        )

    def start(self) -> None:
        """Release a local Foul Play worker after every peer reports ready."""
        if self.spec.worker == "foul-play":
            self.start_path.write_text(f"{self.challenger}\n", encoding="utf-8")

    def start_challenges(self) -> None:
        if not self.challenges_player:
            return
        self.start()

    def metadata(self) -> dict[str, str | int]:
        return {
            "implementation": self.spec.name,
            "repository": self.spec.repository,
            "revision": self.spec.revision,
            "license": self.spec.license,
            "username": self.username,
            "team_preview": (
                "published-search-preview"
                if self.spec.worker == "foul-play"
                else "published-random-preview"
            ),
            **(
                {
                    "search_time_ms": self.foul_play_search_time_ms,
                    "search_parallelism": self.foul_play_parallelism,
                    "search_threads": self.foul_play_search_threads,
                    **(
                        {"teacher_trace": str(self.teacher_trace_path)}
                        if self.capture_teacher_trace
                        else {}
                    ),
                    **(
                        {
                            "student_advisor_url": self.student_advisor_url,
                            "student_action_probability": self.student_action_probability,
                        }
                        if self.student_advisor_url is not None
                        else {}
                    ),
                }
                if self.spec.worker == "foul-play"
                else {}
            ),
        }

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
