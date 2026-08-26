from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import IO


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


class LocalShowdownServer:
    """Bootstrap and manage an official local Pokémon Showdown server."""

    def __init__(
        self,
        directory: str | Path,
        *,
        port: int = 8000,
        bootstrap: bool = True,
        startup_timeout: float = 60.0,
        log_path: str | Path | None = None,
        stop_on_exit: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.port = port
        self.bootstrap = bootstrap
        self.startup_timeout = startup_timeout
        self.log_path = Path(log_path) if log_path is not None else None
        self.stop_on_exit = stop_on_exit
        self.process: subprocess.Popen[str] | None = None
        self._log_stream: IO[str] | None = None
        self.reused_existing_server = False

    def _install(self) -> None:
        executable = self.directory / "pokemon-showdown"
        if not executable.is_file():
            if not self.bootstrap:
                raise FileNotFoundError(
                    f"Pokémon Showdown is not installed at {self.directory}. "
                    "Remove --no-bootstrap-server or clone the official server there."
                )
            if self.directory.exists() and any(self.directory.iterdir()):
                raise FileNotFoundError(
                    f"Showdown directory exists but is incomplete: {self.directory}"
                )
            git = shutil.which("git")
            if git is None:
                raise RuntimeError("Automatic Showdown setup requires git")
            self.directory.parent.mkdir(parents=True, exist_ok=True)
            print(f"Cloning Pokémon Showdown into {self.directory}...", file=sys.stderr)
            subprocess.run(
                [
                    git,
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/smogon/pokemon-showdown.git",
                    str(self.directory),
                ],
                check=True,
            )

        node_modules = self.directory / "node_modules"
        if not node_modules.is_dir():
            if not self.bootstrap:
                raise FileNotFoundError(
                    f"Showdown dependencies are not installed in {self.directory}"
                )
            npm = shutil.which("npm")
            if npm is None:
                raise RuntimeError("Automatic Showdown setup requires npm")
            print("Installing Pokémon Showdown dependencies...", file=sys.stderr)
            subprocess.run([npm, "install"], cwd=self.directory, check=True)

        config = self.directory / "config" / "config.js"
        example = self.directory / "config" / "config-example.js"
        if not config.exists() and example.is_file():
            shutil.copyfile(example, config)

    def prepare(self) -> None:
        """Ensure the official server and validator are available without starting it."""
        self._install()

    def __enter__(self) -> LocalShowdownServer:
        if _port_is_open("127.0.0.1", self.port):
            self.reused_existing_server = True
            print(
                f"Using the existing Showdown server on localhost:{self.port}.",
                file=sys.stderr,
            )
            return self

        self._install()
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("Starting Pokémon Showdown requires Node.js")
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_stream = self.log_path.open("w", encoding="utf-8")
        command = [node, "pokemon-showdown", "start"]
        if self.port != 8000:
            command.append(str(self.port))
        command.append("--no-security")
        print(f"Starting Pokémon Showdown on localhost:{self.port}...", file=sys.stderr)
        self.process = subprocess.Popen(
            command,
            cwd=self.directory,
            stdout=self._log_stream or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Pokémon Showdown exited with status {self.process.returncode}; "
                    f"inspect {self.log_path or 'its console output'}"
                )
            if _port_is_open("127.0.0.1", self.port):
                return self
            time.sleep(0.1)
        self.close()
        raise TimeoutError(
            f"Pokémon Showdown did not listen on port {self.port} within "
            f"{self.startup_timeout:g} seconds"
        )

    def close(self) -> None:
        if self.process is not None and self.stop_on_exit and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
