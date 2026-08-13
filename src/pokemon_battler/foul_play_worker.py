from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--start-file", type=Path, required=True)
    parser.add_argument("--teacher-trace", type=Path)
    parser.add_argument("--student-advisor-url")
    parser.add_argument("--student-action-probability", type=float, default=0.0)
    parser.add_argument("--dagger-seed", type=int, default=42)
    return parser


def main() -> None:
    known, remaining = build_parser().parse_known_args()
    checkout = known.checkout.resolve()
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    sys.argv = [sys.argv[0], *remaining]

    from fp.main import run_foul_play
    from fp.websocket_client import PSWebsocketClient

    if known.teacher_trace is not None:
        from foul_play_teacher_bridge import install_foul_play_teacher_trace

        install_foul_play_teacher_trace(
            known.teacher_trace.resolve(),
            advisor_url=known.student_advisor_url,
            student_action_probability=known.student_action_probability,
            seed=known.dagger_seed,
        )

    async def local_no_security_login(client: PSWebsocketClient) -> str:
        await client.get_id_and_challstr()
        await client.send_message("", [f"/trn {client.username},0,"])
        # The local server runs with --no-security, so an external assertion is
        # neither necessary nor desirable. Give it time to register the name.
        await asyncio.sleep(0.25)
        known.ready_file.write_text(f"{client.username}\n", encoding="utf-8")
        while not known.start_file.is_file():
            await asyncio.sleep(0.05)
        return client.username

    PSWebsocketClient.login = local_no_security_login  # type: ignore[method-assign]
    asyncio.run(run_foul_play())


if __name__ == "__main__":
    main()
