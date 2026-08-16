from __future__ import annotations

import argparse
import asyncio
import logging
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


def _fallback_decision(request: dict, rqid: int, *, team_preview: bool = False) -> list[str]:
    """Return a legal deterministic order if Foul Play's search engine panics."""
    if team_preview:
        return ["/switch 1", str(rqid)]
    force_switch = bool((request.get("forceSwitch") or [False])[0])
    if not force_switch:
        active = request.get("active") or []
        moves = (active[0].get("moves") or []) if active else []
        for move in moves:
            if not move.get("disabled") and int(move.get("pp", 1) or 0) > 0:
                move_id = str(move.get("id") or move.get("move") or "").strip()
                if move_id:
                    return [f"/choose move {move_id}", str(rqid)]
    for index, pokemon in enumerate((request.get("side") or {}).get("pokemon") or [], 1):
        condition = str(pokemon.get("condition") or "")
        if not pokemon.get("active") and not condition.endswith(" fnt") and condition != "0 fnt":
            return [f"/switch {index}", str(rqid)]
    raise RuntimeError("Foul Play search failed and its request had no legal fallback")


def main() -> None:
    known, remaining = build_parser().parse_known_args()
    checkout = known.checkout.resolve()
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    sys.argv = [sys.argv[0], *remaining]

    from fp import run_battle as foul_play_battle
    from fp.main import run_foul_play
    from fp.modes import base as foul_play_base
    from fp.websocket_client import PSWebsocketClient

    if known.teacher_trace is not None:
        from foul_play_teacher_bridge import install_foul_play_teacher_trace

        install_foul_play_teacher_trace(
            known.teacher_trace.resolve(),
            advisor_url=known.student_advisor_url,
            student_action_probability=known.student_action_probability,
            seed=known.dagger_seed,
        )

    original_pick_move = foul_play_battle.async_pick_move

    async def safe_pick_move(battle):
        try:
            return await original_pick_move(battle)
        except Exception:
            logging.getLogger(__name__).exception(
                "Foul Play search failed; submitting a deterministic legal fallback"
            )
            return _fallback_decision(
                battle.request_json,
                battle.rqid,
                team_preview=bool(battle.team_preview),
            )

    foul_play_base.async_pick_move = safe_pick_move
    foul_play_battle.async_pick_move = safe_pick_move

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
