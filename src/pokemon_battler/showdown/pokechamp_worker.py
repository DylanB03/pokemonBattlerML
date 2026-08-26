from __future__ import annotations

import argparse
import asyncio
import atexit
import importlib.machinery
import sys
import types
from pathlib import Path


def _install_unused_dependency_shims() -> None:
    """Satisfy imports used only by PokéChamp's unrelated helper paths."""
    try:
        import openai  # noqa: F401
    except ImportError:
        module = types.ModuleType("openai")
        module.__spec__ = importlib.machinery.ModuleSpec("openai", loader=None)

        class UnavailableOpenAI:
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise RuntimeError("The non-LLM PokéChamp baselines do not use OpenAI")

        class RateLimitError(Exception):
            pass

        module.OpenAI = UnavailableOpenAI  # type: ignore[attr-defined]
        module.RateLimitError = RateLimitError  # type: ignore[attr-defined]
        sys.modules["openai"] = module

    try:
        import pandas  # noqa: F401
    except ImportError:
        module = types.ModuleType("pandas")
        module.__spec__ = importlib.machinery.ModuleSpec("pandas", loader=None)

        class UnavailableDataFrame:
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise RuntimeError("The PokéChamp battle baselines do not use pandas")

        module.DataFrame = UnavailableDataFrame  # type: ignore[attr-defined]
        sys.modules["pandas"] = module

    def stub_class_module(module_name: str, class_name: str) -> None:
        module = types.ModuleType(module_name)
        module.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)
        placeholder = type(class_name, (), {})
        setattr(module, class_name, placeholder)
        sys.modules[module_name] = module

    # These are imported eagerly by PokéChamp's package initializers even though
    # neither selected baseline constructs or calls them.
    stub_class_module("pokechamp.llama_player", "LLAMAPlayer")
    stub_class_module("pokechamp.llm_vgc_player", "LLMVGCPlayer")
    stub_class_module("pokechamp.mcp_player", "MCPPlayer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("one-step", "abyssal"), required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--battle-format", required=True)
    parser.add_argument("--team-file", type=Path, required=True)
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    return parser


async def run(args: argparse.Namespace) -> None:
    _install_unused_dependency_shims()
    from poke_env.concurrency import handle_threaded_coroutines
    from poke_env.player.baselines import AbyssalPlayer, OneStepPlayer
    from poke_env.ps_client.account_configuration import AccountConfiguration
    from poke_env.ps_client.server_configuration import ServerConfiguration

    team = args.team_file.read_text(encoding="utf-8").strip()
    player_class = OneStepPlayer if args.kind == "one-step" else AbyssalPlayer
    player = player_class(
        account_configuration=AccountConfiguration(args.username, None),
        battle_format=args.battle_format,
        save_replays=False,
        server_configuration=ServerConfiguration(
            f"localhost:{args.server_port}",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        team=team,
    )
    await handle_threaded_coroutines(player.ps_client.logged_in.wait())
    args.ready_file.write_text(f"{args.username}\n", encoding="utf-8")
    try:
        await player.accept_challenges(args.challenger, args.games)
    finally:
        await player.ps_client.stop_listening()


def _install_safe_shutdown() -> None:
    from poke_env import concurrency

    broken_handler = getattr(concurrency, "__clear_loop", None)
    if broken_handler is not None:
        atexit.unregister(broken_handler)

    def shutdown() -> None:
        loop = concurrency.POKE_LOOP
        thread = concurrency._t
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)
        if not thread.is_alive() and not loop.is_closed():
            loop.close()

    atexit.register(shutdown)


def main() -> None:
    args = build_parser().parse_args()
    # Importing PokéChamp creates its background event loop.
    _install_unused_dependency_shims()
    from poke_env import concurrency  # noqa: F401

    _install_safe_shutdown()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
