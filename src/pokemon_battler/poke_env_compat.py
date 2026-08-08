from __future__ import annotations

import asyncio
import atexit
from typing import Any


def install_safe_poke_env_shutdown() -> None:
    """Replace poke-env 0.15's blocking Python 3.12 exit handler."""
    from poke_env import concurrency

    if getattr(concurrency, "_pokemon_battler_safe_shutdown", False):
        return
    broken_handler = getattr(concurrency, "__clear_loop", None)
    if broken_handler is not None:
        atexit.unregister(broken_handler)

    loop: asyncio.AbstractEventLoop = concurrency.POKE_LOOP
    thread: Any = concurrency._t

    def shutdown() -> None:
        if loop.is_closed():
            return

        def cancel_and_stop() -> None:
            for task in asyncio.all_tasks(loop):
                task.cancel()
            loop.stop()

        if loop.is_running():
            loop.call_soon_threadsafe(cancel_and_stop)
            # Never hold process shutdown forever if an upstream event loop is
            # already unhealthy. Its thread is daemonized by poke-env.
            thread.join(timeout=2.0)
        if not thread.is_alive() and not loop.is_closed():
            loop.close()

    atexit.register(shutdown)
    concurrency._pokemon_battler_safe_shutdown = True
