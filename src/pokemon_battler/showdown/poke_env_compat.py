from __future__ import annotations

import asyncio
import atexit
from concurrent.futures import Future
from typing import Any


async def close_poke_env_clients(*clients: Any) -> None:
    """Close client sockets and wait for poke-env's listener tasks to finish."""
    if not clients:
        return

    results = await asyncio.gather(
        *(client.stop_listening() for client in clients),
        return_exceptions=True,
    )
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise RuntimeError(
            f"Failed to close {len(errors)} poke-env client(s)"
        ) from errors[0]

    listener_futures: list[Future[Any]] = [
        client._listening_coroutine
        for client in clients
        if hasattr(client, "_listening_coroutine")
    ]
    if listener_futures:
        await asyncio.wait_for(
            asyncio.gather(
                *(asyncio.wrap_future(future) for future in listener_futures),
                return_exceptions=True,
            ),
            timeout=5.0,
        )


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

        async def cancel_pending_tasks() -> None:
            current = asyncio.current_task()
            tasks = [task for task in asyncio.all_tasks(loop) if task is not current]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        if loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(
                    cancel_pending_tasks(), loop
                ).result(timeout=2.0)
            except Exception:
                # Exit must not hang because an upstream task stopped responding.
                pass
            loop.call_soon_threadsafe(loop.stop)
            # Never hold process shutdown forever if an upstream event loop is
            # already unhealthy. Its thread is daemonized by poke-env.
            thread.join(timeout=2.0)
        if not thread.is_alive() and not loop.is_closed():
            loop.close()

    atexit.register(shutdown)
    concurrency._pokemon_battler_safe_shutdown = True
