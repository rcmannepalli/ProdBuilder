"""In-process pub/sub event bus feeding the Monitor panel via SSE."""
from __future__ import annotations

import asyncio
import json
from typing import Any

# project_id -> set of subscriber queues
_subscribers: dict[int, set[asyncio.Queue]] = {}
_loop: asyncio.AbstractEventLoop | None = None


def set_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


def subscribe(project_id: int) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    _subscribers.setdefault(project_id, set()).add(q)
    return q


def unsubscribe(project_id: int, q: asyncio.Queue) -> None:
    subs = _subscribers.get(project_id)
    if subs and q in subs:
        subs.discard(q)


def publish(project_id: int, event: dict[str, Any]) -> None:
    """Publish an event. Safe to call from background (non-async) threads."""
    subs = _subscribers.get(project_id)
    if not subs:
        return
    payload = json.dumps(event, default=str)
    for q in list(subs):
        try:
            if _loop is not None and _loop.is_running():
                _loop.call_soon_threadsafe(_put_nowait, q, payload)
            else:
                _put_nowait(q, payload)
        except Exception:  # noqa: BLE001
            pass


def _put_nowait(q: asyncio.Queue, payload: str) -> None:
    try:
        q.put_nowait(payload)
    except asyncio.QueueFull:
        pass
