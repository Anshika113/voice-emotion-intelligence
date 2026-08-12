"""Dashboard WebSocket fan-out.

Browsers connect to /ws/dashboard and receive JSON messages pushed by the call
orchestrator: live partial transcripts and final transcripts with fused emotion.
"""
from __future__ import annotations

import logging
from typing import Set

from fastapi import WebSocket

logger = logging.getLogger("voice.dashboard")


class ConnectionManager:
    def __init__(self) -> None:
        self._active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)
        logger.info("Dashboard client connected (%d total)", len(self._active))

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)
        logger.info("Dashboard client disconnected (%d total)", len(self._active))

    async def broadcast(self, message: dict) -> None:
        if not self._active:
            return
        dead = []
        for ws in list(self._active):
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 — drop broken clients
                dead.append(ws)
        for ws in dead:
            self._active.discard(ws)


manager = ConnectionManager()


async def broadcast(message: dict) -> None:
    """Module-level convenience used by the call orchestrator."""
    await manager.broadcast(message)
