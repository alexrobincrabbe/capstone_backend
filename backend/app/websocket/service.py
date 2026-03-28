from __future__ import annotations

import json
from json import JSONDecodeError

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .connection import ClientConnection
from .coordinator import RealtimeRoomCoordinator
from .models import ClientEvent

import logging

logger = logging.getLogger("uvicorn.error")


class WebSocketGameService:
    def __init__(self, *, coordinator: RealtimeRoomCoordinator) -> None:
        self.coordinator = coordinator

    async def startup(self) -> None:
        await self.coordinator.startup()

    async def websocket_endpoint(self, websocket: WebSocket) -> None:
        await websocket.accept()
        logger.info("WebSocket accepted")
        connection = ClientConnection(websocket=websocket)

        try:
            while True:
                raw = await websocket.receive_text()
                logger.debug("WebSocket received raw payload: %s", raw)
                try:
                    payload = ClientEvent(**json.loads(raw))
                except (JSONDecodeError, ValidationError, TypeError):
                    logger.warning("Invalid websocket payload received")
                    await websocket.send_json({"type": "error", "message": "Invalid event payload"})
                    continue

                if payload.type == "join":
                    await self.coordinator.handle_join(connection, payload.username)
                    continue

                if connection.username is None:
                    await websocket.send_json({"type": "error", "message": "Join first"})
                    continue

                if payload.type == "chat_message":
                    await self.coordinator.handle_chat_message(connection, payload.text)
                elif payload.type == "start_round":
                    await self.coordinator.handle_start_round(connection)
                elif payload.type == "tap":
                    await self.coordinator.handle_tap(connection)
                else:
                    await websocket.send_json({"type": "error", "message": f"Unknown event: {payload.type}"})

        except WebSocketDisconnect as exc:
            logger.info("WebSocket disconnected (code=%s)", exc.code)
        except Exception:
            logger.exception("Unhandled websocket exception")
        finally:
            await self.coordinator.handle_disconnect(connection)
