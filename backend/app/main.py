from __future__ import annotations

import importlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .room import RoomManager
from .websocket import (
    BotRoomAutomation,
    ChatService,
    GameService,
    NoOpRoomAutomation,
    RealtimeRoomCoordinator,
    RoomBroadcaster,
    RoomAutomation,
    WebSocketGameService,
)


logger = logging.getLogger("uvicorn.error")


def _load_env_file(path: Path) -> bool:
    if not path.exists():
        return False
    loaded_any = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip("'").strip('"')
        if key not in os.environ:
            os.environ[key] = value
            loaded_any = True
    return loaded_any


def _init_env() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    env_candidates = [
        backend_root / ".env",
        Path.cwd() / ".env",
    ]
    loaded = False
    try:
        from dotenv import load_dotenv

        for candidate in env_candidates:
            if candidate.exists() and load_dotenv(candidate):
                loaded = True
    except Exception:
        for candidate in env_candidates:
            if _load_env_file(candidate):
                loaded = True

    if loaded:
        logger.info("Loaded environment from .env")
    else:
        logger.warning("No .env loaded; relying on process environment")


_init_env()


def _load_realtime_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "realtime_config.json"
    if not config_path.exists():
        return {"bot_module": "app.bot"}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_automation(*, game_service: GameService, chat_service: ChatService) -> RoomAutomation:
    config = _load_realtime_config()
    bot_module_name = str(config.get("bot_module") or "app.bot").strip().lower()
    if bot_module_name == "none":
        return NoOpRoomAutomation()

    bot_module = importlib.import_module(bot_module_name)
    bot_controller = bot_module.BotController(bot_module.bot_config_from_env())
    bot_manager = bot_module.BotManager([bot_controller])
    return BotRoomAutomation(
        bot_manager=bot_manager,
        chat_service=chat_service,
        game_service=game_service,
    )


def _memory_db_path() -> Path:
    configured = (os.getenv("BOT_MEMORY_DB_PATH") or "bot_memory.db").strip() or "bot_memory.db"
    candidate = Path(configured)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[1] / candidate


app = FastAPI(title="Tap Game API")
room = RoomManager()
broadcaster = RoomBroadcaster(room)
game_service = GameService(room=room)
chat_service = ChatService(
    broadcaster=broadcaster,
)
automation = _build_automation(game_service=game_service, chat_service=chat_service)
coordinator = RealtimeRoomCoordinator(
    broadcaster=broadcaster,
    chat_service=chat_service,
    game_service=game_service,
    automation=automation,
)
ws_service = WebSocketGameService(coordinator=coordinator)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    await ws_service.startup()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/memories/users")
async def memory_users() -> dict[str, list[str]]:
    db_path = _memory_db_path()
    if not db_path.exists():
        return {"users": []}
    try:
        with sqlite3.connect(str(db_path)) as conn:
            users = [
                str(row[0])
                for row in conn.execute(
                    """
                    SELECT DISTINCT username
                    FROM semantic_memories
                    ORDER BY username COLLATE NOCASE ASC
                    """
                ).fetchall()
                if row and row[0]
            ]
        return {"users": users}
    except sqlite3.Error:
        logger.exception("Failed to load memory users from %s", db_path)
        return {"users": []}


@app.get("/api/memories/{username}")
async def memory_records(username: str, limit: int = 100) -> dict[str, Any]:
    decoded_username = unquote(username).strip()
    if not decoded_username:
        return {"username": "", "memories": []}
    bounded_limit = max(1, min(500, int(limit)))
    db_path = _memory_db_path()
    if not db_path.exists():
        return {"username": decoded_username, "memories": []}
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                """
                SELECT id, memory_text, metadata_json
                FROM semantic_memories
                WHERE username = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (decoded_username, bounded_limit),
            ).fetchall()
        memories = [
            {
                "id": int(row[0]),
                "memoryText": str(row[1] or ""),
                "metadata": json.loads(row[2] or "{}"),
            }
            for row in rows
        ]
        return {"username": decoded_username, "memories": memories}
    except (sqlite3.Error, ValueError, TypeError):
        logger.exception("Failed to load memories for user %s", decoded_username)
        return {"username": decoded_username, "memories": []}


@app.delete("/api/memories/{username}")
async def clear_memory_records(username: str) -> dict[str, Any]:
    decoded_username = unquote(username).strip()
    if not decoded_username:
        return {"username": "", "deleted": 0}
    db_path = _memory_db_path()
    if not db_path.exists():
        return {"username": decoded_username, "deleted": 0}
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.execute(
                """
                DELETE FROM semantic_memories
                WHERE username = ?
                """,
                (decoded_username,),
            )
            deleted = int(cur.rowcount or 0)
            conn.commit()
        return {"username": decoded_username, "deleted": deleted}
    except sqlite3.Error:
        logger.exception("Failed to clear memories for user %s", decoded_username)
        return {"username": decoded_username, "deleted": 0}


app.add_api_websocket_route("/ws", ws_service.websocket_endpoint)
