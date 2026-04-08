# Backend App

FastAPI backend for the tap-game realtime room, bot orchestration, and bot memory APIs.

## What This Service Does

- Hosts the HTTP API and WebSocket endpoint.
- Maintains game-room state and realtime coordination.
- Runs bot automation (`TapBot`) on top of room/game/chat events.
- Stores and serves semantic bot memories from Postgres (Supabase-compatible).

## High-Level Architecture

- **App entrypoint**: `app/main.py`
  - Loads environment variables.
  - Wires core services.
  - Registers routes (`/health`, memory APIs, `/ws`).
- **Realtime core**: `app/websocket/*`
  - `RealtimeRoomCoordinator` coordinates chat/game automation events.
  - `WebSocketGameService` exposes WebSocket handling.
  - `ChatService` / `GameService` handle chat and gameplay state operations.
- **Room model**: `app/room/*`
  - Room and participant state management.
- **Bot module**: `app/bot/*`
  - `BotManager` and `BotController` bridge realtime events to bot systems.
  - Chat bot logic is implemented in `app/bot/chat/*`.
- **Memory DB utility**: `app/memory_db.py`
  - Resolves/normalizes DSN from env (`DATABASE_URL` or `SUPABASE_DATABASE_URL`).

## Runtime Flow

1. Client connects to `/ws`.
2. Coordinator receives events (joins, chat messages, round state changes).
3. Bot automation forwards relevant events to `BotManager`.
4. `BotController` delegates:
   - chat behavior -> `BotChatEngine`
   - tap behavior -> gameplay tap loop
5. Bot response is emitted back via chat service.

## API Surface

- `GET /health` -> health check.
- `GET /api/memories/users` -> list usernames with stored semantic memories.
- `GET /api/memories/{username}?limit=100` -> memory records for a user.
- `DELETE /api/memories/{username}` -> delete all semantic memories for a user.
- `WS /ws` -> realtime game/chat channel.

## Environment Variables

Core:

- `DATABASE_URL` or `SUPABASE_DATABASE_URL` (required for memory features)
- `OPENAI_API_KEY` (required for LLM router/response/embeddings)

Optional bot tuning:

- `BOT_NAME` (default: `TapBot`)
- `BOT_PERSONALITY` (default: `friendly and hype`)
- `BOT_SKILL_LEVEL` (1..10, default: `6`)
- `BOT_RECENT_MESSAGE_LIMIT` (default varies by branch/config; controls retained chat turns)
- `BOT_CONTEXT_HISTORY_LIMIT` (default varies by branch/config; controls non-full-history context)
- `BOT_LLM_ROUTER_MODEL` (default: `gpt-4o-mini`)
- `BOT_LLM_RESPONSE_MODEL` (default: `gpt-4o-mini`)
- `BOT_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `BOT_TRACE` (`true/false`) to enable detailed bot trace logs.

Optional tracing:

- `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, etc.

## Local Development

From the `backend` directory:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

- HTTP: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`
- WebSocket: `ws://127.0.0.1:8000/ws`

## Notes

- The semantic memory table (`semantic_memories`) is created lazily by the chat memory service.
- If bot module loading is disabled via realtime config (`bot_module: "none"`), realtime still works without bot automation.
