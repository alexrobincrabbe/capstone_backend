# FastAPI + React Tap Game Demo

Very small realtime multiplayer demo with one shared room.

## Project structure

- `backend/` - FastAPI + WebSocket server with in-memory room state
- `frontend/` - React + TypeScript client (Vite)

## Backend setup

```bash
cd backend
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend URL: `http://127.0.0.1:8000`
WebSocket endpoint: `ws://127.0.0.1:8000/ws`

## Frontend setup

```bash
cd frontend
npm install
# optional: copy .env.example to .env and change VITE_WS_URL
npm run dev
```

Frontend URL: `http://127.0.0.1:5173`

## How gameplay works

- Join with a username (duplicate usernames are rejected)
- Click **Start Round** to start a 20-second round
- While round is active, click **TAP** to increase your score
- Scores and chat update live for all connected players
- At 20 seconds, round ends and leaderboard remains visible
- Start another round any time after that

## Notes

- Single room only
- No database
- In-memory state only (reset on backend restart)
- Server-authoritative score counting

## Future AI participant hook

This demo bot lives in `backend/app/bot/` (`GameBot` facade).

To swap the placeholder bot chat with a real LLM later:

1. Replace the message construction inside `GameBot.maybe_reply_to_chat(...)` (the section that currently sets `reply = ...`).
2. Optionally also replace the join/round templates in `on_player_joined`, `on_round_started`, and `on_round_ended` if you want the AI to handle those too.

The bot's chat still flows through the same broadcast pipeline as human messages, so the frontend protocol does not need to change.

## Bot configuration (optional)

The bot is configured via environment variables on the backend:

- `BOT_NAME` (default: `TapBot`)
- `BOT_PERSONALITY` (default: `friendly and hype`)
- `BOT_SKILL_LEVEL` (default: `6`, clamped to `1..10`)
