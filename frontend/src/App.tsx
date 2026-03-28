import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { ChatMessage, RoomState, ServerEvent } from "./types";
import "./styles.css";

const API_WS_URL =
  import.meta.env.VITE_WS_URL ?? "ws://127.0.0.1:8000/ws";

const emptyState: RoomState = {
  players: [],
  scores: {},
  status: "waiting",
  roundEndTime: null,
};

function App() {
  const [usernameInput, setUsernameInput] = useState("");
  const [username, setUsername] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [roomState, setRoomState] = useState<RoomState>(emptyState);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [countdown, setCountdown] = useState<number>(0);
  const socketRef = useRef<WebSocket | null>(null);

  const leaderboard = useMemo(() => {
    return [...Object.entries(roomState.scores)].sort((a, b) => b[1] - a[1]);
  }, [roomState.scores]);

  const botNames = useMemo(() => {
    return new Set(roomState.players.filter((p) => p.isBot).map((p) => p.name));
  }, [roomState.players]);

  useEffect(() => {
    if (!username) return;

    const socket = new WebSocket(API_WS_URL);
    socketRef.current = socket;

    socket.onopen = () => {
      socket.send(JSON.stringify({ type: "join", username }));
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data) as ServerEvent;

      if (data.type === "error") {
        setError((data as { message?: string }).message ?? "Unknown error");
        return;
      }

      if (data.type === "chat_message" && data.message) {
        setChatMessages((prev) => [...prev, data.message!].slice(-200));
        return;
      }

      if (data.type === "room_state" && data.state) {
        setRoomState(data.state);
      }
    };

    socket.onclose = () => {
      setError("Disconnected from server");
    };

    return () => {
      socket.close();
      socketRef.current = null;
    };
  }, [username]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      if (!roomState.roundEndTime || roomState.status !== "active") {
        setCountdown(0);
        return;
      }
      const left = Math.max(0, Math.ceil(roomState.roundEndTime - Date.now() / 1000));
      setCountdown(left);
    }, 250);

    return () => window.clearInterval(interval);
  }, [roomState.roundEndTime, roomState.status]);

  function send(type: string, payload: Record<string, unknown> = {}) {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    socketRef.current.send(JSON.stringify({ type, ...payload }));
  }

  function onJoin(e: FormEvent) {
    e.preventDefault();
    const trimmed = usernameInput.trim();
    if (!trimmed) return;
    setError(null);
    setChatMessages([]);
    setUsername(trimmed);
  }

  function onSendChat(e: FormEvent) {
    e.preventDefault();
    const trimmed = chatInput.trim();
    if (!trimmed) return;
    send("chat_message", { text: trimmed });
    setChatInput("");
  }

  if (!username) {
    return (
      <main className="page">
        <section className="card join-card">
          <h1>Tap Game Demo</h1>
          <form onSubmit={onJoin} className="join-form">
            <input
              value={usernameInput}
              onChange={(e) => setUsernameInput(e.target.value)}
              placeholder="Enter username"
            />
            <button type="submit">Join Room</button>
          </form>
          {error && <p className="error">{error}</p>}
        </section>
      </main>
    );
  }

  return (
    <main className="page">
      <h1>Tap Game Demo</h1>
      <p>Signed in as <strong>{username}</strong></p>
      {error && <p className="error">{error}</p>}

      <section className="layout">
        <div className="card">
          <h2>Game</h2>
          <p>Status: <strong>{roomState.status}</strong></p>
          <p>Countdown: <strong>{countdown}s</strong></p>
          <button
            className="tap-button"
            disabled={roomState.status !== "active"}
            onClick={() => send("tap")}
          >
            TAP
          </button>
          <button onClick={() => send("start_round")}>Start Round</button>
        </div>

        <div className="card">
          <h2>Players ({roomState.players.length})</h2>
          <ul>
            {roomState.players.map((player) => (
              <li key={player.id}>
                {player.name}
                {player.isBot ? " (bot)" : ""}
              </li>
            ))}
          </ul>

          <h2>Leaderboard</h2>
          <ol>
            {leaderboard.map(([player, score]) => (
              <li key={player}>
                {player}
                {botNames.has(player) ? " (bot)" : ""}: {score}
              </li>
            ))}
          </ol>
        </div>

        <div className="card chat-card">
          <h2>Chat</h2>
          <div className="chat-log">
            {chatMessages.map((m, idx) => (
              <p
                key={`${m.timestamp}-${idx}`}
                className={m.system ? "system" : m.isBot ? "bot-msg" : ""}
              >
                <strong>{m.sender}{m.isBot ? " (bot)" : ""}:</strong> {m.text}
              </p>
            ))}
          </div>
          <form onSubmit={onSendChat} className="chat-form">
            <input
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Type message"
            />
            <button type="submit">Send</button>
          </form>
        </div>
      </section>
    </main>
  );
}

export default App;
