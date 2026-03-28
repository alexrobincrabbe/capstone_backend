import { useEffect, useMemo, useState } from "react";
import { MemoryRecord } from "./types";
import "./styles.css";

const API_HTTP_URL =
  import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

function MemoryBrowser() {
  const [users, setUsers] = useState<string[]>([]);
  const [selectedUser, setSelectedUser] = useState<string>("");
  const [memories, setMemories] = useState<MemoryRecord[]>([]);
  const [loadingUsers, setLoadingUsers] = useState(false);
  const [loadingMemories, setLoadingMemories] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadUsers() {
    setLoadingUsers(true);
    setError(null);
    try {
      const res = await fetch(`${API_HTTP_URL}/api/memories/users`);
      if (!res.ok) {
        throw new Error("failed users request");
      }
      const data = (await res.json()) as { users?: string[] };
      const loadedUsers = Array.isArray(data.users) ? data.users : [];
      setUsers(loadedUsers);
      setSelectedUser((prev) => {
        if (prev && loadedUsers.includes(prev)) return prev;
        return loadedUsers[0] ?? "";
      });
    } catch {
      setError("Failed to load memory users");
    } finally {
      setLoadingUsers(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    loadUsers().catch(() => {
      if (!cancelled) setError("Failed to load memory users");
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedUser) {
      setMemories([]);
      return;
    }
    let cancelled = false;
    async function loadMemories() {
      setLoadingMemories(true);
      setError(null);
      try {
        const res = await fetch(
          `${API_HTTP_URL}/api/memories/${encodeURIComponent(selectedUser)}?limit=200`
        );
        const data = (await res.json()) as { memories?: MemoryRecord[] };
        if (cancelled) return;
        setMemories(Array.isArray(data.memories) ? data.memories : []);
      } catch {
        if (!cancelled) setError("Failed to load memories for selected user");
      } finally {
        if (!cancelled) setLoadingMemories(false);
      }
    }
    loadMemories();
    return () => {
      cancelled = true;
    };
  }, [selectedUser]);

  const emptyText = useMemo(() => {
    if (!selectedUser) return "No users found";
    if (loadingMemories) return "Loading memories...";
    if (memories.length === 0) return "No memories for this user yet";
    return "";
  }, [selectedUser, loadingMemories, memories.length]);

  async function onClearSelectedUser() {
    if (!selectedUser || clearing) return;
    const ok = window.confirm(`Clear all memories for ${selectedUser}?`);
    if (!ok) return;
    setClearing(true);
    setError(null);
    try {
      const res = await fetch(`${API_HTTP_URL}/api/memories/${encodeURIComponent(selectedUser)}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        throw new Error("failed clear request");
      }
      await loadUsers();
    } catch {
      setError("Failed to clear memories for selected user");
    } finally {
      setClearing(false);
    }
  }

  return (
    <main className="page">
      <div className="page-header">
        <h1>Bot Memory Browser</h1>
        <a href="/">Back to game</a>
      </div>
      {error && <p className="error">{error}</p>}
      <section className="layout">
        <div className="card">
          <h2>Users</h2>
          {loadingUsers && <p>Loading users...</p>}
          {!loadingUsers && users.length === 0 && <p>No memory users yet.</p>}
          <ul>
            {users.map((name) => (
              <li key={name}>
                <button
                  type="button"
                  className={name === selectedUser ? "selected-user" : ""}
                  onClick={() => setSelectedUser(name)}
                >
                  {name}
                </button>
              </li>
            ))}
          </ul>
        </div>
        <div className="card">
          <h2>Memories {selectedUser ? `for ${selectedUser}` : ""}</h2>
          {selectedUser && (
            <button
              type="button"
              className="danger-button"
              onClick={onClearSelectedUser}
              disabled={clearing}
            >
              {clearing ? "Clearing..." : "Clear Memories"}
            </button>
          )}
          {emptyText && <p>{emptyText}</p>}
          {!emptyText && (
            <ul className="memory-list">
              {memories.map((m) => (
                <li key={m.id}>
                  <p>{m.memoryText}</p>
                  <small>{JSON.stringify(m.metadata)}</small>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </main>
  );
}

export default MemoryBrowser;
