from __future__ import annotations

import os


def normalize_postgres_url(url: str) -> str:
    u = url.strip()
    if u.startswith("postgres://"):
        u = "postgresql://" + u[len("postgres://") :]
    return u


def memory_database_dsn_from_environ() -> str | None:
    raw = (os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DATABASE_URL") or "").strip()
    if not raw:
        return None
    return normalize_postgres_url(raw)
