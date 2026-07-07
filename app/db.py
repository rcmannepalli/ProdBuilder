"""DuckDB persistence layer.

Single-process app: we serialize all access through one connection behind a
re-entrant lock so background build tasks and web requests never collide.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Iterable

import duckdb

from .config import DB_PATH

_lock = threading.RLock()
_conn: duckdb.DuckDBPyConnection | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        _conn = duckdb.connect(str(DB_PATH))
        _init_schema(_conn)
    return _conn


SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS seq_projects START 1;
CREATE SEQUENCE IF NOT EXISTS seq_reqs START 1;
CREATE SEQUENCE IF NOT EXISTS seq_settings START 1;
CREATE SEQUENCE IF NOT EXISTS seq_phases START 1;
CREATE SEQUENCE IF NOT EXISTS seq_tasks START 1;
CREATE SEQUENCE IF NOT EXISTS seq_test_runs START 1;
CREATE SEQUENCE IF NOT EXISTS seq_events START 1;
CREATE SEQUENCE IF NOT EXISTS seq_proposals START 1;
CREATE SEQUENCE IF NOT EXISTS seq_providers START 1;

CREATE TABLE IF NOT EXISTS projects (
    id BIGINT PRIMARY KEY DEFAULT nextval('seq_projects'),
    name TEXT NOT NULL,
    target_folder TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'idle',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS requirements_versions (
    id BIGINT PRIMARY KEY DEFAULT nextval('seq_reqs'),
    project_id BIGINT NOT NULL,
    version_no INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    structured_json TEXT,
    diff_from_prev TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    id BIGINT PRIMARY KEY DEFAULT nextval('seq_settings'),
    project_id BIGINT NOT NULL UNIQUE,
    ollama_base_url TEXT,
    ollama_api_key_enc TEXT,
    model_map_json TEXT,
    provider_map_json TEXT,
    max_fix_attempts INTEGER DEFAULT 3,
    auto_apply_changes BOOLEAN DEFAULT FALSE,
    poll_interval_sec INTEGER DEFAULT 60,
    test_command TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS phases (
    id BIGINT PRIMARY KEY DEFAULT nextval('seq_phases'),
    project_id BIGINT NOT NULL,
    order_no INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    deliverables_json TEXT,
    test_plan_json TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id BIGINT PRIMARY KEY DEFAULT nextval('seq_tasks'),
    phase_id BIGINT NOT NULL,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    file_paths_json TEXT,
    detail TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS test_runs (
    id BIGINT PRIMARY KEY DEFAULT nextval('seq_test_runs'),
    phase_id BIGINT NOT NULL,
    attempt_no INTEGER NOT NULL,
    passed BOOLEAN,
    summary TEXT,
    stdout TEXT,
    stderr TEXT,
    duration_ms INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_events (
    id BIGINT PRIMARY KEY DEFAULT nextval('seq_events'),
    project_id BIGINT NOT NULL,
    phase_id BIGINT,
    agent_name TEXT,
    event_type TEXT,
    message TEXT,
    payload_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS providers (
    id BIGINT PRIMARY KEY DEFAULT nextval('seq_providers'),
    project_id BIGINT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'ollama_cloud',
    base_url TEXT,
    api_key_enc TEXT,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS change_proposals (
    id BIGINT PRIMARY KEY DEFAULT nextval('seq_proposals'),
    project_id BIGINT NOT NULL,
    source TEXT NOT NULL,
    summary TEXT,
    diff_json TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL
);
"""


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    for stmt in SCHEMA.strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(s)
    # Lightweight migrations for pre-existing databases.
    for mig in (
        "ALTER TABLE settings ADD COLUMN IF NOT EXISTS provider_map_json TEXT",
    ):
        try:
            conn.execute(mig)
        except Exception:  # noqa: BLE001 - column may already exist
            pass


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _cols(cur) -> list[str]:
    return [d[0] for d in cur.description]


def query(sql: str, params: Iterable[Any] = ()) -> list[dict]:
    with _lock:
        cur = get_conn().execute(sql, list(params))
        cols = _cols(cur)
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def query_one(sql: str, params: Iterable[Any] = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: Iterable[Any] = ()) -> None:
    with _lock:
        get_conn().execute(sql, list(params))


def insert_returning_id(sql: str, params: Iterable[Any]) -> int:
    with _lock:
        cur = get_conn().execute(sql + " RETURNING id", list(params))
        row = cur.fetchone()
        return int(row[0])


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def loads(s: str | None, default: Any = None) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default
