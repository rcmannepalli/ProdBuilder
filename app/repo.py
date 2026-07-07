"""Domain repository helpers built on top of the DuckDB layer."""
from __future__ import annotations

from typing import Any

from . import db, secrets
from .config import (
    DEFAULT_MAX_FIX_ATTEMPTS,
    DEFAULT_MODEL_MAP,
    DEFAULT_OLLAMA_URL,
    DEFAULT_POLL_INTERVAL_SEC,
    DEFAULT_TEST_COMMAND,
)

# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

def create_project(name: str, target_folder: str) -> int:
    ts = db.now()
    pid = db.insert_returning_id(
        "INSERT INTO projects (name, target_folder, status, created_at, updated_at)"
        " VALUES (?, ?, 'idle', ?, ?)",
        (name, target_folder, ts, ts),
    )
    ensure_settings(pid)
    return pid


def list_projects() -> list[dict]:
    return db.query("SELECT * FROM projects ORDER BY updated_at DESC")


def get_project(pid: int) -> dict | None:
    return db.query_one("SELECT * FROM projects WHERE id = ?", (pid,))


def set_project_status(pid: int, status: str) -> None:
    db.execute(
        "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
        (status, db.now(), pid),
    )


def touch_project(pid: int) -> None:
    db.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (db.now(), pid))


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------

def latest_requirements(pid: int) -> dict | None:
    return db.query_one(
        "SELECT * FROM requirements_versions WHERE project_id = ?"
        " ORDER BY version_no DESC LIMIT 1",
        (pid,),
    )


def query_prev_requirements(pid: int, version_no: int) -> dict | None:
    return db.query_one(
        "SELECT * FROM requirements_versions WHERE project_id = ? AND version_no < ?"
        " ORDER BY version_no DESC LIMIT 1",
        (pid, version_no),
    )


def add_requirements_version(
    pid: int, raw_text: str, structured_json: str | None = None,
    diff_from_prev: str | None = None,
) -> int:
    prev = latest_requirements(pid)
    version_no = (prev["version_no"] + 1) if prev else 1
    return db.insert_returning_id(
        "INSERT INTO requirements_versions"
        " (project_id, version_no, raw_text, structured_json, diff_from_prev, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (pid, version_no, raw_text, structured_json, diff_from_prev, db.now()),
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def ensure_settings(pid: int) -> None:
    existing = db.query_one("SELECT id FROM settings WHERE project_id = ?", (pid,))
    if existing:
        return
    db.execute(
        "INSERT INTO settings (project_id, ollama_base_url, ollama_api_key_enc,"
        " model_map_json, max_fix_attempts, auto_apply_changes, poll_interval_sec,"
        " test_command, updated_at) VALUES (?, ?, '', ?, ?, ?, ?, ?, ?)",
        (
            pid, DEFAULT_OLLAMA_URL, db.dumps(DEFAULT_MODEL_MAP),
            DEFAULT_MAX_FIX_ATTEMPTS, False, DEFAULT_POLL_INTERVAL_SEC,
            DEFAULT_TEST_COMMAND, db.now(),
        ),
    )


def get_settings(pid: int) -> dict:
    ensure_settings(pid)
    row = db.query_one("SELECT * FROM settings WHERE project_id = ?", (pid,))
    assert row is not None
    row["model_map"] = db.loads(row.get("model_map_json"), dict(DEFAULT_MODEL_MAP))
    row["api_key"] = secrets.decrypt(row.get("ollama_api_key_enc") or "")
    row["api_key_masked"] = secrets.mask(row["api_key"])
    return row


def update_settings(pid: int, **fields: Any) -> None:
    ensure_settings(pid)
    sets, params = [], []
    if "ollama_base_url" in fields:
        sets.append("ollama_base_url = ?")
        params.append(fields["ollama_base_url"])
    if "api_key" in fields and fields["api_key"] is not None:
        # Only overwrite when a non-empty new key is supplied.
        if fields["api_key"] != "":
            sets.append("ollama_api_key_enc = ?")
            params.append(secrets.encrypt(fields["api_key"]))
    if "model_map" in fields:
        sets.append("model_map_json = ?")
        params.append(db.dumps(fields["model_map"]))
    for key in ("max_fix_attempts", "poll_interval_sec"):
        if key in fields:
            sets.append(f"{key} = ?")
            params.append(int(fields[key]))
    if "auto_apply_changes" in fields:
        sets.append("auto_apply_changes = ?")
        params.append(bool(fields["auto_apply_changes"]))
    if "test_command" in fields:
        sets.append("test_command = ?")
        params.append(fields["test_command"])
    if not sets:
        return
    sets.append("updated_at = ?")
    params.append(db.now())
    params.append(pid)
    db.execute(f"UPDATE settings SET {', '.join(sets)} WHERE project_id = ?", params)


# ---------------------------------------------------------------------------
# Phases & tasks
# ---------------------------------------------------------------------------

def clear_plan(pid: int) -> None:
    phase_ids = [r["id"] for r in db.query(
        "SELECT id FROM phases WHERE project_id = ?", (pid,))]
    for phid in phase_ids:
        db.execute("DELETE FROM tasks WHERE phase_id = ?", (phid,))
    db.execute("DELETE FROM phases WHERE project_id = ?", (pid,))


def add_phase(pid: int, order_no: int, name: str, description: str,
              deliverables: list, test_plan: list) -> int:
    return db.insert_returning_id(
        "INSERT INTO phases (project_id, order_no, name, description,"
        " deliverables_json, test_plan_json, status)"
        " VALUES (?, ?, ?, ?, ?, ?, 'pending')",
        (pid, order_no, name, description,
         db.dumps(deliverables), db.dumps(test_plan)),
    )


def list_phases(pid: int) -> list[dict]:
    rows = db.query(
        "SELECT * FROM phases WHERE project_id = ? ORDER BY order_no", (pid,))
    for r in rows:
        r["deliverables"] = db.loads(r.get("deliverables_json"), [])
        r["test_plan"] = db.loads(r.get("test_plan_json"), [])
        r["tasks"] = list_tasks(r["id"])
    return rows


def get_phase(phase_id: int) -> dict | None:
    r = db.query_one("SELECT * FROM phases WHERE id = ?", (phase_id,))
    if r:
        r["deliverables"] = db.loads(r.get("deliverables_json"), [])
        r["test_plan"] = db.loads(r.get("test_plan_json"), [])
    return r


def set_phase_status(phase_id: int, status: str) -> None:
    ts = db.now()
    if status == "building":
        db.execute("UPDATE phases SET status = ?, started_at = COALESCE(started_at, ?)"
                   " WHERE id = ?", (status, ts, phase_id))
    elif status == "done":
        db.execute("UPDATE phases SET status = ?, completed_at = ? WHERE id = ?",
                   (status, ts, phase_id))
    else:
        db.execute("UPDATE phases SET status = ? WHERE id = ?", (status, phase_id))


def add_task(phase_id: int, title: str, kind: str,
             file_paths: list | None = None, detail: str = "") -> int:
    ts = db.now()
    return db.insert_returning_id(
        "INSERT INTO tasks (phase_id, title, kind, status, file_paths_json,"
        " detail, created_at, updated_at) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)",
        (phase_id, title, kind, db.dumps(file_paths or []), detail, ts, ts),
    )


def list_tasks(phase_id: int) -> list[dict]:
    rows = db.query("SELECT * FROM tasks WHERE phase_id = ? ORDER BY id", (phase_id,))
    for r in rows:
        r["file_paths"] = db.loads(r.get("file_paths_json"), [])
    return rows


def set_task_status(task_id: int, status: str, detail: str | None = None) -> None:
    if detail is not None:
        db.execute("UPDATE tasks SET status = ?, detail = ?, updated_at = ? WHERE id = ?",
                   (status, detail, db.now(), task_id))
    else:
        db.execute("UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                   (status, db.now(), task_id))


# ---------------------------------------------------------------------------
# Test runs
# ---------------------------------------------------------------------------

def add_test_run(phase_id: int, attempt_no: int, passed: bool, summary: str,
                 stdout: str, stderr: str, duration_ms: int) -> int:
    return db.insert_returning_id(
        "INSERT INTO test_runs (phase_id, attempt_no, passed, summary, stdout,"
        " stderr, duration_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (phase_id, attempt_no, passed, summary, stdout[-8000:], stderr[-8000:],
         duration_ms, db.now()),
    )


def latest_test_run(phase_id: int) -> dict | None:
    return db.query_one(
        "SELECT * FROM test_runs WHERE phase_id = ? ORDER BY id DESC LIMIT 1",
        (phase_id,))


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def add_event(pid: int, agent_name: str, event_type: str, message: str,
              phase_id: int | None = None, payload: Any = None) -> dict:
    eid = db.insert_returning_id(
        "INSERT INTO agent_events (project_id, phase_id, agent_name, event_type,"
        " message, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (pid, phase_id, agent_name, event_type, message,
         db.dumps(payload) if payload is not None else None, db.now()),
    )
    return {
        "id": eid, "project_id": pid, "phase_id": phase_id,
        "agent_name": agent_name, "event_type": event_type,
        "message": message, "created_at": db.now(),
    }


def list_events(pid: int, after_id: int = 0, limit: int = 300) -> list[dict]:
    return db.query(
        "SELECT * FROM agent_events WHERE project_id = ? AND id > ?"
        " ORDER BY id ASC LIMIT ?",
        (pid, after_id, limit))


# ---------------------------------------------------------------------------
# Change proposals
# ---------------------------------------------------------------------------

def add_proposal(pid: int, source: str, summary: str, diff: Any) -> int:
    return db.insert_returning_id(
        "INSERT INTO change_proposals (project_id, source, summary, diff_json, status,"
        " created_at) VALUES (?, ?, ?, ?, 'pending', ?)",
        (pid, source, summary, db.dumps(diff), db.now()),
    )


def list_proposals(pid: int, status: str | None = None) -> list[dict]:
    if status:
        rows = db.query(
            "SELECT * FROM change_proposals WHERE project_id = ? AND status = ?"
            " ORDER BY id DESC", (pid, status))
    else:
        rows = db.query(
            "SELECT * FROM change_proposals WHERE project_id = ? ORDER BY id DESC",
            (pid,))
    for r in rows:
        r["diff"] = db.loads(r.get("diff_json"), {})
    return rows


def set_proposal_status(cp_id: int, status: str) -> None:
    db.execute("UPDATE change_proposals SET status = ? WHERE id = ?", (status, cp_id))


def get_proposal(cp_id: int) -> dict | None:
    r = db.query_one("SELECT * FROM change_proposals WHERE id = ?", (cp_id,))
    if r:
        r["diff"] = db.loads(r.get("diff_json"), {})
    return r
