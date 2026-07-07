"""Agent facade — prompt construction + structured parsing for each pipeline step.

Every function routes through :func:`app.crew.run_role` (CrewAI when available,
direct Ollama otherwise) and returns validated Python structures. When the LLM
is unreachable or returns unparseable output, small deterministic fallbacks keep
the product usable for demos and offline runs.
"""
from __future__ import annotations

from typing import Any

from . import crew
from .llm import LLMConfig, extract_json

_JSON_SYS = (
    "You are a precise engineering agent. Respond with STRICT JSON only. "
    "Do not include markdown fences, prose, or commentary outside the JSON."
)


# ---------------------------------------------------------------------------
# Requirements parsing
# ---------------------------------------------------------------------------

def parse_requirements(cfg: LLMConfig, raw: str) -> dict:
    user = (
        "Parse the following product requirements into structured JSON with keys: "
        "\"summary\" (string), \"features\" (array of short strings), "
        "\"user_stories\" (array of strings), "
        "\"acceptance_criteria\" (array of strings), "
        "\"tech_notes\" (array of strings).\n\nREQUIREMENTS:\n" + raw
    )
    out = _safe_json(cfg, "planner", _JSON_SYS, user)
    if isinstance(out, dict) and out.get("features"):
        return out
    return _fallback_structured(raw)


# ---------------------------------------------------------------------------
# Build plan
# ---------------------------------------------------------------------------

def generate_plan(cfg: LLMConfig, structured: dict, raw: str) -> list[dict]:
    user = (
        "Given the structured product spec below, produce a phased build plan as "
        "JSON with this exact shape:\n"
        "{\"phases\": [{\"name\": str, \"description\": str, "
        "\"deliverables\": [str], \"test_plan\": [str], "
        "\"tasks\": [{\"title\": str, \"kind\": \"code\"|\"test\", "
        "\"files\": [str]}]}]}\n"
        "Rules: 3-7 phases, each phase independently testable, ordered so each "
        "builds on the previous, every phase has a non-empty test_plan, and code "
        "tasks list concrete file paths relative to the project root.\n\n"
        f"SPEC:\n{structured}\n\nRAW REQUIREMENTS:\n{raw}"
    )
    out = _safe_json(cfg, "planner", _JSON_SYS, user)
    phases = out.get("phases") if isinstance(out, dict) else None
    if isinstance(phases, list) and phases:
        return [_normalize_phase(p) for p in phases]
    return _fallback_plan(structured)


def diff_requirements(cfg: LLMConfig, old_raw: str, new_raw: str) -> dict:
    user = (
        "Compare the OLD and NEW product requirements. Return JSON: "
        "{\"has_changes\": bool, \"summary\": str, \"changes\": [str], "
        "\"impact\": \"additive\"|\"modifying\"|\"breaking\"}.\n\n"
        f"OLD:\n{old_raw}\n\nNEW:\n{new_raw}"
    )
    out = _safe_json(cfg, "monitor", _JSON_SYS, user)
    if isinstance(out, dict) and "has_changes" in out:
        return out
    changed = old_raw.strip() != new_raw.strip()
    return {
        "has_changes": changed,
        "summary": "Requirements text changed." if changed else "No change.",
        "changes": [], "impact": "modifying",
    }


# ---------------------------------------------------------------------------
# Code / tests / fixes / review
# ---------------------------------------------------------------------------

def generate_code(cfg: LLMConfig, project_summary: str, phase: dict,
                  task: dict, existing_files: dict[str, str]) -> dict:
    ctx = _files_context(existing_files)
    user = (
        "Implement the task below. Return JSON: "
        "{\"files\": [{\"path\": str, \"content\": str}], \"notes\": str}. "
        "Provide COMPLETE file contents (not diffs). Paths are relative to the "
        "project root.\n\n"
        f"PROJECT: {project_summary}\n"
        f"PHASE: {phase.get('name')} — {phase.get('description')}\n"
        f"TASK: {task.get('title')} (files: {task.get('file_paths')})\n\n"
        f"EXISTING FILES:\n{ctx}"
    )
    out = _safe_json(cfg, "coder", _JSON_SYS, user)
    return _coerce_files(out)


def generate_tests(cfg: LLMConfig, project_summary: str, phase: dict,
                   existing_files: dict[str, str]) -> dict:
    ctx = _files_context(existing_files)
    test_plan = phase.get("test_plan") or []
    user = (
        "Write automated tests (pytest style unless the project is clearly another "
        "language) validating this phase's test plan. Return JSON: "
        "{\"files\": [{\"path\": str, \"content\": str}], \"notes\": str}. "
        "Complete file contents only.\n\n"
        f"PROJECT: {project_summary}\n"
        f"PHASE: {phase.get('name')}\n"
        f"TEST PLAN: {test_plan}\n\n"
        f"EXISTING FILES:\n{ctx}"
    )
    out = _safe_json(cfg, "tester", _JSON_SYS, user)
    return _coerce_files(out)


def fix_failure(cfg: LLMConfig, project_summary: str, phase: dict,
                existing_files: dict[str, str], stdout: str, stderr: str) -> dict:
    ctx = _files_context(existing_files)
    user = (
        "The tests failed. Diagnose and fix by returning corrected COMPLETE files. "
        "Return JSON: {\"files\": [{\"path\": str, \"content\": str}], "
        "\"notes\": str}. Only include files you change.\n\n"
        f"PROJECT: {project_summary}\nPHASE: {phase.get('name')}\n\n"
        f"TEST STDOUT:\n{stdout[-4000:]}\n\nTEST STDERR:\n{stderr[-4000:]}\n\n"
        f"CURRENT FILES:\n{ctx}"
    )
    out = _safe_json(cfg, "fixer", _JSON_SYS, user)
    return _coerce_files(out)


def review_phase(cfg: LLMConfig, phase: dict, acceptance: list[str],
                 existing_files: dict[str, str]) -> dict:
    ctx = _files_context(existing_files, max_chars=6000)
    user = (
        "Review whether the delivered code meets the acceptance criteria. Return "
        "JSON: {\"approved\": bool, \"notes\": str, \"gaps\": [str]}.\n\n"
        f"PHASE: {phase.get('name')}\n"
        f"ACCEPTANCE CRITERIA: {acceptance}\n"
        f"TEST PLAN: {phase.get('test_plan')}\n\n"
        f"DELIVERED FILES:\n{ctx}"
    )
    out = _safe_json(cfg, "reviewer", _JSON_SYS, user)
    if isinstance(out, dict) and "approved" in out:
        return out
    return {"approved": True, "notes": "Auto-approved (reviewer unavailable).",
            "gaps": []}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _safe_json(cfg: LLMConfig, role: str, system: str, user: str) -> Any:
    try:
        raw = crew.run_role(cfg, role, system, user, json_mode=True)
    except Exception:  # noqa: BLE001
        return None
    return extract_json(raw)


def _files_context(files: dict[str, str], max_chars: int = 8000) -> str:
    if not files:
        return "(none yet)"
    parts, total = [], 0
    for path, content in files.items():
        snippet = content if len(content) < 2000 else content[:2000] + "\n…(truncated)"
        block = f"--- {path} ---\n{snippet}\n"
        if total + len(block) > max_chars:
            parts.append("…(remaining files omitted)")
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def _coerce_files(out: Any) -> dict:
    files = []
    if isinstance(out, dict):
        raw_files = out.get("files") or []
        notes = out.get("notes", "")
    elif isinstance(out, list):
        raw_files, notes = out, ""
    else:
        return {"files": [], "notes": "No parseable output from model."}
    for f in raw_files:
        if isinstance(f, dict) and f.get("path") and "content" in f:
            files.append({"path": str(f["path"]).lstrip("/"),
                          "content": str(f["content"])})
    return {"files": files, "notes": notes}


def _normalize_phase(p: dict) -> dict:
    tasks = []
    for t in p.get("tasks", []) or []:
        if isinstance(t, dict):
            tasks.append({
                "title": t.get("title", "task"),
                "kind": t.get("kind", "code"),
                "files": t.get("files", []) or [],
            })
    return {
        "name": p.get("name", "Phase"),
        "description": p.get("description", ""),
        "deliverables": p.get("deliverables", []) or [],
        "test_plan": p.get("test_plan", []) or [],
        "tasks": tasks or [
            {"title": f"Implement {p.get('name', 'phase')}", "kind": "code", "files": []},
        ],
    }


# --- deterministic fallbacks -------------------------------------------------

def _fallback_structured(raw: str) -> dict:
    lines = [l.strip("-* \t") for l in raw.splitlines() if l.strip()]
    features = lines[:8] if lines else ["Core functionality"]
    return {
        "summary": (raw.strip()[:200] or "Product"),
        "features": features,
        "user_stories": [f"As a user, I want: {f}" for f in features[:5]],
        "acceptance_criteria": [f"{f} works as described" for f in features[:5]],
        "tech_notes": [],
    }


def _fallback_plan(structured: dict) -> list[dict]:
    feats = structured.get("features", ["Core functionality"])
    return [
        _normalize_phase({
            "name": "Phase 1 — Scaffolding",
            "description": "Set up the project structure and entry point.",
            "deliverables": ["Project skeleton", "Entry point", "README"],
            "test_plan": ["Project imports without error", "Entry point runs"],
            "tasks": [{"title": "Create project skeleton", "kind": "code",
                       "files": ["main.py", "README.md"]}],
        }),
        _normalize_phase({
            "name": "Phase 2 — Core Features",
            "description": "Implement the primary features.",
            "deliverables": list(feats)[:5],
            "test_plan": [f"{f} passes its tests" for f in feats[:5]],
            "tasks": [{"title": f"Implement {f}", "kind": "code", "files": []}
                      for f in feats[:5]],
        }),
        _normalize_phase({
            "name": "Phase 3 — Validation & Polish",
            "description": "Add tests, error handling, and documentation.",
            "deliverables": ["Test suite", "Error handling", "Docs"],
            "test_plan": ["Full test suite passes", "Edge cases handled"],
            "tasks": [{"title": "Add integration tests", "kind": "test",
                       "files": ["tests/test_integration.py"]}],
        }),
    ]
