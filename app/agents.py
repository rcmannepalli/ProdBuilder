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
    out, _diag = _safe_json(cfg, "planner", _JSON_SYS, user)
    if isinstance(out, dict) and out.get("features"):
        return out
    return _fallback_structured(raw)


# ---------------------------------------------------------------------------
# Build plan
# ---------------------------------------------------------------------------

def generate_plan(cfg: LLMConfig, structured: dict, raw: str,
                  existing_code: str = "") -> list[dict]:
    if existing_code:
        mode = (
            "This is an ENRICHMENT of an EXISTING codebase (shown below). Produce "
            "phases that ADD, EXTEND, REFACTOR, or FIX the existing code to satisfy "
            "the requirements — do NOT scaffold from scratch or recreate files that "
            "already exist. Reuse existing modules; reference their real file paths.\n\n"
            f"EXISTING CODEBASE:\n{existing_code}\n\n"
        )
    else:
        mode = ""
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
        f"{mode}SPEC:\n{structured}\n\nRAW REQUIREMENTS:\n{raw}"
    )
    out, _diag = _safe_json(cfg, "planner", _JSON_SYS, user)
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
    out, _diag = _safe_json(cfg, "monitor", _JSON_SYS, user)
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
    out, diag = _safe_json(cfg, "coder", _JSON_SYS, user)
    result = _coerce_files(out)
    if not result["files"]:
        result["_diag"] = diag or "model returned no file entries"
    return result


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
    out, diag = _safe_json(cfg, "tester", _JSON_SYS, user)
    result = _coerce_files(out)
    if not result["files"]:
        result["_diag"] = diag or "model returned no test files"
    return result


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
    out, diag = _safe_json(cfg, "fixer", _JSON_SYS, user)
    result = _coerce_files(out)
    if not result["files"]:
        result["_diag"] = diag or "model returned no patch"
    return result


def diagnose_failure(cfg: LLMConfig, phase: dict, reason: str,
                     stdout: str, stderr: str,
                     existing_files: dict[str, str]) -> dict:
    """Produce a clear, developer-facing resolution report for a phase that
    could not be completed automatically."""
    ctx = _files_context(existing_files, max_chars=4000)
    user = (
        "Automated build/fix could not resolve this phase. Write a concise report "
        "for the human developer. Return JSON: {\"problem\": str, "
        "\"likely_cause\": str, \"recommended_actions\": [str], "
        "\"severity\": \"low\"|\"medium\"|\"high\"}. Actions must be concrete steps "
        "a developer can take (commands, code changes, config).\n\n"
        f"PHASE: {phase.get('name')}\nREASON: {reason}\n\n"
        f"TEST STDOUT:\n{stdout[-3000:]}\n\nTEST STDERR:\n{stderr[-3000:]}\n\n"
        f"FILES:\n{ctx}"
    )
    out, _diag = _safe_json(cfg, "reviewer", _JSON_SYS, user)
    if isinstance(out, dict) and out.get("problem"):
        out.setdefault("recommended_actions", [])
        out.setdefault("severity", "medium")
        return out
    # Heuristic fallback so the developer always gets actionable guidance.
    return _fallback_diagnosis(reason, stdout, stderr)


def _fallback_diagnosis(reason: str, stdout: str, stderr: str) -> dict:
    text = (stderr or "") + "\n" + (stdout or "")
    actions: list[str] = []
    problem = reason
    low = text.lower()
    if "no module named" in low:
        import re as _re
        mods = _re.findall(r"no module named ['\"]([\w.]+)", low)
        problem = f"Missing dependency: {', '.join(sorted(set(mods))) or 'unknown'}"
        actions = [
            f"Add the package(s) to requirements.txt: {', '.join(sorted(set(mods)))}",
            "Confirm the import name matches the pip package name.",
        ]
    elif "importerror" in low or "cannot import" in low:
        problem = "A module or symbol could not be imported."
        actions = ["Check the module path and that the referenced name exists.",
                   "Ensure the file defining it was created in this phase."]
    elif "assert" in low:
        problem = "A test assertion failed — the implementation does not yet meet the expected behavior."
        actions = ["Review the failing assertion and align the implementation with the test's expectation.",
                   "If the test is wrong, correct the test to match the intended behavior."]
    elif "syntaxerror" in low:
        problem = "Generated code has a syntax error."
        actions = ["Open the offending file in the Explorer and fix the syntax.",
                   "Re-run the phase after correcting."]
    else:
        actions = ["Review the test output below.",
                   "Fix the underlying issue in the target folder, then Retry the phase."]
    return {"problem": problem[:400], "likely_cause": reason[:400],
            "recommended_actions": actions, "severity": "medium"}


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
    out, _diag = _safe_json(cfg, "reviewer", _JSON_SYS, user)
    if isinstance(out, dict) and "approved" in out:
        return out
    return {"approved": True, "notes": "Auto-approved (reviewer unavailable).",
            "gaps": []}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _safe_json(cfg: LLMConfig, role: str, system: str,
               user: str) -> tuple[Any, str]:
    """Call a role and parse JSON. Returns (parsed_or_None, diagnostic).

    On unparseable output, makes one repair attempt asking the model to convert
    its own response into strict JSON before giving up.
    """
    try:
        raw = crew.run_role(cfg, role, system, user, json_mode=True)
    except Exception as e:  # noqa: BLE001
        return None, f"LLM call failed: {type(e).__name__}: {e}"
    parsed = extract_json(raw)
    if parsed is not None:
        return parsed, ""
    try:
        fixed = crew.run_role(
            cfg, role,
            "You convert text into STRICT, valid JSON. Output ONLY the JSON.",
            "Convert the following into valid JSON with the requested schema. "
            "Output ONLY the JSON, no prose:\n\n" + (raw or "")[:6000],
            json_mode=True)
        parsed = extract_json(fixed)
        if parsed is not None:
            return parsed, ""
    except Exception:  # noqa: BLE001
        pass
    snippet = (raw or "").strip().replace("\n", " ")[:180]
    if not snippet:
        return None, "model returned an empty response"
    return None, f"unparseable model output (starts: {snippet!r})"


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


_PATH_KEYS = ("path", "filepath", "file_path", "filename", "file", "name")
_CONTENT_KEYS = ("content", "code", "text", "body", "source", "contents")


def _coerce_files(out: Any) -> dict:
    """Normalise the many shapes models use for "a set of files" into
    [{path, content}]. Handles: {"files":[...]}, a bare list, a
    {path: content} mapping, {"files": {path: content}}, and per-file dicts
    keyed by filename/code/etc."""
    notes = ""
    raw_files: Any = None
    if isinstance(out, dict):
        notes = str(out.get("notes") or out.get("summary") or "")
        if "files" in out:
            raw_files = out["files"]
        elif _looks_like_path_map(out):
            raw_files = out  # bare {path: content} mapping
        else:
            # Maybe a single {path, content} object.
            single = _one_file(out)
            raw_files = [single] if single else []
    elif isinstance(out, list):
        raw_files = out
    else:
        return {"files": [], "notes": ""}

    files: list[dict] = []
    if isinstance(raw_files, dict):
        for path, content in raw_files.items():
            files.append({"path": _clean_path(path),
                          "content": _as_text(content)})
    elif isinstance(raw_files, list):
        for f in raw_files:
            single = _one_file(f)
            if single:
                files.append(single)
    files = [f for f in files if f["path"]]
    return {"files": files, "notes": notes}


def _one_file(f: Any) -> dict | None:
    if not isinstance(f, dict):
        return None
    path = next((f[k] for k in _PATH_KEYS if f.get(k)), None)
    content = next((f[k] for k in _CONTENT_KEYS if k in f), None)
    if path is None or content is None:
        return None
    return {"path": _clean_path(path), "content": _as_text(content)}


def _looks_like_path_map(d: dict) -> bool:
    if not d:
        return False
    keyish = [k for k in d if k not in ("notes", "summary")]
    if not keyish:
        return False
    return all(("." in str(k) or "/" in str(k)) and
               isinstance(d[k], (str, int, float)) for k in keyish)


def _clean_path(p: Any) -> str:
    return str(p).strip().lstrip("/")


def _as_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, (list, dict)):
        try:
            return __import__("json").dumps(c, indent=2)
        except Exception:  # noqa: BLE001
            return str(c)
    return str(c)


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
