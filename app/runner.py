"""Background build runner — the autonomous phase-loop state machine.

Each project's build runs in its own daemon thread. A parallel monitor thread
re-reads requirements on an interval and raises change proposals. Control
(pause/resume/stop) is via threading events. All state is persisted to DuckDB so
a restart resumes from the last known phase.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from . import agents, executor, events, repo
from .llm import LLMConfig


class Control:
    def __init__(self) -> None:
        self.paused = threading.Event()
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None
        self.monitor_thread: threading.Thread | None = None
        self.last_seen_version: int = 0


_controls: dict[int, Control] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public control API
# ---------------------------------------------------------------------------

def is_running(pid: int) -> bool:
    c = _controls.get(pid)
    return bool(c and c.thread and c.thread.is_alive())


def status(pid: int) -> str:
    c = _controls.get(pid)
    if not c or not c.thread or not c.thread.is_alive():
        return "stopped"
    return "paused" if c.paused.is_set() else "running"


def start_build(pid: int) -> None:
    with _lock:
        if is_running(pid):
            return
        ctrl = Control()
        _controls[pid] = ctrl
        ctrl.thread = threading.Thread(target=_run_build, args=(pid, ctrl), daemon=True)
        ctrl.monitor_thread = threading.Thread(
            target=_run_monitor, args=(pid, ctrl), daemon=True)
        ctrl.thread.start()
        ctrl.monitor_thread.start()


def pause_build(pid: int) -> None:
    c = _controls.get(pid)
    if c:
        c.paused.set()
        _emit(pid, "System", "control", "Build paused.")
        repo.set_project_status(pid, "paused")


def resume_build(pid: int) -> None:
    c = _controls.get(pid)
    if c:
        c.paused.clear()
        _emit(pid, "System", "control", "Build resumed.")
        repo.set_project_status(pid, "running")


def stop_build(pid: int) -> None:
    c = _controls.get(pid)
    if c:
        c.stopped.set()
        c.paused.clear()
        _emit(pid, "System", "control", "Stop requested.")


# ---------------------------------------------------------------------------
# Build loop
# ---------------------------------------------------------------------------

def _cfg_for(pid: int) -> LLMConfig:
    return LLMConfig.from_settings(repo.get_settings(pid))


def _wait_if_paused(ctrl: Control) -> None:
    while ctrl.paused.is_set() and not ctrl.stopped.is_set():
        time.sleep(0.4)


def _run_build(pid: int, ctrl: Control) -> None:
    try:
        repo.set_project_status(pid, "running")
        _emit(pid, "System", "build", "Build started.")
        cfg = _cfg_for(pid)
        settings = repo.get_settings(pid)
        project = repo.get_project(pid)
        target = project["target_folder"]
        reqs = repo.latest_requirements(pid)
        summary = _project_summary(reqs)
        ctrl.last_seen_version = reqs["version_no"] if reqs else 0

        phases = repo.list_phases(pid)
        if not phases:
            _emit(pid, "System", "build", "No plan found — generate a plan first.")
            repo.set_project_status(pid, "idle")
            return

        for phase in phases:
            if ctrl.stopped.is_set():
                break
            if phase["status"] == "done":
                continue
            _wait_if_paused(ctrl)
            ok = _run_phase(pid, ctrl, cfg, settings, target, summary, phase)
            if not ok:
                repo.set_project_status(pid, "needs_attention")
                _emit(pid, "System", "build",
                      f"Phase '{phase['name']}' needs attention — pausing build.",
                      phase_id=phase["id"])
                return

        if ctrl.stopped.is_set():
            _emit(pid, "System", "build", "Build stopped by user.")
            repo.set_project_status(pid, "idle")
        else:
            _emit(pid, "System", "build", "All phases complete. ✅")
            repo.set_project_status(pid, "done")
    except Exception as e:  # noqa: BLE001
        _emit(pid, "System", "error", f"Build crashed: {type(e).__name__}: {e}")
        repo.set_project_status(pid, "needs_attention")


def _run_phase(pid: int, ctrl: Control, cfg: LLMConfig, settings: dict,
               target: str, summary: str, phase: dict) -> bool:
    phase_id = phase["id"]
    repo.set_phase_status(phase_id, "building")
    _emit(pid, "Architect", "phase", f"▶ Phase '{phase['name']}' started.",
          phase_id=phase_id)

    tasks = repo.list_tasks(phase_id)
    code_tasks = [t for t in tasks if t["kind"] == "code"] or tasks

    # 1. Coding tasks
    for task in code_tasks:
        if ctrl.stopped.is_set():
            return False
        _wait_if_paused(ctrl)
        repo.set_task_status(task["id"], "running")
        _emit(pid, "Coder", "task", f"Coding: {task['title']}", phase_id=phase_id)
        existing = executor.read_files(target)
        result = agents.generate_code(cfg, summary, phase, task, existing)
        if result["files"]:
            written = executor.write_files(target, result["files"])
            repo.set_task_status(task["id"], "done",
                                 detail=f"Wrote {len(written)} file(s)")
            _emit(pid, "Coder", "files", f"Wrote: {', '.join(written)}",
                  phase_id=phase_id, payload={"files": written})
        else:
            reason = result.get("_diag") or "model returned no file entries"
            repo.set_task_status(task["id"], "blocked", detail=reason[:200])
            _emit(pid, "Coder", "warn",
                  f"No files produced for '{task['title']}' — {reason}",
                  phase_id=phase_id, payload={"reason": reason})

    # 2. Test authoring
    repo.set_phase_status(phase_id, "test_writing")
    _emit(pid, "Test Writer", "task", "Authoring tests…", phase_id=phase_id)
    existing = executor.read_files(target)
    tests = agents.generate_tests(cfg, summary, phase, existing)
    if tests["files"]:
        written = executor.write_files(target, tests["files"])
        _emit(pid, "Test Writer", "files", f"Wrote tests: {', '.join(written)}",
              phase_id=phase_id, payload={"files": written})
    else:
        reason = tests.get("_diag") or "model returned no test files"
        _emit(pid, "Test Writer", "warn", f"No tests produced — {reason}",
              phase_id=phase_id, payload={"reason": reason})

    # Guard: if the target folder still has no source files, testing is futile.
    if not _has_source(target):
        _escalate(pid, cfg, target, phase,
                  "The Coder agent produced no source files for this phase.")
        return False

    # 2b. Environment: create the project venv and install dependencies so the
    # Test agent runs against a real, isolated environment.
    if settings.get("use_venv", True):
        _emit(pid, "Environment", "env", "Preparing virtual environment…",
              phase_id=phase_id)
        ok, tool, msg = executor.ensure_venv(target)
        _emit(pid, "Environment", "env", f"[{tool}] {msg}", phase_id=phase_id)
        iok, imsg = executor.install_requirements(target)
        _emit(pid, "Environment", "env", f"Dependencies: {imsg}",
              phase_id=phase_id)
        # Proactively install third-party libraries the code imports, so
        # validation isn't blocked on a missing dependency.
        imports = executor.scan_imports(target)
        if imports:
            _emit(pid, "Environment", "env",
                  f"Installing imported libraries: {', '.join(imports)}",
                  phase_id=phase_id)
            pok, pmsg = executor.pip_install(target, imports)
            _emit(pid, "Environment", "env",
                  ("Imported libraries ready" if pok else f"Install issue: {pmsg}"),
                  phase_id=phase_id)

    # 3. Test + fix loop
    max_attempts = int(settings.get("max_fix_attempts") or 3)
    test_cmd = settings.get("test_command") or "python -m pytest -q"
    passed = _test_and_fix(pid, ctrl, cfg, settings, target, summary, phase,
                           test_cmd, max_attempts)
    if ctrl.stopped.is_set():
        return False
    if not passed:
        last = repo.latest_test_run(phase_id) or {}
        _escalate(pid, cfg, target, phase,
                  f"Tests still failing after {max_attempts} automated fix attempts.",
                  stdout=last.get("stdout", ""), stderr=last.get("stderr", ""))
        return False

    # 4. Review gate
    repo.set_phase_status(phase_id, "review")
    _emit(pid, "Reviewer", "task", "Reviewing acceptance criteria…",
          phase_id=phase_id)
    acceptance = phase.get("test_plan") or []
    review = agents.review_phase(cfg, phase, acceptance, executor.read_files(target))
    if review.get("approved"):
        repo.set_phase_status(phase_id, "done")
        _emit(pid, "Reviewer", "phase", f"✔ Phase '{phase['name']}' approved.",
              phase_id=phase_id, payload=review)
        return True
    gaps = review.get("gaps") or []
    _emit(pid, "Reviewer", "warn",
          f"Review found gaps: {'; '.join(gaps) or review.get('notes', '')}",
          phase_id=phase_id, payload=review)
    _escalate(pid, cfg, target, phase,
              "Tests pass but the Reviewer found unmet acceptance criteria: "
              + ("; ".join(gaps) or review.get("notes", "")))
    return False


def _run_tests_autoinstall(pid: int, target: str, test_cmd: str, settings: dict,
                           phase_id: int, max_installs: int = 3):
    """Run tests; if they fail purely because a third-party module is missing,
    install it into the venv and re-run (bounded). Returns the final RunResult.
    Dependency installs never consume an LLM fix attempt."""
    rr = executor.run_tests(target, test_cmd)
    if not settings.get("use_venv", True):
        return rr
    for _ in range(max_installs):
        if rr.passed:
            break
        missing = executor.extract_missing_modules(rr.stdout + rr.stderr)
        if not missing:
            break
        _emit(pid, "Environment", "env",
              f"Installing missing dependencies: {', '.join(missing)}",
              phase_id=phase_id)
        iok, imsg = executor.pip_install(target, missing)
        _emit(pid, "Environment", "env",
              ("Installed; re-testing…" if iok else f"Install issue: {imsg}"),
              phase_id=phase_id)
        if not iok:
            break
        rr = executor.run_tests(target, test_cmd)
    return rr


def _test_and_fix(pid: int, ctrl: Control, cfg: LLMConfig, settings: dict,
                  target: str, summary: str, phase: dict, test_cmd: str,
                  max_attempts: int) -> bool:
    phase_id = phase["id"]
    for attempt in range(1, max_attempts + 1):
        if ctrl.stopped.is_set():
            return False
        _wait_if_paused(ctrl)
        repo.set_phase_status(phase_id, "testing")
        _emit(pid, "Validator", "test", f"Running tests (attempt {attempt})…",
              phase_id=phase_id)
        rr = _run_tests_autoinstall(pid, target, test_cmd, settings, phase_id)
        repo.add_test_run(phase_id, attempt, rr.passed, rr.summary,
                          rr.stdout, rr.stderr, rr.duration_ms)
        if rr.passed:
            _emit(pid, "Validator", "test", f"✔ Tests passed. {rr.summary}",
                  phase_id=phase_id)
            return True
        _emit(pid, "Validator", "test", f"✗ Tests failed: {rr.summary}",
              phase_id=phase_id, payload={"summary": rr.summary})
        # Surface the actual failure detail so it isn't a black box.
        detail = _failure_detail(rr.stdout, rr.stderr)
        if detail:
            _emit(pid, "Validator", "output", detail, phase_id=phase_id,
                  payload={"stdout_tail": rr.stdout[-2000:],
                           "stderr_tail": rr.stderr[-2000:]})
        if attempt >= max_attempts:
            _emit(pid, "Fixer", "warn",
                  f"Exhausted {max_attempts} fix attempts — see the failure "
                  "output above. Phase marked needs-attention.",
                  phase_id=phase_id)
            return False
        repo.set_phase_status(phase_id, "fixing")
        _emit(pid, "Fixer", "task", f"Attempting fix {attempt}/{max_attempts}…",
              phase_id=phase_id)
        fix = agents.fix_failure(cfg, summary, phase, executor.read_files(target),
                                 rr.stdout, rr.stderr)
        if fix["files"]:
            written = executor.write_files(target, fix["files"])
            _emit(pid, "Fixer", "files", f"Patched: {', '.join(written)}",
                  phase_id=phase_id, payload={"files": written})
        else:
            reason = fix.get("_diag") or "model returned no patch"
            _emit(pid, "Fixer", "warn", f"Fixer produced no patch — {reason}",
                  phase_id=phase_id, payload={"reason": reason})
    return False


# ---------------------------------------------------------------------------
# Requirements monitor
# ---------------------------------------------------------------------------

def _run_monitor(pid: int, ctrl: Control) -> None:
    settings = repo.get_settings(pid)
    interval = max(15, int(settings.get("poll_interval_sec") or 60))
    while not ctrl.stopped.is_set() and ctrl.thread and ctrl.thread.is_alive():
        for _ in range(interval * 2):  # ~0.5s granularity
            if ctrl.stopped.is_set():
                return
            time.sleep(0.5)
        try:
            _monitor_tick(pid, ctrl)
        except Exception as e:  # noqa: BLE001
            _emit(pid, "Monitor", "error", f"Monitor error: {e}")


def _monitor_tick(pid: int, ctrl: Control) -> None:
    latest = repo.latest_requirements(pid)
    if not latest:
        return
    if latest["version_no"] <= ctrl.last_seen_version:
        return
    prev = repo.query_prev_requirements(pid, latest["version_no"])
    old_raw = prev["raw_text"] if prev else ""
    cfg = _cfg_for(pid)
    diff = agents.diff_requirements(cfg, old_raw, latest["raw_text"])
    ctrl.last_seen_version = latest["version_no"]
    if not diff.get("has_changes"):
        return
    cp_id = repo.add_proposal(pid, "monitor", diff.get("summary", "Change detected"),
                              diff)
    _emit(pid, "Monitor", "proposal",
          f"Requirement change detected: {diff.get('summary', '')}",
          payload={"proposal_id": cp_id, "impact": diff.get("impact")})
    settings = repo.get_settings(pid)
    if settings.get("auto_apply_changes"):
        apply_proposal(pid, cp_id)


def apply_proposal(pid: int, cp_id: int) -> None:
    cp = repo.get_proposal(cp_id)
    if not cp:
        return
    diff = cp.get("diff") or {}
    changes = diff.get("changes") or [diff.get("summary", "Requirement update")]
    order_no = len(repo.list_phases(pid)) + 1
    phase_id = repo.add_phase(
        pid, order_no, f"Requirement Update #{cp_id}",
        diff.get("summary", "Incremental requirement change"),
        deliverables=list(changes),
        test_plan=[f"Change satisfied: {c}" for c in changes],
    )
    for c in changes:
        repo.add_task(phase_id, f"Apply: {c}", "code")
    repo.set_proposal_status(cp_id, "applied")
    _emit(pid, "Architect", "plan",
          f"Applied change proposal #{cp_id} as a new phase.",
          phase_id=phase_id)


def reject_proposal(pid: int, cp_id: int) -> None:
    repo.set_proposal_status(cp_id, "rejected")
    _emit(pid, "Monitor", "proposal", f"Change proposal #{cp_id} rejected.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _escalate(pid: int, cfg: LLMConfig, target: str, phase: dict, reason: str,
              stdout: str = "", stderr: str = "") -> None:
    """Build a clear developer resolution report, store it on the phase, and
    surface it prominently. Used whenever a phase cannot be completed
    automatically."""
    phase_id = phase["id"]
    _emit(pid, "Reviewer", "task",
          "Preparing a resolution report for the developer…", phase_id=phase_id)
    try:
        diag = agents.diagnose_failure(cfg, phase, reason, stdout, stderr,
                                       executor.read_files(target))
    except Exception as e:  # noqa: BLE001
        diag = {"problem": reason, "likely_cause": f"{type(e).__name__}: {e}",
                "recommended_actions": ["Review the activity log and target files."],
                "severity": "medium"}
    repo.set_phase_attention(phase_id, diag)
    repo.set_phase_status(phase_id, "needs_attention")
    actions = " | ".join(diag.get("recommended_actions", [])[:3])
    _emit(pid, "System", "attention",
          f"⚠ NEEDS DEVELOPER ATTENTION — {diag.get('problem', reason)}. "
          f"Recommended: {actions}",
          phase_id=phase_id, payload=diag)


def _project_summary(reqs: dict | None) -> str:
    if not reqs:
        return "Unnamed product."
    return (reqs.get("raw_text") or "")[:1200]


_SOURCE_EXT = (".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
               ".rb", ".php", ".c", ".cpp", ".cs", ".html", ".css", ".sql")


def _has_source(target: str) -> bool:
    """True if the target folder contains any non-test source file."""
    for path in executor.list_tree(target):
        low = path.lower()
        if low.endswith("/"):
            continue
        if "test" in low:
            continue
        if low.endswith(_SOURCE_EXT):
            return True
    return False


def _failure_detail(stdout: str, stderr: str) -> str:
    """Extract the most informative lines from a failed test run."""
    text = (stderr or "").strip() or (stdout or "").strip()
    if not text:
        return ""
    lines = [l for l in text.splitlines() if l.strip()]
    # Prefer lines that name the actual error.
    key = [l for l in lines if any(m in l for m in (
        "Error", "error:", "assert", "Assertion", "Traceback", "FAILED",
        "ModuleNotFound", "ImportError", "SyntaxError", "No module named",
        "collected", "cannot import"))]
    picked = (key or lines)[-6:]
    return " · ".join(l.strip()[:160] for l in picked)


def _emit(pid: int, agent: str, event_type: str, message: str,
          phase_id: int | None = None, payload: Any = None) -> None:
    ev = repo.add_event(pid, agent, event_type, message, phase_id=phase_id,
                        payload=payload)
    events.publish(pid, ev)
