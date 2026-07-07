"""ProdBuilder FastAPI application — routes, HTMX partials, and SSE."""
from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import agents, events, executor, repo
from .config import APP_NAME, APP_TAGLINE, BASE_DIR, DEFAULT_MODEL_MAP
from .db import get_conn
from .llm import LLMConfig, OllamaClient
from . import runner

app = FastAPI(title=APP_NAME)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")),
          name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
templates.env.globals.update(APP_NAME=APP_NAME, APP_TAGLINE=APP_TAGLINE)

ROLES = ["planner", "coder", "tester", "fixer", "reviewer", "monitor"]


@app.on_event("startup")
async def _startup() -> None:
    get_conn()  # initialise schema
    events.set_loop(asyncio.get_event_loop())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_project(request: Request) -> dict | None:
    pid = request.query_params.get("p")
    if pid:
        proj = repo.get_project(int(pid))
        if proj:
            return proj
    projects = repo.list_projects()
    return projects[0] if projects else None


def _dashboard_ctx(request: Request, project: dict | None) -> dict:
    ctx: dict = {
        "request": request,
        "projects": repo.list_projects(),
        "project": project,
        "roles": ROLES,
        "default_model_map": DEFAULT_MODEL_MAP,
    }
    if project:
        pid = project["id"]
        ctx.update({
            "settings": repo.get_settings(pid),
            "requirements": repo.latest_requirements(pid),
            "phases": repo.list_phases(pid),
            "proposals": repo.list_proposals(pid, status="pending"),
            "run_status": runner.status(pid),
        })
    return ctx


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    project = _current_project(request)
    return templates.TemplateResponse(request, "index.html",
                                      _dashboard_ctx(request, project))


@app.post("/projects", response_class=HTMLResponse)
async def create_project(request: Request, name: str = Form(...),
                         target_folder: str = Form(...)):
    pid = repo.create_project(name.strip(), target_folder.strip())
    resp = HTMLResponse("")
    resp.headers["HX-Redirect"] = f"/?p={pid}"
    return resp


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------

@app.post("/projects/{pid}/requirements", response_class=HTMLResponse)
async def save_requirements(request: Request, pid: int,
                            requirements: str = Form("")):
    latest = repo.latest_requirements(pid)
    if not latest or latest["raw_text"] != requirements:
        repo.add_requirements_version(pid, requirements)
        repo.touch_project(pid)
    return templates.TemplateResponse(request, "partials/saved.html",
                                      {"request": request, "label": "Requirements saved"})


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@app.post("/projects/{pid}/settings", response_class=HTMLResponse)
async def save_settings(request: Request, pid: int):
    form = await request.form()
    model_map = {role: (form.get(f"model_{role}") or "").strip() for role in ROLES}
    model_map = {k: v for k, v in model_map.items() if v}
    repo.update_settings(
        pid,
        ollama_base_url=(form.get("ollama_base_url") or "").strip(),
        api_key=(form.get("api_key") or "").strip(),
        model_map=model_map or dict(DEFAULT_MODEL_MAP),
        max_fix_attempts=form.get("max_fix_attempts") or 3,
        poll_interval_sec=form.get("poll_interval_sec") or 60,
        auto_apply_changes=bool(form.get("auto_apply_changes")),
        test_command=(form.get("test_command") or "pytest -q").strip(),
    )
    return templates.TemplateResponse(request, "partials/saved.html",
                                      {"request": request, "label": "Settings saved"})


@app.post("/projects/{pid}/settings/test", response_class=HTMLResponse)
async def test_connection(request: Request, pid: int):
    s = repo.get_settings(pid)
    cfg = LLMConfig(s["ollama_base_url"], s["api_key"], s["model_map"])
    ok, msg = await asyncio.to_thread(OllamaClient(cfg).test_connection)
    return templates.TemplateResponse(request, 
        "partials/conn_result.html",
        {"request": request, "ok": ok, "message": msg})


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@app.post("/projects/{pid}/plan", response_class=HTMLResponse)
async def generate_plan(request: Request, pid: int):
    reqs = repo.latest_requirements(pid)
    raw = reqs["raw_text"] if reqs else ""
    s = repo.get_settings(pid)
    cfg = LLMConfig(s["ollama_base_url"], s["api_key"], s["model_map"])

    def _build():
        structured = agents.parse_requirements(cfg, raw)
        plan = agents.generate_plan(cfg, structured, raw)
        repo.clear_plan(pid)
        for i, ph in enumerate(plan, start=1):
            phase_id = repo.add_phase(pid, i, ph["name"], ph["description"],
                                      ph["deliverables"], ph["test_plan"])
            for t in ph["tasks"]:
                repo.add_task(phase_id, t["title"], t["kind"], t.get("files"))
        return structured

    structured = await asyncio.to_thread(_build)
    if reqs:
        repo.add_requirements_version(pid, raw,
                                      structured_json=json.dumps(structured))
    repo.add_event(pid, "Architect", "plan",
                   f"Generated build plan with {len(repo.list_phases(pid))} phases.")
    return templates.TemplateResponse(request, 
        "partials/plan.html",
        {"request": request, "project": repo.get_project(pid),
         "phases": repo.list_phases(pid), "run_status": runner.status(pid)})


@app.get("/projects/{pid}/plan", response_class=HTMLResponse)
async def get_plan(request: Request, pid: int):
    return templates.TemplateResponse(request, 
        "partials/plan.html",
        {"request": request, "project": repo.get_project(pid),
         "phases": repo.list_phases(pid), "run_status": runner.status(pid)})


@app.post("/projects/{pid}/phases/{phase_id}/retry", response_class=HTMLResponse)
async def retry_phase(request: Request, pid: int, phase_id: int):
    repo.set_phase_status(phase_id, "pending")
    for t in repo.list_tasks(phase_id):
        repo.set_task_status(t["id"], "pending")
    repo.add_event(pid, "System", "control", "Phase reset to pending.",
                   phase_id=phase_id)
    return await get_plan(request, pid)


# ---------------------------------------------------------------------------
# Build controls
# ---------------------------------------------------------------------------

@app.post("/projects/{pid}/build/{action}", response_class=HTMLResponse)
async def build_control(request: Request, pid: int, action: str):
    if action == "start":
        runner.start_build(pid)
    elif action == "pause":
        runner.pause_build(pid)
    elif action == "resume":
        runner.resume_build(pid)
    elif action == "stop":
        runner.stop_build(pid)
    return templates.TemplateResponse(request, 
        "partials/controls.html",
        {"request": request, "project": repo.get_project(pid),
         "run_status": runner.status(pid)})


@app.get("/projects/{pid}/status", response_class=HTMLResponse)
async def status_pill(request: Request, pid: int):
    return templates.TemplateResponse(request, 
        "partials/controls.html",
        {"request": request, "project": repo.get_project(pid),
         "run_status": runner.status(pid)})


# ---------------------------------------------------------------------------
# Monitor / events
# ---------------------------------------------------------------------------

@app.get("/projects/{pid}/monitor", response_class=HTMLResponse)
async def monitor(request: Request, pid: int, after: int = 0):
    return templates.TemplateResponse(request, 
        "partials/monitor.html",
        {"request": request, "project_id": pid,
         "events": repo.list_events(pid, after_id=after)})


def _render_event_html(ev: dict) -> str:
    """Render one event to a single-line HTML fragment for SSE (htmx sse ext)."""
    html = templates.get_template("partials/event_line.html").render(ev=ev)
    return " ".join(html.split("\n"))


@app.get("/projects/{pid}/events")
async def sse_events(request: Request, pid: int):
    async def stream():
        q = events.subscribe(pid)
        try:
            # Replay a short backlog so a fresh connection isn't empty.
            for ev in repo.list_events(pid)[-40:]:
                yield f"event: message\ndata: {_render_event_html(ev)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    ev = json.loads(payload)
                    yield f"event: message\ndata: {_render_event_html(ev)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            events.unsubscribe(pid, q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------

@app.get("/projects/{pid}/proposals", response_class=HTMLResponse)
async def get_proposals(request: Request, pid: int):
    return templates.TemplateResponse(request, 
        "partials/proposals.html",
        {"request": request, "project_id": pid,
         "proposals": repo.list_proposals(pid, status="pending")})


@app.post("/projects/{pid}/proposals/{cp_id}/{action}", response_class=HTMLResponse)
async def proposal_action(request: Request, pid: int, cp_id: int, action: str):
    if action == "approve":
        runner.apply_proposal(pid, cp_id)
    elif action == "reject":
        runner.reject_proposal(pid, cp_id)
    return templates.TemplateResponse(request, 
        "partials/proposals.html",
        {"request": request, "project_id": pid,
         "proposals": repo.list_proposals(pid, status="pending")})


# ---------------------------------------------------------------------------
# Files / diffs
# ---------------------------------------------------------------------------

@app.get("/projects/{pid}/files", response_class=HTMLResponse)
async def get_files(request: Request, pid: int, path: str = ""):
    project = repo.get_project(pid)
    target = project["target_folder"] if project else ""
    tree = executor.list_tree(target) if target else []
    content = ""
    if path:
        files = executor.read_files(target, limit=500)
        content = files.get(path, "(binary or missing file)")
    return templates.TemplateResponse(request, 
        "partials/files.html",
        {"request": request, "project_id": pid, "tree": tree,
         "selected": path, "content": content})


@app.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok", "app": APP_NAME})
