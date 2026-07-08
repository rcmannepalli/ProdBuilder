"""ProdBuilder FastAPI application — routes, HTMX partials, and SSE."""
from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import agents, events, executor, llm, logging_conf, repo
from .config import (APP_NAME, APP_TAGLINE, BASE_DIR, DEFAULT_MODEL_MAP,
                     PROVIDER_KINDS)
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
    logging_conf.configure()
    get_conn()  # initialise schema
    events.set_loop(asyncio.get_event_loop())
    logging_conf.logger.info("%s started", APP_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _emit_event(pid: int, agent: str, event_type: str, message: str) -> None:
    """Persist an event and push it to the live Monitor stream."""
    ev = repo.add_event(pid, agent, event_type, message)
    events.publish(pid, ev)


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
        "provider_kinds": PROVIDER_KINDS,
        "log_level": logging_conf.current_level(),
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
    provider_map = {}
    for role in ROLES:
        pv = (form.get(f"provider_{role}") or "").strip()
        if pv:
            provider_map[role] = int(pv)
    repo.update_settings(
        pid,
        model_map=model_map or dict(DEFAULT_MODEL_MAP),
        provider_map=provider_map,
        max_fix_attempts=form.get("max_fix_attempts") or 3,
        poll_interval_sec=form.get("poll_interval_sec") or 60,
        auto_apply_changes=bool(form.get("auto_apply_changes")),
        use_venv=bool(form.get("use_venv")),
        test_command=(form.get("test_command") or "python -m pytest -q").strip(),
    )
    return templates.TemplateResponse(request, "partials/saved.html",
                                      {"request": request, "label": "Settings saved"})


# --- providers -------------------------------------------------------------

@app.post("/projects/{pid}/providers", response_class=HTMLResponse)
async def add_provider(request: Request, pid: int):
    form = await request.form()
    kind = (form.get("kind") or "ollama_cloud").strip()
    name = (form.get("name") or PROVIDER_KINDS.get(kind, {}).get("label", kind)).strip()
    base_url = (form.get("base_url") or
                PROVIDER_KINDS.get(kind, {}).get("base_url", "")).strip()
    repo.add_provider(pid, name, kind, base_url, (form.get("api_key") or "").strip())
    return _providers_partial(request, pid)


@app.post("/projects/{pid}/providers/{prov_id}/toggle", response_class=HTMLResponse)
async def toggle_provider(request: Request, pid: int, prov_id: int):
    prov = repo.get_provider(prov_id)
    if prov:
        repo.update_provider(prov_id, enabled=not prov["enabled"])
    return _providers_partial(request, pid)


@app.post("/projects/{pid}/providers/{prov_id}/delete", response_class=HTMLResponse)
async def remove_provider(request: Request, pid: int, prov_id: int):
    repo.delete_provider(prov_id)
    return _providers_partial(request, pid)


@app.post("/projects/{pid}/providers/{prov_id}/test", response_class=HTMLResponse)
async def test_provider(request: Request, pid: int, prov_id: int):
    prov = repo.get_provider(prov_id)
    model = repo.get_settings(pid)["model_map"].get("planner") or DEFAULT_MODEL_MAP["planner"]
    ok, msg = await asyncio.to_thread(
        llm.test_endpoint, prov["base_url"], prov["api_key"], model)
    return templates.TemplateResponse(request, "partials/conn_result.html",
                                      {"request": request, "ok": ok, "message": msg})


def _providers_partial(request: Request, pid: int):
    s = repo.get_settings(pid)
    return templates.TemplateResponse(request, "partials/providers.html", {
        "request": request, "project": repo.get_project(pid),
        "settings": s, "roles": ROLES, "provider_kinds": PROVIDER_KINDS,
        "default_model_map": DEFAULT_MODEL_MAP,
    })


@app.post("/projects/{pid}/settings/test", response_class=HTMLResponse)
async def test_connection(request: Request, pid: int):
    cfg = LLMConfig.from_settings(repo.get_settings(pid))
    ok, msg = await asyncio.to_thread(OllamaClient(cfg).test_connection)
    return templates.TemplateResponse(request,
        "partials/conn_result.html",
        {"request": request, "ok": ok, "message": msg})


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@app.post("/projects/{pid}/plan", response_class=HTMLResponse)
async def generate_plan(request: Request, pid: int):
    project = repo.get_project(pid)
    target = project["target_folder"] if project else ""
    reqs = repo.latest_requirements(pid)
    raw = reqs["raw_text"] if reqs else ""

    # If a PRD exists in the target folder and no requirements are saved, use it.
    prd_name, prd_text = executor.find_prd(target) if target else ("", "")
    if prd_text and not raw.strip():
        raw = prd_text
        repo.add_requirements_version(pid, raw)
        reqs = repo.latest_requirements(pid)
        _emit_event(pid, "Architect", "plan", f"Imported requirements from {prd_name}.")

    # Enrichment mode when the folder already contains source.
    existing_code = executor.codebase_summary(target) if target and _has_code(target) else ""
    cfg = LLMConfig.from_settings(repo.get_settings(pid))
    mode = "Enriching existing codebase" if existing_code else "Generating build plan"
    _emit_event(pid, "Architect", "plan", f"{mode} from requirements…")

    def _build():
        structured = agents.parse_requirements(cfg, raw)
        plan = agents.generate_plan(cfg, structured, raw, existing_code=existing_code)
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
                   f"Generated {len(repo.list_phases(pid))}-phase "
                   f"{'enrichment ' if existing_code else ''}plan.")
    return templates.TemplateResponse(request,
        "partials/plan.html",
        {"request": request, "project": repo.get_project(pid),
         "phases": repo.list_phases(pid), "run_status": runner.status(pid)})


def _has_code(target: str) -> bool:
    exts = (".py", ".js", ".ts", ".go", ".rs", ".java", ".rb", ".php")
    return any(p.lower().endswith(exts) for p in executor.list_tree(target))


@app.post("/projects/{pid}/import-prd", response_class=HTMLResponse)
async def import_prd(request: Request, pid: int):
    project = repo.get_project(pid)
    target = project["target_folder"] if project else ""
    name, text = executor.find_prd(target) if target else ("", "")
    if text:
        repo.add_requirements_version(pid, text)
        label = f"Imported {name}"
    else:
        label = "No PRD found in target folder"
    ctx = _dashboard_ctx(request, project)
    ctx["prd_label"] = label
    return templates.TemplateResponse(request, "partials/sidebar_requirements.html", ctx)


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


@app.post("/projects/{pid}/logs/clear", response_class=HTMLResponse)
async def clear_activity(request: Request, pid: int):
    repo.clear_events(pid)
    logging_conf.logger.info("Activity log cleared for project %s", pid)
    return templates.TemplateResponse(request, "partials/monitor.html",
                                      {"request": request, "project_id": pid,
                                       "events": []})


@app.post("/log-level", response_class=HTMLResponse)
async def set_log_level(request: Request):
    form = await request.form()
    level = logging_conf.set_level((form.get("level") or "INFO"))
    if form.get("clear"):
        logging_conf.clear_log_file()
    return templates.TemplateResponse(request, "partials/log_level.html",
                                      {"request": request,
                                       "level": level,
                                       "cleared": bool(form.get("clear"))})


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
# Explorer (nested tree) + editor
# ---------------------------------------------------------------------------

@app.get("/projects/{pid}/tree", response_class=HTMLResponse)
async def get_tree(request: Request, pid: int, selected: str = ""):
    project = repo.get_project(pid)
    target = project["target_folder"] if project else ""
    tree = executor.build_tree(target) if target else []
    return templates.TemplateResponse(request, "partials/tree.html", {
        "request": request, "project_id": pid, "nodes": tree,
        "selected": selected, "empty": not tree})


@app.get("/projects/{pid}/file", response_class=HTMLResponse)
async def get_file(request: Request, pid: int, path: str = ""):
    project = repo.get_project(pid)
    target = project["target_folder"] if project else ""
    content, status = ("", "missing")
    if target and path:
        content, status = executor.read_single_file(target, path)
    lang = _lang_for(path)
    return templates.TemplateResponse(request, "partials/editor.html", {
        "request": request, "project_id": pid, "path": path,
        "content": content, "status": status, "lang": lang,
        "line_count": content.count("\n") + 1 if content else 0})


def _lang_for(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {
        "py": "Python", "js": "JavaScript", "ts": "TypeScript",
        "tsx": "TypeScript", "jsx": "JavaScript", "html": "HTML", "css": "CSS",
        "json": "JSON", "md": "Markdown", "sql": "SQL", "go": "Go", "rs": "Rust",
        "java": "Java", "yml": "YAML", "yaml": "YAML", "toml": "TOML",
        "sh": "Shell", "txt": "Text",
    }.get(ext, ext.upper() or "Text")


@app.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok", "app": APP_NAME})
