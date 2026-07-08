# ProdBuilder — Autonomous Product Engineering

ProdBuilder turns a product requirements description into a running, tested
codebase. It behaves like an autonomous "Claude Code / Copilot Workspace"-style
agent — but powered entirely by **Ollama Cloud** models orchestrated through
**CrewAI**.

Give it (1) product requirements, (2) a target folder, and (3) your Ollama Cloud
settings, and it will:

1. **Plan** — a Product Manager + Architect agent turn requirements into a
   phased build plan (phases → deliverables → per-phase test plans → tasks).
2. **Build** — a Coder agent writes each module into the target folder.
3. **Test** — a Test Writer agent generates tests; a deterministic Validator
   runs them.
4. **Fix** — on failure, a Fixer agent patches the code in a bounded retry loop.
5. **Review** — a Reviewer agent checks the acceptance criteria before a phase is
   marked done, then advances to the next phase.
6. **Monitor & adapt** — a background Requirements Monitor re-reads requirements
   on an interval, detects drift, and raises **change proposals** you can approve
   (or auto-apply) to evolve the product incrementally.

Everything streams live to a compact, enterprise-styled **Monitor** panel.

---

## Tech stack

| Layer | Choice |
|---|---|
| Web framework | FastAPI |
| Templating | Jinja2 |
| Interactivity | HTMX (server-driven partials) + Alpine.js (local UI state) |
| Styling | Tailwind CSS (compact, corporate theme) |
| Storage | DuckDB (embedded) |
| Agents | CrewAI (with a direct-Ollama fallback so the loop runs even without CrewAI) |
| LLMs | Any OpenAI-compatible platform — Ollama Cloud, local Ollama, llama.cpp, or a generic endpoint — assignable per agent role |
| Live updates | Server-Sent Events (SSE) |

### Interface

A VS Code-style workspace: an **activity bar** (Explorer · Requirements · Build
Plan · Monitor · Settings), a **collapsible sidebar**, a central **editor** that
shows the generated files, and a live **Monitor** panel. The Explorer is a real
nested file/folder tree of the target folder — click any file to view its
contents (with line numbers) in the editor as the agents write it.

### Isolated test environments

The Test/Validator step runs in a **per-project virtual environment** created
inside the target folder (`.venv`, via `uv` when available, else stdlib `venv`).
It installs `requirements.txt` + `pytest`, and when a test fails with
`ModuleNotFoundError` it **auto-installs the missing package** and re-runs —
without spending an LLM fix attempt. Toggle with *"Test in an isolated venv"* in
Settings.

### Enrich an existing app / read a PRD

Point a project at a folder that already contains code and ProdBuilder switches
to **enrichment mode**: the planner is given a digest of the existing codebase
and produces phases that extend/refactor it rather than scaffolding from scratch.
If a `prd.md` / `PRD.md` (or `docs/PRD.md`, `requirements.md`) exists in the
target folder, use **Import PRD** in the Requirements panel to load it as the
requirements — or just generate a plan and it's picked up automatically.

### Logs

Clear a project's activity log from the Monitor footer. Website logging is
configurable between **INFO** (verbose) and **ERROR** (quiet) in Settings, with a
one-click **Clear log** for the rotating log file under the data directory.

### Multiple LLM platforms

Add several platforms in Settings (Ollama Cloud, local Ollama, llama.cpp, or any
OpenAI-compatible server), enable the ones you want, and assign a platform +
model **per role** — e.g. Ollama Cloud for planning and a local llama.cpp for the
coding/fix loop. All platforms speak the OpenAI `/v1` protocol, so the client is
uniform.

See [`docs/PRD.md`](docs/PRD.md) for the full product requirements document and
phased build plan.

---

## Quick start

```bash
# 1. Create a virtualenv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # includes CrewAI

# (Minimal, no CrewAI — the direct-Ollama path still works:)
# pip install fastapi "uvicorn[standard]" jinja2 python-multipart duckdb httpx cryptography

# 2. Run the app
python run.py                            # http://127.0.0.1:8000
# or: uvicorn app.main:app --reload
```

Then in the browser:

1. **New Project** → give it a name and an absolute target folder.
2. **Settings** (⚙️ in the activity bar) → under **LLM Platforms**, add a platform
   (e.g. Ollama Cloud with your base URL + API key), then assign a **platform and
   model per role** (planner / coder / tester / fixer / reviewer / monitor).
   Click **Test** on a platform to verify the connection.
3. **Requirements** (📝) → describe the product. It autosaves.
4. **Build Plan** (📋) → **Regenerate** to have the Architect agent produce the
   phased plan.
5. **Start Build** → watch the agents build, test, fix, and validate each phase
   in the **Monitor**; open the **Explorer** (🗂) to browse the generated files
   in the editor as they are written.

---

## Configuration

Settings are per-project and stored in DuckDB; the API key is **encrypted at
rest** (Fernet) and never returned to the browser in plaintext — only a masked
hint is shown.

| Setting | Purpose |
|---|---|
| LLM Platforms | Add/enable multiple OpenAI-compatible platforms (Ollama Cloud, local Ollama, llama.cpp, generic) |
| Platform + model per role | Mix platforms and models — e.g. Ollama Cloud for planning, a local llama.cpp coder for the build loop |
| Max fix attempts | Bounds the Fixer retry loop before a phase is escalated to `needs attention` |
| Poll interval | How often the Requirements Monitor re-checks for drift |
| Auto-apply changes | Auto-merge change proposals vs. require manual approval |
| Test command | Command the Validator runs (default `pytest -q`) |

Environment overrides: `PRODBUILDER_DATA` (data dir), `PRODBUILDER_DEFAULT_MODEL`,
`PRODBUILDER_OLLAMA_URL`, `HOST`, `PORT`, `RELOAD`.

---

## Safety

- **Sandboxed writes/execution** — all file writes and the test command are
  scoped to the project's target folder, with path-traversal guards.
- **Command allow-list** — the Validator only runs whitelisted commands
  (`pytest`, `python`, `npm`, …); destructive tokens (`rm`, `curl`, `sudo`,
  shell operators) are refused.
- **Bounded loops** — the fix loop is capped; a phase that can't pass is marked
  `needs_attention` rather than looping forever or being falsely marked done.
- **Resumable** — build state lives in DuckDB, so restarting the server resumes
  from the last completed phase.

---

## Architecture

```
app/
  main.py        FastAPI routes, HTMX partials, SSE endpoint
  config.py      Paths, defaults, model map
  db.py          DuckDB connection + schema (serialized via one lock)
  repo.py        Domain repository (projects, settings, phases, tasks, events…)
  secrets.py     Encrypt/mask API keys at rest
  llm.py         OpenAI-compatible client + per-role endpoint resolution
  crew.py        CrewAI agent/crew factory (per-role LLM) with direct fallback
  agents.py      Prompt construction + tolerant structured parsing per step
  executor.py    Sandboxed writes, guarded test execution, nested file tree
  events.py      In-process pub/sub feeding SSE
  runner.py      Background phase-loop state machine + requirements monitor
  templates/     Jinja2 VS Code-style shell + HTMX partials
                 (tree, editor, providers, plan, monitor, proposals, …)
  static/app.css Corporate design system layered on Tailwind
tests/           Unit + end-to-end build-loop tests (21)
docs/PRD.md      Product requirements & phased build plan
```

### Build state machine (per phase)

```
pending → building → test_writing → testing ─┬─ pass → review ─┬─ approved → done
                                              │                 └─ gaps    → needs_attention
                                              └─ fail → fixing → testing (bounded retries)
                                                                  └─ exhausted → needs_attention
```

---

## Testing

```bash
source .venv/bin/activate
python -m pytest -q
```

The suite covers the secret store, repository round-trips, the sandbox executor
(including the path-traversal guard and command allow-list), the JSON extraction
helpers, and two end-to-end runner tests that actually build a module, run
`pytest`, repair a deliberately broken build, and validate it.
