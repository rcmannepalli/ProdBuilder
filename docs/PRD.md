# ProdBuilder — AI Autonomous Product Builder
### Product Requirements Document & Phased Build Plan (v0.1 — for review)

---

## 1. Vision

ProdBuilder is a self-hosted web app that turns a product requirements
description into a running codebase. It behaves like an autonomous
"Claude Code / GitHub Copilot Workspace"-style agent, but powered entirely by
**Ollama Cloud** models orchestrated through **CrewAI**.

Given:
- Product requirements (free text / structured)
- A target folder on disk
- LLM settings (Ollama Cloud API key, base URL, model-per-role mapping)

ProdBuilder will:
1. Generate a **build plan** — phases, deliverables, acceptance/test criteria.
2. **Build each module** phase-by-phase, writing code into the target folder.
3. **Test and fix** each module autonomously (generate tests → run → repair
   failures → re-run) until the phase's validation gate passes.
4. Only then advance to the next phase.
5. Continuously **re-read the product requirements** in the background and
   detect drift/changes, proposing (or auto-applying, per settings) plan
   adjustments and incremental features.
6. Show everything live in a **Monitor** panel: agent activity, file diffs,
   command output, test results, phase/task status — with controls
   (pause/resume/approve/rollback).

---

## 2. Goals / Non-Goals

**Goals**
- Turn a requirements doc into a working, tested codebase with minimal
  human intervention, using local/cloud Ollama models only (no OpenAI/Anthropic
  dependency required to run).
- Make the control loop transparent and interruptible — every agent action
  is visible and reversible.
- Support incremental evolution: requirements can change mid-build; the
  system reconciles rather than restarting.
- Keep the UI compact, keyboard-friendly, single-screen where possible.

**Non-Goals (v1)**
- Not a multi-tenant SaaS — single user / single machine, local target
  folder.
- Not a general chat IDE — no manual chat-driven file editing UI (that can
  come later); the loop is plan-driven and autonomous.
- Not responsible for deployment/hosting of the *built* product — it stops
  at "builds, passes tests, runs locally."
- No sandboxed multi-language execution matrix in v1 — start with
  Python/FastAPI-style generated projects, design the executor to be
  pluggable for other stacks later.

---

## 3. Tech Stack

| Layer | Choice |
|---|---|
| Backend framework | FastAPI |
| Templates | Jinja2 |
| Frontend interactivity | HTMX (server-driven partial updates) + Alpine.js (local UI state) |
| Styling | Tailwind CSS (compact, dense layout) |
| Local DB | DuckDB (embedded, file-based — projects, plans, runs, logs, requirement versions) |
| Agent orchestration | CrewAI (crews/agents/tasks/processes) |
| LLM runtime | Ollama Cloud (OpenAI-compatible endpoint) via CrewAI's LLM abstraction |
| Live updates | Server-Sent Events (SSE) endpoint feeding the Monitor panel (HTMX `hx-sse` / `sse-swap`) |
| Sandboxed execution | Subprocess runner scoped to the target folder, with timeouts + captured stdout/stderr |
| Packaging | Single FastAPI app, `uvicorn`, no external services required beyond Ollama Cloud |

Why SSE over WebSockets: one-directional log/event stream, simpler with
HTMX, no extra JS framework needed — fits the "compact, minimal JS" goal.

---

## 4. Personas / Primary Use Case

Single developer/operator who:
1. Pastes/writes product requirements.
2. Points ProdBuilder at an empty (or existing) target folder.
3. Configures Ollama Cloud credentials + which model handles which role
   (e.g. a bigger model for planning/architecture, a faster one for
   coding/test-fixing loops).
4. Clicks **Start Build**, watches the Monitor, occasionally approves a
   plan change or reviews a diff, and gets a working, tested codebase.

---

## 5. Agent Architecture (CrewAI)

Each role is a CrewAI `Agent` with its own goal/backstory and an assigned
LLM (so the user can mix models — e.g. `qwen2.5-coder` for coding,
a lighter model for monitoring). Crews run per phase, not all at once.

| Agent | Responsibility |
|---|---|
| **Product Manager (PM) Agent** | Parses raw requirements into structured features/user stories/acceptance criteria. Detects requirement changes on each monitoring pass and produces a diff summary. |
| **Architect / Planner Agent** | Produces the phased build plan: phases → modules → deliverables → file-level task list → test plan per phase. Re-plans incrementally when PM Agent reports drift. |
| **Coder Agent** | Implements the current phase's tasks: creates/edits files in the target folder. Works task-by-task, one module at a time. |
| **Test Writer Agent** | Generates/updates unit + integration tests for the module just built, per the phase's test plan. |
| **Test Runner / Validator** (tool, not LLM) | Executes the test suite (pytest, etc.) in the target folder via subprocess; returns pass/fail + logs. Deterministic, not an agent. |
| **Fixer Agent** | On test failure, receives failing test output + relevant source, proposes and applies a patch; loop bounded by a max-retry setting per phase. |
| **Reviewer / QA Agent** | Final pass over a phase before it's marked "validated": checks acceptance criteria from the PM Agent are actually met, not just "tests green." |
| **Requirements Monitor Agent** | Background job (interval-based) that re-reads the live requirements input, diffs against the last-seen version stored in DuckDB, and raises "incremental change" tasks that the Planner triages into the plan (new phase, modify current phase, or backlog). |

**Process model:** CrewAI `Process.sequential` within a phase
(PM → Plan → Code → Test-write → Run → Fix-loop → Review), with the
Fix-loop as a bounded sub-loop (config: `max_fix_attempts`, default 3)
before the phase is flagged `needs_attention` for human review instead of
silently failing forward.

---

## 6. Core Workflow (State Machine)

```
 [Requirements Input] --PM Agent--> [Structured Spec]
        |
        v
 [Architect Agent] --> [Build Plan: Phases[ Deliverables, Tasks, Test Plan ]]
        |
        v
 ┌───────────────── per Phase loop ─────────────────┐
 │ PENDING -> BUILDING (Coder Agent writes modules)  │
 │        -> TEST_WRITING (Test Writer Agent)        │
 │        -> TESTING (Validator runs suite)          │
 │            ├─ pass -> REVIEW (Reviewer Agent)     │
 │            │        ├─ pass -> DONE -> next phase │
 │            │        └─ fail -> BUILDING (revise)  │
 │            └─ fail -> FIXING (Fixer Agent)         │
 │                     -> TESTING (retry, bounded)    │
 │                     -> NEEDS_ATTENTION (if maxed)  │
 └────────────────────────────────────────────────────┘
        |
        v  (continuously, in parallel with the phase loop)
 [Requirements Monitor Agent] --diff--> [Change Proposal]
        |                                     |
        |                     auto-apply? ----+---- ask user (Monitor panel action)
        v
 [Architect Agent re-plans affected phase(s)] --> merges into Build Plan
```

Every state transition is written to DuckDB and emitted as an SSE event
for the Monitor panel.

---

## 7. Data Model (DuckDB)

- `projects (id, name, target_folder, status, created_at, updated_at)`
- `requirements_versions (id, project_id, version_no, raw_text, structured_json, created_at, diff_from_prev)`
- `settings (id, project_id, ollama_base_url, ollama_api_key_ref, model_map_json, max_fix_attempts, auto_apply_changes, poll_interval_sec)`
- `phases (id, project_id, order_no, name, description, deliverables_json, test_plan_json, status, started_at, completed_at)`
- `tasks (id, phase_id, title, kind[code|test|fix|review], status, file_paths_json, created_at, updated_at)`
- `test_runs (id, phase_id, attempt_no, passed, summary, stdout, stderr, duration_ms, created_at)`
- `agent_events (id, project_id, phase_id, agent_name, event_type, message, payload_json, created_at)` — feeds the Monitor
- `change_proposals (id, project_id, source[monitor|manual], summary, diff_json, status[pending|applied|rejected], created_at)`

`ollama_api_key_ref` stores a reference to an OS-keyring/encrypted-file
secret, never the raw key, in DuckDB.

---

## 8. API Surface (FastAPI)

- `GET /` — compact dashboard (settings + requirements + plan + monitor, one screen)
- `POST /projects` — create project (name, target folder)
- `POST /projects/{id}/requirements` — submit/update requirements text
- `POST /projects/{id}/settings` — Ollama URL/key/model map/limits
- `POST /projects/{id}/plan` — trigger Architect Agent to (re)generate the build plan
- `GET /projects/{id}/plan` — fetch current plan (HTMX partial)
- `POST /projects/{id}/build/start` — start the phase-loop runner (background task)
- `POST /projects/{id}/build/pause` / `/resume` / `/stop`
- `POST /projects/{id}/phases/{phase_id}/retry`
- `POST /projects/{id}/change-proposals/{cp_id}/approve` / `/reject`
- `GET /projects/{id}/events` — SSE stream for the Monitor panel
- `GET /projects/{id}/files` — read-only tree/diff viewer of the target folder

---

## 9. UI Layout (compact, single screen)

```
┌─────────────────────────────────────────────────────────────────┐
│ ProdBuilder            [Project: ▾]      [● Running] [Pause][Stop]│
├───────────────┬─────────────────────────────┬────────────────────┤
│ Requirements  │  Build Plan (phases)         │ Monitor / Activity │
│ (textarea,    │  ▸ Phase 1  ✔ done           │ live agent log     │
│  compact,     │  ▾ Phase 2  ⟳ building       │ (SSE stream,       │
│  autosave)    │     - task: coder…  ✔        │  auto-scroll,      │
│               │     - task: tests…  ⟳        │  filter by agent)  │
│ Settings ▾    │  ▸ Phase 3  ⋯ pending        │                    │
│  - base url   │                              │ [Change Proposals] │
│  - api key    │  [Regenerate Plan]           │  pending: 1  [▾]   │
│  - model map  │                              │                    │
│  - limits     │                              │ [File Diffs Tab]   │
└───────────────┴─────────────────────────────┴────────────────────┘
```

- Left rail: requirements editor + collapsible settings (all in one
  narrow column, no separate settings page needed for v1).
- Center: phase/task tree, collapsible, status badges, "Regenerate Plan"
  and per-phase "Retry" actions.
- Right rail: tabbed Monitor — **Activity** (agent chatter/log lines),
  **Change Proposals** (approve/reject incremental changes), **Diffs**
  (unified diff viewer of files touched in the current/last phase).
- Everything server-rendered via Jinja2 partials, swapped with HTMX;
  Alpine.js only for local toggles (collapse/expand, tabs, modals) —
  no client-side state duplication of server truth.

---

## 10. Settings Model

Per project (persisted, key stored securely):
- `ollama_base_url` (e.g. Ollama Cloud endpoint)
- `ollama_api_key`
- `model_map`: `{ planner: "...", coder: "...", tester: "...", fixer: "...", reviewer: "...", monitor: "..." }`
- `max_fix_attempts` (default 3)
- `auto_apply_changes` (bool — auto-merge Requirements Monitor proposals vs. require approval)
- `poll_interval_sec` (Requirements Monitor cadence, default 60s)
- `test_command` override (default auto-detected: `pytest`, else configurable)

---

## 11. Non-Functional Requirements

- **Observability**: every agent call logged (prompt/response summary,
  tokens if reported, duration) to `agent_events` and streamed to Monitor.
- **Safety**: all file writes and shell/test execution are sandboxed to
  the configured target folder (path traversal guarded); destructive
  commands (`rm -rf`, `git push`, etc.) are never auto-run by agents —
  only the whitelisted test/build commands.
- **Resilience**: bounded retry loops everywhere (fix-loop, LLM call
  retries with backoff); a phase that exceeds retries goes to
  `needs_attention`, never silently marked done.
- **Resumability**: build state lives in DuckDB, so stopping/restarting
  the server resumes from the last known phase/task state.
- **Extensibility**: LLM client is an interface (`LLMProvider`) with an
  Ollama Cloud implementation now, so other OpenAI-compatible providers
  can be added later without touching agent logic.
- **Secrets**: API key never rendered back to the browser in plaintext
  after save; stored encrypted at rest.

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LLM produces broken/incomplete code repeatedly | Bounded fix-loop + `needs_attention` escalation instead of infinite retries |
| Requirements Monitor causes plan thrash (constant re-planning) | Debounce: only act on diffs above a similarity threshold; batch changes per poll interval |
| Long-running background builds vs. request/response model | Use FastAPI `BackgroundTasks` / a lightweight in-process task runner + SSE, not blocking requests |
| Model quality varies a lot by role | Per-role model mapping in settings so user can tune (bigger model for planning, faster for iteration) |
| Target folder collisions / accidental overwrite of existing project | Require explicit "existing folder" confirmation + always operate through a diff-preview before destructive edits when folder is non-empty |
| DuckDB single-writer concurrency under background tasks + web requests | Single-process app, serialize writes through one connection/queue |

---

## 13. Phased Build Plan (for building ProdBuilder itself)

Each phase below ends with an explicit validation gate — mirroring the
same discipline ProdBuilder will enforce on the products *it* builds.

### Phase 0 — Project Scaffolding
- Deliverables: repo layout, `pyproject`/`requirements.txt`, FastAPI app
  skeleton, Jinja2 + Tailwind (CDN or compiled) + HTMX + Alpine wired up,
  DuckDB schema migrations, base layout template.
- Test plan: app boots, `/` renders, DB tables created on startup.

### Phase 1 — Settings & Project Management
- Deliverables: create/select project, settings form (Ollama URL/key/model
  map/limits), secret storage, requirements textarea with autosave.
- Test plan: CRUD round-trip tests for project/settings; secret is never
  returned in plaintext on GET.

### Phase 2 — Ollama Cloud + CrewAI Integration Layer
- Deliverables: `LLMProvider` abstraction, CrewAI agent/crew factory using
  per-role model map, a smoke-test call to Ollama Cloud from settings
  ("Test Connection" button).
- Test plan: mocked LLM unit tests; live "Test Connection" verified
  manually against real Ollama Cloud credentials.

### Phase 3 — Planner: Requirements → Build Plan
- Deliverables: PM Agent + Architect Agent pipeline producing structured
  phases/deliverables/test plans, persisted to DuckDB, rendered in the
  center panel.
- Test plan: given a sample requirements doc, plan generation produces
  well-formed phases with non-empty test plans; regeneration is
  idempotent-safe (versioned, not silently overwritten).

### Phase 4 — Build Loop: Coder + Test Writer + Validator
- Deliverables: phase-loop runner (background task), Coder Agent writing
  files into target folder, Test Writer Agent generating tests, Validator
  executing them via subprocess, status transitions persisted.
- Test plan: run against a trivial sample spec ("build a CLI that adds two
  numbers") end-to-end, confirm files created, tests generated, tests
  pass, phase marked done.

### Phase 5 — Fixer + Reviewer Loop
- Deliverables: Fixer Agent bounded retry loop on failing tests, Reviewer
  Agent acceptance-criteria check before phase completion,
  `needs_attention` escalation path + UI affordance to view/retry it.
- Test plan: seed a deliberately failing test, confirm Fixer resolves it
  within retry budget; seed an unfixable case, confirm escalation (not
  infinite loop, not false "done").

### Phase 6 — Monitor Panel (SSE) + Activity Log
- Deliverables: `agent_events` emission from every agent/tool call, SSE
  endpoint, HTMX-driven live Activity tab, filter by agent/phase.
- Test plan: manual verification that starting a build streams live
  events with no page reload; reconnect behavior after browser refresh.

### Phase 7 — Diff Viewer + Change Proposals (Requirements Monitor)
- Deliverables: file tree + unified diff viewer for target folder;
  Requirements Monitor Agent background poller; Change Proposal
  generation, approve/reject UI, auto-apply setting.
- Test plan: edit the requirements mid-build, confirm a change proposal
  appears within one poll interval; approve it and confirm the plan is
  incrementally updated (not fully regenerated) when the diff is additive.

### Phase 8 — Controls, Resumability, Polish
- Deliverables: pause/resume/stop controls wired to the runner; resume
  from DuckDB state after process restart; compact UI pass (spacing,
  keyboard shortcuts, empty/error states); README + setup docs.
- Test plan: kill the server mid-build, restart, confirm it resumes at
  the correct phase/task without corrupting state or redoing completed
  phases.

### Phase 9 (stretch) — Multi-language / pluggable executors
- Deliverables: abstract the test-runner/build-command detection so
  non-Python target projects (Node, etc.) are supported.
- Test plan: build a trivial Node project end-to-end through the same
  loop.

---

## 14. Open Questions for You

1. **Model roles**: do you already have specific Ollama Cloud model names
   in mind per role (planner/coder/tester/fixer/reviewer), or should v1
   default all roles to one model with per-role override optional?
2. **Auto-apply changes**: should Requirements Monitor changes ever
   auto-apply by default, or always require approval in v1 (safer)?
3. **Existing (non-empty) target folders**: is v1 scoped to *new* projects
   only, or must it also handle "build into an existing codebase" from
   day one?
4. **Git**: should ProdBuilder `git init`/commit per phase automatically
   (nice audit trail), or leave git entirely to the user for now?
5. **Test command detection**: is `pytest` as the default/only test runner
   acceptable for v1, or do you need a configurable command from the
   start (e.g. `npm test`) even though non-Python is a Phase 9 stretch?
6. **Deployment target**: confirm this runs as a single local
   `uvicorn` process for now (no Docker/compose requirement for v1)?

---

Once you confirm/adjust the above (especially section 14), I'll proceed to
implement starting at **Phase 0**.
