"""Live agent-workflow state for the graphical pipeline view.

Derives, from the persisted event stream and phase table, which agent is working
now, what each agent last did (its input/feedback), and a recent timeline.
"""
from __future__ import annotations

from . import repo

# Ordered pipeline. (agent name as emitted in events, icon, short role).
PIPELINE = [
    ("Architect", "📐", "Plan & re-plan"),
    ("Coder", "⌨️", "Write modules"),
    ("Test Writer", "🧪", "Author tests"),
    ("Environment", "📦", "Install deps"),
    ("Validator", "✅", "Run tests"),
    ("Fixer", "🔧", "Fix failures"),
    ("Reviewer", "🔎", "Verify criteria"),
]

_ACTIVE_PHASE_STATUSES = {"building", "test_writing", "testing", "fixing", "review"}


def build_state(pid: int) -> dict:
    from . import runner  # local import to avoid a cycle

    events = repo.list_events(pid, limit=400)
    run_status = runner.status(pid)
    last = events[-1] if events else None
    active_agent = last["agent_name"] if (last and run_status == "running") else None

    # Latest event per pipeline agent.
    last_by_agent: dict[str, dict] = {}
    for ev in events:
        name = ev.get("agent_name") or ""
        for agent, _, _ in PIPELINE:
            if name == agent:
                last_by_agent[agent] = ev

    nodes = []
    for agent, icon, role in PIPELINE:
        ev = last_by_agent.get(agent)
        if active_agent == agent:
            status = "active"
        elif ev and ev.get("event_type") in ("warn", "error", "attention"):
            status = "error"
        elif ev:
            status = "done"
        else:
            status = "idle"
        nodes.append({
            "agent": agent, "icon": icon, "role": role, "status": status,
            "last_msg": (ev or {}).get("message", ""),
            "last_time": _hhmmss((ev or {}).get("created_at", "")),
        })

    # Current phase context.
    phases = repo.list_phases(pid)
    current = next((p for p in phases if p["status"] in _ACTIVE_PHASE_STATUSES), None)
    if not current:
        current = next((p for p in reversed(phases)
                        if p["status"] in ("needs_attention", "done")), None)
    done = sum(1 for p in phases if p["status"] == "done")

    timeline = [{
        "agent": e.get("agent_name", ""),
        "agent_key": (e.get("agent_name") or "System").split(" ")[0],
        "type": e.get("event_type", ""),
        "message": e.get("message", ""),
        "time": _hhmmss(e.get("created_at", "")),
    } for e in events[-18:]][::-1]

    return {
        "run_status": run_status,
        "active_agent": active_agent,
        "nodes": nodes,
        "current_phase": current,
        "phase_total": len(phases),
        "phase_done": done,
        "timeline": timeline,
    }


def _hhmmss(iso: str) -> str:
    return iso[11:19] if iso and len(iso) > 19 else ""
