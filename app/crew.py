"""CrewAI orchestration layer.

When CrewAI is installed we run each role as a genuine CrewAI Agent executing a
Task, with a per-role LLM pointed at Ollama Cloud. When it is not installed we
fall back to a direct Ollama chat call so the build loop keeps working. Both
paths share the same :class:`~app.llm.LLMConfig`.
"""
from __future__ import annotations

from .llm import LLMConfig, OllamaClient

try:  # pragma: no cover - exercised only when crewai is present
    from crewai import Agent, Crew, Process, Task
    from crewai import LLM as CrewLLM

    CREWAI_AVAILABLE = True
except Exception:  # noqa: BLE001
    CREWAI_AVAILABLE = False


# Role -> (persona role title, goal, backstory) for CrewAI agents.
ROLE_PERSONAS: dict[str, tuple[str, str, str]] = {
    "planner": (
        "Principal Software Architect",
        "Decompose product requirements into a rigorous, phased build plan with "
        "deliverables and per-phase test plans.",
        "You have shipped dozens of production systems and think in terms of "
        "incremental, independently verifiable milestones.",
    ),
    "coder": (
        "Senior Software Engineer",
        "Implement the current task by producing complete, runnable source files.",
        "You write clean, idiomatic, well-structured code and never leave TODO "
        "stubs where real logic is required.",
    ),
    "tester": (
        "Test Engineer",
        "Author thorough automated tests that validate the module's acceptance "
        "criteria.",
        "You believe untested code is broken code and cover happy paths and edge "
        "cases alike.",
    ),
    "fixer": (
        "Debugging Specialist",
        "Diagnose failing tests from their output and patch the source so they pass.",
        "You read stack traces surgically and make the smallest correct change.",
    ),
    "reviewer": (
        "Engineering Reviewer / QA Lead",
        "Verify that the delivered module actually meets the stated acceptance "
        "criteria, not merely that tests are green.",
        "You are meticulous and refuse to sign off on incomplete work.",
    ),
    "monitor": (
        "Product Manager",
        "Track changing requirements and surface incremental change proposals.",
        "You keep the build aligned with what the stakeholder actually wants.",
    ),
}


def _crew_llm(cfg: LLMConfig, role: str):
    """Build a CrewAI LLM for a role targeting its assigned OpenAI-compatible
    platform (Ollama Cloud, local Ollama, llama.cpp, …)."""
    model = cfg.model_for(role)
    base_url, api_key = cfg.endpoint_for(role)
    base = (base_url or "").rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    # All supported platforms speak the OpenAI protocol; use the openai prefix.
    return CrewLLM(
        model=f"openai/{model}",
        base_url=base,
        api_key=api_key or "ollama",
    )


def run_role(cfg: LLMConfig, role: str, system: str, user: str,
             json_mode: bool = False, temperature: float = 0.2) -> str:
    """Run one role's prompt and return the raw text output."""
    if CREWAI_AVAILABLE:
        try:
            return _run_via_crewai(cfg, role, system, user, temperature)
        except Exception:  # noqa: BLE001 - degrade to direct call
            pass
    return OllamaClient(cfg).chat(role, system, user, temperature=temperature,
                                  json_mode=json_mode)


def _run_via_crewai(cfg: LLMConfig, role: str, system: str, user: str,
                    temperature: float) -> str:
    persona = ROLE_PERSONAS.get(role, ("Specialist", "Complete the task.", ""))
    agent = Agent(
        role=persona[0],
        goal=persona[1],
        backstory=f"{persona[2]}\n\n{system}",
        llm=_crew_llm(cfg, role),
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description=user,
        expected_output="The requested artifact, exactly as instructed.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential,
                verbose=False)
    result = crew.kickoff()
    return str(getattr(result, "raw", result))
