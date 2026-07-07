"""Unit tests for the ProdBuilder backend core."""
from pathlib import Path

from app import repo, secrets
from app.agents import _coerce_files, _fallback_plan, _fallback_structured
from app.executor import (command_allowed, list_tree, read_files, run_tests,
                          write_files)
from app.llm import LLMConfig, extract_json, _v1_url


# --- secrets ---------------------------------------------------------------

def test_secret_roundtrip_and_mask():
    token = secrets.encrypt("sk-supersecret-1234")
    assert token and token != "sk-supersecret-1234"
    assert secrets.decrypt(token) == "sk-supersecret-1234"
    masked = secrets.mask("sk-supersecret-1234")
    assert masked.endswith("1234") and "supersecret" not in masked


# --- projects / settings / requirements ------------------------------------

def test_project_settings_requirements_roundtrip(tmp_path):
    pid = repo.create_project("Test Proj", str(tmp_path / "out"))
    proj = repo.get_project(pid)
    assert proj["name"] == "Test Proj"

    repo.update_settings(pid, ollama_base_url="https://ollama.com",
                         api_key="secret-key-xyz", max_fix_attempts=5)
    s = repo.get_settings(pid)
    assert s["ollama_base_url"] == "https://ollama.com"
    assert s["max_fix_attempts"] == 5
    # Plaintext key never stored raw; masked hint hides the body.
    assert s["ollama_api_key_enc"] != "secret-key-xyz"
    assert s["api_key"] == "secret-key-xyz"
    assert "secret-key-xyz" not in s["api_key_masked"]

    repo.add_requirements_version(pid, "Build a widget")
    repo.add_requirements_version(pid, "Build a better widget")
    latest = repo.latest_requirements(pid)
    assert latest["version_no"] == 2
    prev = repo.query_prev_requirements(pid, latest["version_no"])
    assert prev["raw_text"] == "Build a widget"


def test_settings_blank_key_preserves_existing(tmp_path):
    pid = repo.create_project("Keep Key", str(tmp_path / "o"))
    repo.update_settings(pid, api_key="first-key")
    repo.update_settings(pid, api_key="")  # blank must not overwrite
    assert repo.get_settings(pid)["api_key"] == "first-key"


# --- plan / phases / events ------------------------------------------------

def test_plan_phases_tasks_and_events(tmp_path):
    pid = repo.create_project("Planned", str(tmp_path / "o"))
    ph = repo.add_phase(pid, 1, "Phase 1", "desc", ["d1"], ["t1"])
    repo.add_task(ph, "code it", "code", ["main.py"])
    repo.set_phase_status(ph, "done")
    phases = repo.list_phases(pid)
    assert phases[0]["status"] == "done"
    assert phases[0]["tasks"][0]["title"] == "code it"

    repo.add_event(pid, "Coder", "task", "hello")
    evs = repo.list_events(pid)
    assert any(e["message"] == "hello" for e in evs)


def test_change_proposals(tmp_path):
    pid = repo.create_project("CP", str(tmp_path / "o"))
    cp = repo.add_proposal(pid, "monitor", "changed", {"changes": ["a", "b"]})
    assert len(repo.list_proposals(pid, status="pending")) == 1
    repo.set_proposal_status(cp, "applied")
    assert repo.list_proposals(pid, status="pending") == []


# --- executor sandbox ------------------------------------------------------

def test_write_read_and_traversal_guard(tmp_path):
    target = str(tmp_path / "sandbox")
    write_files(target, [{"path": "pkg/mod.py", "content": "x = 1\n"}])
    files = read_files(target)
    assert files["pkg/mod.py"] == "x = 1\n"
    assert any("pkg/mod.py" in t for t in list_tree(target))

    import pytest
    with pytest.raises(ValueError):
        write_files(target, [{"path": "../escape.py", "content": "nope"}])


def test_command_allowlist():
    assert command_allowed("pytest -q")
    assert command_allowed("python -m pytest")
    assert not command_allowed("rm -rf /")
    assert not command_allowed("curl http://evil")
    assert not command_allowed("pytest && rm x")


def test_run_tests_executes(tmp_path):
    target = str(tmp_path / "runnable")
    write_files(target, [{"path": "test_ok.py",
                          "content": "def test_ok():\n    assert 1 + 1 == 2\n"}])
    result = run_tests(target, "python -m pytest -q", timeout=60)
    assert result.passed, result.stderr


# --- llm helpers -----------------------------------------------------------

def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('noise before {"a": 3} noise after') == {"a": 3}
    assert extract_json("[1, 2, 3]") == [1, 2, 3]
    assert extract_json("totally not json") is None


def test_v1_url_normalisation():
    assert _v1_url("https://ollama.com") == "https://ollama.com/v1/chat/completions"
    assert _v1_url("https://ollama.com/") == "https://ollama.com/v1/chat/completions"
    assert _v1_url("https://x/v1") == "https://x/v1/chat/completions"


def test_model_for_falls_back():
    cfg = LLMConfig("u", "k", {"coder": "m1"})
    assert cfg.model_for("coder") == "m1"
    assert cfg.model_for("unknown") == "m1"  # falls back to first available


# --- agents fallbacks ------------------------------------------------------

def test_fallback_structured_and_plan():
    structured = _fallback_structured("- feature a\n- feature b")
    assert structured["features"]
    plan = _fallback_plan(structured)
    assert len(plan) >= 3
    assert all(p["test_plan"] for p in plan)


def test_coerce_files_filters_invalid():
    out = _coerce_files({"files": [
        {"path": "a.py", "content": "x"},
        {"path": "", "content": "skip"},
        {"nope": True},
    ], "notes": "n"})
    assert out["files"] == [{"path": "a.py", "content": "x"}]
    assert out["notes"] == "n"
