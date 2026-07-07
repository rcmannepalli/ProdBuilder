"""Integration test: drive the autonomous build loop end-to-end with stub agents.

We stub the LLM-backed agent facade so the Coder writes a real module, the Test
Writer writes a real passing test, and (for the failure path) the Fixer repairs
a deliberately broken module. This exercises the runner state machine, the
sandboxed executor, and the real pytest run — no network required.
"""
import time

from app import agents, repo, runner


def _wait_until(pred, timeout=30.0):
    start = time.time()
    while time.time() - start < timeout:
        if pred():
            return True
        time.sleep(0.2)
    return False


def test_build_loop_builds_and_validates(tmp_path, monkeypatch):
    target = str(tmp_path / "adder")
    pid = repo.create_project("Adder", target)
    repo.add_requirements_version(pid, "Build add(a,b) that returns a+b")
    phase_id = repo.add_phase(pid, 1, "Core", "adder", ["add fn"],
                              ["add(2,3)==5"])
    repo.add_task(phase_id, "Implement add", "code", ["adder.py"])

    def fake_code(cfg, summary, phase, task, existing):
        return {"files": [{"path": "adder.py",
                           "content": "def add(a, b):\n    return a + b\n"}],
                "notes": "ok"}

    def fake_tests(cfg, summary, phase, existing):
        return {"files": [{"path": "test_adder.py",
                           "content": "from adder import add\n\n"
                                      "def test_add():\n    assert add(2, 3) == 5\n"}],
                "notes": "ok"}

    monkeypatch.setattr(agents, "generate_code", fake_code)
    monkeypatch.setattr(agents, "generate_tests", fake_tests)
    monkeypatch.setattr(agents, "review_phase",
                        lambda *a, **k: {"approved": True, "notes": "lgtm", "gaps": []})

    runner.start_build(pid)
    assert _wait_until(lambda: repo.get_project(pid)["status"] == "done"), \
        repo.get_project(pid)["status"]

    ph = repo.get_phase(phase_id)
    assert ph["status"] == "done"
    tr = repo.latest_test_run(phase_id)
    assert tr and tr["passed"]
    # The module and its test were actually written to the sandbox.
    files = {f for f in __import__("os").listdir(target)}
    assert "adder.py" in files and "test_adder.py" in files


def test_fixer_repairs_failing_tests(tmp_path, monkeypatch):
    target = str(tmp_path / "broken")
    pid = repo.create_project("Broken", target)
    repo.add_requirements_version(pid, "Build add that returns a+b")
    phase_id = repo.add_phase(pid, 1, "Core", "adder", ["add fn"], ["add works"])
    repo.add_task(phase_id, "Implement add", "code", ["adder.py"])

    # Coder ships a bug (subtraction); Test Writer writes a correct test.
    monkeypatch.setattr(agents, "generate_code", lambda *a, **k: {
        "files": [{"path": "adder.py", "content": "def add(a, b):\n    return a - b\n"}],
        "notes": ""})
    monkeypatch.setattr(agents, "generate_tests", lambda *a, **k: {
        "files": [{"path": "test_adder.py",
                   "content": "from adder import add\n\n"
                              "def test_add():\n    assert add(2, 3) == 5\n"}],
        "notes": ""})
    # Fixer corrects the bug.
    monkeypatch.setattr(agents, "fix_failure", lambda *a, **k: {
        "files": [{"path": "adder.py", "content": "def add(a, b):\n    return a + b\n"}],
        "notes": "fixed"})
    monkeypatch.setattr(agents, "review_phase",
                        lambda *a, **k: {"approved": True, "notes": "", "gaps": []})
    repo.update_settings(pid, max_fix_attempts=3, test_command="python -m pytest -q")

    runner.start_build(pid)
    assert _wait_until(lambda: repo.get_project(pid)["status"] == "done"), \
        repo.get_project(pid)["status"]
    runs = [r for r in __import__("app.repo", fromlist=["db"]).db.query(
        "SELECT * FROM test_runs WHERE phase_id = ? ORDER BY attempt_no", (phase_id,))]
    # At least one failing attempt followed by a pass.
    assert any(not r["passed"] for r in runs) and runs[-1]["passed"]
