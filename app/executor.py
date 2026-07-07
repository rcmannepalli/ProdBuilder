"""Sandboxed file writes and test execution scoped to a project's target folder."""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# Commands agents are permitted to run. Everything else is refused.
ALLOWED_COMMANDS = {"pytest", "python", "python3", "npm", "node", "go", "cargo",
                    "make", "ruff", "mypy", "unittest"}

_BLOCKED_TOKENS = {"rm", "sudo", "curl", "wget", "git", "ssh", "scp", ">", ">>",
                   "&&", "||", ";", "|", "$(", "`"}


@dataclass
class RunResult:
    passed: bool
    summary: str
    stdout: str
    stderr: str
    duration_ms: int


def _safe_target(target_folder: str) -> Path:
    root = Path(target_folder).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_within(root: Path, rel_path: str) -> Path:
    candidate = (root / rel_path).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"Path traversal blocked: {rel_path}")
    return candidate


def write_files(target_folder: str, files: list[dict]) -> list[str]:
    """Write [{path, content}] into the target folder; returns written paths."""
    root = _safe_target(target_folder)
    written: list[str] = []
    for f in files:
        rel = str(f.get("path", "")).lstrip("/")
        if not rel:
            continue
        dest = _resolve_within(root, rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f.get("content", ""), encoding="utf-8")
        written.append(rel)
    return written


def read_files(target_folder: str, limit: int = 60) -> dict[str, str]:
    """Read text files from the target folder into a {rel_path: content} map."""
    root = _safe_target(target_folder)
    out: dict[str, str] = {}
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv",
                 ".pytest_cache", "dist", "build"}
    for path in sorted(root.rglob("*")):
        if len(out) >= limit:
            break
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.stat().st_size > 200_000:
            continue
        try:
            out[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
    return out


def read_single_file(target_folder: str, rel_path: str,
                     max_bytes: int = 400_000) -> tuple[str, str]:
    """Read one file's content. Returns (content, status) where status is
    'ok', 'missing', 'binary', or 'too_large'."""
    root = _safe_target(target_folder)
    try:
        dest = _resolve_within(root, rel_path.lstrip("/"))
    except ValueError:
        return "", "missing"
    if not dest.is_file():
        return "", "missing"
    if dest.stat().st_size > max_bytes:
        return "", "too_large"
    try:
        return dest.read_text(encoding="utf-8"), "ok"
    except (UnicodeDecodeError, OSError):
        return "", "binary"


def list_tree(target_folder: str, limit: int = 400) -> list[str]:
    root = _safe_target(target_folder)
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}
    out = []
    for path in sorted(root.rglob("*")):
        if len(out) >= limit:
            break
        if any(part in skip_dirs for part in path.parts):
            continue
        rel = str(path.relative_to(root))
        out.append(rel + ("/" if path.is_dir() else ""))
    return out


def build_tree(target_folder: str, limit: int = 800) -> list[dict]:
    """Return a nested tree of the target folder for a VS Code-style explorer.

    Each node: {name, path, type: 'dir'|'file', children?: [...], ext}.
    Directories are sorted first, then files, both alphabetically.
    """
    root = _safe_target(target_folder)
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv",
                 ".pytest_cache", "dist", "build", ".mypy_cache"}
    tree: dict = {"name": "", "path": "", "type": "dir", "children": {}}
    count = 0
    for path in sorted(root.rglob("*")):
        if count >= limit:
            break
        parts = path.relative_to(root).parts
        if any(p in skip_dirs for p in parts):
            continue
        count += 1
        node = tree
        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            children = node["children"]
            if part not in children:
                is_dir = (not is_last) or path.is_dir()
                rel = "/".join(parts[:i + 1])
                children[part] = {
                    "name": part, "path": rel,
                    "type": "dir" if is_dir else "file",
                    "ext": path.suffix.lstrip(".") if not is_dir else "",
                    "children": {},
                }
            node = children[part]

    def _finalize(node: dict) -> list[dict]:
        items = list(node["children"].values())
        for it in items:
            if it["type"] == "dir":
                it["children"] = _finalize(it)
            else:
                it.pop("children", None)
        items.sort(key=lambda x: (x["type"] != "dir", x["name"].lower()))
        return items

    return _finalize(tree)


def command_allowed(cmd: str) -> bool:
    tokens = shlex.split(cmd)
    if not tokens:
        return False
    if any(tok in _BLOCKED_TOKENS for tok in tokens):
        return False
    return tokens[0] in ALLOWED_COMMANDS


def run_tests(target_folder: str, test_command: str,
              timeout: int = 300) -> RunResult:
    """Execute the configured test command inside the target folder."""
    root = _safe_target(target_folder)
    if not command_allowed(test_command):
        return RunResult(False, "Refused: command not allowed", "",
                         f"Command '{test_command}' is not in the allow-list.", 0)
    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    start = time.time()
    try:
        proc = subprocess.run(
            shlex.split(test_command), cwd=str(root), capture_output=True,
            text=True, timeout=timeout, env=env,
        )
        dur = int((time.time() - start) * 1000)
        passed = proc.returncode == 0
        summary = _summarize(proc.stdout, proc.stderr, passed)
        return RunResult(passed, summary, proc.stdout, proc.stderr, dur)
    except subprocess.TimeoutExpired as e:
        dur = int((time.time() - start) * 1000)
        return RunResult(False, f"Timed out after {timeout}s", e.stdout or "",
                         "Test run timed out.", dur)
    except FileNotFoundError:
        dur = int((time.time() - start) * 1000)
        return RunResult(False, "Test runner not found", "",
                         "The test command binary was not found on PATH.", dur)


def _summarize(stdout: str, stderr: str, passed: bool) -> str:
    tail = (stdout or "").strip().splitlines()[-1:] or [""]
    status = "PASSED" if passed else "FAILED"
    return f"{status}: {tail[0][:160]}" if tail[0] else status
