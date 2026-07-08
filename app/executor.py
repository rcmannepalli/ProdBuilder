"""Sandboxed file writes and test execution scoped to a project's target folder."""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
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


_PRD_NAMES = ["prd.md", "PRD.md", "prd.txt", "PRD.txt", "requirements.md",
              "docs/prd.md", "docs/PRD.md"]


def find_prd(target_folder: str) -> tuple[str, str]:
    """Look for a product requirements doc in the target folder.
    Returns (filename, content) or ('', '')."""
    root = _safe_target(target_folder)
    for name in _PRD_NAMES:
        candidate = root / name
        if candidate.is_file():
            try:
                return name, candidate.read_text(encoding="utf-8")[:20000]
            except (UnicodeDecodeError, OSError):
                continue
    return "", ""


def codebase_summary(target_folder: str, max_files: int = 40) -> str:
    """A compact digest of an existing codebase for enrichment context:
    the file tree plus short heads of the most relevant source files."""
    root = _safe_target(target_folder)
    paths = [p for p in list_tree(target_folder) if not p.endswith("/")]
    if not paths:
        return ""
    lines = ["FILE TREE:", *[f"  {p}" for p in paths[:120]], "", "KEY FILES:"]
    files = read_files(target_folder, limit=max_files)
    for path, content in list(files.items())[:max_files]:
        head = content[:800]
        lines.append(f"--- {path} ---\n{head}\n")
    return "\n".join(lines)[:12000]


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


# ---------------------------------------------------------------------------
# Per-project virtual environment + dependency management
# ---------------------------------------------------------------------------

def _has_uv() -> bool:
    return shutil.which("uv") is not None


def _venv_dir(root: Path) -> Path:
    return root / ".venv"


def venv_python(root: Path) -> Path:
    """Path to the venv's python interpreter (POSIX or Windows layout)."""
    vdir = _venv_dir(root)
    win = vdir / "Scripts" / "python.exe"
    return win if win.exists() else vdir / "bin" / "python"


def _bin_dir(root: Path) -> Path:
    vdir = _venv_dir(root)
    win = vdir / "Scripts"
    return win if win.exists() else vdir / "bin"


def _run(cmd: list[str], cwd: Path, timeout: int, env: dict | None = None
         ) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                              timeout=timeout, env=env)
        return proc.returncode == 0, (proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except FileNotFoundError as e:
        return False, str(e)


def ensure_venv(target_folder: str, timeout: int = 240) -> tuple[bool, str, str]:
    """Ensure a virtual environment exists in the target folder.

    Uses `uv` when available (fast), else stdlib venv. Returns
    (ok, tool, message). Idempotent — reuses an existing .venv.
    """
    root = _safe_target(target_folder)
    if venv_python(root).exists():
        return True, "uv" if _has_uv() else "venv", "Reusing existing .venv"
    if _has_uv():
        ok, out = _run(["uv", "venv", ".venv"], root, timeout)
        return ok, "uv", ("Created .venv with uv" if ok else out[-400:])
    ok, out = _run([sys.executable, "-m", "venv", ".venv"], root, timeout)
    return ok, "venv", ("Created .venv" if ok else out[-400:])


def venv_env(target_folder: str) -> dict:
    """Environment with the project's venv activated (if it exists) and the
    project root on PYTHONPATH so tests can import top-level modules."""
    root = _safe_target(target_folder)
    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) + (os.pathsep + existing_pp if existing_pp else "")
    vpy = venv_python(root)
    if vpy.exists():
        env["VIRTUAL_ENV"] = str(_venv_dir(root))
        env["PATH"] = str(_bin_dir(root)) + os.pathsep + env.get("PATH", "")
        env.pop("PYTHONHOME", None)
    return env


def pip_install(target_folder: str, packages: list[str],
                timeout: int = 300) -> tuple[bool, str]:
    """Install packages into the project's venv (uv pip if available)."""
    root = _safe_target(target_folder)
    pkgs = [p for p in packages if _SAFE_PKG.match(p)]
    if not pkgs:
        return True, "nothing to install"
    env = venv_env(target_folder)
    vpy = venv_python(root)
    if _has_uv():
        cmd = ["uv", "pip", "install", "--python", str(vpy), *pkgs]
    else:
        cmd = [str(vpy), "-m", "pip", "install", "--disable-pip-version-check",
               "-q", *pkgs]
    ok, out = _run(cmd, root, timeout, env=env)
    return ok, out[-600:]


def install_requirements(target_folder: str, timeout: int = 600
                         ) -> tuple[bool, str]:
    """Install requirements.txt (if present) plus pytest, into the venv."""
    root = _safe_target(target_folder)
    env = venv_env(target_folder)
    vpy = venv_python(root)
    msgs = []
    req = root / "requirements.txt"
    if req.exists():
        if _has_uv():
            cmd = ["uv", "pip", "install", "--python", str(vpy), "-r",
                   "requirements.txt"]
        else:
            cmd = [str(vpy), "-m", "pip", "install", "--disable-pip-version-check",
                   "-q", "-r", "requirements.txt"]
        ok, out = _run(cmd, root, timeout, env=env)
        msgs.append("requirements.txt installed" if ok else out[-400:])
    # Always ensure pytest is available for the test runner.
    ok2, _ = _run(
        (["uv", "pip", "install", "--python", str(vpy), "pytest"] if _has_uv()
         else [str(vpy), "-m", "pip", "install", "-q", "pytest"]),
        root, 180, env=env)
    msgs.append("pytest ready" if ok2 else "pytest install failed")
    return ok2, "; ".join(msgs)


# import-name -> pip package for common mismatches
_IMPORT_TO_PACKAGE = {
    "yaml": "pyyaml", "cv2": "opencv-python", "PIL": "pillow",
    "bs4": "beautifulsoup4", "sklearn": "scikit-learn", "dotenv": "python-dotenv",
    "jose": "python-jose", "jwt": "pyjwt", "psycopg2": "psycopg2-binary",
    "dateutil": "python-dateutil", "OpenSSL": "pyopenssl", "attr": "attrs",
    "google": "google-api-python-client", "serial": "pyserial",
}
_SAFE_PKG = re.compile(r"^[A-Za-z0-9_.\-]+(\[[A-Za-z0-9_,\-]+\])?$")
_STDLIB_HINT = {"os", "sys", "json", "re", "math", "typing", "pathlib", "asyncio",
                "dataclasses", "datetime", "collections", "itertools", "functools",
                "unittest", "sqlite3", "subprocess", "logging", "enum"}


def extract_missing_modules(text: str) -> list[str]:
    """Parse ModuleNotFoundError names from test output and map to pip packages."""
    names = set(re.findall(r"No module named ['\"]([A-Za-z0-9_]+)", text or ""))
    packages = []
    for name in names:
        if name in _STDLIB_HINT:
            continue
        packages.append(_IMPORT_TO_PACKAGE.get(name, name))
    return sorted(set(packages))


def run_tests(target_folder: str, test_command: str,
              timeout: int = 300) -> RunResult:
    """Execute the configured test command inside the target folder, using the
    project's virtual environment when one exists."""
    root = _safe_target(target_folder)
    if not command_allowed(test_command):
        return RunResult(False, "Refused: command not allowed", "",
                         f"Command '{test_command}' is not in the allow-list.", 0)
    env = venv_env(target_folder)
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
