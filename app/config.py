"""Application configuration and paths."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PRODBUILDER_DATA", BASE_DIR / "data"))
DB_PATH = DATA_DIR / "prodbuilder.duckdb"
SECRET_KEY_PATH = DATA_DIR / "secret.key"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default per-role model map. All roles default to one model; override in UI.
DEFAULT_MODEL = os.environ.get("PRODBUILDER_DEFAULT_MODEL", "qwen2.5-coder:32b")
DEFAULT_OLLAMA_URL = os.environ.get(
    "PRODBUILDER_OLLAMA_URL", "https://ollama.com"
)

DEFAULT_MODEL_MAP = {
    "planner": DEFAULT_MODEL,
    "coder": DEFAULT_MODEL,
    "tester": DEFAULT_MODEL,
    "fixer": DEFAULT_MODEL,
    "reviewer": DEFAULT_MODEL,
    "monitor": DEFAULT_MODEL,
}

DEFAULT_MAX_FIX_ATTEMPTS = 3
DEFAULT_POLL_INTERVAL_SEC = 60
DEFAULT_TEST_COMMAND = "pytest -q"

APP_NAME = "ProdBuilder"
APP_TAGLINE = "Autonomous Product Engineering"
