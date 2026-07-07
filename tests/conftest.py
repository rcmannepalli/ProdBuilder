"""Test fixtures — isolate DuckDB and secrets to a temp dir per session."""
import os
import tempfile
from pathlib import Path

import pytest

# Point the app at a throwaway data dir BEFORE importing app modules.
_TMP = tempfile.mkdtemp(prefix="prodbuilder-test-")
os.environ["PRODBUILDER_DATA"] = _TMP


@pytest.fixture(scope="session")
def data_dir():
    return Path(_TMP)
