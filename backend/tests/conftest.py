import os
import tempfile
from pathlib import Path

import pytest

# app.core.config builds a module-level `settings` singleton the moment it's
# first imported, pointed at ROUTETRACKER_DATA_DIR. That import happens
# during test collection (routes_*.py import app.core.state at module
# scope), before any test function -- and therefore before any
# monkeypatch.setenv -- runs. Setting the env var here, at conftest module
# scope, is what actually lands before that first import.
os.environ.setdefault("ROUTETRACKER_DATA_DIR", tempfile.mkdtemp(prefix="routetracker_test_data_"))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def gopro_telemetry_txt() -> Path:
    path = FIXTURES_DIR / "GX010411_telemetry.txt"
    assert path.exists(), "expected sample fixture committed in tests/fixtures/"
    return path


@pytest.fixture
def sample_gpx() -> Path:
    path = FIXTURES_DIR / "Krakow - Zarki.gpx"
    assert path.exists(), "expected sample fixture committed in tests/fixtures/"
    return path
