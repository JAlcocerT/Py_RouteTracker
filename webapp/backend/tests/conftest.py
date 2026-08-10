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

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def gopro_telemetry_txt(repo_root: Path) -> Path:
    path = repo_root / "Z_GoPro" / "GX010411_telemetry.txt"
    assert path.exists(), "expected sample fixture committed in Z_GoPro/"
    return path


@pytest.fixture
def sample_gpx(repo_root: Path) -> Path:
    path = repo_root / "Data_My_Routes" / "Krakow - Zarki.gpx"
    assert path.exists(), "expected sample fixture committed in Data_My_Routes/"
    return path
