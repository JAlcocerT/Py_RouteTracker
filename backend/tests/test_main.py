"""Regression test for a real bug the Docker acceptance pass caught:
app.mount("/", StaticFiles(...)) was registered before the plain
@app.get("/api/health") route. Starlette matches routes in registration
order and a Mount("/") matches every path as a prefix, so the mount
shadowed /api/health (and would have shadowed any other route registered
after it) into a 404 from StaticFiles instead of ever reaching our handler.
This never failed locally because backend/static doesn't exist in
dev, so the mount was skipped entirely -- it only broke once the Docker
image (which does have a built frontend) was actually run.
"""

from __future__ import annotations

import pytest
from starlette.routing import Mount

from app.main import app
from fastapi.testclient import TestClient


def test_api_routes_are_registered_before_any_catch_all_mount():
    routes = app.router.routes
    mount_indices = [i for i, r in enumerate(routes) if isinstance(r, Mount)]
    if not mount_indices:
        pytest.skip("no static frontend built in this environment; nothing to shadow")

    first_mount = min(mount_indices)
    api_route_indices = [i for i, r in enumerate(routes) if getattr(r, "path", "").startswith("/api")]
    assert api_route_indices, "expected at least one /api route to be registered"
    assert max(api_route_indices) < first_mount, "an /api route was registered after the catch-all mount and would be shadowed"


def test_health_endpoint_is_reachable():
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
