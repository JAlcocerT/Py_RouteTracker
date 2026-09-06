"""Static-file server for the built frontend.

Every compute-heavy feature (video trim/join, telemetry extraction, lap
detection, HUD rendering, compositing) runs entirely in the visiting
browser now -- see frontend/src/lib/. This backend has nothing left to
compute; it exists only to serve the built React app, so this app/ package
can keep being `docker run` the same way it always has (see backend/readme.md).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="PitLane")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Starlette matches routes in registration order, and a Mount("/") matches
# every path as a prefix -- it must be registered last, or it shadows every
# route above (including /api/health), swallowing them into 404s from
# StaticFiles instead of ever reaching our handler. See test_main.py.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "static"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
