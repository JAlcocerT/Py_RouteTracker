from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import routes_jobs, routes_laps, routes_render, routes_videos

app = FastAPI(title="Py_RouteTracker Telemetry Overlay")

app.include_router(routes_videos.router)
app.include_router(routes_laps.router)
app.include_router(routes_render.router)
app.include_router(routes_jobs.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Starlette matches routes in registration order, and a Mount("/") matches
# every path as a prefix -- it must be registered last, or it shadows every
# route above (including /api/health), swallowing them into 404s from
# StaticFiles instead of reaching our handlers.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "static"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
