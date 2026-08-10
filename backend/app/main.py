from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import routes_jobs, routes_laps, routes_render, routes_videos
from app.core.cleanup import RetentionSweeper
from app.core.state import job_manager, settings, video_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    sweeper = RetentionSweeper(video_store, settings, job_manager, interval_seconds=settings.sweep_interval_seconds)
    sweeper.start()
    yield
    sweeper.stop()


app = FastAPI(title="Py_RouteTracker Telemetry Overlay", lifespan=lifespan)

app.include_router(routes_videos.router)
app.include_router(routes_laps.router)
app.include_router(routes_render.router)
app.include_router(routes_jobs.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "retention_minutes": settings.retention_minutes}


# Starlette matches routes in registration order, and a Mount("/") matches
# every path as a prefix -- it must be registered last, or it shadows every
# route above (including /api/health), swallowing them into 404s from
# StaticFiles instead of reaching our handlers.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "static"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
