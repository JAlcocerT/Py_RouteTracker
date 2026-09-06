# --- stage 1: build the React frontend ---
FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- stage 2: backend runtime ---
# Every compute-heavy feature (trim, join, telemetry extraction, lap
# detection, HUD rendering, compositing) now runs client-side in the
# browser -- see frontend/src/lib/ -- so this stage has nothing left to do
# but serve the built frontend. No ffmpeg/exiftool, no data volume: nothing
# is ever stored server-side anymore either.
FROM python:3.12-slim AS backend

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-install-project

COPY backend/app ./app
COPY --from=frontend-build /frontend/dist ./static

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7000"]
