# --- stage 1: build the React frontend ---
FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- stage 2: backend runtime ---
FROM python:3.12-slim AS backend

# ffmpeg provides both ffmpeg and ffprobe; libimage-exiftool-perl provides exiftool.
# None of the original overlay/*.py scripts ever verified these were installed --
# app.core.binaries checks for them at call time and fails with a clear error
# instead of a buried subprocess traceback.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-install-project

COPY backend/app ./app
COPY --from=frontend-build /frontend/dist ./static

ENV PATH="/app/.venv/bin:$PATH"
ENV ROUTETRACKER_DATA_DIR=/data

VOLUME ["/data"]
EXPOSE 7000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7000"]
