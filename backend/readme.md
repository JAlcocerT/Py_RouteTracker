# Backend

A minimal FastAPI app that serves the built frontend as static files. See the
root `README.md` for the product-level picture, and `frontend/src/lib/` for
where the actual work happens now.

Every feature that used to run here -- video trim/join, GoPro/GPX telemetry
extraction, lap detection, HUD rendering, ffmpeg compositing, the
distributed-render job queue -- now runs entirely client-side, in the
visiting browser, via WebCodecs/Canvas2D/mediabunny/mp4box.js. Nothing is
ever uploaded to this server; there's no video processing here to speed up,
secure, or scale. `app/main.py` is the whole backend.

## Local development

```sh
uv sync
uv run uvicorn app.main:app --reload --port 7000
```

There's no built frontend at `./static` in this mode, so `/` 404s (see
`main.py`'s comment on why the catch-all mount only gets registered when
`static/` actually exists) -- run the frontend itself via `cd frontend && npm
run dev` instead, which proxies nothing to this backend anymore since there's
no API left for it to call.

## Tests

```sh
uv run pytest
```

Two tests, both about not regressing the one thing this app still needs to
get right: that `/api/health` (and any future `/api/*` route) is registered
*before* the catch-all static-file mount, since Starlette matches routes in
registration order and a `Mount("/")` would otherwise shadow everything
after it into 404s. See `tests/test_main.py`.
