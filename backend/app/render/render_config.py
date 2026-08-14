"""Pure mapping from the JSON render-request payload (as stored in a render
job and handed to a worker) to a RenderConfig. Deliberately dependency-free
beyond hud_layers -- app.worker_main (which runs standalone, on any machine,
with no coordinator state) imports this directly rather than going through
app.render.coordinator, which pulls in Settings/JobManager/VideoStore and
would otherwise make a plain worker process create a spurious local
`./data/` directory just from the import.
"""

from __future__ import annotations

from app.render.hud_layers import RenderConfig

_DEFAULTS = RenderConfig()


def render_config_from_payload(payload: dict) -> RenderConfig:
    # A remote worker (app.worker_main) is a separately versioned Docker
    # image -- it can lag or lead the coordinator that built this payload
    # (e.g. a widget field added/removed on one side but not yet on the
    # other, or a stale cached `:latest` image on the worker's machine).
    # `.get(..., default)` degrades to that field's normal default instead
    # of a bare KeyError that names a field the reader may never have heard
    # of; unrecognized extra keys are simply ignored.
    widgets = payload.get("widgets", {})
    style = payload.get("style", {})
    return RenderConfig(
        enable_speedo=widgets.get("speedo", _DEFAULTS.enable_speedo),
        enable_gg=widgets.get("gg", _DEFAULTS.enable_gg),
        enable_minimap=widgets.get("minimap", _DEFAULTS.enable_minimap),
        max_expected_speed_kmh=style.get("max_expected_speed_kmh", _DEFAULTS.max_expected_speed_kmh),
        limit_g=style.get("limit_g", _DEFAULTS.limit_g),
        theme=style.get("theme", _DEFAULTS.theme),
    )
