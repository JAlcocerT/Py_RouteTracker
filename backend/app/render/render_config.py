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


def render_config_from_payload(payload: dict) -> RenderConfig:
    widgets = payload["widgets"]
    style = payload["style"]
    return RenderConfig(
        enable_speedo=widgets["speedo"],
        enable_gg=widgets["gg"],
        enable_minimap=widgets["minimap"],
        max_expected_speed_kmh=style["max_expected_speed_kmh"],
        limit_g=style["limit_g"],
        theme=style["theme"],
    )
