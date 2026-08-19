"""Static-file export for the hosted curated replay viewer."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from voyager.replay.catalog import ReplayCatalog, catalog_entry_payload
from voyager.replay.serialization import canonical_json_bytes


def export_static_viewer(replay_root: Path, output: Path) -> None:
    """Export catalog and replay endpoints into a static-api directory."""

    destination = output.resolve()
    if destination.name != "static-api":
        raise ValueError("Static viewer output directory must be named 'static-api'.")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    catalog = ReplayCatalog([("stage7-curated", replay_root)])
    entries = catalog.scan()
    _write(
        destination / "catalog.json",
        {
            "items": [catalog_entry_payload(entry) for entry in entries],
            "next_cursor": None,
            "total": len(entries),
        },
    )
    for entry in entries:
        loader = catalog.loader(entry.manifest.replay_id)
        replay_output = destination / "replays" / entry.manifest.replay_id
        manifest = {
            **entry.manifest.model_dump(mode="json"),
            "catalog_source": entry.source,
            "available_series": [
                str(registry_entry["id"])
                for registry_entry in entry.manifest.registries.get("metrics", [])
                if "id" in registry_entry
            ],
        }
        _write(replay_output / "manifest.json", manifest)
        _write(replay_output / "initial.json", loader.initial())
        records = loader.timeline(1, entry.manifest.world_steps)
        _write(replay_output / "timeline.json", {"records": records})
        _write(
            replay_output / "events.json",
            {"events": [event for record in records for event in record.get("events", [])]},
        )
        _write(replay_output / "metrics.json", loader.metrics())
        _write(replay_output / "camera.json", loader.camera())


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))
