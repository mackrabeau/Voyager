"""Static hosted replay export contract."""

from __future__ import annotations

import json
from pathlib import Path

from voyager.replay.static_export import export_static_viewer

ROOT = Path(__file__).resolve().parents[1]


def test_static_export_contains_all_curated_replay_data(tmp_path: Path) -> None:
    output = tmp_path / "static-api"
    export_static_viewer(ROOT / "benchmarks/replays/stage7_curated_v1", output)
    catalog = json.loads((output / "catalog.json").read_text())
    assert catalog["total"] == 5
    assert catalog["items"][0]["replay_id"] == "island_feed_forward_progression_v1"
    for item in catalog["items"]:
        replay = output / "replays" / item["replay_id"]
        manifest = json.loads((replay / "manifest.json").read_text())
        timeline = json.loads((replay / "timeline.json").read_text())
        assert len(timeline["records"]) == manifest["world_steps"]
        assert (replay / "initial.json").is_file()
        assert (replay / "events.json").is_file()
        assert (replay / "metrics.json").is_file()
        assert (replay / "camera.json").is_file()


def test_static_export_creates_real_pages_for_public_spa_routes(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    index = site / "index.html"
    index.write_text("<main>Voyager</main>", encoding="utf-8")

    export_static_viewer(
        ROOT / "benchmarks/replays/stage7_curated_v1",
        site / "static-api",
        spa_index=index,
    )

    assert (site / "compare/index.html").read_bytes() == index.read_bytes()
    for replay_id in (
        "island_feed_forward_progression_v1",
        "island_learned_failure_v1",
        "island_legal_random_matched_v1",
        "island_oracle_rescue_v1",
        "island_recurrent_progression_v1",
    ):
        assert (site / "replays" / replay_id / "index.html").read_bytes() == index.read_bytes()
        assert (site / "present" / replay_id / "index.html").read_bytes() == index.read_bytes()
