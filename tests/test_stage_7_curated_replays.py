"""Frozen Stage 7 replay selection, provenance, and reconstruction tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voyager.replay.catalog import ReplayCatalog
from voyager.replay.island_curated import build_curated_island_selection, sha256_file
from voyager.replay.loader import ReplayLoader
from voyager.server.app import _comparison_incompatibilities, create_app

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "benchmarks/replays/stage7_curated_v1"


def test_curated_collection_has_five_declared_roles() -> None:
    selection = _read(CURATED / "selection.json")
    specs = selection["replays"]
    assert selection["use_for_quantitative_claims"] is False
    assert [spec["role"] for spec in specs] == [
        "oracle_rescue",
        "matched_legal_random",
        "feed_forward_progression",
        "recurrent_progression",
        "representative_learned_failure",
    ]
    assert specs[1]["evaluation_seed"] == specs[2]["evaluation_seed"] == 20_007


def test_curated_collection_manifest_hashes_every_declared_artifact() -> None:
    collection = _read(CURATED / "collection.json")
    for artifact in collection["artifacts"]:
        path = CURATED / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert sha256_file(path) == artifact["sha256"]


def test_curated_replays_match_frozen_outcomes_and_deeply_reconstruct() -> None:
    selection = _read(CURATED / "selection.json")
    for spec in selection["replays"]:
        loader = ReplayLoader(CURATED / spec["replay_id"])
        summary = loader.manifest.terminal_summary
        expected = spec["expected_episode"]
        assert loader.validate(deep=True)["checked_ticks"] == summary["world_steps"] + 1
        assert sorted(summary["achievements"]) == sorted(expected["achievements"])
        assert summary["achievement_steps"] == expected["achievement_steps"]
        assert summary["rescue_success"] == expected["rescue_success"]
        assert summary["invalid_actions"] == expected["invalid_actions"]
        assert summary["survivors"] == expected["final_active_agents"]
        assert loader.manifest.source.benchmark_episode is not None
        assert loader.manifest.source.benchmark_episode["use_for_quantitative_claims"] is False


def test_learned_replays_pin_checkpoint_hashes_and_honest_outcomes() -> None:
    selection = _read(CURATED / "selection.json")
    by_role = {spec["role"]: spec for spec in selection["replays"]}
    for role in ("feed_forward_progression", "recurrent_progression", "representative_learned_failure"):
        spec = by_role[role]
        manifest = ReplayLoader(CURATED / spec["replay_id"]).manifest
        assert manifest.source.checkpoint_sha256 == spec["checkpoint_sha256"]
        assert manifest.source.training_seed == spec["training_seed"]
        assert manifest.terminal_summary["rescue_success"] is False
    assert ReplayLoader(CURATED / "island_oracle_rescue_v1").manifest.terminal_summary[
        "rescue_success"
    ] is True


def test_curated_catalog_prioritizes_learned_showcase() -> None:
    catalog = ReplayCatalog([("stage7-curated", CURATED)])
    entries = catalog.scan()
    assert len(entries) == 5
    assert entries[0].manifest.replay_id == "island_feed_forward_progression_v1"
    assert entries[0].source == "stage7-curated"


def test_selection_regenerates_from_local_frozen_results_when_available() -> None:
    experiment = ROOT / "results/stage7/procedural_baselines_v1"
    if not experiment.exists():
        return
    assert build_curated_island_selection(experiment) == _read(CURATED / "selection.json")


def test_default_api_exposes_stage7_and_allows_matched_comparison() -> None:
    app = create_app(frontend_dir=ROOT / "web/dist")
    catalog = app.state.catalog
    assert {entry.manifest.replay_id for entry in catalog.query(tag="stage7")}.issuperset(
        {
            "island_feed_forward_progression_v1",
            "island_legal_random_matched_v1",
            "island_oracle_rescue_v1",
        }
    )
    assert not _comparison_incompatibilities(
        catalog.loader("island_legal_random_matched_v1"),
        catalog.loader("island_feed_forward_progression_v1"),
    )


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
