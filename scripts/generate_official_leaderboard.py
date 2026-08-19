"""Generate the official v0.1.0 leaderboard submissions from frozen raw results."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from scripts.build_release_assets import VERSION, build_release_assets
from voyager.benchmark.island import load_island_seed_manifests
from voyager.benchmark.leaderboard import SCHEMA_VERSION, score_episode_records
from voyager.sim.island_achievements import ISLAND_ACHIEVEMENT_VERSION

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "results/stage7/procedural_baselines_v1"
FINAL = EXPERIMENT / "final"
SUBMISSIONS = ROOT / "leaderboard/submissions"
RELEASE_BASE = f"https://github.com/fawzmehfil/Voyager/releases/download/{VERSION}"
VIEWER_BASE = "https://fawzmehfil.github.io/Voyager/replays"
SOURCE_REPOSITORY = "https://github.com/fawzmehfil/Voyager"


def main() -> None:
    summary = _read(ROOT / "benchmarks/reference/voyager_island_v1_baselines_v1/summary.json")
    git_revision = str(summary["git_revision"])
    manifest = load_island_seed_manifests()["test"]
    with tempfile.TemporaryDirectory(prefix="voyager-release-assets-") as temporary:
        release_artifacts = build_release_assets(Path(temporary))
        release_hashes = {path.name: _sha256(path) for path in release_artifacts}

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    reference_specs = (
        (
            "legal-random",
            "Legal Random",
            "Uniform sampling from the public legal-action mask.",
            "legal_random.json",
            "legal_random",
            "island_legal_random_matched_v1",
        ),
        (
            "scripted-oracle",
            "Scripted Public-Observation Oracle",
            "A deterministic solvability reference using only the public structured observation.",
            "scripted_oracle.json",
            "scripted_oracle",
            "island_oracle_rescue_v1",
        ),
    )
    for submission_id, name, description, filename, policy_key, replay_id in reference_specs:
        raw = _read(FINAL / filename)
        records = [_record(None, episode) for episode in raw["episodes"]]
        statistics = summary["policies"][policy_key]
        payload = _submission(
            submission_id=submission_id,
            name=name,
            category="scripted_reference",
            description=description,
            git_revision=git_revision,
            test_manifest_sha256=manifest.sha256,
            training_seeds=[],
            training_budget=0,
            checkpoints=[],
            statistics=statistics,
            records=records,
            replay_id=replay_id,
            release_hashes=release_hashes,
            disclosures={
                "external_data": (False, "No external data."),
                "pretrained_models": (False, "No pretrained models."),
                "privileged_information": (
                    False,
                    "Legal random uses the public action mask; the oracle uses public observations only.",
                ),
            },
        )
        _write(SUBMISSIONS / f"{submission_id}.json", payload)

    learned_specs = (
        (
            "feed-forward-ppo",
            "Feed-Forward PPO",
            "feed_forward_ppo",
            "island_feed_forward_progression_v1",
        ),
        (
            "recurrent-ppo",
            "Recurrent PPO",
            "recurrent_ppo",
            "island_recurrent_progression_v1",
        ),
    )
    for submission_id, name, algorithm, replay_id in learned_specs:
        records: list[dict[str, Any]] = []
        checkpoints = []
        for training_seed in (0, 1, 2):
            raw = _read(FINAL / "learned" / algorithm / f"seed_{training_seed}.json")
            records.extend(
                _record(training_seed, episode)
                for episode in raw["evaluation"]["stochastic"]["episodes"]
            )
            archive_name = (
                f"voyager-island-{VERSION}-checkpoint-"
                f"{algorithm.replace('_', '-')}-seed-{training_seed}.tar.gz"
            )
            weights = (
                EXPERIMENT
                / "runs"
                / algorithm
                / f"seed_{training_seed}"
                / "checkpoints/best/model.weights.h5"
            )
            checkpoints.append(
                {
                    "training_seed": training_seed,
                    "url": f"{RELEASE_BASE}/{archive_name}",
                    "archive_sha256": release_hashes[archive_name],
                    "weights_sha256": _sha256(weights),
                }
            )
        payload = _submission(
            submission_id=submission_id,
            name=name,
            category="learned_official_budget",
            description=(
                "Shared-policy PPO baseline trained on the frozen island training split with "
                "seeded-stochastic held-out inference."
            ),
            git_revision=git_revision,
            test_manifest_sha256=manifest.sha256,
            training_seeds=[0, 1, 2],
            training_budget=1_000_000,
            checkpoints=checkpoints,
            statistics=summary["families"][algorithm]["modes"]["stochastic"],
            records=records,
            replay_id=replay_id,
            release_hashes=release_hashes,
            disclosures={
                "external_data": (False, "No external data; only the frozen training split."),
                "pretrained_models": (False, "Networks were initialized from scratch."),
                "privileged_information": (
                    False,
                    "Training and evaluation use the declared public structured observations.",
                ),
            },
        )
        _write(SUBMISSIONS / f"{submission_id}.json", payload)


def _submission(
    *,
    submission_id: str,
    name: str,
    category: str,
    description: str,
    git_revision: str,
    test_manifest_sha256: str,
    training_seeds: list[int],
    training_budget: int,
    checkpoints: list[dict[str, Any]],
    statistics: dict[str, Any],
    records: list[dict[str, Any]],
    replay_id: str,
    release_hashes: dict[str, str],
    disclosures: dict[str, tuple[bool, str]],
) -> dict[str, Any]:
    computed = score_episode_records(records)
    score = statistics["achievement_geometric_mean"]
    achievements = statistics["achievements"]
    raw_archive = f"voyager-island-{VERSION}-raw-episode-evaluations.tar.gz"
    replay_manifest = _read(
        ROOT / "benchmarks/replays/stage7_curated_v1" / replay_id / "manifest.json"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "submission_id": submission_id,
        "status": "official",
        "submitted_at": "2026-08-18",
        "benchmark": {
            "id": "VoyagerIsland-v1",
            "environment_version": "voyager_island_v1",
            "achievement_version": ISLAND_ACHIEVEMENT_VERSION,
            "generator_version": "voyager_island_generator_v1",
            "split": "test",
            "test_manifest_sha256": test_manifest_sha256,
        },
        "method": {
            "name": name,
            "category": category,
            "description": description,
            "policy_source": {
                "repository_url": SOURCE_REPOSITORY,
                "commit": git_revision,
            },
            "training_seeds": training_seeds,
            "training_budget_agent_steps": training_budget,
            **{
                key: {"used": value[0], "description": value[1]}
                for key, value in disclosures.items()
            },
        },
        "artifacts": {
            "checkpoints": checkpoints,
            "episode_records": {
                "url": f"{RELEASE_BASE}/{raw_archive}",
                "sha256": release_hashes[raw_archive],
            },
            "representative_replay": {
                "url": f"{VIEWER_BASE}/{replay_id}",
                "manifest_sha256": replay_manifest["manifest_sha256"],
            },
        },
        "results": {
            "episodes_per_training_seed": 100,
            "aggregate_score": {
                "value": computed["score"],
                "ci_low": score["ci_low"],
                "ci_high": score["ci_high"],
                "confidence": 0.95,
            },
            "achievement_rates": {
                achievement: achievements[achievement]["success_rate"]
                for achievement in computed["achievement_rates"]
            },
            "rescue_rate": statistics["metrics"]["rescue_rate"]["mean"],
        },
        "episode_records": records,
    }
    return payload


def _record(training_seed: int | None, episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "training_seed": training_seed,
        "evaluation_seed": episode["seed"],
        "achievements": episode["achievements"],
        "rescue_success": episode["rescue_success"],
    }


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object in {path}.")
    return value


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
