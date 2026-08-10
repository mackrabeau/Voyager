from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from voyager.benchmark.island import ISLAND_TEST_SEEDS, load_island_seed_manifests
from voyager.benchmark.island_results import (
    ALGORITHMS,
    EXPERIMENT_VERSION,
    FINAL_EVALUATION_VERSION,
    ISLAND_RESULTS_BENCHMARK_ID,
    TRAINING_SEEDS,
    export_island_results,
    load_frozen_island_results,
    verify_island_result_reference,
)
from voyager.sim.island_achievements import (
    ISLAND_ACHIEVEMENTS,
    achievement_success_rates,
    geometric_mean_score,
    grouped_scores,
)


def test_official_result_export_is_deterministic_and_honest(tmp_path: Path) -> None:
    experiment = _synthetic_experiment(tmp_path / "experiment")
    first = export_island_results(experiment, tmp_path / "first", bootstrap_samples=200)
    second = export_island_results(experiment, tmp_path / "second", bootstrap_samples=200)
    assert _directory_bytes(first) == _directory_bytes(second)
    assert verify_island_result_reference(experiment, first)["checked_artifacts"] == 6

    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark_id"] == ISLAND_RESULTS_BENCHMARK_ID
    assert summary["release_gate"]["checks"]["learned_significantly_exceeds_random"]
    assert summary["release_gate"]["checks"]["learned_middle_capability_at_least_25"]
    assert not summary["release_gate"]["checks"]["learned_rescue_at_least_25"]
    assert not summary["release_gate"]["passed"]
    comparison = summary["families"]["feed_forward_ppo"]["paired_vs_legal_random"]
    assert comparison["paired_islands"]
    assert comparison["resampled_training_seeds"]
    assert comparison["ci_low"] > 0.0
    assert "It is not supported as a task solved by PPO" in (first / "report.md").read_text()


def test_result_loader_rejects_corrupted_final_artifact(tmp_path: Path) -> None:
    experiment = _synthetic_experiment(tmp_path / "experiment")
    learned = experiment / "final" / "learned" / "feed_forward_ppo" / "seed_0.json"
    learned.write_text(learned.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_frozen_island_results(experiment)


def test_reference_verification_detects_manual_changes(tmp_path: Path) -> None:
    experiment = _synthetic_experiment(tmp_path / "experiment")
    output = export_island_results(experiment, tmp_path / "reference", bootstrap_samples=100)
    report = output / "report.md"
    report.write_text(report.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not reproducible"):
        verify_island_result_reference(experiment, output)


def test_export_refuses_to_overwrite_reference(tmp_path: Path) -> None:
    experiment = _synthetic_experiment(tmp_path / "experiment")
    output = export_island_results(experiment, tmp_path / "reference", bootstrap_samples=10)
    with pytest.raises(FileExistsError, match="already exists"):
        export_island_results(experiment, output, bootstrap_samples=10)


def _synthetic_experiment(root: Path) -> Path:
    manifests = load_island_seed_manifests()
    revision = "1" * 40
    protocol = {
        "contract": EXPERIMENT_VERSION,
        "official": True,
        "git_revision": revision,
        "manifests": {split: {"sha256": manifest.sha256} for split, manifest in manifests.items()},
    }
    _write_json(root / "protocol.json", protocol)
    final = root / "final"
    random_episodes = [
        _episode(seed, {"collect_food"} if index % 4 == 0 else set())
        for index, seed in enumerate(ISLAND_TEST_SEEDS)
    ]
    oracle_episodes = [
        _episode(seed, set(ISLAND_ACHIEVEMENTS), active=2, rescue=True)
        for seed in ISLAND_TEST_SEEDS
    ]
    _write_json(
        final / "legal_random.json",
        _comparator_payload("legal_random", random_episodes, manifests["test"].sha256),
    )
    oracle_payload = _comparator_payload(
        "scripted_oracle", oracle_episodes, manifests["test"].sha256
    )
    oracle_payload["gate"] = {"passed": True, "checks": {"synthetic": True}}
    _write_json(final / "scripted_oracle.json", oracle_payload)

    learned_runs = []
    for algorithm in ALGORITHMS:
        for training_seed in TRAINING_SEEDS:
            checkpoint_metadata = _run_files(
                root,
                algorithm=algorithm,
                training_seed=training_seed,
                revision=revision,
            )
            artifact = Path("learned") / algorithm / f"seed_{training_seed}.json"
            learned_runs.append(
                {
                    "algorithm": algorithm,
                    "training_seed": training_seed,
                    "checkpoint": str(
                        Path("runs") / algorithm / f"seed_{training_seed}" / "checkpoints/best"
                    ),
                    "artifact": str(artifact),
                }
            )
            learned_achievements = {
                "collect_food",
                "collect_wood",
                "collect_stone",
                "deposit_wood",
                "deposit_stone",
                "build_workbench",
                "both_survive_first_night",
            }
            if algorithm == "feed_forward_ppo":
                learned_achievements |= {"craft_axe", "craft_spear", "build_campfire"}
            stochastic = [
                _episode(seed, learned_achievements if index < 80 else {"collect_food"})
                for index, seed in enumerate(ISLAND_TEST_SEEDS)
            ]
            deterministic = [
                _episode(seed, learned_achievements if index < 60 else set())
                for index, seed in enumerate(ISLAND_TEST_SEEDS)
            ]
            _write_json(
                final / artifact,
                {
                    "contract": FINAL_EVALUATION_VERSION,
                    "algorithm": algorithm,
                    "training_seed": training_seed,
                    "split": "test",
                    "manifest_sha256": manifests["test"].sha256,
                    "primary_inference": "seeded_stochastic",
                    "evaluation": {
                        "checkpoint_metadata": checkpoint_metadata,
                        "checkpoint": "ignored-absolute-source-path",
                        "stochastic": {
                            "episodes": stochastic,
                            "summary": _summary(stochastic),
                        },
                        "deterministic": {
                            "episodes": deterministic,
                            "summary": _summary(deterministic),
                        },
                    },
                },
            )
    _write_json(
        final / "final_evaluation.json",
        {
            "contract": FINAL_EVALUATION_VERSION,
            "experiment_contract": EXPERIMENT_VERSION,
            "git_revision": revision,
            "test_manifest_sha256": manifests["test"].sha256,
            "test_results_used_for_selection": False,
            "test_seeds": list(ISLAND_TEST_SEEDS),
            "learned_runs": learned_runs,
        },
    )
    _write_artifact_manifest(final)
    return root


def _run_files(
    root: Path, *, algorithm: str, training_seed: int, revision: str
) -> dict[str, object]:
    run = root / "runs" / algorithm / f"seed_{training_seed}"
    checkpoint = run / "checkpoints" / "best"
    checkpoint.mkdir(parents=True)
    weights = checkpoint / "model.weights.h5"
    weights.write_bytes(f"{algorithm}:{training_seed}".encode())
    metadata = {
        "git_revision": revision,
        "training_seed": training_seed,
        "weights_file": weights.name,
        "weights_sha256": _sha256(weights),
        "model_type": "recurrent_gru" if algorithm == "recurrent_ppo" else "feed_forward",
        "reward_version": "voyager_island_progression_reward_v4",
    }
    _write_json(checkpoint / "metadata.json", metadata)
    _write_json(run / "status.json", {"status": "finalized"})
    _write_json(
        run / "selection.json",
        {
            "locked": True,
            "test_seeds_accessed": False,
            "best_checkpoint": "checkpoints/best",
            "best_milestone_agent_transitions": 400_000,
        },
    )
    _write_json(
        run / "run_config.json",
        {
            "official": True,
            "git_revision": revision,
            "algorithm": algorithm,
            "training_seed": training_seed,
        },
    )
    _write_artifact_manifest(run)
    return metadata


def _episode(
    seed: int,
    achievements: set[str],
    *,
    active: int = 0,
    rescue: bool = False,
) -> dict[str, object]:
    return {
        "seed": seed,
        "achievements": sorted(achievements),
        "actions": 100,
        "invalid_actions": 0,
        "rescue_success": rescue,
        "joint_survival": active == 2,
        "final_active_agents": active,
        "conservation_errors": {},
    }


def _summary(episodes: list[dict[str, object]]) -> dict[str, object]:
    achievement_sets = [set(row["achievements"]) for row in episodes]
    rates = achievement_success_rates(achievement_sets)
    return {
        "episodes": len(episodes),
        "achievement_success_rates": rates,
        "achievement_geometric_mean": geometric_mean_score(rates),
        "group_scores": grouped_scores(rates),
        "invalid_action_rate": sum(int(row["invalid_actions"]) for row in episodes)
        / sum(int(row["actions"]) for row in episodes),
        "rescue_rate": sum(bool(row["rescue_success"]) for row in episodes) / len(episodes),
        "joint_survival_rate": sum(bool(row["joint_survival"]) for row in episodes) / len(episodes),
        "mean_final_active_agents": sum(int(row["final_active_agents"]) for row in episodes)
        / len(episodes),
        "ledger_reconciliation_rate": 1.0,
    }


def _comparator_payload(
    policy: str,
    episodes: list[dict[str, object]],
    manifest_hash: str,
) -> dict[str, object]:
    return {
        "contract": FINAL_EVALUATION_VERSION,
        "policy": policy,
        "split": "test",
        "manifest_sha256": manifest_hash,
        "episodes": episodes,
        "summary": _summary(episodes),
    }


def _write_artifact_manifest(root: Path) -> None:
    artifacts = [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    _write_json(root / "artifact_manifest.json", {"artifacts": artifacts})


def _directory_bytes(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
