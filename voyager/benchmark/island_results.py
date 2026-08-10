"""Deterministic statistical exports for the frozen VoyagerIsland-v1 evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev
from typing import Any

import numpy as np

from voyager.benchmark.island import ISLAND_TEST_SEEDS, load_island_seed_manifests
from voyager.sim.island_achievements import (
    ISLAND_ACHIEVEMENT_GROUPS,
    ISLAND_ACHIEVEMENTS,
    geometric_mean_score,
    grouped_scores,
)

ISLAND_RESULTS_VERSION = "voyager_island_results_v1"
ISLAND_RESULTS_BENCHMARK_ID = "voyager_island_v1_procedural_baselines_v1"
FINAL_EVALUATION_VERSION = "voyager_island_final_evaluation_v1"
EXPERIMENT_VERSION = "voyager_island_procedural_baselines_v1"
ALGORITHMS = ("feed_forward_ppo", "recurrent_ppo")
INFERENCE_MODES = ("stochastic", "deterministic")
TRAINING_SEEDS = (0, 1, 2)
PRIMARY_INFERENCE_MODE = "stochastic"
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class IslandEpisodeTable:
    """Numeric public outcomes for one policy on the matched test islands."""

    seeds: tuple[int, ...]
    indicators: np.ndarray
    achievement_counts: np.ndarray
    invalid_actions: np.ndarray
    actions: np.ndarray
    rescue: np.ndarray
    joint_survival: np.ndarray
    final_active_agents: np.ndarray
    ledger_reconciles: np.ndarray


@dataclass(frozen=True, slots=True)
class FrozenIslandResults:
    """Validated source material for one completed official experiment matrix."""

    experiment_root: Path
    git_revision: str
    test_manifest_sha256: str
    source_hashes: dict[str, str]
    random: IslandEpisodeTable
    oracle: IslandEpisodeTable
    oracle_gate: dict[str, object]
    learned: dict[tuple[str, int, str], IslandEpisodeTable]
    runs: dict[tuple[str, int], dict[str, object]]


def load_frozen_island_results(experiment_root: str | Path) -> FrozenIslandResults:
    """Load and deeply validate the finalized six-run held-out evaluation."""

    root = Path(experiment_root).resolve()
    final_dir = root / "final"
    source_hashes = {
        f"final/{name}": digest for name, digest in _verify_artifact_manifest(final_dir).items()
    }
    index = _read_mapping(final_dir / "final_evaluation.json")
    if index.get("contract") != FINAL_EVALUATION_VERSION:
        raise ValueError("Unexpected Stage 7 final-evaluation contract.")
    if index.get("experiment_contract") != EXPERIMENT_VERSION:
        raise ValueError("Unexpected Stage 7 procedural experiment contract.")
    if index.get("test_results_used_for_selection") is not False:
        raise ValueError("Held-out test results were marked as used for checkpoint selection.")
    test_seeds = _integer_tuple(index.get("test_seeds"), "test_seeds")
    if test_seeds != ISLAND_TEST_SEEDS:
        raise ValueError("Final evaluation does not contain the frozen 100-island test split.")

    manifests = load_island_seed_manifests()
    test_hash = str(index.get("test_manifest_sha256", ""))
    if test_hash != manifests["test"].sha256:
        raise ValueError("Final evaluation test-manifest hash does not match the tracked split.")
    git_revision = _nonempty_string(index.get("git_revision"), "git_revision")
    protocol = _read_mapping(root / "protocol.json")
    source_hashes["protocol.json"] = _sha256(root / "protocol.json")
    _validate_protocol(protocol, git_revision=git_revision, manifests=manifests)

    random_payload = _read_mapping(final_dir / "legal_random.json")
    random = _load_policy_table(
        random_payload,
        expected_policy="legal_random",
        expected_seeds=test_seeds,
        expected_manifest_hash=test_hash,
    )
    oracle_payload = _read_mapping(final_dir / "scripted_oracle.json")
    oracle = _load_policy_table(
        oracle_payload,
        expected_policy="scripted_oracle",
        expected_seeds=test_seeds,
        expected_manifest_hash=test_hash,
    )
    oracle_gate = _mapping_copy(oracle_payload.get("gate"), "scripted oracle gate")

    expected_runs = {(algorithm, seed) for algorithm in ALGORITHMS for seed in TRAINING_SEEDS}
    learned_index = index.get("learned_runs")
    if not isinstance(learned_index, list):
        raise TypeError("Final evaluation learned_runs must be a list.")
    learned: dict[tuple[str, int, str], IslandEpisodeTable] = {}
    runs: dict[tuple[str, int], dict[str, object]] = {}
    seen: set[tuple[str, int]] = set()
    for raw_entry in learned_index:
        entry = _mapping_copy(raw_entry, "learned run entry")
        algorithm = _choice(entry.get("algorithm"), ALGORITHMS, "algorithm")
        training_seed = _integer(entry.get("training_seed"), "training_seed")
        key = (algorithm, training_seed)
        if key not in expected_runs or key in seen:
            raise ValueError(f"Unexpected or duplicate learned run: {key!r}.")
        seen.add(key)
        artifact = _safe_relative_path(entry.get("artifact"), "learned artifact")
        payload = _read_mapping(final_dir / artifact)
        if payload.get("contract") != FINAL_EVALUATION_VERSION:
            raise ValueError(f"Unexpected learned evaluation contract for {key!r}.")
        if payload.get("algorithm") != algorithm or payload.get("training_seed") != training_seed:
            raise ValueError(f"Learned evaluation identity mismatch for {key!r}.")
        if payload.get("split") != "test" or payload.get("manifest_sha256") != test_hash:
            raise ValueError(f"Learned evaluation split mismatch for {key!r}.")
        if payload.get("primary_inference") != "seeded_stochastic":
            raise ValueError(f"Learned evaluation primary inference mismatch for {key!r}.")
        evaluation = _mapping_copy(payload.get("evaluation"), "learned evaluation")
        metadata = _mapping_copy(evaluation.get("checkpoint_metadata"), "checkpoint metadata")
        if metadata.get("git_revision") != git_revision:
            raise ValueError(f"Checkpoint Git revision mismatch for {key!r}.")
        if metadata.get("training_seed") != training_seed:
            raise ValueError(f"Checkpoint training seed mismatch for {key!r}.")
        for mode in INFERENCE_MODES:
            mode_payload = _mapping_copy(evaluation.get(mode), f"{mode} evaluation")
            learned[(algorithm, training_seed, mode)] = _load_mode_table(
                mode_payload,
                expected_seeds=test_seeds,
                label=f"{algorithm} seed {training_seed} {mode}",
            )
        runs[key] = _load_run_provenance(
            root,
            algorithm=algorithm,
            training_seed=training_seed,
            metadata=metadata,
            git_revision=git_revision,
        )
    if seen != expected_runs:
        missing = sorted(expected_runs - seen)
        raise ValueError(f"Final evaluation is missing official runs: {missing!r}.")

    return FrozenIslandResults(
        experiment_root=root,
        git_revision=git_revision,
        test_manifest_sha256=test_hash,
        source_hashes=source_hashes,
        random=random,
        oracle=oracle,
        oracle_gate=oracle_gate,
        learned=learned,
        runs=runs,
    )


def build_island_result_summary(
    frozen: FrozenIslandResults,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, object]:
    """Build deterministic per-policy, family, and paired-comparison statistics."""

    _validate_bootstrap(bootstrap_samples, confidence)
    policies: dict[str, dict[str, object]] = {}
    random_summary = _policy_statistics(
        frozen.random,
        samples=bootstrap_samples,
        confidence=confidence,
        rng=np.random.default_rng(bootstrap_seed),
    )
    oracle_summary = _policy_statistics(
        frozen.oracle,
        samples=bootstrap_samples,
        confidence=confidence,
        rng=np.random.default_rng(bootstrap_seed + 1),
    )
    policies["legal_random"] = {
        "kind": "comparator",
        "primary": True,
        **random_summary,
    }
    policies["scripted_oracle"] = {
        "kind": "solvability_oracle",
        "primary": True,
        **oracle_summary,
    }

    comparisons: dict[str, dict[str, object]] = {}
    offset = 10
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        for seed in TRAINING_SEEDS:
            for mode_index, mode in enumerate(INFERENCE_MODES):
                policy_id = _policy_id(algorithm, seed, mode)
                table = frozen.learned[(algorithm, seed, mode)]
                policies[policy_id] = {
                    "kind": "learned",
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "inference_mode": mode,
                    "primary": mode == PRIMARY_INFERENCE_MODE,
                    "checkpoint": frozen.runs[(algorithm, seed)],
                    **_policy_statistics(
                        table,
                        samples=bootstrap_samples,
                        confidence=confidence,
                        rng=np.random.default_rng(
                            bootstrap_seed + offset + algorithm_index * 20 + seed * 2 + mode_index
                        ),
                    ),
                }
            primary_id = _policy_id(algorithm, seed, PRIMARY_INFERENCE_MODE)
            comparisons[primary_id] = _paired_score_comparison(
                [frozen.learned[(algorithm, seed, PRIMARY_INFERENCE_MODE)]],
                frozen.random,
                samples=bootstrap_samples,
                confidence=confidence,
                rng=np.random.default_rng(bootstrap_seed + 100 + algorithm_index * 10 + seed),
            )

    families: dict[str, dict[str, object]] = {}
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        family_modes: dict[str, object] = {}
        for mode_index, mode in enumerate(INFERENCE_MODES):
            tables = [frozen.learned[(algorithm, seed, mode)] for seed in TRAINING_SEEDS]
            family_modes[mode] = _family_statistics(
                tables,
                samples=bootstrap_samples,
                confidence=confidence,
                rng=np.random.default_rng(bootstrap_seed + 200 + algorithm_index * 10 + mode_index),
            )
        comparison = _paired_score_comparison(
            [frozen.learned[(algorithm, seed, PRIMARY_INFERENCE_MODE)] for seed in TRAINING_SEEDS],
            frozen.random,
            samples=bootstrap_samples,
            confidence=confidence,
            rng=np.random.default_rng(bootstrap_seed + 300 + algorithm_index),
        )
        comparisons[f"{algorithm}_family"] = comparison
        families[algorithm] = {
            "training_seeds": list(TRAINING_SEEDS),
            "primary_inference_mode": PRIMARY_INFERENCE_MODE,
            "modes": family_modes,
            "paired_vs_legal_random": comparison,
        }

    release_gate = _release_gate(
        frozen=frozen,
        policies=policies,
        families=families,
        comparisons=comparisons,
    )
    return {
        "schema_version": ISLAND_RESULTS_VERSION,
        "benchmark_id": ISLAND_RESULTS_BENCHMARK_ID,
        "experiment_contract": EXPERIMENT_VERSION,
        "final_evaluation_contract": FINAL_EVALUATION_VERSION,
        "git_revision": frozen.git_revision,
        "primary_inference_mode": PRIMARY_INFERENCE_MODE,
        "test_split": {
            "episodes_per_policy": len(ISLAND_TEST_SEEDS),
            "manifest_sha256": frozen.test_manifest_sha256,
            "seeds": list(ISLAND_TEST_SEEDS),
            "used_for_checkpoint_selection": False,
        },
        "statistics": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "confidence": confidence,
            "policy_intervals": "paired island bootstrap",
            "family_intervals": "hierarchical training-seed and paired-island bootstrap",
            "comparison_metric": "smoothed achievement geometric-mean score difference",
        },
        "policies": policies,
        "families": families,
        "comparisons_vs_legal_random": comparisons,
        "release_gate": release_gate,
        "conclusion": (
            "PPO learns the early shared economy and materially exceeds legal random, but "
            "neither learned baseline reaches shelter, hunting, cooking, beacon construction, "
            "or rescue. The learned-rescue release target is not met."
        ),
    }


def export_island_results(
    experiment_root: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Path:
    """Atomically create a compact immutable reference package."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Island result reference already exists: {destination}.")
    frozen = load_frozen_island_results(experiment_root)
    summary = build_island_result_summary(
        frozen,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        _write_json(temporary / "summary.json", summary)
        _write_policy_csv(temporary / "policies.csv", summary)
        _write_achievement_csv(temporary / "achievements.csv", summary)
        _write_comparison_csv(temporary / "comparisons.csv", summary)
        (temporary / "report.md").write_text(_markdown_report(summary), encoding="utf-8")
        artifacts = _artifact_rows(temporary)
        _write_json(
            temporary / "manifest.json",
            {
                "schema_version": ISLAND_RESULTS_VERSION,
                "benchmark_id": ISLAND_RESULTS_BENCHMARK_ID,
                "git_revision": frozen.git_revision,
                "source_final_artifacts": frozen.source_hashes,
                "artifacts": artifacts,
            },
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def verify_island_result_reference(
    experiment_root: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Regenerate a reference package and require byte-identical tracked artifacts."""

    destination = Path(output_dir).resolve()
    manifest = _read_mapping(destination / "manifest.json")
    if manifest.get("schema_version") != ISLAND_RESULTS_VERSION:
        raise ValueError("Unexpected island result reference schema.")
    summary = _read_mapping(destination / "summary.json")
    statistics = _mapping_copy(summary.get("statistics"), "statistics")
    with tempfile.TemporaryDirectory(prefix="voyager-island-results-") as directory:
        regenerated = export_island_results(
            experiment_root,
            Path(directory) / "reference",
            bootstrap_samples=_integer(statistics.get("bootstrap_samples"), "bootstrap_samples"),
            bootstrap_seed=_integer(statistics.get("bootstrap_seed"), "bootstrap_seed"),
            confidence=_number(statistics.get("confidence"), "confidence"),
        )
        expected = {
            path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()
        }
        actual = {path.name: path.read_bytes() for path in regenerated.iterdir() if path.is_file()}
        if expected != actual:
            shared = set(expected) & set(actual)
            changed = sorted(
                set(expected) ^ set(actual)
                | {key for key in shared if expected[key] != actual[key]}
            )
            raise ValueError(f"Island result reference is not reproducible: {changed!r}.")
    return {
        "status": "valid",
        "benchmark_id": ISLAND_RESULTS_BENCHMARK_ID,
        "checked_artifacts": len(expected),
        "path": str(destination),
    }


def _load_policy_table(
    payload: Mapping[str, object],
    *,
    expected_policy: str,
    expected_seeds: tuple[int, ...],
    expected_manifest_hash: str,
) -> IslandEpisodeTable:
    if payload.get("contract") != FINAL_EVALUATION_VERSION:
        raise ValueError(f"Unexpected final-evaluation contract for {expected_policy}.")
    if payload.get("policy") != expected_policy:
        raise ValueError(f"Comparator identity mismatch for {expected_policy}.")
    if payload.get("split") != "test" or payload.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError(f"Comparator split mismatch for {expected_policy}.")
    return _load_mode_table(payload, expected_seeds=expected_seeds, label=expected_policy)


def _load_mode_table(
    payload: Mapping[str, object],
    *,
    expected_seeds: tuple[int, ...],
    label: str,
) -> IslandEpisodeTable:
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError(f"{label} episodes must be a list.")
    table = _episode_table(episodes, expected_seeds=expected_seeds, label=label)
    summary = _mapping_copy(payload.get("summary"), f"{label} summary")
    _validate_stored_summary(table, summary, label=label)
    return table


def _episode_table(
    episodes: Sequence[object],
    *,
    expected_seeds: tuple[int, ...],
    label: str,
) -> IslandEpisodeTable:
    rows = [_mapping_copy(row, f"{label} episode") for row in episodes]
    seeds = tuple(_integer(row.get("seed"), "episode seed") for row in rows)
    if seeds != expected_seeds:
        raise ValueError(f"{label} episodes do not match the frozen ordered test seeds.")
    indicators = np.zeros((len(rows), len(ISLAND_ACHIEVEMENTS)), dtype=np.float64)
    invalid: list[float] = []
    actions: list[float] = []
    rescue: list[float] = []
    joint: list[float] = []
    active: list[float] = []
    ledger: list[float] = []
    achievement_set = set(ISLAND_ACHIEVEMENTS)
    for row_index, row in enumerate(rows):
        raw_achievements = row.get("achievements")
        if not isinstance(raw_achievements, list) or not all(
            isinstance(value, str) for value in raw_achievements
        ):
            raise TypeError(f"{label} achievements must be a string list.")
        achievements = set(raw_achievements)
        if len(achievements) != len(raw_achievements) or not achievements <= achievement_set:
            raise ValueError(f"{label} contains duplicate or unknown achievements.")
        for achievement_index, achievement in enumerate(ISLAND_ACHIEVEMENTS):
            indicators[row_index, achievement_index] = float(achievement in achievements)
        invalid_value = _integer(row.get("invalid_actions"), "invalid_actions")
        action_value = _integer(row.get("actions"), "actions")
        if invalid_value < 0 or action_value <= 0 or invalid_value > action_value:
            raise ValueError(f"{label} contains invalid action accounting.")
        invalid.append(float(invalid_value))
        actions.append(float(action_value))
        rescue.append(float(_boolean(row.get("rescue_success"), "rescue_success")))
        joint.append(float(_boolean(row.get("joint_survival"), "joint_survival")))
        final_active = _integer(row.get("final_active_agents"), "final_active_agents")
        if final_active < 0 or final_active > 2:
            raise ValueError(f"{label} contains an invalid final active-agent count.")
        active.append(float(final_active))
        conservation = row.get("conservation_errors")
        if not isinstance(conservation, Mapping):
            raise TypeError(f"{label} conservation_errors must be a mapping.")
        ledger.append(float(not conservation))
    return IslandEpisodeTable(
        seeds=seeds,
        indicators=indicators,
        achievement_counts=indicators.sum(axis=1),
        invalid_actions=np.asarray(invalid, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float64),
        rescue=np.asarray(rescue, dtype=np.float64),
        joint_survival=np.asarray(joint, dtype=np.float64),
        final_active_agents=np.asarray(active, dtype=np.float64),
        ledger_reconciles=np.asarray(ledger, dtype=np.float64),
    )


def _validate_stored_summary(
    table: IslandEpisodeTable,
    summary: Mapping[str, object],
    *,
    label: str,
) -> None:
    point = _point_statistics(table)
    checks = {
        "episodes": len(table.seeds),
        "achievement_geometric_mean": point["achievement_geometric_mean"],
        "invalid_action_rate": point["invalid_action_rate"],
        "rescue_rate": point["rescue_rate"],
        "joint_survival_rate": point["joint_survival_rate"],
        "mean_final_active_agents": point["mean_final_active_agents"],
        "ledger_reconciliation_rate": point["ledger_reconciliation_rate"],
    }
    for name, expected in checks.items():
        actual = summary.get(name)
        if name == "episodes":
            if actual != expected:
                raise ValueError(f"{label} stored {name} does not reconstruct.")
        elif not math.isclose(_number(actual, name), float(expected), abs_tol=1e-12):
            raise ValueError(f"{label} stored {name} does not reconstruct.")
    raw_rates = _mapping_copy(summary.get("achievement_success_rates"), "achievement rates")
    for achievement, expected in point["achievement_success_rates"].items():
        if not math.isclose(
            _number(raw_rates.get(achievement), achievement), expected, abs_tol=1e-12
        ):
            raise ValueError(
                f"{label} stored achievement rate does not reconstruct: {achievement}."
            )


def _load_run_provenance(
    root: Path,
    *,
    algorithm: str,
    training_seed: int,
    metadata: Mapping[str, object],
    git_revision: str,
) -> dict[str, object]:
    run_dir = root / "runs" / algorithm / f"seed_{training_seed}"
    status = _read_mapping(run_dir / "status.json")
    if status.get("status") != "finalized":
        raise ValueError(f"Official run is not finalized: {run_dir}.")
    _verify_artifact_manifest(run_dir)
    selection = _read_mapping(run_dir / "selection.json")
    config = _read_mapping(run_dir / "run_config.json")
    if not selection.get("locked") or selection.get("test_seeds_accessed") is not False:
        raise ValueError(f"Official selection is not cleanly locked: {run_dir}.")
    if config.get("git_revision") != git_revision or config.get("official") is not True:
        raise ValueError(f"Official run provenance mismatch: {run_dir}.")
    if config.get("algorithm") != algorithm or config.get("training_seed") != training_seed:
        raise ValueError(f"Official run identity mismatch: {run_dir}.")
    checkpoint_relative = _safe_relative_path(selection.get("best_checkpoint"), "best checkpoint")
    checkpoint = (run_dir / checkpoint_relative).resolve()
    if run_dir.resolve() not in checkpoint.parents:
        raise ValueError("Selected checkpoint escapes its run directory.")
    checkpoint_metadata = _read_mapping(checkpoint / "metadata.json")
    if checkpoint_metadata != dict(metadata):
        raise ValueError(f"Embedded and on-disk checkpoint metadata differ: {run_dir}.")
    weights_name = _safe_relative_path(metadata.get("weights_file"), "weights file")
    weights = checkpoint / weights_name
    weights_hash = _nonempty_string(metadata.get("weights_sha256"), "weights_sha256")
    if _sha256(weights) != weights_hash:
        raise ValueError(f"Selected checkpoint weights failed validation: {weights}.")
    return {
        "selected_milestone_agent_transitions": _integer(
            selection.get("best_milestone_agent_transitions"),
            "best_milestone_agent_transitions",
        ),
        "weights_sha256": weights_hash,
        "model_type": _nonempty_string(metadata.get("model_type"), "model_type"),
        "training_reward_version": _nonempty_string(
            metadata.get("reward_version"), "reward_version"
        ),
        "checkpoint_git_revision": _nonempty_string(
            metadata.get("git_revision"), "checkpoint git_revision"
        ),
        "source_artifacts": {
            "artifact_manifest.json": _sha256(run_dir / "artifact_manifest.json"),
            "run_config.json": _sha256(run_dir / "run_config.json"),
            "selection.json": _sha256(run_dir / "selection.json"),
            "status.json": _sha256(run_dir / "status.json"),
            str(checkpoint_relative / "metadata.json"): _sha256(checkpoint / "metadata.json"),
            str(checkpoint_relative / weights_name): weights_hash,
        },
    }


def _validate_protocol(
    protocol: Mapping[str, object],
    *,
    git_revision: str,
    manifests: Mapping[str, Any],
) -> None:
    if protocol.get("contract") != EXPERIMENT_VERSION or protocol.get("official") is not True:
        raise ValueError("Experiment protocol is not the frozen official contract.")
    if protocol.get("git_revision") != git_revision:
        raise ValueError("Experiment and final-evaluation Git revisions differ.")
    rows = _mapping_copy(protocol.get("manifests"), "protocol manifests")
    for split, manifest in manifests.items():
        row = _mapping_copy(rows.get(split), f"{split} manifest")
        if row.get("sha256") != manifest.sha256:
            raise ValueError(f"Protocol {split} manifest no longer matches the tracked file.")


def _point_statistics(table: IslandEpisodeTable) -> dict[str, Any]:
    rates_array = table.indicators.mean(axis=0)
    rates = {
        achievement: float(rates_array[index])
        for index, achievement in enumerate(ISLAND_ACHIEVEMENTS)
    }
    return {
        "episodes": len(table.seeds),
        "achievement_success_rates": rates,
        "achievement_geometric_mean": geometric_mean_score(rates),
        "group_scores": grouped_scores(rates),
        "mean_achievement_count": float(table.achievement_counts.mean()),
        "invalid_action_rate": float(table.invalid_actions.sum() / table.actions.sum()),
        "rescue_rate": float(table.rescue.mean()),
        "joint_survival_rate": float(table.joint_survival.mean()),
        "mean_final_active_agents": float(table.final_active_agents.mean()),
        "ledger_reconciliation_rate": float(table.ledger_reconciles.mean()),
    }


def _policy_statistics(
    table: IslandEpisodeTable,
    *,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    point = _point_statistics(table)
    indexes = rng.integers(0, len(table.seeds), size=(samples, len(table.seeds)))
    rate_replicates = table.indicators[indexes].mean(axis=1)
    score_replicates = _score_rows(rate_replicates)
    achievement_rows = {
        achievement: {
            "success_rate": point["achievement_success_rates"][achievement],
            "ci_low": _quantile(rate_replicates[:, index], confidence, lower=True),
            "ci_high": _quantile(rate_replicates[:, index], confidence, lower=False),
        }
        for index, achievement in enumerate(ISLAND_ACHIEVEMENTS)
    }
    metric_values = {
        "mean_achievement_count": table.achievement_counts[indexes].mean(axis=1),
        "rescue_rate": table.rescue[indexes].mean(axis=1),
        "joint_survival_rate": table.joint_survival[indexes].mean(axis=1),
        "mean_final_active_agents": table.final_active_agents[indexes].mean(axis=1),
        "ledger_reconciliation_rate": table.ledger_reconciles[indexes].mean(axis=1),
        "invalid_action_rate": table.invalid_actions[indexes].sum(axis=1)
        / table.actions[indexes].sum(axis=1),
    }
    metrics = {
        name: {
            "mean": point[name],
            "ci_low": _quantile(values, confidence, lower=True),
            "ci_high": _quantile(values, confidence, lower=False),
        }
        for name, values in metric_values.items()
    }
    return {
        "episodes": len(table.seeds),
        "achievement_geometric_mean": {
            "mean": point["achievement_geometric_mean"],
            "ci_low": _quantile(score_replicates, confidence, lower=True),
            "ci_high": _quantile(score_replicates, confidence, lower=False),
        },
        "achievements": achievement_rows,
        "groups": point["group_scores"],
        "metrics": metrics,
    }


def _family_statistics(
    tables: Sequence[IslandEpisodeTable],
    *,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    _validate_family_tables(tables)
    indicators = np.asarray([table.indicators for table in tables])
    rates = indicators.mean(axis=(0, 1))
    point_score = _score_rows(rates[None, :])[0]
    seed_rates = indicators.mean(axis=1)
    seed_scores = _score_rows(seed_rates)
    policy_indexes, island_indexes = _hierarchical_indexes(
        rng,
        samples=samples,
        policy_count=len(tables),
        island_count=len(tables[0].seeds),
    )
    rate_replicates = _hierarchical_rate_replicates(
        indicators,
        policy_indexes=policy_indexes,
        island_indexes=island_indexes,
    )
    score_replicates = _score_rows(rate_replicates)
    achievement_rows = {
        achievement: {
            "success_rate": float(rates[index]),
            "ci_low": _quantile(rate_replicates[:, index], confidence, lower=True),
            "ci_high": _quantile(rate_replicates[:, index], confidence, lower=False),
            "training_seed_std": _sample_std(seed_rates[:, index]),
            "training_seed_range": [
                float(seed_rates[:, index].min()),
                float(seed_rates[:, index].max()),
            ],
        }
        for index, achievement in enumerate(ISLAND_ACHIEVEMENTS)
    }
    metric_arrays = {
        "mean_achievement_count": np.asarray([table.achievement_counts for table in tables]),
        "rescue_rate": np.asarray([table.rescue for table in tables]),
        "joint_survival_rate": np.asarray([table.joint_survival for table in tables]),
        "mean_final_active_agents": np.asarray([table.final_active_agents for table in tables]),
        "ledger_reconciliation_rate": np.asarray([table.ledger_reconciles for table in tables]),
    }
    metrics: dict[str, object] = {}
    for name, values in metric_arrays.items():
        replicates = _hierarchical_mean_replicates(
            values,
            policy_indexes=policy_indexes,
            island_indexes=island_indexes,
        )
        seed_means = values.mean(axis=1)
        metrics[name] = {
            "mean": float(values.mean()),
            "ci_low": _quantile(replicates, confidence, lower=True),
            "ci_high": _quantile(replicates, confidence, lower=False),
            "training_seed_std": _sample_std(seed_means),
            "training_seed_range": [float(seed_means.min()), float(seed_means.max())],
        }
    invalid = np.asarray([table.invalid_actions for table in tables])
    actions = np.asarray([table.actions for table in tables])
    invalid_replicates = _hierarchical_ratio_replicates(
        invalid,
        actions,
        policy_indexes=policy_indexes,
        island_indexes=island_indexes,
    )
    seed_invalid = invalid.sum(axis=1) / actions.sum(axis=1)
    metrics["invalid_action_rate"] = {
        "mean": float(invalid.sum() / actions.sum()),
        "ci_low": _quantile(invalid_replicates, confidence, lower=True),
        "ci_high": _quantile(invalid_replicates, confidence, lower=False),
        "training_seed_std": _sample_std(seed_invalid),
        "training_seed_range": [float(seed_invalid.min()), float(seed_invalid.max())],
    }
    group_rates = {
        group: float(np.mean([rates[ISLAND_ACHIEVEMENTS.index(name)] for name in names]))
        for group, names in ISLAND_ACHIEVEMENT_GROUPS.items()
    }
    return {
        "training_seed_count": len(tables),
        "episodes_per_training_seed": len(tables[0].seeds),
        "achievement_geometric_mean": {
            "mean": float(point_score),
            "ci_low": _quantile(score_replicates, confidence, lower=True),
            "ci_high": _quantile(score_replicates, confidence, lower=False),
            "training_seed_mean": float(seed_scores.mean()),
            "training_seed_std": _sample_std(seed_scores),
            "training_seed_range": [float(seed_scores.min()), float(seed_scores.max())],
        },
        "achievements": achievement_rows,
        "groups": group_rates,
        "metrics": metrics,
    }


def _paired_score_comparison(
    learned_tables: Sequence[IslandEpisodeTable],
    random_table: IslandEpisodeTable,
    *,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    _validate_family_tables(learned_tables)
    if learned_tables[0].seeds != random_table.seeds:
        raise ValueError("Learned and random evaluations must use matched ordered islands.")
    learned = np.asarray([table.indicators for table in learned_tables])
    policy_indexes, island_indexes = _hierarchical_indexes(
        rng,
        samples=samples,
        policy_count=len(learned_tables),
        island_count=len(random_table.seeds),
    )
    learned_replicates = _hierarchical_rate_replicates(
        learned,
        policy_indexes=policy_indexes,
        island_indexes=island_indexes,
    )
    random_replicates = random_table.indicators[island_indexes].mean(axis=1)
    differences = _score_rows(learned_replicates) - _score_rows(random_replicates)
    learned_score = _score_rows(learned.mean(axis=(0, 1))[None, :])[0]
    random_score = _score_rows(random_table.indicators.mean(axis=0)[None, :])[0]
    low = _quantile(differences, confidence, lower=True)
    high = _quantile(differences, confidence, lower=False)
    return {
        "baseline": "legal_random",
        "metric": "achievement_geometric_mean",
        "mean_difference": float(learned_score - random_score),
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "paired_islands": True,
        "resampled_training_seeds": len(learned_tables) > 1,
        "significantly_exceeds_random": low > 0.0,
    }


def _release_gate(
    *,
    frozen: FrozenIslandResults,
    policies: Mapping[str, Mapping[str, object]],
    families: Mapping[str, Mapping[str, object]],
    comparisons: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    random_policy = policies["legal_random"]
    random_achievements = _mapping_copy(random_policy.get("achievements"), "random achievements")
    learned_primary = [
        row
        for row in policies.values()
        if row.get("kind") == "learned" and row.get("primary") is True
    ]
    middle_achievements = (
        "build_workbench",
        "craft_axe",
        "craft_spear",
        "build_campfire",
        "build_shelter",
        "hunt_deer",
        "cook_meat",
    )
    learned_middle = max(
        _achievement_rate(_mapping_copy(row.get("achievements"), "achievements"), name)
        for row in learned_primary
        for name in middle_achievements
    )
    learned_rescue = max(
        _achievement_rate(_mapping_copy(row.get("achievements"), "achievements"), "rescue_both")
        for row in learned_primary
    )
    family_primary = [
        _mapping_copy(
            _mapping_copy(row.get("modes"), "family modes").get(PRIMARY_INFERENCE_MODE),
            "family primary",
        )
        for row in families.values()
    ]
    all_ledgers = all(
        math.isclose(
            _metric_mean(
                _mapping_copy(row.get("metrics"), "metrics"), "ledger_reconciliation_rate"
            ),
            1.0,
        )
        for row in [
            *learned_primary,
            *family_primary,
            policies["legal_random"],
            policies["scripted_oracle"],
        ]
    )
    checks = {
        "oracle_solvability_gate": bool(frozen.oracle_gate.get("passed")),
        "random_never_rescues": _number(
            _mapping_copy(random_achievements["rescue_both"], "random rescue").get("success_rate"),
            "random rescue",
        )
        == 0.0,
        "random_rarely_builds_workbench": _number(
            _mapping_copy(random_achievements["build_workbench"], "random workbench").get(
                "success_rate"
            ),
            "random workbench",
        )
        <= 0.05,
        "learned_significantly_exceeds_random": any(
            bool(row.get("significantly_exceeds_random")) for row in comparisons.values()
        ),
        "learned_middle_capability_at_least_25": learned_middle >= 0.25,
        "learned_rescue_at_least_25": learned_rescue >= 0.25,
        "all_ledgers_reconcile": all_ledgers,
        "large_reproducibility_artifacts_published": False,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed_max_middle_capability": learned_middle,
        "observed_max_learned_rescue": learned_rescue,
        "note": (
            "The compact statistics are frozen here. Checkpoints, raw episode records, curves, "
            "and curated learned-policy replays remain release artifacts."
        ),
    }


def _hierarchical_indexes(
    rng: np.random.Generator,
    *,
    samples: int,
    policy_count: int,
    island_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        rng.integers(0, policy_count, size=(samples, policy_count)),
        rng.integers(0, island_count, size=(samples, island_count)),
    )


def _hierarchical_rate_replicates(
    values: np.ndarray,
    *,
    policy_indexes: np.ndarray,
    island_indexes: np.ndarray,
) -> np.ndarray:
    replicates = np.empty((len(policy_indexes), values.shape[2]), dtype=np.float64)
    sample_policies = policy_indexes[:, :, None]
    sample_islands = island_indexes[:, None, :]
    for achievement_index in range(values.shape[2]):
        selected = values[:, :, achievement_index][sample_policies, sample_islands]
        replicates[:, achievement_index] = selected.mean(axis=(1, 2))
    return replicates


def _hierarchical_mean_replicates(
    values: np.ndarray,
    *,
    policy_indexes: np.ndarray,
    island_indexes: np.ndarray,
) -> np.ndarray:
    return values[policy_indexes[:, :, None], island_indexes[:, None, :]].mean(axis=(1, 2))


def _hierarchical_ratio_replicates(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    policy_indexes: np.ndarray,
    island_indexes: np.ndarray,
) -> np.ndarray:
    selected_numerators = numerators[policy_indexes[:, :, None], island_indexes[:, None, :]]
    selected_denominators = denominators[policy_indexes[:, :, None], island_indexes[:, None, :]]
    return selected_numerators.sum(axis=(1, 2)) / selected_denominators.sum(axis=(1, 2))


def _score_rows(rates: np.ndarray) -> np.ndarray:
    return np.exp(np.log(np.clip(rates, 0.0, 1.0) + 0.01).mean(axis=1)) - 0.01


def _validate_family_tables(tables: Sequence[IslandEpisodeTable]) -> None:
    if not tables:
        raise ValueError("At least one learned table is required.")
    if any(table.seeds != tables[0].seeds for table in tables[1:]):
        raise ValueError("Training-seed policies must use matched ordered islands.")


def _validate_bootstrap(samples: int, confidence: float) -> None:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one.")


def _sample_std(values: np.ndarray) -> float:
    return float(stdev(float(value) for value in values)) if values.size > 1 else 0.0


def _quantile(values: np.ndarray, confidence: float, *, lower: bool) -> float:
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha if lower else 1.0 - alpha))


def _write_policy_csv(path: Path, summary: Mapping[str, object]) -> None:
    policies = _mapping_copy(summary.get("policies"), "policies")
    fields = (
        "policy_id",
        "kind",
        "algorithm",
        "training_seed",
        "inference_mode",
        "primary",
        "episodes",
        "achievement_score",
        "score_ci_low",
        "score_ci_high",
        "mean_achievement_count",
        "invalid_action_rate",
        "first_night_rate",
        "rescue_rate",
        "joint_survival_rate",
        "mean_final_active_agents",
        "selected_milestone_agent_transitions",
        "weights_sha256",
    )
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for policy_id, raw_policy in policies.items():
            policy = _mapping_copy(raw_policy, "policy")
            score = _mapping_copy(policy.get("achievement_geometric_mean"), "score")
            metrics = _mapping_copy(policy.get("metrics"), "metrics")
            achievements = _mapping_copy(policy.get("achievements"), "achievements")
            checkpoint = (
                _mapping_copy(policy.get("checkpoint"), "checkpoint")
                if policy.get("checkpoint") is not None
                else {}
            )
            writer.writerow(
                {
                    "policy_id": policy_id,
                    "kind": policy.get("kind", ""),
                    "algorithm": policy.get("algorithm", ""),
                    "training_seed": policy.get("training_seed", ""),
                    "inference_mode": policy.get("inference_mode", ""),
                    "primary": policy.get("primary", ""),
                    "episodes": policy.get("episodes", ""),
                    "achievement_score": score.get("mean", ""),
                    "score_ci_low": score.get("ci_low", ""),
                    "score_ci_high": score.get("ci_high", ""),
                    "mean_achievement_count": _metric_mean(metrics, "mean_achievement_count"),
                    "invalid_action_rate": _metric_mean(metrics, "invalid_action_rate"),
                    "first_night_rate": _achievement_rate(achievements, "both_survive_first_night"),
                    "rescue_rate": _metric_mean(metrics, "rescue_rate"),
                    "joint_survival_rate": _metric_mean(metrics, "joint_survival_rate"),
                    "mean_final_active_agents": _metric_mean(metrics, "mean_final_active_agents"),
                    "selected_milestone_agent_transitions": checkpoint.get(
                        "selected_milestone_agent_transitions", ""
                    ),
                    "weights_sha256": checkpoint.get("weights_sha256", ""),
                }
            )


def _write_achievement_csv(path: Path, summary: Mapping[str, object]) -> None:
    fields = (
        "scope_id",
        "scope_type",
        "algorithm",
        "training_seed",
        "inference_mode",
        "achievement",
        "success_rate",
        "ci_low",
        "ci_high",
        "training_seed_std",
        "training_seed_min",
        "training_seed_max",
    )
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        policies = _mapping_copy(summary.get("policies"), "policies")
        for policy_id, raw_policy in policies.items():
            policy = _mapping_copy(raw_policy, "policy")
            achievements = _mapping_copy(policy.get("achievements"), "achievements")
            for achievement in ISLAND_ACHIEVEMENTS:
                row = _mapping_copy(achievements.get(achievement), achievement)
                writer.writerow(
                    {
                        "scope_id": policy_id,
                        "scope_type": "policy",
                        "algorithm": policy.get("algorithm", ""),
                        "training_seed": policy.get("training_seed", ""),
                        "inference_mode": policy.get("inference_mode", ""),
                        "achievement": achievement,
                        "success_rate": row.get("success_rate"),
                        "ci_low": row.get("ci_low"),
                        "ci_high": row.get("ci_high"),
                        "training_seed_std": "",
                        "training_seed_min": "",
                        "training_seed_max": "",
                    }
                )
        families = _mapping_copy(summary.get("families"), "families")
        for algorithm, raw_family in families.items():
            family = _mapping_copy(raw_family, "family")
            modes = _mapping_copy(family.get("modes"), "family modes")
            for mode in INFERENCE_MODES:
                mode_row = _mapping_copy(modes.get(mode), "family mode")
                achievements = _mapping_copy(mode_row.get("achievements"), "achievements")
                for achievement in ISLAND_ACHIEVEMENTS:
                    row = _mapping_copy(achievements.get(achievement), achievement)
                    seed_range = row.get("training_seed_range")
                    if not isinstance(seed_range, list) or len(seed_range) != 2:
                        raise TypeError("Family training_seed_range must contain two values.")
                    writer.writerow(
                        {
                            "scope_id": f"{algorithm}_family_{mode}",
                            "scope_type": "algorithm_family",
                            "algorithm": algorithm,
                            "training_seed": "",
                            "inference_mode": mode,
                            "achievement": achievement,
                            "success_rate": row.get("success_rate"),
                            "ci_low": row.get("ci_low"),
                            "ci_high": row.get("ci_high"),
                            "training_seed_std": row.get("training_seed_std"),
                            "training_seed_min": seed_range[0],
                            "training_seed_max": seed_range[1],
                        }
                    )


def _write_comparison_csv(path: Path, summary: Mapping[str, object]) -> None:
    comparisons = _mapping_copy(summary.get("comparisons_vs_legal_random"), "comparisons")
    fields = (
        "policy_or_family",
        "baseline",
        "metric",
        "mean_difference",
        "ci_low",
        "ci_high",
        "confidence",
        "paired_islands",
        "resampled_training_seeds",
        "significantly_exceeds_random",
    )
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, raw_row in comparisons.items():
            row = _mapping_copy(raw_row, "comparison")
            writer.writerow(
                {"policy_or_family": name, **{field: row.get(field) for field in fields[1:]}}
            )


def _markdown_report(summary: Mapping[str, object]) -> str:
    policies = _mapping_copy(summary.get("policies"), "policies")
    families = _mapping_copy(summary.get("families"), "families")
    random_score = _score_value(policies["legal_random"])
    oracle_score = _score_value(policies["scripted_oracle"])
    ff = _family_primary(families, "feed_forward_ppo")
    recurrent = _family_primary(families, "recurrent_ppo")
    gate = _mapping_copy(summary.get("release_gate"), "release gate")
    checks = _mapping_copy(gate.get("checks"), "release checks")
    lines = [
        "# VoyagerIsland-v1 Official Baseline Results",
        "",
        "This report freezes the first official procedural evaluation. Checkpoints were selected",
        "only on the 50-island development split; the 100-island test split was opened once after",
        "all six runs were locked. Seeded-stochastic inference is the primary learned-policy result.",
        "",
        "## Headline results",
        "",
        "| Policy | Public score | 95% CI | Training-seed SD | Rescue |",
        "|---|---:|---:|---:|---:|",
        _report_row("Legal random", policies["legal_random"]),
        _report_family_row("Feed-forward PPO", ff),
        _report_family_row("Recurrent PPO", recurrent),
        _report_row("Scripted oracle", policies["scripted_oracle"]),
        "",
        f"Legal random scored `{random_score:.3f}` and the scripted oracle scored `{oracle_score:.3f}`.",
        "Feed-forward PPO learned gathering, delivery, the workbench, tools, and campfire construction",
        "in multiple training seeds. Recurrent PPO did not improve on feed-forward PPO under the frozen",
        "one-million-transition budget.",
        "",
        "No learned policy built shelter, hunted, cooked meat, built the beacon, or rescued both agents",
        "on a held-out island. The scripted oracle rescued both agents on 96% of islands, so the mechanics",
        "are solvable even though the learned full task remains unsolved by these baselines.",
        "",
        "## Frozen release-gate outcome",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            f"Overall release gate: **{'PASS' if gate.get('passed') else 'FAIL'}**.",
            "",
            "The environment is therefore supported as a difficult benchmark with demonstrable early-economy",
            "learning and a clearly measured late-game frontier. It is not supported as a task solved by PPO.",
            "",
            "## Statistical method",
            "",
            "Per-policy intervals resample matched test islands. Algorithm-family intervals resample both the",
            "three independent training seeds and the matched islands. Comparisons against legal random reuse",
            "the same island resample on both sides. This preserves pairing and avoids treating 300 episodes as",
            "300 independently trained policies. The public score is the frozen smoothed geometric mean over all",
            "15 achievements. Deterministic argmax results are diagnostic and never replace the stochastic result.",
            "",
            "See `summary.json` for complete statistics and provenance, and the CSV files for tables suitable",
            "for analysis and plotting.",
        ]
    )
    return "\n".join(lines) + "\n"


def _report_row(label: str, raw_policy: object) -> str:
    policy = _mapping_copy(raw_policy, "report policy")
    score = _mapping_copy(policy.get("achievement_geometric_mean"), "score")
    rescue = _metric_mean(_mapping_copy(policy.get("metrics"), "metrics"), "rescue_rate")
    return (
        f"| {label} | {_number(score.get('mean'), 'score mean'):.3f} | "
        f"[{_number(score.get('ci_low'), 'score ci_low'):.3f}, "
        f"{_number(score.get('ci_high'), 'score ci_high'):.3f}] | — | {rescue:.1%} |"
    )


def _report_family_row(label: str, raw_family: Mapping[str, object]) -> str:
    score = _mapping_copy(raw_family.get("achievement_geometric_mean"), "family score")
    rescue = _metric_mean(_mapping_copy(raw_family.get("metrics"), "metrics"), "rescue_rate")
    return (
        f"| {label} | {_number(score.get('mean'), 'family score mean'):.3f} | "
        f"[{_number(score.get('ci_low'), 'family score ci_low'):.3f}, "
        f"{_number(score.get('ci_high'), 'family score ci_high'):.3f}] | "
        f"{_number(score.get('training_seed_std'), 'training seed std'):.3f} | "
        f"{rescue:.1%} |"
    )


def _family_primary(families: Mapping[str, object], algorithm: str) -> dict[str, object]:
    family = _mapping_copy(families.get(algorithm), "family")
    modes = _mapping_copy(family.get("modes"), "family modes")
    return _mapping_copy(modes.get(PRIMARY_INFERENCE_MODE), "family primary mode")


def _score_value(raw_policy: object) -> float:
    policy = _mapping_copy(raw_policy, "policy")
    score = _mapping_copy(policy.get("achievement_geometric_mean"), "score")
    return _number(score.get("mean"), "score mean")


def _metric_mean(metrics: Mapping[str, object], name: str) -> float:
    row = _mapping_copy(metrics.get(name), name)
    return _number(row.get("mean"), f"{name} mean")


def _achievement_rate(achievements: Mapping[str, object], name: str) -> float:
    row = _mapping_copy(achievements.get(name), name)
    return _number(row.get("success_rate"), f"{name} success_rate")


def _policy_id(algorithm: str, training_seed: int, mode: str) -> str:
    return f"{algorithm}_seed_{training_seed}_{mode}"


def _artifact_rows(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.json"
    ]


def _verify_artifact_manifest(root: Path) -> dict[str, str]:
    manifest_path = root / "artifact_manifest.json"
    manifest = _read_mapping(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError(f"Artifact manifest has no artifacts list: {root}.")
    hashes = {"artifact_manifest.json": _sha256(manifest_path)}
    seen: set[str] = set()
    for raw_row in artifacts:
        row = _mapping_copy(raw_row, "artifact row")
        relative = _safe_relative_path(row.get("path"), "artifact path")
        name = str(relative)
        if name in seen:
            raise ValueError(f"Duplicate artifact path in {root}: {name}.")
        seen.add(name)
        path = root / relative
        expected_hash = _nonempty_string(row.get("sha256"), "artifact sha256")
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"Artifact failed SHA-256 validation: {path}.")
        expected_size = row.get("size", row.get("bytes"))
        if expected_size is not None and path.stat().st_size != _integer(expected_size, "size"):
            raise ValueError(f"Artifact failed size validation: {path}.")
        hashes[name] = expected_hash
    return dict(sorted(hashes.items()))


def _safe_relative_path(value: object, label: str) -> Path:
    raw = _nonempty_string(value, label)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path.")
    return path


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list.")
    return tuple(_integer(item, label) for item in value)


def _mapping_copy(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def _choice(value: object, choices: Sequence[str], label: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{label} must be one of {tuple(choices)!r}.")
    return value


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer.")
    return value


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be Boolean.")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string.")
    return value


def _read_mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping_copy(value, str(path))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
