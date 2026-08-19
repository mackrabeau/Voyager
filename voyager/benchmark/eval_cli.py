"""Evaluate an importable Voyager policy and emit a submission draft."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from voyager.benchmark.island import load_island_seed_manifests
from voyager.benchmark.leaderboard import SCHEMA_VERSION, score_episode_records
from voyager.sim.island_achievements import ISLAND_ACHIEVEMENT_VERSION
from voyager.training.island_evaluation import IslandPolicy, evaluate_island_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="Import path in module:factory form.")
    parser.add_argument("--split", choices=("development", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-name")
    parser.add_argument(
        "--category",
        choices=("learned_official_budget", "external_knowledge", "scripted_reference"),
        default="external_knowledge",
    )
    parser.add_argument("--training-seed", type=int, action="append", default=[])
    parser.add_argument("--training-budget", type=int, default=0)
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args()

    manifests = load_island_seed_manifests()
    manifest = manifests[args.split]
    seeds = manifest.seeds if args.episodes is None else manifest.seeds[: args.episodes]
    factory = _load_factory(args.policy)
    records: list[dict[str, Any]] = []
    training_seeds: list[int | None] = args.training_seed or [None]
    for training_seed in training_seeds:
        def policy_for_evaluation(
            evaluation_seed: int,
            selected_training_seed: int | None = training_seed,
        ) -> IslandPolicy:
            return _create_policy(
                factory,
                training_seed=selected_training_seed,
                evaluation_seed=evaluation_seed,
            )

        episodes, _summary = evaluate_island_policy(
            policy_for_evaluation,
            seeds=seeds,
            procedural=True,
        )
        records.extend(
            {
                "training_seed": training_seed,
                "evaluation_seed": episode.seed,
                "achievements": list(episode.achievements),
                "rescue_success": episode.rescue_success,
            }
            for episode in episodes
        )
    computed = score_episode_records(records)
    ci_low, ci_high = _bootstrap_interval(records)
    method_name = args.method_name or args.policy
    payload = {
        "schema_version": SCHEMA_VERSION,
        "submission_id": "replace-with-method-slug",
        "status": "draft",
        "submitted_at": "1970-01-01",
        "benchmark": {
            "id": "VoyagerIsland-v1",
            "environment_version": "voyager_island_v1",
            "achievement_version": ISLAND_ACHIEVEMENT_VERSION,
            "generator_version": manifest.generator_version,
            "split": args.split,
            "test_manifest_sha256": manifests["test"].sha256,
        },
        "method": {
            "name": method_name,
            "category": args.category,
            "description": "REPLACE BEFORE SUBMISSION",
            "policy_source": {
                "repository_url": "https://example.invalid/replace-me",
                "commit": "0" * 40,
            },
            "training_seeds": args.training_seed,
            "training_budget_agent_steps": args.training_budget,
            "external_data": {"used": False, "description": "REPLACE BEFORE SUBMISSION"},
            "pretrained_models": {"used": False, "description": "REPLACE BEFORE SUBMISSION"},
            "privileged_information": {"used": False, "description": "REPLACE BEFORE SUBMISSION"},
        },
        "artifacts": {
            "checkpoints": [],
            "episode_records": {
                "url": "https://example.invalid/replace-with-artifact-url",
                "sha256": "0" * 64,
            },
            "representative_replay": {
                "url": "https://example.invalid/replace-with-replay-url",
                "manifest_sha256": "0" * 64,
            },
        },
        "results": {
            "episodes_per_training_seed": len(seeds),
            "aggregate_score": {
                "value": computed["score"],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "confidence": 0.95,
            },
            "achievement_rates": computed["achievement_rates"],
            "rescue_rate": computed["rescue_rate"],
        },
        "episode_records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Submission draft: {args.output.resolve()}")
    print("Complete the disclosure and artifact placeholders before opening a leaderboard PR.")


def _load_factory(spec: str) -> Callable[..., IslandPolicy]:
    if ":" not in spec:
        raise ValueError("--policy must use module:factory syntax.")
    module_name, attribute = spec.rsplit(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"Policy factory {spec!r} is not callable.")
    return cast(Callable[..., IslandPolicy], value)


def _create_policy(
    factory: Callable[..., IslandPolicy],
    *,
    training_seed: int | None,
    evaluation_seed: int,
) -> IslandPolicy:
    signature = inspect.signature(factory)
    if training_seed is not None:
        try:
            signature.bind(training_seed=training_seed, evaluation_seed=evaluation_seed)
        except TypeError as exc:
            raise TypeError(
                "A multi-seed factory must accept training_seed and evaluation_seed keywords."
            ) from exc
        return factory(training_seed=training_seed, evaluation_seed=evaluation_seed)
    try:
        signature.bind(evaluation_seed)
    except TypeError:
        signature.bind()
        return factory()
    return factory(evaluation_seed)


def _bootstrap_interval(
    records: list[dict[str, Any]], *, samples: int = 10_000
) -> tuple[float, float]:
    achievements = tuple(score_episode_records(records)["achievement_rates"])
    indicators = np.asarray(
        [[achievement in record["achievements"] for achievement in achievements] for record in records],
        dtype=np.float64,
    )
    rng = np.random.default_rng(0)
    indexes = rng.integers(0, len(records), size=(samples, len(records)))
    rates = indicators[indexes].mean(axis=1)
    scores = np.exp(np.log(rates + 0.01).mean(axis=1)) - 0.01
    return float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))


if __name__ == "__main__":
    main()
