"""Validation and deterministic rendering for data-only leaderboard submissions."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voyager.benchmark.island import ISLAND_TEST_SEEDS, load_island_seed_manifests
from voyager.sim.island_achievements import ISLAND_ACHIEVEMENTS, geometric_mean_score

SCHEMA_VERSION = "voyager_leaderboard_submission_v1"
BENCHMARK_ID = "VoyagerIsland-v1"
ENVIRONMENT_VERSION = "voyager_island_v1"
OFFICIAL_BUDGET = 1_000_000
CATEGORIES = (
    "learned_official_budget",
    "external_knowledge",
    "scripted_reference",
)


def validate_submission(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate the JSON schema and frozen benchmark semantics."""

    from jsonschema import Draft202012Validator, FormatChecker

    errors = sorted(
        Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        rendered = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ValueError(f"Schema validation failed: {rendered}")

    if payload["status"] == "draft":
        raise ValueError("Draft evaluations must be completed before leaderboard submission.")
    benchmark = payload["benchmark"]
    manifests = load_island_seed_manifests()
    if benchmark["test_manifest_sha256"] != manifests["test"].sha256:
        raise ValueError("Submission does not pin the frozen test manifest.")

    method = payload["method"]
    category = method["category"]
    training_seeds = method["training_seeds"]
    budget = method["training_budget_agent_steps"]
    if category == "learned_official_budget":
        if len(training_seeds) != 3 or len(set(training_seeds)) != 3:
            raise ValueError("Official-budget learned submissions require three training seeds.")
        if budget != OFFICIAL_BUDGET:
            raise ValueError("Official-budget learned submissions require exactly 1M agent steps.")
    elif category == "scripted_reference" and (training_seeds or budget != 0):
        raise ValueError("Scripted references must declare no training seeds and a zero budget.")
    elif category == "external_knowledge" and training_seeds and len(set(training_seeds)) < 3:
        raise ValueError("Learned external-knowledge submissions require three training seeds.")

    _reject_placeholders(payload)
    records = payload["episode_records"]
    _validate_episode_seeds(records, training_seeds)
    computed = score_episode_records(records)
    reported = payload["results"]
    _close(float(reported["aggregate_score"]["value"]), computed["score"], "aggregate score")
    _close(float(reported["rescue_rate"]), computed["rescue_rate"], "rescue rate")
    for achievement in ISLAND_ACHIEVEMENTS:
        _close(
            float(reported["achievement_rates"][achievement]),
            computed["achievement_rates"][achievement],
            f"achievement rate {achievement}",
        )
    interval = reported["aggregate_score"]
    if not float(interval["ci_low"]) <= computed["score"] <= float(interval["ci_high"]):
        raise ValueError("Aggregate score is outside its reported confidence interval.")
    if not math.isclose(
        computed["achievement_rates"]["rescue_both"],
        computed["rescue_rate"],
        abs_tol=1e-12,
    ):
        raise ValueError("rescue_both achievement rate and rescue rate differ.")


def score_episode_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute the public point estimate from episode outcomes only."""

    if not records:
        raise ValueError("At least one episode record is required.")
    rates = {
        achievement: sum(achievement in record["achievements"] for record in records)
        / len(records)
        for achievement in ISLAND_ACHIEVEMENTS
    }
    return {
        "score": geometric_mean_score(rates),
        "achievement_rates": rates,
        "rescue_rate": sum(bool(record["rescue_success"]) for record in records)
        / len(records),
    }


def load_submissions(directory: Path, schema_path: Path) -> list[dict[str, Any]]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    submissions: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_submission(payload, schema)
        if path.stem != payload["submission_id"]:
            raise ValueError(f"{path.name} must match submission_id={payload['submission_id']!r}.")
        submissions.append(payload)
    if not submissions:
        raise ValueError("No leaderboard submissions found.")
    return submissions


def render_generated_tables(submissions: Sequence[Mapping[str, Any]]) -> str:
    """Render the three non-comparable leaderboard sections."""

    sections = (
        (
            "learned_official_budget",
            "Learned agents under the official interaction budget",
            "Ranked by the frozen public score. Three training seeds and 1M agent transitions are required.",
        ),
        (
            "external_knowledge",
            "External-knowledge / foundation-model agents",
            "Reported separately because external data or pretrained models change the comparison.",
        ),
        (
            "scripted_reference",
            "Scripted references and oracle (not ranked)",
            "Diagnostic comparators only; these entries are excluded from learned-agent ranking.",
        ),
    )
    blocks: list[str] = []
    for category, title, note in sections:
        rows = [item for item in submissions if item["method"]["category"] == category]
        if category != "scripted_reference":
            rows.sort(key=lambda item: float(item["results"]["aggregate_score"]["value"]), reverse=True)
        blocks.extend([f"### {title}", "", note, ""])
        if not rows:
            blocks.extend(["_No accepted submissions yet._", ""])
            continue
        blocks.extend(
            [
                "| Method | Score (95% CI) | Rescue | Replay | Policy source |",
                "|---|---:|---:|---|---|",
            ]
        )
        for item in rows:
            score = item["results"]["aggregate_score"]
            method = item["method"]
            artifacts = item["artifacts"]
            commit = method["policy_source"]["commit"]
            blocks.append(
                "| {name} | {value:.3f} ({low:.3f}–{high:.3f}) | {rescue:.0%} | "
                "[replay]({replay}) | [source@{short}]({source}/tree/{commit}) |".format(
                    name=str(method["name"]).replace("|", "\\|"),
                    value=float(score["value"]),
                    low=float(score["ci_low"]),
                    high=float(score["ci_high"]),
                    rescue=float(item["results"]["rescue_rate"]),
                    replay=artifacts["representative_replay"]["url"],
                    source=method["policy_source"]["repository_url"].rstrip("/"),
                    short=commit[:8],
                    commit=commit,
                )
            )
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def update_readme(readme: str, generated: str) -> str:
    start = "<!-- BEGIN GENERATED LEADERBOARD -->"
    end = "<!-- END GENERATED LEADERBOARD -->"
    if start not in readme or end not in readme:
        raise ValueError("Leaderboard README is missing generated-table markers.")
    prefix, remainder = readme.split(start, 1)
    _old, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n\n{generated.rstrip()}\n\n{end}{suffix}"


def _validate_episode_seeds(
    records: Sequence[Mapping[str, Any]], training_seeds: Sequence[int]
) -> None:
    expected = set(ISLAND_TEST_SEEDS)
    grouped: dict[int | None, list[int]] = defaultdict(list)
    for record in records:
        grouped[record["training_seed"]].append(int(record["evaluation_seed"]))
    expected_groups: set[int | None] = set(training_seeds) if training_seeds else {None}
    if set(grouped) != expected_groups:
        raise ValueError("Episode records do not match the declared training seeds.")
    for training_seed, evaluation_seeds in grouped.items():
        if len(evaluation_seeds) != len(expected) or set(evaluation_seeds) != expected:
            raise ValueError(
                f"Training seed {training_seed!r} must contain every frozen test seed exactly once."
            )


def _reject_placeholders(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    if "example.invalid" in serialized or "REPLACE BEFORE SUBMISSION" in serialized:
        raise ValueError("Submission still contains voyager-eval draft placeholders.")
    if "0" * 64 in serialized:
        raise ValueError("Submission contains a placeholder SHA-256 digest.")


def _close(reported: float, computed: float, label: str) -> None:
    if not math.isclose(reported, computed, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Reported {label} {reported} does not recompute to {computed}.")
