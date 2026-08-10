"""Deterministic curation rules for the frozen Stage 7 replay collection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voyager.benchmark.island_results import (
    ISLAND_RESULTS_BENCHMARK_ID,
    load_frozen_island_results,
)
from voyager.replay.serialization import sha256_value

CURATED_ISLAND_REPLAY_VERSION = "voyager_island_curated_replays_v1"
CURATED_RECORDED_AT = "2026-08-07T00:00:00+00:00"

_PROGRESSION_TIERS = (
    ("rescue_both",),
    ("build_beacon",),
    ("build_shelter", "cook_meat", "hunt_deer"),
    ("build_campfire",),
    ("craft_axe", "craft_spear"),
    ("build_workbench",),
    ("deposit_wood", "deposit_stone"),
    ("collect_food", "collect_wood", "collect_stone"),
)


def build_curated_island_selection(experiment_root: str | Path) -> dict[str, Any]:
    """Select five honest examples from the immutable official evaluation matrix."""

    root = Path(experiment_root).resolve()
    frozen = load_frozen_island_results(root)
    final = root / "final"
    random_payload = _read(final / "legal_random.json")
    oracle_payload = _read(final / "scripted_oracle.json")
    learned = {
        (algorithm, seed): _read(final / "learned" / algorithm / f"seed_{seed}.json")
        for algorithm in ("feed_forward_ppo", "recurrent_ppo")
        for seed in (0, 1, 2)
    }

    feed_forward = _strongest_progression(learned, algorithm="feed_forward_ppo")
    recurrent = _strongest_progression(learned, algorithm="recurrent_ppo")
    oracle = min(
        (
            episode
            for episode in _episodes(oracle_payload)
            if episode["rescue_success"]
            and len(episode["achievements"]) == 15
            and int(episode["invalid_actions"]) == 0
            and not episode["conservation_errors"]
        ),
        key=lambda episode: int(episode["seed"]),
    )
    matched_random = next(
        episode
        for episode in _episodes(random_payload)
        if int(episode["seed"]) == int(feed_forward["episode"]["seed"])
    )
    weak_feed_forward_seed = min(
        range(3),
        key=lambda seed: float(
            learned[("feed_forward_ppo", seed)]["evaluation"]["stochastic"]["summary"][
                "achievement_geometric_mean"
            ]
        ),
    )
    failure = _representative_episode(
        _episodes(
            learned[("feed_forward_ppo", weak_feed_forward_seed)]["evaluation"]["stochastic"]
        )
    )

    specs = [
        _comparator_spec(
            "island_oracle_rescue_v1",
            "oracle_rescue",
            "scripted_oracle",
            oracle,
            "lowest held-out seed with rescue, all achievements, and a reconciled ledger",
            frozen.source_hashes["final/scripted_oracle.json"],
        ),
        _comparator_spec(
            "island_legal_random_matched_v1",
            "matched_legal_random",
            "legal_random",
            matched_random,
            "same held-out island as the feed-forward progression replay",
            frozen.source_hashes["final/legal_random.json"],
        ),
        _learned_spec(root, frozen, learned, feed_forward, "feed_forward_progression"),
        _learned_spec(root, frozen, learned, recurrent, "recurrent_progression"),
        _learned_spec(
            root,
            frozen,
            learned,
            {"training_seed": weak_feed_forward_seed, "episode": failure},
            "representative_learned_failure",
        ),
    ]
    return {
        "schema_version": CURATED_ISLAND_REPLAY_VERSION,
        "benchmark_id": ISLAND_RESULTS_BENCHMARK_ID,
        "git_revision": frozen.git_revision,
        "test_manifest_sha256": frozen.test_manifest_sha256,
        "recorded_at": CURATED_RECORDED_AT,
        "use_for_quantitative_claims": False,
        "selection_note": (
            "These post-evaluation examples are selected for inspection only; all benchmark "
            "claims remain computed from all 100 held-out islands and three training seeds."
        ),
        "replays": specs,
    }


def _strongest_progression(
    learned: Mapping[tuple[str, int], Mapping[str, Any]], *, algorithm: str
) -> dict[str, Any]:
    candidates = [
        (seed, episode)
        for seed in range(3)
        for episode in _episodes(learned[(algorithm, seed)]["evaluation"]["stochastic"])
    ]
    seed, episode = max(candidates, key=lambda row: _progression_key(row[1], row[0]))
    return {"training_seed": seed, "episode": episode}


def _progression_key(episode: Mapping[str, Any], training_seed: int) -> tuple[int, ...]:
    achievements = {str(value) for value in episode["achievements"]}
    deepest = max(
        (
            len(_PROGRESSION_TIERS) - index
            for index, tier in enumerate(_PROGRESSION_TIERS)
            if achievements.intersection(tier)
        ),
        default=0,
    )
    return (
        deepest,
        len(achievements),
        int(bool(episode["joint_survival"])),
        -int(episode["invalid_actions"]),
        -training_seed,
        -int(episode["seed"]),
    )


def _representative_episode(episodes: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    ordered_counts = sorted(len(episode["achievements"]) for episode in episodes)
    median_count = ordered_counts[(len(ordered_counts) - 1) // 2]
    return min(
        episodes,
        key=lambda episode: (
            abs(len(episode["achievements"]) - median_count),
            int(episode["invalid_actions"]),
            int(episode["seed"]),
        ),
    )


def _learned_spec(
    root: Path,
    frozen: Any,
    learned: Mapping[tuple[str, int], Mapping[str, Any]],
    selected: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    algorithm = "recurrent_ppo" if role.startswith("recurrent") else "feed_forward_ppo"
    training_seed = int(selected["training_seed"])
    episode = selected["episode"]
    payload = learned[(algorithm, training_seed)]
    relative_checkpoint = Path(
        str(payload["evaluation"]["checkpoint"])
    ).resolve().relative_to(root).as_posix()
    replay_id = {
        "feed_forward_progression": "island_feed_forward_progression_v1",
        "recurrent_progression": "island_recurrent_progression_v1",
        "representative_learned_failure": "island_learned_failure_v1",
    }[role]
    source_key = f"final/learned/{algorithm}/seed_{training_seed}.json"
    return {
        "replay_id": replay_id,
        "role": role,
        "policy_kind": "learned",
        "policy_id": algorithm,
        "inference_mode": "seeded_stochastic",
        "training_seed": training_seed,
        "evaluation_seed": int(episode["seed"]),
        "checkpoint": relative_checkpoint,
        "checkpoint_sha256": frozen.runs[(algorithm, training_seed)]["weights_sha256"],
        "source_artifact": source_key,
        "source_artifact_sha256": frozen.source_hashes[source_key],
        "selection_rule": (
            "lexicographically deepest technology tier, achievement count, joint survival, "
            "lower invalid count, training seed, then island seed"
            if role != "representative_learned_failure"
            else "median-achievement episode from the weakest feed-forward training seed"
        ),
        "expected_episode": _outcome(episode),
    }


def _comparator_spec(
    replay_id: str,
    role: str,
    policy_id: str,
    episode: Mapping[str, Any],
    rule: str,
    source_sha256: str,
) -> dict[str, Any]:
    return {
        "replay_id": replay_id,
        "role": role,
        "policy_kind": "scripted" if policy_id == "scripted_oracle" else "random",
        "policy_id": policy_id,
        "inference_mode": "decentralized_deterministic" if policy_id == "scripted_oracle" else "seeded_stochastic",
        "evaluation_seed": int(episode["seed"]),
        "source_artifact": f"final/{policy_id}.json" if policy_id == "scripted_oracle" else "final/legal_random.json",
        "source_artifact_sha256": source_sha256,
        "selection_rule": rule,
        "expected_episode": _outcome(episode),
    }


def _outcome(episode: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(episode["seed"]),
        "achievements": list(episode["achievements"]),
        "achievement_steps": dict(episode["achievement_steps"]),
        "joint_survival": bool(episode["joint_survival"]),
        "rescue_success": bool(episode["rescue_success"]),
        "invalid_actions": int(episode["invalid_actions"]),
        "actions": int(episode["actions"]),
        "final_active_agents": int(episode["final_active_agents"]),
        "conservation_errors": dict(episode["conservation_errors"]),
    }


def selection_fingerprint(spec: Mapping[str, Any], git_revision: str) -> str:
    return sha256_value({"selection": dict(spec), "git_revision": git_revision})


def _episodes(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = payload["episodes"]
    if not isinstance(value, list):
        raise TypeError("Official evaluation episodes must be a list.")
    return value


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object at {path}.")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
