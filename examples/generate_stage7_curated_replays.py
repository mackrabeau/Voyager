"""Generate the frozen Stage 7 curated Replay 2.3 collection."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from voyager.benchmark.island_results import load_frozen_island_results
from voyager.policies.island_scripted import ScriptedIslandOracle
from voyager.replay.island import record_island_policy_replay
from voyager.replay.island_curated import (
    build_curated_island_selection,
    selection_fingerprint,
    sha256_file,
)
from voyager.replay.loader import ReplayLoader
from voyager.replay.serialization import write_json
from voyager.training.island_evaluation import (
    FeedForwardCheckpointIslandPolicy,
    IslandPolicy,
    LegalRandomIslandPolicy,
    RecurrentCheckpointIslandPolicy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("results/stage7/procedural_baselines_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/replays/stage7_curated_v1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(args.experiment_root, args.output_root, overwrite=args.overwrite)


def generate(experiment_root: Path, output_root: Path, *, overwrite: bool) -> Path:
    experiment = experiment_root.resolve()
    destination = output_root.resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Curated replay collection already exists: {destination}")
    selection = build_curated_island_selection(experiment)
    frozen = load_frozen_island_results(experiment)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for spec in selection["replays"]:
            _record_spec(temporary, experiment, frozen.git_revision, selection, spec)
        write_json(temporary / "selection.json", selection)
        artifacts = [
            {
                "path": path.relative_to(temporary).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        ]
        write_json(
            temporary / "collection.json",
            {
                "schema_version": selection["schema_version"],
                "benchmark_id": selection["benchmark_id"],
                "git_revision": selection["git_revision"],
                "replay_ids": [spec["replay_id"] for spec in selection["replays"]],
                "artifacts": artifacts,
            },
        )
        for spec in selection["replays"]:
            ReplayLoader(temporary / spec["replay_id"]).validate(deep=True)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Stage 7 curated replays: {destination}")
    for spec in selection["replays"]:
        outcome = spec["expected_episode"]
        print(
            f"- {spec['replay_id']}: seed={spec['evaluation_seed']} "
            f"achievements={len(outcome['achievements'])} "
            f"rescue={outcome['rescue_success']}"
        )
    return destination


def _record_spec(
    output_root: Path,
    experiment: Path,
    git_revision: str,
    selection: dict[str, Any],
    spec: dict[str, Any],
) -> None:
    seed = int(spec["evaluation_seed"])
    policy_id = str(spec["policy_id"])
    checkpoint = spec.get("checkpoint")
    if policy_id == "scripted_oracle":
        policy: IslandPolicy = ScriptedIslandOracle()
    elif policy_id == "legal_random":
        policy = LegalRandomIslandPolicy(seed)
    elif policy_id == "feed_forward_ppo":
        policy = FeedForwardCheckpointIslandPolicy(
            experiment / str(checkpoint), deterministic=False, seed=seed
        )
    elif policy_id == "recurrent_ppo":
        policy = RecurrentCheckpointIslandPolicy(
            experiment / str(checkpoint), deterministic=False, seed=seed
        )
    else:
        raise ValueError(f"Unsupported curated policy: {policy_id}")

    source = {
        "policy_kind": spec["policy_kind"],
        "policy_id": policy_id,
        "inference_mode": spec["inference_mode"],
        "evaluation_seed": seed,
        "git_revision": git_revision,
        "benchmark_id": selection["benchmark_id"],
        "benchmark_episode": {
            "role": spec["role"],
            "selection_rule": spec["selection_rule"],
            "source_artifact": spec["source_artifact"],
            "source_artifact_sha256": spec["source_artifact_sha256"],
            "use_for_quantitative_claims": False,
        },
        "source_fingerprint": selection_fingerprint(spec, git_revision),
    }
    if checkpoint is not None:
        source.update(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": spec["checkpoint_sha256"],
                "training_seed": spec["training_seed"],
            }
        )
    tags = ["stage7", "island-benchmark", "curated", str(spec["role"])]
    if spec["role"] == "feed_forward_progression":
        tags.extend(["showcase", "learned", "matched-comparison"])
    elif spec["role"] == "matched_legal_random":
        tags.append("matched-comparison")
    elif spec["policy_kind"] == "learned":
        tags.append("learned")
        if spec["role"] == "representative_learned_failure":
            tags.append("failure-case")
    elif spec["role"] == "oracle_rescue":
        tags.extend(["oracle", "rescue"])
    record_island_policy_replay(
        output_root,
        replay_id=str(spec["replay_id"]),
        policy=policy,
        seed=seed,
        source=source,
        tags=tags,
        recorded_at=str(selection["recorded_at"]),
        expected_episode=spec["expected_episode"],
    )


if __name__ == "__main__":
    main()
