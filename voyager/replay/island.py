"""Record public-observation VoyagerIsland-v1 policies as Replay 2.3."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voyager.envs.island import (
    ISLAND_ENVIRONMENT_VERSION,
    ISLAND_OBSERVATION_VERSION,
    ISLAND_REWARD_VERSION,
    VoyagerIslandEnv,
)
from voyager.policies.island_scripted import ScriptedIslandOracle
from voyager.sim.constants import Terrain
from voyager.sim.island_achievements import ISLAND_ACHIEVEMENT_VERSION, ISLAND_ACHIEVEMENTS
from voyager.sim.island_registry import ISLAND_ACTION_VERSION, IslandAction
from voyager.training.island_evaluation import IslandPolicy

from .civilization import _state_hash
from .director import generate_camera_cues
from .recorder import _artifact_index, _registries, _state_delta, _world_snapshot
from .schema import (
    ISLAND_REPLAY_SCHEMA_VERSION,
    SNAPSHOT_INTERVAL,
    TIMELINE_CHUNK_SIZE,
    validate_manifest,
)
from .serialization import sha256_value, write_json, write_json_gz

ISLAND_ORACLE_REPLAY_ID = "island_benchmark_oracle_v1"
ISLAND_ORACLE_REPLAY_SEED = 3


def record_island_oracle_replay(
    output_root: str | Path = "runs/replays",
    *,
    seed: int = ISLAND_ORACLE_REPLAY_SEED,
    overwrite: bool = False,
) -> Path:
    """Atomically write and deeply validate the canonical Replay 2.3 artifact."""

    root = Path(output_root).resolve()
    target = root / ISLAND_ORACLE_REPLAY_ID
    if target.exists() and not overwrite:
        raise FileExistsError(f"Replay already exists at {target}.")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{ISLAND_ORACLE_REPLAY_ID}.", dir=root))
    try:
        _record_into(
            temporary,
            replay_id=ISLAND_ORACLE_REPLAY_ID,
            policy=ScriptedIslandOracle(),
            seed=seed,
            source={
                "policy_kind": "scripted",
                "policy_id": "island_public_observation_oracle_v1",
                "inference_mode": "decentralized_deterministic",
                "evaluation_seed": seed,
                "source_fingerprint": sha256_value(
                    {"policy": "island_public_observation_oracle_v1", "seed": seed}
                ),
            },
            tags=["stage7", "island-benchmark", "oracle", "rescue"],
            recorded_at=datetime.now(UTC).isoformat(),
            require_complete_oracle=True,
        )
        from .loader import ReplayLoader

        ReplayLoader(temporary).validate(deep=True)
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def record_island_policy_replay(
    output_root: str | Path,
    *,
    replay_id: str,
    policy: IslandPolicy,
    seed: int,
    source: Mapping[str, Any],
    tags: list[str],
    recorded_at: str,
    expected_episode: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically record one policy and require exact frozen-outcome agreement."""

    root = Path(output_root).resolve()
    target = root / replay_id
    if target.exists() and not overwrite:
        raise FileExistsError(f"Replay already exists at {target}.")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{replay_id}.", dir=root))
    try:
        terminal = _record_into(
            temporary,
            replay_id=replay_id,
            policy=policy,
            seed=seed,
            source=source,
            tags=tags,
            recorded_at=recorded_at,
            require_complete_oracle=False,
        )
        if expected_episode is not None:
            _require_expected_episode(terminal, expected_episode)
        from .loader import ReplayLoader

        ReplayLoader(temporary).validate(deep=True)
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _record_into(
    directory: Path,
    *,
    replay_id: str,
    policy: IslandPolicy,
    seed: int,
    source: Mapping[str, Any],
    tags: list[str],
    recorded_at: str,
    require_complete_oracle: bool,
) -> dict[str, Any]:
    env = VoyagerIslandEnv(procedural=True)
    observations, infos = env.reset(seed=seed)
    policy.reset(tuple(env.possible_agents))
    state = env.world.state
    assert state is not None
    initial_world = _world_snapshot(env)
    initial = {
        "tick": 0,
        "width": env.map_size,
        "height": env.map_size,
        "terrain": [[Terrain(int(cell)).name.lower() for cell in row] for row in state.terrain],
        **initial_world,
        "weather": {"kind": "clear", "storm_active": False},
    }
    initial["extensions"] = {
        **initial.get("extensions", {}),
        "state_hash": _state_hash(initial_world),
        "replay_profile": "voyager_island_v1",
    }
    previous = initial_world
    timeline: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, Any]] = {0: initial}
    all_events: list[dict[str, Any]] = []
    metrics: dict[str, list[float]] = defaultdict(list)
    reward_components: dict[str, float] = defaultdict(float)
    invalid_actions = 0
    dense_return = 0.0

    while env.agents:
        acting_observations = observations
        actions = policy.act(observations, infos)
        observations, rewards, _terminations, _truncations, step_infos = env.step(actions)
        infos.update(step_infos)
        state = env.world.state
        assert state is not None
        world = _world_snapshot(env)
        state_hash = _state_hash(world)
        world["extensions"] = {**world.get("extensions", {}), "state_hash": state_hash}
        action_records: list[dict[str, Any]] = []
        for agent_id, selected in sorted(actions.items()):
            info = step_infos[agent_id]
            invalid = bool(info.get("invalid_action", False))
            invalid_actions += int(invalid)
            action = IslandAction(selected)
            action_records.append(
                {
                    "agent_id": agent_id,
                    "role": "survivor",
                    "selected_action": action.name.lower(),
                    "selected_action_id": int(action),
                    "raw_action": action.name.lower(),
                    "raw_action_id": int(action),
                    "action_mask": acting_observations[agent_id]["action_mask"]
                    .astype(int)
                    .tolist(),
                    "invalid_probability_mass": 0.0,
                    "invalid": invalid,
                    "reward": float(rewards[agent_id]),
                    "reward_components": dict(info.get("reward_components", {})),
                    "dense_reward_components": dict(info.get("reward_components", {})),
                }
            )
            dense_return += float(rewards[agent_id])
            for name, value in info.get("reward_components", {}).items():
                reward_components[str(name)] += float(value)
        events = [dict(event) for event in state.events]
        all_events.extend(events)
        night = 200 <= state.step_count % 300 < 300
        record = {
            "tick": state.step_count,
            "actions": action_records,
            "events": events,
            "state_delta": _state_delta(previous, world),
            "achievements": sorted(
                {
                    achievement
                    for info in step_infos.values()
                    for achievement in info.get("new_achievements", ())
                }
            ),
            "weather": {"kind": "night" if night else "clear", "storm_active": False},
            "extensions": {"state_hash": state_hash},
        }
        timeline.append(record)
        previous = world
        if state.step_count % SNAPSHOT_INTERVAL == 0 or not env.agents:
            snapshots[state.step_count] = {
                "tick": state.step_count,
                **world,
                "weather": record["weather"],
            }
        metrics["tick"].append(float(state.step_count))
        survivors = float(sum(agent.alive for agent in state.agents.values()))
        metrics["active"].append(survivors)
        metrics["survivors"].append(survivors)
        metrics["achievements"].append(float(len(state.achievements)))
        metrics["camp_food"].append(float(state.camp.stockpile.get("food", 0)))
        metrics["shelter_progress"].append(float(state.structures["shelter"].progress))
        metrics["mean_health"].append(
            float(sum(agent.health for agent in state.agents.values()) / len(state.agents))
        )

    write_json_gz(directory / "initial.json.gz", initial)
    for index in range(0, len(timeline), TIMELINE_CHUNK_SIZE):
        records = timeline[index : index + TIMELINE_CHUNK_SIZE]
        start, end = records[0]["tick"], records[-1]["tick"]
        write_json_gz(
            directory / "timeline" / f"{start:06d}-{end:06d}.json.gz",
            {"start_tick": start, "end_tick": end, "records": records},
        )
    for tick, snapshot in sorted(snapshots.items()):
        write_json_gz(directory / "snapshots" / f"{tick:06d}.json.gz", snapshot)
    write_json_gz(
        directory / "metrics.json.gz",
        {
            "global": metrics,
            "reward_components": dict(sorted(reward_components.items())),
            "extensions": {},
        },
    )
    write_json_gz(
        directory / "ledger.json.gz",
        {
            "entries": state.ledger,
            "contributions": env.world.metrics()["contributions"],
            "conservation": env.world.reconcile_v2_ledger(),
        },
    )
    write_json(
        directory / "camera.json",
        generate_camera_cues(
            all_events,
            terminal_tick=state.step_count,
            camp=initial["camp"],
            agents=initial["agents"],
        ),
    )
    terminal_summary = {
        "world_steps": state.step_count,
        "agent_steps": sum(len(record["actions"]) for record in timeline),
        "survivors": sum(agent.alive for agent in state.agents.values()),
        "deaths": state.deaths,
        "dense_return": dense_return,
        "rescue_success": state.rescue_success,
        "achievements": sorted(state.achievements),
        "achievement_steps": dict(sorted(state.achievement_steps.items())),
        "camp_stockpile": dict(state.camp.stockpile),
        "structures": {
            name: {"complete": structure.complete, "progress": structure.progress}
            for name, structure in sorted(state.structures.items())
        },
        "shelter_progress": state.structures["shelter"].progress,
        "invalid_actions": invalid_actions,
        "conservation": env.world.reconcile_v2_ledger(),
        "state_hash": _state_hash(previous),
    }
    if require_complete_oracle and set(state.achievements) != set(ISLAND_ACHIEVEMENTS):
        raise RuntimeError("Canonical island replay must demonstrate every achievement.")
    if require_complete_oracle and invalid_actions:
        raise RuntimeError("Canonical island replay must contain zero invalid scripted actions.")
    if require_complete_oracle:
        terminal_summary["invalid_scripted_actions"] = invalid_actions
    if terminal_summary["conservation"]:
        raise RuntimeError("Canonical island replay must reconcile its resource ledger.")
    registries = _registries(initial)
    registries["actions"] = [
        {"id": action.name.lower(), "value": int(action), "label": action.name.title()}
        for action in IslandAction
    ]
    registries["achievements"] = [
        {"id": achievement, "label": achievement.replace("_", " ").title()}
        for achievement in ISLAND_ACHIEVEMENTS
    ]
    payload: dict[str, Any] = {
        "replay_id": replay_id,
        "versions": {
            "replay": ISLAND_REPLAY_SCHEMA_VERSION,
            "environment": ISLAND_ENVIRONMENT_VERSION,
            "scenario": state.scenario_id,
            "reward": ISLAND_REWARD_VERSION,
            "observation": ISLAND_OBSERVATION_VERSION,
            "action": ISLAND_ACTION_VERSION,
            "achievement": ISLAND_ACHIEVEMENT_VERSION,
        },
        "status": "complete",
        "source": dict(source),
        "environment_config": {
            "num_agents": 2,
            "map_size": 48,
            "max_steps": 1_200,
            "local_view_size": 7,
            "inventory_capacity": 10,
            "procedural": True,
        },
        "tick_rate": 4,
        "world_steps": state.step_count,
        "agent_steps": terminal_summary["agent_steps"],
        "recorded_at": recorded_at,
        "tags": tags,
        "terminal_summary": terminal_summary,
        "registries": registries,
        "artifacts": _artifact_index(directory),
        "extensions": {
            "privileged_policy": False,
            "playback_seconds": 300,
            "ledger_artifact": "ledger.json.gz",
        },
    }
    provisional = validate_manifest(payload)
    normalized = provisional.model_dump(mode="json")
    normalized.pop("manifest_sha256", None)
    normalized["manifest_sha256"] = sha256_value(normalized)
    write_json(directory / "manifest.json", validate_manifest(normalized).model_dump(mode="json"))
    return terminal_summary


def _require_expected_episode(
    terminal: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Reject a curated replay if it is not the frozen official episode."""

    comparable = {
        "achievements": sorted(terminal["achievements"]),
        "achievement_steps": dict(terminal["achievement_steps"]),
        "rescue_success": bool(terminal["rescue_success"]),
        "invalid_actions": int(terminal["invalid_actions"]),
        "final_active_agents": int(terminal["survivors"]),
        "actions": int(terminal["agent_steps"]),
        "conservation_errors": dict(terminal["conservation"]),
    }
    frozen = {
        "achievements": sorted(expected["achievements"]),
        "achievement_steps": dict(expected["achievement_steps"]),
        "rescue_success": bool(expected["rescue_success"]),
        "invalid_actions": int(expected["invalid_actions"]),
        "final_active_agents": int(expected["final_active_agents"]),
        "actions": int(expected["actions"]),
        "conservation_errors": dict(expected["conservation_errors"]),
    }
    if comparable != frozen:
        raise RuntimeError(
            "Recorded replay does not reproduce its frozen official episode: "
            f"recorded={comparable!r}, frozen={frozen!r}."
        )
