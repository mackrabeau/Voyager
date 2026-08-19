"""Release-package tests that guard wheel-only resources and metadata."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import gymnasium as gym
from fastapi.testclient import TestClient
from pettingzoo.test import parallel_api_test

import voyager
from voyager.benchmark.island import load_island_seed_manifests
from voyager.envs.island_v1 import centralized_env, parallel_env
from voyager.server.app import PACKAGE_FRONTEND, PACKAGE_REPLAYS, create_app


def test_packaged_seed_manifests_match_repository_contracts() -> None:
    package_root = resources.files("voyager.data.manifests")
    repository_root = Path("benchmarks/manifests")
    for filename in ("island_v1_train.json", "island_v1_dev.json", "island_v1_test.json"):
        assert package_root.joinpath(filename).read_bytes() == (
            repository_root / filename
        ).read_bytes()


def test_default_manifest_loader_uses_packaged_resources() -> None:
    manifests = load_island_seed_manifests()
    assert manifests["train"].path.name == "island_v1_train.json"
    assert manifests["development"].path.name == "island_v1_dev.json"
    assert manifests["test"].path.name == "island_v1_test.json"


def test_named_public_factories_are_checker_compliant() -> None:
    parallel_api_test(parallel_env(procedural=False), num_cycles=25)
    gym.utils.env_checker.check_env(centralized_env(procedural=False))
    registered = gym.make("VoyagerIslandCentralized-v1", procedural=False)
    observation, _info = registered.reset(seed=0)
    assert registered.observation_space.contains(observation)
    registered.close()


def test_pettingzoo_benchmark_id_is_not_misregistered_with_gymnasium() -> None:
    assert voyager.__version__ == "0.1.0"
    assert "VoyagerIsland-v1" not in gym.envs.registry


def test_packaged_viewer_and_stage7_replays_are_self_contained() -> None:
    assert (PACKAGE_FRONTEND / "index.html").is_file()
    assert (PACKAGE_REPLAYS / "collection.json").is_file()
    client = TestClient(
        create_app(replay_roots=[("packaged-stage7", PACKAGE_REPLAYS)])
    )
    assert client.get("/").status_code == 200
    health = client.get("/healthz").json()
    assert health == {"status": "ok", "replays": 5}
