"""Release-package tests that guard wheel-only resources and metadata."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from voyager.benchmark.island import load_island_seed_manifests


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
