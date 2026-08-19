"""Named public factories for the frozen VoyagerIsland-v1 interfaces."""

from __future__ import annotations

from typing import Any

from pettingzoo.utils.conversions import parallel_to_aec

from voyager.envs.island import VoyagerIslandCentralizedEnv, VoyagerIslandEnv


def parallel_env(**kwargs: Any) -> VoyagerIslandEnv:
    """Create the canonical PettingZoo parallel environment."""

    return VoyagerIslandEnv(**kwargs)


def env(**kwargs: Any) -> Any:
    """Create the canonical PettingZoo AEC environment."""

    return parallel_to_aec(parallel_env(**kwargs))


def centralized_env(**kwargs: Any) -> VoyagerIslandCentralizedEnv:
    """Create the checker-compliant centralized Gymnasium environment."""

    return VoyagerIslandCentralizedEnv(**kwargs)


__all__ = ["centralized_env", "env", "parallel_env"]
