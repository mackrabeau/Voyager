"""Environment wrappers and registration helpers for Voyager."""

from voyager.envs.civilization import (
    CivilizationFlattenedActionWrapper,
    VoyagerCivilizationEnv,
)
from voyager.envs.civilization_v2 import (
    CivilizationV2FlattenedActionWrapper,
    VoyagerCivilizationV2Env,
)
from voyager.envs.island import VoyagerIslandCentralizedEnv, VoyagerIslandEnv
from voyager.envs.island_v1 import (
    centralized_env as make_island_centralized_env,
)
from voyager.envs.island_v1 import (
    env as make_island_aec_env,
)
from voyager.envs.island_v1 import (
    parallel_env as make_island_parallel_env,
)
from voyager.envs.parallel_env import VoyagerParallelEnv
from voyager.envs.placeholders import MultiAgentPlaceholderEnv
from voyager.envs.registration import register_envs
from voyager.envs.single_agent import VoyagerSingleAgentEnv

__all__ = [
    "CivilizationFlattenedActionWrapper",
    "CivilizationV2FlattenedActionWrapper",
    "MultiAgentPlaceholderEnv",
    "VoyagerCivilizationEnv",
    "VoyagerCivilizationV2Env",
    "VoyagerIslandCentralizedEnv",
    "VoyagerIslandEnv",
    "VoyagerParallelEnv",
    "VoyagerSingleAgentEnv",
    "make_island_aec_env",
    "make_island_centralized_env",
    "make_island_parallel_env",
    "register_envs",
]
