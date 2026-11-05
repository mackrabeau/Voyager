"""Abstract goals for hierarchical VoyagerIsland-v1 agents.

Each goal is checked from public observations only (before obs, action, after obs), so
the same check runs in training and in decentralized evaluation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import IntEnum

import numpy as np

from voyager.sim.island_registry import ISLAND_WORK_ACTIONS, IslandAction

Observation = Mapping[str, np.ndarray]

# Observation indices (see VoyagerIslandEnv._observation).
_FOOD, _WOOD, _STONE, _RAW_MEAT, _COOKED_MEAT = range(5)  # "inventory"
_HUNGER, _ENERGY, _SHELTERED = 1, 2, 4  # "self_state"
_BOARD_COOKED = 4  # "public_board"
_BOARD_PROGRESS = slice(5, 9)  # workbench, campfire, shelter, beacon progress

_EPS = 1e-6


class IslandGoal(IntEnum):
    GATHER_WOOD = 0
    GATHER_STONE = 1
    GATHER_FOOD = 2
    DEPOSIT = 3
    BUILD = 4
    CRAFT = 5
    HUNT = 6
    COOK = 7
    EAT = 8
    REST = 9
    SHELTER = 10


GOAL_COUNT = len(IslandGoal)
ISLAND_GOAL_VERSION = "voyager_island_goals_v1"


def _increased(before: Observation, after: Observation, key: str, index: int | slice) -> bool:
    return float(np.sum(after[key][index])) > float(np.sum(before[key][index])) + _EPS


def _decreased(before: Observation, after: Observation, key: str, index: int | slice) -> bool:
    return float(np.sum(after[key][index])) < float(np.sum(before[key][index])) - _EPS


def _cooked_total(observation: Observation) -> float:
    return float(observation["inventory"][_COOKED_MEAT] + observation["public_board"][_BOARD_COOKED])


_CHECKS: dict[IslandGoal, Callable[[Observation, int, Observation], bool]] = {
    IslandGoal.GATHER_WOOD: lambda b, a, n: _increased(b, n, "inventory", _WOOD),
    IslandGoal.GATHER_STONE: lambda b, a, n: _increased(b, n, "inventory", _STONE),
    IslandGoal.GATHER_FOOD: lambda b, a, n: _increased(b, n, "inventory", _FOOD),
    IslandGoal.DEPOSIT: lambda b, a, n: (
        a == IslandAction.DEPOSIT_ALL and _decreased(b, n, "inventory", slice(None))
    ),
    IslandGoal.BUILD: lambda b, a, n: (
        a in ISLAND_WORK_ACTIONS and _increased(b, n, "public_board", _BOARD_PROGRESS)
    ),
    IslandGoal.CRAFT: lambda b, a, n: _increased(b, n, "tools", slice(None)),
    IslandGoal.HUNT: lambda b, a, n: _increased(b, n, "inventory", _RAW_MEAT),
    IslandGoal.COOK: lambda b, a, n: _cooked_total(n) > _cooked_total(b) + _EPS,
    IslandGoal.EAT: lambda b, a, n: (
        a == IslandAction.EAT and _decreased(b, n, "self_state", _HUNGER)
    ),
    IslandGoal.REST: lambda b, a, n: (
        a == IslandAction.REST and _increased(b, n, "self_state", _ENERGY)
    ),
    IslandGoal.SHELTER: lambda b, a, n: (
        n["self_state"][_SHELTERED] > 0.5 and b["self_state"][_SHELTERED] <= 0.5
    ),
}


def goal_achieved(goal: int, before: Observation, action: int, after: Observation) -> bool:
    """True when the step before -> after (taking `action`) completes `goal`."""

    return _CHECKS[IslandGoal(goal)](before, int(action), after)


def goal_one_hot(goals: np.ndarray | list[int]) -> np.ndarray:
    return np.eye(GOAL_COUNT, dtype=np.float32)[np.asarray(goals, dtype=np.int64)]
