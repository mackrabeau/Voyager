"""Installed-package smoke rollout for VoyagerIsland-v1."""

from __future__ import annotations

import argparse
import json
from typing import Any

from voyager.envs.island_v1 import parallel_env


def random_rollout(seed: int = 0) -> dict[str, Any]:
    """Complete one legal-random episode and return its public terminal summary."""

    environment = parallel_env(procedural=True)
    _observations, infos = environment.reset(seed=seed)
    for index, agent in enumerate(environment.possible_agents):
        environment.action_space(agent).seed(seed + index)
    steps = 0
    while environment.agents:
        actions = {
            agent: environment.action_space(agent).sample(
                mask=infos[agent]["action_mask"]
            )
            for agent in environment.agents
        }
        _observations, _rewards, _terminations, _truncations, infos = environment.step(
            actions
        )
        steps += 1
    terminal = next(iter(infos.values()))
    summary = {
        "environment": "VoyagerIsland-v1",
        "policy": "legal_random",
        "seed": seed,
        "world_steps": steps,
        "achievements": terminal["achievements"],
        "rescue_success": terminal["rescue_success"],
    }
    environment.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(random_rollout(args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
