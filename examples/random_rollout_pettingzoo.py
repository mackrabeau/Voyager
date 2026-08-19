"""Run one dependency-free policy rollout through the PettingZoo parallel API."""

from voyager.envs.island_v1 import parallel_env


def main() -> None:
    environment = parallel_env(procedural=True)
    _observations, infos = environment.reset(seed=0)
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
    environment.close()
    print(f"PettingZoo rollout complete: {steps} world steps")


if __name__ == "__main__":
    main()
