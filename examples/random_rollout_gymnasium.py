"""Run one dependency-free policy rollout through the centralized Gymnasium API."""

import gymnasium as gym

import voyager  # noqa: F401  # Registers VoyagerIslandCentralized-v1.


def main() -> None:
    environment = gym.make("VoyagerIslandCentralized-v1")
    _observation, _info = environment.reset(seed=0)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _observation, _reward, terminated, truncated, _info = environment.step(
            environment.action_space.sample()
        )
        steps += 1
    environment.close()
    print(f"Gymnasium rollout complete: {steps} world steps")


if __name__ == "__main__":
    main()
