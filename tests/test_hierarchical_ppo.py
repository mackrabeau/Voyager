from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voyager.envs.island import VoyagerIslandEnv
from voyager.sim.island_registry import IslandAction
from voyager.training.environments import ISLAND_V1_TRAINING_ENVIRONMENT
from voyager.training.island_goals import GOAL_COUNT, IslandGoal, goal_achieved
from voyager.training.island_reward import ISLAND_TRAINING_REWARD_V4


def _observation() -> dict[str, np.ndarray]:
    env = VoyagerIslandEnv(procedural=True)
    observations, _ = env.reset(seed=0)
    return {k: np.array(v, copy=True) for k, v in observations["agent_0"].items()}


def test_goal_checks_follow_observation_changes() -> None:
    before = _observation()
    after = {k: v.copy() for k, v in before.items()}
    after["inventory"][1] += 0.1  # +1 wood
    assert goal_achieved(IslandGoal.GATHER_WOOD, before, IslandAction.INTERACT, after)
    assert not goal_achieved(IslandGoal.GATHER_STONE, before, IslandAction.INTERACT, after)

    # Depositing needs the deposit action and a lighter inventory.
    assert goal_achieved(IslandGoal.DEPOSIT, after, IslandAction.DEPOSIT_ALL, before)
    assert not goal_achieved(IslandGoal.DEPOSIT, after, IslandAction.NOOP, before)

    built = {k: v.copy() for k, v in before.items()}
    built["public_board"][5] += 0.25
    assert goal_achieved(IslandGoal.BUILD, before, IslandAction.WORK_WORKBENCH, built)
    assert not goal_achieved(IslandGoal.BUILD, before, IslandAction.NOOP, built)

    sheltered = {k: v.copy() for k, v in before.items()}
    sheltered["self_state"][4] = 1.0
    assert goal_achieved(IslandGoal.SHELTER, before, IslandAction.USE_SHELTER, sheltered)
    assert not goal_achieved(IslandGoal.SHELTER, sheltered, IslandAction.NOOP, sheltered)


def test_semi_mdp_gae_uses_option_discounts() -> None:
    from voyager.training.hierarchical_ppo import _ManagerTransition, _semi_mdp_gae

    zeros = np.zeros(1, np.float32)
    first = _ManagerTransition("a", zeros, zeros, 0, 0.0, 0.0, 1.0, 0.5, 2.0, False)
    second = _ManagerTransition("a", zeros, zeros, 0, 0.0, 2.0, 1.0, 0.5, 0.0, True)
    advantages, returns = _semi_mdp_gae([first, second], gae_lambda=1.0)
    assert np.allclose(advantages, [1.0 + 0.5 * (1.0 - 2.0 + 2.0), -1.0])
    assert np.allclose(returns, advantages + [0.0, 2.0])


def test_trainer_runs_and_checkpoint_reloads(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    from voyager.training.hierarchical_ppo import (
        HierarchicalPPOConfig,
        HierarchicalPPOTrainer,
        load_hierarchical_policy,
    )
    from voyager.training.island_evaluation import evaluate_island_policy

    config = HierarchicalPPOConfig(
        total_steps=256,
        environment_id=ISLAND_V1_TRAINING_ENVIRONMENT,
        reward_contract=ISLAND_TRAINING_REWARD_V4,
        rollout_steps=32,
        num_agents=2,
        map_size=48,
        max_steps=1_200,
        seed=5,
        hidden_sizes=(16,),
        manager_hidden_sizes=(16,),
        minibatch_size=16,
        train_epochs=1,
        max_option_steps=4,
        manager_batch_size=16,
        manager_minibatch_size=8,
        manager_epochs=1,
        checkpoint_dir=str(tmp_path),
        checkpoint_every=0,
    )
    trainer = HierarchicalPPOTrainer(config)
    batch = trainer.collect_rollout()
    assert batch.observations.shape[1] == trainer.input_dim + GOAL_COUNT
    # Options last at most 4 ticks, so 32 ticks x 2 agents close at least 16.
    assert len(trainer.manager_buffer) >= 16

    stats = trainer.train()
    assert stats[-1].agent_steps == 256
    assert trainer.last_manager_losses
    assert trainer.goal_stats.chosen.sum() > 0

    policy = load_hierarchical_policy(tmp_path / "latest", deterministic=False, seed=0)
    episodes, _ = evaluate_island_policy(lambda seed: policy, seeds=(1,), procedural=True)
    assert len(episodes) == 1
    trainer.env.close()


def test_every_goal_is_reached_on_an_oracle_rescue() -> None:
    from voyager.policies.island_scripted import ScriptedIslandOracle

    env = VoyagerIslandEnv(procedural=True)
    observations, infos = env.reset(seed=20000)
    oracle = ScriptedIslandOracle()
    oracle.reset(tuple(env.possible_agents))
    reached: set[IslandGoal] = set()
    while env.agents:
        actions = oracle.act(observations, infos)
        next_observations, _, _, _, infos = env.step(actions)
        for agent_id, action in actions.items():
            if agent_id in next_observations:
                reached.update(
                    goal
                    for goal in IslandGoal
                    if goal_achieved(
                        goal, observations[agent_id], action, next_observations[agent_id]
                    )
                )
        observations = next_observations
    assert reached == set(IslandGoal)
