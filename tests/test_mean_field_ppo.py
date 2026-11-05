from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voyager.training.environments import ISLAND_V1_TRAINING_ENVIRONMENT
from voyager.training.island_reward import ISLAND_TRAINING_REWARD_V4
from voyager.training.mean_field_ppo import (
    CRITIC_WEIGHTS_FILE,
    MeanFieldPPOConfig,
    mean_of_others,
)


def test_mean_of_others_excludes_self() -> None:
    rows = np.eye(3, dtype=np.float32)
    assert np.allclose(mean_of_others(rows), [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    assert np.allclose(mean_of_others(rows[:1]), 0.0)


def test_config_rejects_unknown_advantage_mode() -> None:
    with pytest.raises(ValueError, match="advantage_mode"):
        MeanFieldPPOConfig(advantage_mode="nope").validate()  # type: ignore[arg-type]


@pytest.mark.parametrize("advantage_mode", ["counterfactual", "gae"])
def test_trainer_runs_and_checkpoint_loads_as_feed_forward(
    tmp_path: Path, advantage_mode: str
) -> None:
    pytest.importorskip("tensorflow")
    from voyager.training.checkpoints import load_policy_checkpoint
    from voyager.training.island_evaluation import FeedForwardCheckpointIslandPolicy
    from voyager.training.mean_field_ppo import MeanFieldPPOTrainer

    config = MeanFieldPPOConfig(
        total_steps=64,
        environment_id=ISLAND_V1_TRAINING_ENVIRONMENT,
        reward_contract=ISLAND_TRAINING_REWARD_V4,
        rollout_steps=16,
        num_agents=2,
        map_size=48,
        max_steps=1_200,
        seed=3,
        hidden_sizes=(16,),
        critic_hidden_sizes=(16,),
        minibatch_size=8,
        train_epochs=1,
        checkpoint_dir=str(tmp_path),
        checkpoint_every=0,
        advantage_mode=advantage_mode,  # type: ignore[arg-type]
        team_reward=True,
    )
    trainer = MeanFieldPPOTrainer(config)
    batch = trainer.collect_rollout()
    assert batch.mean_actions.shape == (batch.actions.shape[0], trainer.action_count)
    # With two agents, a_bar_i is exactly the partner's one-hot action.
    assert np.allclose(batch.mean_actions[0::2].argmax(axis=1), batch.actions[1::2])
    assert np.isfinite(batch.advantages).all() and np.isfinite(batch.returns).all()

    stats = trainer.train()
    assert stats[-1].agent_steps == 64
    latest = tmp_path / "latest"
    assert (latest / CRITIC_WEIGHTS_FILE).exists()
    _, metadata = load_policy_checkpoint(latest)
    assert metadata["algorithm"] == "mean_field_ppo"
    FeedForwardCheckpointIslandPolicy(latest, deterministic=True, seed=0)
    trainer.env.close()
