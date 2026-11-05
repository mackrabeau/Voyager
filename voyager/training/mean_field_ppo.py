"""Mean-field counterfactual PPO (MF-PPO) for Voyager.

Decentralized actor pi(a_i | o_i), shared by all agents (same network as PPOTrainer).
Mean-field critic Q(o_i, a_i, a_bar_i), where a_bar_i is the mean one-hot action of the
other agents (Yang et al., 2018). Policy advantage is either:

- "counterfactual": Q(o, a_i, a_bar) - sum_b pi(b|o) Q(o, b, a_bar)   (COMA-style)
- "gae":            GAE on V(o, a_bar) = sum_b pi(b|o) Q(o, b, a_bar)

The critic is only used in training; evaluation runs the actor alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from voyager.training.advantages import compute_gae
from voyager.training.masking import stack_action_masks
from voyager.training.model import build_mean_field_critic
from voyager.training.obs import flatten_observations
from voyager.training.ppo import PPOConfig, PPOTrainer, _normalize_advantages

AdvantageMode = Literal["counterfactual", "gae"]
CRITIC_WEIGHTS_FILE = "critic.weights.h5"


@dataclass(frozen=True, slots=True)
class MeanFieldPPOConfig(PPOConfig):
    """PPOConfig plus mean-field critic options."""

    advantage_mode: AdvantageMode = "counterfactual"
    team_reward: bool = False
    critic_hidden_sizes: tuple[int, ...] = (128, 128)
    critic_learning_rate: float = 3e-4

    def validate(self) -> None:
        PPOConfig.validate(self)
        if self.advantage_mode not in ("counterfactual", "gae"):
            raise ValueError(f"Unsupported advantage_mode: {self.advantage_mode!r}.")
        if not self.critic_hidden_sizes:
            raise ValueError("critic_hidden_sizes must not be empty.")
        if self.critic_learning_rate <= 0.0:
            raise ValueError("critic_learning_rate must be positive.")


@dataclass(frozen=True, slots=True)
class MeanFieldRolloutBatch:
    observations: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    mean_actions: np.ndarray
    old_log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray  # TD(lambda) targets for Q(o, a, a_bar)
    agent_ids: tuple[str, ...]


def mean_of_others(rows: np.ndarray) -> np.ndarray:
    """Row i -> mean of every other row. A lone agent gets zeros."""

    count = rows.shape[0]
    if count <= 1:
        return np.zeros_like(rows, dtype=np.float32)
    return ((rows.sum(axis=0, keepdims=True) - rows) / (count - 1)).astype(np.float32)


class MeanFieldPPOTrainer(PPOTrainer):
    """PPO with a decentralized actor and a mean-field action-value critic."""

    config: MeanFieldPPOConfig

    def __init__(self, config: MeanFieldPPOConfig) -> None:
        super().__init__(config)
        # self.model is the actor. It keeps the standard actor-critic layout so the
        # existing feed-forward evaluation and checkpoint loaders work unchanged;
        # its value head is simply never trained.
        self.critic = build_mean_field_critic(
            input_dim=self.input_dim,
            action_count=self.action_count,
            hidden_sizes=config.critic_hidden_sizes,
            seed=config.seed + 1,
        )
        self.critic_optimizer = self.tf.keras.optimizers.Adam(
            learning_rate=config.critic_learning_rate
        )

    def collect_rollout(self, max_agent_steps: int | None = None) -> MeanFieldRolloutBatch:  # type: ignore[override]
        if max_agent_steps is not None and max_agent_steps <= 0:
            raise ValueError("max_agent_steps must be positive when provided.")

        rows: dict[str, list[Any]] = {
            key: []
            for key in (
                "obs", "mask", "action", "mean_action", "log_prob", "q_taken",
                "baseline", "reward", "done", "next_value", "agent",
            )
        }

        for _step in range(self.config.rollout_steps):
            if not self.env.agents:
                self._reset_env()

            agent_ids = tuple(self.env.agents)
            if (
                max_agent_steps is not None
                and rows["obs"]
                and len(rows["obs"]) + len(agent_ids) > max_agent_steps
            ):
                break

            started = time.perf_counter()
            flat_obs = flatten_observations(self.observations, agent_ids, self.observation_encoder)
            action_masks = self._masks(self.infos, agent_ids)
            logits, _ = self.model(flat_obs, training=False)
            masked_logits = self._mask_logits(logits, action_masks)
            actions = self._sample_actions(masked_logits)
            log_probs = self._selected_log_probs(masked_logits, actions).numpy()
            probabilities = self.tf.nn.softmax(masked_logits).numpy()

            mean_actions = mean_of_others(np.eye(self.action_count, dtype=np.float32)[actions])
            q_values = self.critic([flat_obs, mean_actions], training=False).numpy()
            q_taken = q_values[np.arange(len(agent_ids)), actions]
            baseline = np.sum(probabilities * q_values, axis=1)
            self.timing_seconds["actor_inference"] += time.perf_counter() - started

            started = time.perf_counter()
            next_observations, rewards, terminations, truncations, step_infos = self.env.step(
                {agent_id: int(actions[i]) for i, agent_id in enumerate(agent_ids)}
            )
            self.timing_seconds["environment_step"] += time.perf_counter() - started
            self.world_steps += 1

            started = time.perf_counter()
            next_values = self._bootstrap_values(next_observations, step_infos)
            self.timing_seconds["next_value_inference"] += time.perf_counter() - started

            team_reward = float(np.mean([rewards[a] for a in agent_ids]))
            for i, agent_id in enumerate(agent_ids):
                done = bool(terminations[agent_id] or truncations[agent_id])
                rows["obs"].append(flat_obs[i])
                rows["mask"].append(action_masks[i])
                rows["action"].append(int(actions[i]))
                rows["mean_action"].append(mean_actions[i])
                rows["log_prob"].append(float(log_probs[i]))
                rows["q_taken"].append(float(q_taken[i]))
                rows["baseline"].append(float(baseline[i]))
                rows["reward"].append(
                    team_reward if self.config.team_reward else float(rewards[agent_id])
                )
                rows["done"].append(done)
                rows["next_value"].append(0.0 if done else next_values.get(agent_id, 0.0))
                rows["agent"].append(agent_id)

            self.observations = next_observations
            self.infos.update(step_infos)

        started = time.perf_counter()
        rewards_array = np.asarray(rows["reward"], dtype=np.float32)
        dones_array = np.asarray(rows["done"], dtype=np.float32)
        q_taken_array = np.asarray(rows["q_taken"], dtype=np.float32)
        baseline_array = np.asarray(rows["baseline"], dtype=np.float32)
        next_values_array = np.asarray(rows["next_value"], dtype=np.float32)
        agents = rows["agent"]

        # Q targets: lambda-returns with Q(o, a, a_bar) as the per-step estimate and
        # V(o', a_bar') as the bootstrap.
        _, q_targets = self._gae_by_agent(
            rewards_array, dones_array, q_taken_array, next_values_array, agents
        )
        if self.config.advantage_mode == "counterfactual":
            advantages = q_taken_array - baseline_array
        else:
            advantages, _ = self._gae_by_agent(
                rewards_array, dones_array, baseline_array, next_values_array, agents
            )
        self.timing_seconds["batch_and_gae"] += time.perf_counter() - started

        return MeanFieldRolloutBatch(
            observations=np.asarray(rows["obs"], dtype=np.float32),
            action_masks=np.asarray(rows["mask"], dtype=np.bool_),
            actions=np.asarray(rows["action"], dtype=np.int32),
            mean_actions=np.asarray(rows["mean_action"], dtype=np.float32),
            old_log_probs=np.asarray(rows["log_prob"], dtype=np.float32),
            rewards=rewards_array,
            dones=dones_array,
            advantages=_normalize_advantages(advantages),
            returns=q_targets,
            agent_ids=tuple(agents),
        )

    def update_policy(  # type: ignore[override]
        self,
        batch: MeanFieldRolloutBatch,
        entropy_coef: float | None = None,
    ) -> dict[str, float]:
        tf = self.tf
        if entropy_coef is None:
            entropy_coef = self.config.entropy_coefficient(self.agent_steps)
        indices = np.arange(int(batch.actions.shape[0]))
        loss_rows: list[dict[str, float]] = []

        for _epoch in range(self.config.train_epochs):
            self.rng.shuffle(indices)
            for start in range(0, len(indices), self.config.minibatch_size):
                mb = indices[start : start + self.config.minibatch_size]
                obs = tf.convert_to_tensor(batch.observations[mb])
                masks = tf.convert_to_tensor(batch.action_masks[mb])
                actions = tf.convert_to_tensor(batch.actions[mb])
                mean_actions = tf.convert_to_tensor(batch.mean_actions[mb])
                old_log_probs = tf.convert_to_tensor(batch.old_log_probs[mb])
                advantages = tf.convert_to_tensor(batch.advantages[mb])
                q_targets = tf.convert_to_tensor(batch.returns[mb])

                with tf.GradientTape() as actor_tape:
                    logits, _ = self.model(obs, training=True)
                    masked_logits = self._mask_logits(logits, masks)
                    ratio = tf.exp(self._selected_log_probs(masked_logits, actions) - old_log_probs)
                    clipped = tf.clip_by_value(
                        ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                    )
                    policy_loss = -tf.reduce_mean(
                        tf.minimum(ratio * advantages, clipped * advantages)
                    )
                    entropy = self._entropy(masked_logits)
                    actor_loss = policy_loss - entropy_coef * entropy
                actor_variables = self.model.trainable_variables
                self.optimizer.apply_gradients(
                    (g, v)
                    for g, v in zip(actor_tape.gradient(actor_loss, actor_variables), actor_variables)
                    if g is not None  # unused value head
                )

                with tf.GradientTape() as critic_tape:
                    q_values = self.critic([obs, mean_actions], training=True)
                    q_taken = tf.reduce_sum(
                        q_values * tf.one_hot(actions, self.action_count), axis=1
                    )
                    critic_loss = tf.reduce_mean(tf.square(q_targets - q_taken))
                critic_variables = self.critic.trainable_variables
                self.critic_optimizer.apply_gradients(
                    zip(critic_tape.gradient(critic_loss, critic_variables), critic_variables)
                )

                loss_rows.append(
                    {
                        "policy_loss": float(policy_loss.numpy()),
                        "value_loss": float(critic_loss.numpy()),
                        "entropy": float(entropy.numpy()),
                        "total_loss": float(
                            actor_loss.numpy() + self.config.value_coef * critic_loss.numpy()
                        ),
                    }
                )

        return {
            key: float(np.mean([row[key] for row in loss_rows]))
            for key in ("policy_loss", "value_loss", "entropy", "total_loss")
        }

    def _masks(self, infos: Any, agent_ids: tuple[str, ...]) -> np.ndarray:
        masks = stack_action_masks(infos, agent_ids, self.action_count)
        if not self.config.use_action_mask:
            return np.ones_like(masks, dtype=np.bool_)
        return masks

    def _bootstrap_values(
        self,
        observations: dict[str, dict[str, np.ndarray]],
        infos: Any,
    ) -> dict[str, float]:
        """V(o', a_bar') where a_bar' is the others' expected (policy-mean) action."""

        if not observations:
            return {}
        agent_ids = tuple(observations)
        flat_obs = flatten_observations(observations, agent_ids, self.observation_encoder)
        logits, _ = self.model(flat_obs, training=False)
        probabilities = self.tf.nn.softmax(
            self._mask_logits(logits, self._masks(infos, agent_ids))
        ).numpy()
        q_values = self.critic(
            [flat_obs, mean_of_others(probabilities)], training=False
        ).numpy()
        values = np.sum(probabilities * q_values, axis=1)
        return {agent_id: float(values[i]) for i, agent_id in enumerate(agent_ids)}

    def _gae_by_agent(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        agent_ids: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        agents = np.asarray(agent_ids)
        for agent_id in sorted(set(agent_ids)):
            idx = np.flatnonzero(agents == agent_id)
            advantages[idx], returns[idx] = compute_gae(
                rewards[idx], dones[idx], values[idx], next_values[idx],
                gamma=self.config.gamma, gae_lambda=self.config.gae_lambda,
            )
        return advantages, returns

    def save_named_checkpoint(self, name: str, update: int) -> str:
        path = super().save_named_checkpoint(name, update)
        self._save_critic(path)
        return path

    def _save_checkpoint(self, update: int) -> str | None:
        path = super()._save_checkpoint(update)
        if path is not None:
            self._save_critic(path)
            if self.config.checkpoint_every > 0 and update % self.config.checkpoint_every == 0:
                self._save_critic(Path(path).parent / f"update_{update:05d}")
        return path

    def _save_critic(self, checkpoint_dir: str | Path) -> None:
        self.critic.save_weights(str(Path(checkpoint_dir) / CRITIC_WEIGHTS_FILE))

    def _checkpoint_metadata(self, update: int) -> dict[str, object]:
        metadata = super()._checkpoint_metadata(update)
        metadata.update(
            {
                "algorithm": "mean_field_ppo",
                "advantage_mode": self.config.advantage_mode,
                "team_reward": self.config.team_reward,
                "critic_hidden_sizes": list(self.config.critic_hidden_sizes),
                "critic_learning_rate": self.config.critic_learning_rate,
                "critic_weights_file": CRITIC_WEIGHTS_FILE,
            }
        )
        return metadata
