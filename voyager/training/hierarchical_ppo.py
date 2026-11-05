"""Hierarchical mean-field PPO (HMF-PPO) for VoyagerIsland-v1.

Manager: every option, each agent picks an abstract goal g ~ pi_M(g | o). Its critic
V_M(o, g_bar) also sees g_bar, the mean one-hot goal of the other agents (mean field over
the team's goal distribution). Trained with PPO on semi-MDP transitions:
    (o_start, g, R = sum_k gamma^k r_k, gamma^K, o_end).

Worker: goal-conditioned PPO actor-critic pi_W(a | o, g). Reward is the environment's
training reward plus `goal_bonus` when the goal is reached. An option ends when the goal
is reached, after `max_option_steps`, or when the agent's episode ends.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from voyager.training.island_goals import (
    GOAL_COUNT,
    ISLAND_GOAL_VERSION,
    goal_achieved,
    goal_one_hot,
)
from voyager.training.masking import mask_numpy_logits, stack_action_masks
from voyager.training.mean_field_ppo import mean_of_others
from voyager.training.model import build_actor_critic, build_goal_manager, require_tensorflow
from voyager.training.obs import flatten_observation, flatten_observations
from voyager.training.ppo import (
    PPOConfig,
    PPOTrainer,
    RolloutBatch,
    _normalize_advantages,
)

MANAGER_WEIGHTS_FILE = "manager.weights.h5"


@dataclass(frozen=True, slots=True)
class HierarchicalPPOConfig(PPOConfig):
    goal_bonus: float = 1.0
    env_reward_coef: float = 1.0
    max_option_steps: int = 40
    manager_hidden_sizes: tuple[int, ...] = (128, 128)
    manager_learning_rate: float = 3e-4
    manager_batch_size: int = 128
    manager_minibatch_size: int = 64
    manager_epochs: int = 4
    manager_entropy_coef: float = 0.01

    def validate(self) -> None:
        PPOConfig.validate(self)
        for name in (
            "max_option_steps", "manager_batch_size", "manager_minibatch_size", "manager_epochs",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.manager_learning_rate <= 0.0:
            raise ValueError("manager_learning_rate must be positive.")


@dataclass(slots=True)
class _Option:
    goal: int
    start_obs: np.ndarray
    goal_mean: np.ndarray
    log_prob: float
    value: float
    reward: float = 0.0
    discount: float = 1.0
    steps: int = 0


@dataclass(slots=True)
class _ManagerTransition:
    agent_id: str
    obs: np.ndarray
    goal_mean: np.ndarray
    goal: int
    log_prob: float
    value: float
    reward: float
    discount: float  # gamma^K over the option
    next_value: float
    done: bool


@dataclass(slots=True)
class _GoalStats:
    chosen: np.ndarray = field(default_factory=lambda: np.zeros(GOAL_COUNT, dtype=np.int64))
    achieved: np.ndarray = field(default_factory=lambda: np.zeros(GOAL_COUNT, dtype=np.int64))


class HierarchicalPPOTrainer(PPOTrainer):
    """self.model is the worker; self.manager picks goals."""

    config: HierarchicalPPOConfig

    def __init__(self, config: HierarchicalPPOConfig) -> None:
        super().__init__(config)
        self.worker_input_dim = self.input_dim + GOAL_COUNT
        self.model = build_actor_critic(
            input_dim=self.worker_input_dim,
            action_count=self.action_count,
            hidden_sizes=config.hidden_sizes,
            seed=config.seed,
        )
        self.optimizer = self.tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.manager = build_goal_manager(
            input_dim=self.input_dim,
            goal_count=GOAL_COUNT,
            hidden_sizes=config.manager_hidden_sizes,
            seed=config.seed + 1,
        )
        self.manager_optimizer = self.tf.keras.optimizers.Adam(
            learning_rate=config.manager_learning_rate
        )
        self.options: dict[str, _Option] = {}
        self.manager_buffer: list[_ManagerTransition] = []
        self.goal_stats = _GoalStats()
        self.last_manager_losses: dict[str, float] = {}

    # ---------------------------------------------------------------- rollout

    def collect_rollout(self, max_agent_steps: int | None = None) -> RolloutBatch:
        if max_agent_steps is not None and max_agent_steps <= 0:
            raise ValueError("max_agent_steps must be positive when provided.")
        rows: dict[str, list[Any]] = {
            key: []
            for key in (
                "obs", "mask", "action", "log_prob", "value", "reward", "done",
                "next_value", "agent",
            )
        }
        gamma = self.config.gamma

        for _step in range(self.config.rollout_steps):
            if not self.env.agents:
                self._reset_env()
                self.options.clear()
            agent_ids = tuple(self.env.agents)
            if (
                max_agent_steps is not None
                and rows["obs"]
                and len(rows["obs"]) + len(agent_ids) > max_agent_steps
            ):
                break

            started = time.perf_counter()
            flat_obs = flatten_observations(self.observations, agent_ids, self.observation_encoder)
            self._start_options(agent_ids, flat_obs)
            goals = np.asarray([self.options[a].goal for a in agent_ids], dtype=np.int64)
            worker_obs = np.concatenate([flat_obs, goal_one_hot(goals)], axis=1)
            action_masks = self._masks(self.infos, agent_ids)
            logits, values = self.model(worker_obs, training=False)
            masked_logits = self._mask_logits(logits, action_masks)
            actions = self._sample_actions(masked_logits)
            log_probs = self._selected_log_probs(masked_logits, actions).numpy()
            values_np = np.squeeze(values.numpy(), axis=1)
            self.timing_seconds["actor_inference"] += time.perf_counter() - started

            started = time.perf_counter()
            next_observations, rewards, terminations, truncations, step_infos = self.env.step(
                {a: int(actions[i]) for i, a in enumerate(agent_ids)}
            )
            self.timing_seconds["environment_step"] += time.perf_counter() - started
            self.world_steps += 1

            started = time.perf_counter()
            worker_next_values = self._worker_values(next_observations)
            ended: list[str] = []
            for i, agent_id in enumerate(agent_ids):
                option = self.options[agent_id]
                env_reward = float(rewards[agent_id])
                done = bool(terminations[agent_id] or truncations[agent_id])
                alive = agent_id in next_observations
                success = alive and goal_achieved(
                    option.goal,
                    self.observations[agent_id],
                    int(actions[i]),
                    next_observations[agent_id],
                )
                option.reward += option.discount * env_reward
                option.discount *= gamma
                option.steps += 1
                timeout = option.steps >= self.config.max_option_steps

                worker_reward = (
                    self.config.env_reward_coef * env_reward
                    + self.config.goal_bonus * float(success)
                )
                worker_done = done or success or timeout or not alive
                if timeout and not (done or success) and alive:
                    # Truncated option: bootstrap through the reward and end the worker episode.
                    worker_reward += gamma * worker_next_values[agent_id]

                rows["obs"].append(worker_obs[i])
                rows["mask"].append(action_masks[i])
                rows["action"].append(int(actions[i]))
                rows["log_prob"].append(float(log_probs[i]))
                rows["value"].append(float(values_np[i]))
                rows["reward"].append(worker_reward)
                rows["done"].append(worker_done)
                rows["next_value"].append(
                    0.0 if worker_done else worker_next_values.get(agent_id, 0.0)
                )
                rows["agent"].append(agent_id)

                if success:
                    self.goal_stats.achieved[option.goal] += 1
                if worker_done:
                    ended.append(agent_id)

            self._end_options(ended, next_observations, terminations, truncations)
            self.timing_seconds["batch_and_gae"] += time.perf_counter() - started
            self.observations = next_observations
            self.infos.update(step_infos)

        started = time.perf_counter()
        rewards_array = np.asarray(rows["reward"], dtype=np.float32)
        dones_array = np.asarray(rows["done"], dtype=np.float32)
        values_array = np.asarray(rows["value"], dtype=np.float32)
        next_values_array = np.asarray(rows["next_value"], dtype=np.float32)
        advantages, returns = self._advantages_by_agent(
            rewards=rewards_array,
            dones=dones_array,
            values=values_array,
            next_values=next_values_array,
            agent_ids=rows["agent"],
        )
        self.timing_seconds["batch_and_gae"] += time.perf_counter() - started
        return RolloutBatch(
            observations=np.asarray(rows["obs"], dtype=np.float32),
            action_masks=np.asarray(rows["mask"], dtype=np.bool_),
            actions=np.asarray(rows["action"], dtype=np.int32),
            old_log_probs=np.asarray(rows["log_prob"], dtype=np.float32),
            values=values_array,
            rewards=rewards_array,
            dones=dones_array,
            next_values=next_values_array,
            advantages=_normalize_advantages(advantages),
            returns=returns,
            agent_ids=tuple(rows["agent"]),
        )

    def _start_options(self, agent_ids: tuple[str, ...], flat_obs: np.ndarray) -> None:
        needs = [i for i, a in enumerate(agent_ids) if a not in self.options]
        if not needs:
            return
        obs = flat_obs[needs]
        logits, _ = self.manager([obs, np.zeros((len(needs), GOAL_COUNT), np.float32)])
        goals = self._sample_actions(logits)
        log_probs = self.tf.nn.log_softmax(logits).numpy()[np.arange(len(needs)), goals]
        for row, index in enumerate(needs):
            self.options[agent_ids[index]] = _Option(
                goal=int(goals[row]),
                start_obs=obs[row],
                goal_mean=np.zeros(GOAL_COUNT, np.float32),
                log_prob=float(log_probs[row]),
                value=0.0,
            )
            self.goal_stats.chosen[int(goals[row])] += 1
        # g_bar is taken once every agent has a goal, then valued by the critic.
        goal_means = self._goal_means(agent_ids)
        new_ids = [agent_ids[i] for i in needs]
        means = np.stack([goal_means[a] for a in new_ids])
        _, values = self.manager([obs, means])
        for row, agent_id in enumerate(new_ids):
            self.options[agent_id].goal_mean = means[row]
            self.options[agent_id].value = float(values.numpy()[row, 0])

    def _end_options(
        self,
        ended: list[str],
        next_observations: Mapping[str, Any],
        terminations: Mapping[str, bool],
        truncations: Mapping[str, bool],
    ) -> None:
        if not ended:
            return
        continuing = [a for a in ended if a in next_observations and not (
            terminations[a] or truncations[a]
        )]
        bootstrap: dict[str, float] = {}
        if continuing:
            goal_means = self._goal_means(tuple(self.options))
            obs = flatten_observations(next_observations, continuing, self.observation_encoder)
            means = np.stack([goal_means[a] for a in continuing])
            _, values = self.manager([obs, means])
            bootstrap = {a: float(values.numpy()[i, 0]) for i, a in enumerate(continuing)}
        for agent_id in ended:
            option = self.options.pop(agent_id)
            self.manager_buffer.append(
                _ManagerTransition(
                    agent_id=agent_id,
                    obs=option.start_obs,
                    goal_mean=option.goal_mean,
                    goal=option.goal,
                    log_prob=option.log_prob,
                    value=option.value,
                    reward=option.reward,
                    discount=option.discount,
                    next_value=bootstrap.get(agent_id, 0.0),
                    done=agent_id not in bootstrap,
                )
            )

    def _goal_means(self, agent_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
        ids = [a for a in agent_ids if a in self.options]
        if not ids:
            return {}
        means = mean_of_others(goal_one_hot([self.options[a].goal for a in ids]))
        return {a: means[i] for i, a in enumerate(ids)}

    def _worker_values(self, observations: Mapping[str, Any]) -> dict[str, float]:
        ids = [a for a in observations if a in self.options]
        if not ids:
            return {}
        flat = flatten_observations(observations, ids, self.observation_encoder)
        goals = goal_one_hot([self.options[a].goal for a in ids])
        _, values = self.model(np.concatenate([flat, goals], axis=1), training=False)
        return {a: float(values.numpy()[i, 0]) for i, a in enumerate(ids)}

    def _masks(self, infos: Any, agent_ids: tuple[str, ...]) -> np.ndarray:
        masks = stack_action_masks(infos, agent_ids, self.action_count)
        return masks if self.config.use_action_mask else np.ones_like(masks, dtype=np.bool_)

    # ---------------------------------------------------------------- updates

    def update_policy(
        self,
        batch: RolloutBatch,
        entropy_coef: float | None = None,
    ) -> dict[str, float]:
        losses = super().update_policy(batch, entropy_coef=entropy_coef)
        if len(self.manager_buffer) >= self.config.manager_batch_size:
            self.last_manager_losses = self._update_manager(self.manager_buffer)
            self.manager_buffer = []
        return losses

    def _update_manager(self, transitions: list[_ManagerTransition]) -> dict[str, float]:
        tf = self.tf
        advantages, returns = _semi_mdp_gae(transitions, self.config.gae_lambda)
        advantages = _normalize_advantages(advantages)
        obs = np.stack([t.obs for t in transitions]).astype(np.float32)
        means = np.stack([t.goal_mean for t in transitions]).astype(np.float32)
        goals = np.asarray([t.goal for t in transitions], dtype=np.int32)
        old_log_probs = np.asarray([t.log_prob for t in transitions], dtype=np.float32)
        indices = np.arange(len(transitions))
        loss_rows: list[dict[str, float]] = []

        for _epoch in range(self.config.manager_epochs):
            self.rng.shuffle(indices)
            for start in range(0, len(indices), self.config.manager_minibatch_size):
                mb = indices[start : start + self.config.manager_minibatch_size]
                with tf.GradientTape() as tape:
                    logits, values = self.manager([obs[mb], means[mb]], training=True)
                    log_probs = tf.reduce_sum(
                        tf.one_hot(goals[mb], GOAL_COUNT) * tf.nn.log_softmax(logits), axis=1
                    )
                    ratio = tf.exp(log_probs - old_log_probs[mb])
                    adv = tf.convert_to_tensor(advantages[mb])
                    clipped = tf.clip_by_value(
                        ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                    )
                    policy_loss = -tf.reduce_mean(tf.minimum(ratio * adv, clipped * adv))
                    value_loss = tf.reduce_mean(
                        tf.square(returns[mb] - tf.squeeze(values, axis=1))
                    )
                    entropy = self._entropy(logits)
                    loss = (
                        policy_loss
                        + self.config.value_coef * value_loss
                        - self.config.manager_entropy_coef * entropy
                    )
                variables = self.manager.trainable_variables
                self.manager_optimizer.apply_gradients(
                    zip(tape.gradient(loss, variables), variables)
                )
                loss_rows.append(
                    {
                        "manager_policy_loss": float(policy_loss.numpy()),
                        "manager_value_loss": float(value_loss.numpy()),
                        "manager_entropy": float(entropy.numpy()),
                    }
                )
        return {k: float(np.mean([r[k] for r in loss_rows])) for k in loss_rows[0]}

    # ------------------------------------------------------------ checkpoints

    def save_named_checkpoint(self, name: str, update: int) -> str:
        path = super().save_named_checkpoint(name, update)
        self._save_manager(path)
        return path

    def _save_checkpoint(self, update: int) -> str | None:
        path = super()._save_checkpoint(update)
        if path is not None:
            self._save_manager(path)
            if self.config.checkpoint_every > 0 and update % self.config.checkpoint_every == 0:
                self._save_manager(Path(path).parent / f"update_{update:05d}")
        return path

    def _save_manager(self, checkpoint_dir: str | Path) -> None:
        self.manager.save_weights(str(Path(checkpoint_dir) / MANAGER_WEIGHTS_FILE))

    def _checkpoint_metadata(self, update: int) -> dict[str, object]:
        metadata = super()._checkpoint_metadata(update)
        metadata.update(
            {
                "algorithm": "hierarchical_mean_field_ppo",
                "model_type": "hierarchical_goal",
                "input_dim": self.input_dim,
                "worker_input_dim": self.worker_input_dim,
                "goal_count": GOAL_COUNT,
                "goal_version": ISLAND_GOAL_VERSION,
                "max_option_steps": self.config.max_option_steps,
                "goal_bonus": self.config.goal_bonus,
                "env_reward_coef": self.config.env_reward_coef,
                "manager_hidden_sizes": list(self.config.manager_hidden_sizes),
                "manager_learning_rate": self.config.manager_learning_rate,
                "manager_batch_size": self.config.manager_batch_size,
                "manager_entropy_coef": self.config.manager_entropy_coef,
                "manager_weights_file": MANAGER_WEIGHTS_FILE,
                "goal_chosen_counts": self.goal_stats.chosen.tolist(),
                "goal_achieved_counts": self.goal_stats.achieved.tolist(),
                "last_manager_losses": dict(self.last_manager_losses),
            }
        )
        return metadata


def _semi_mdp_gae(
    transitions: list[_ManagerTransition], gae_lambda: float
) -> tuple[np.ndarray, np.ndarray]:
    """GAE over options; each step discounts by its own gamma^K."""

    advantages = np.zeros(len(transitions), dtype=np.float32)
    by_agent: dict[str, list[int]] = {}
    for index, transition in enumerate(transitions):
        by_agent.setdefault(transition.agent_id, []).append(index)
    for indices in by_agent.values():
        running = 0.0
        for index in reversed(indices):
            t = transitions[index]
            nonterminal = 0.0 if t.done else 1.0
            delta = t.reward + t.discount * nonterminal * t.next_value - t.value
            running = delta + t.discount * gae_lambda * nonterminal * running
            advantages[index] = running
    values = np.asarray([t.value for t in transitions], dtype=np.float32)
    return advantages, (advantages + values).astype(np.float32)


# -------------------------------------------------------------------- evaluation


class HierarchicalIslandPolicy:
    """Decentralized manager + worker policy implementing the IslandPolicy protocol."""

    def __init__(
        self,
        manager: Any,
        worker: Any,
        observation_encoder: str,
        *,
        max_option_steps: int,
        deterministic: bool,
        seed: int,
    ) -> None:
        self.manager = manager
        self.worker = worker
        self.encoder = observation_encoder
        self.max_option_steps = max_option_steps
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self.reset(())

    def reset(self, possible_agents: tuple[str, ...]) -> None:
        _ = possible_agents
        self.goals: dict[str, int] = {}
        self.steps: dict[str, int] = {}
        self.previous: dict[str, tuple[Mapping[str, np.ndarray], int]] = {}

    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        infos: Mapping[str, Mapping[str, object]],
    ) -> dict[str, int]:
        _ = infos
        actions: dict[str, int] = {}
        for agent_id, observation in observations.items():
            flat = flatten_observation(observation, self.encoder)[None, :]
            if self._option_over(agent_id, observation):
                logits, _ = self.manager([flat, np.zeros((1, GOAL_COUNT), np.float32)])
                self.goals[agent_id] = self._choose(np.asarray(logits[0], np.float32))
                self.steps[agent_id] = 0
            goal = goal_one_hot([self.goals[agent_id]])
            logits, _ = self.worker(np.concatenate([flat, goal], axis=1), training=False)
            masked = mask_numpy_logits(
                np.asarray(logits[0], np.float32),
                np.asarray(observation["action_mask"], np.int8),
            )
            action = self._choose(masked)
            actions[agent_id] = action
            self.steps[agent_id] += 1
            self.previous[agent_id] = (
                {key: np.array(value, copy=True) for key, value in observation.items()},
                action,
            )
        return actions

    def _option_over(self, agent_id: str, observation: Mapping[str, np.ndarray]) -> bool:
        if agent_id not in self.goals or agent_id not in self.previous:
            return True
        if self.steps[agent_id] >= self.max_option_steps:
            return True
        before, action = self.previous[agent_id]
        return goal_achieved(self.goals[agent_id], before, action, observation)

    def _choose(self, logits: np.ndarray) -> int:
        if self.deterministic:
            return int(np.argmax(logits))
        shifted = np.exp(logits - np.max(logits))
        return int(self.rng.choice(len(shifted), p=shifted / shifted.sum()))


def load_hierarchical_policy(
    checkpoint: str | Path,
    *,
    deterministic: bool,
    seed: int,
) -> HierarchicalIslandPolicy:
    """Load a saved hierarchical checkpoint as a decentralized evaluation policy."""

    require_tensorflow()
    path = Path(checkpoint)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_type") != "hierarchical_goal":
        raise ValueError("Checkpoint is not a hierarchical goal policy.")
    worker = build_actor_critic(
        input_dim=int(metadata["worker_input_dim"]),
        action_count=int(metadata["action_count"]),
        hidden_sizes=tuple(int(v) for v in metadata["hidden_sizes"]),
    )
    worker.load_weights(str(path / str(metadata["weights_file"])))
    manager = build_goal_manager(
        input_dim=int(metadata["input_dim"]),
        goal_count=int(metadata["goal_count"]),
        hidden_sizes=tuple(int(v) for v in metadata["manager_hidden_sizes"]),
    )
    manager.load_weights(str(path / str(metadata["manager_weights_file"])))
    return HierarchicalIslandPolicy(
        manager,
        worker,
        str(metadata["observation_encoder"]),
        max_option_steps=int(metadata["max_option_steps"]),
        deterministic=deterministic,
        seed=seed,
    )


__all__ = [
    "HierarchicalIslandPolicy",
    "HierarchicalPPOConfig",
    "HierarchicalPPOTrainer",
    "load_hierarchical_policy",
]
