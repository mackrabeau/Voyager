# VoyagerIsland-v1 Benchmark and Environment Card

## Summary

`VoyagerIsland-v1` is a deterministic two-agent reinforcement-learning environment for measuring
long-horizon team capability acquisition in a finite procedural survival economy. It asks whether
a learning method can train a decentralized pair to compose exploration, resource collection,
return navigation, public investment, survival, and rescue under a fixed interaction budget.

The benchmark is released as a research preview. Its first official procedural experiment is
complete and frozen. The early economy is learnable by ordinary PPO, the complete task is solvable
by a public-observation oracle, and learned rescue remains an unsolved baseline frontier.

## Intended research use

Voyager is intended to compare learning methods for:

- long-horizon temporal composition;
- exploration and return navigation under partial observability;
- shared resource allocation and public infrastructure investment;
- decentralized execution with shared returns;
- memory and recurrent policy architectures;
- multi-agent credit assignment;
- procedural-world generalization; and
- achievement-based evaluation independent of training reward.

It is not currently designed to measure negotiation, deception, natural-language communication,
human social behavior, market formation, governance, or general social intelligence. Both agents
are symmetric and can perform every action. Consequently, v1 is a shared-objective team benchmark,
not a claim that social interdependence or emergent specialization has been isolated experimentally.

## Frozen public contract

| Component | Version |
|---|---|
| Environment | `VoyagerIsland-v1` / `voyager_island_v1` |
| Scenario | `voyager_island_benchmark_v1` |
| Observation | `voyager_island_observation_v1` |
| Action | `voyager_island_action_v1` |
| Public reward | `voyager_island_achievement_reward_v1` |
| PPO training reward | `voyager_island_progression_reward_v4` |
| Achievements | `voyager_island_achievements_v1` |
| Generator | `voyager_island_generator_v1` |
| Replay | `stage7_replay_2.3.0` |
| Result export | `voyager_island_results_v1` |

The canonical API is a PettingZoo `ParallelEnv`. A centralized Gymnasium wrapper is supplied for
algorithms that emit both actions jointly, without changing simulation semantics.

## Environment

### Population and horizon

- Two symmetric agents with decentralized local observations.
- One procedurally generated 48×48 island.
- A maximum of 1,200 world ticks.
- Four 300-tick day/night cycles; night occupies ticks 200–299 of each cycle.
- Early termination if both agents die or the rescue extraction succeeds.
- One initial nonperishable wreck ration per agent.

### Economy and progression

The world has finite berries, wood, stone, two deer, shared camp storage, an axe, a spear, and four
public structures. Nothing respawns in the v1 economy.

```text
explore → gather → return → deposit → workbench → axe and spear
        → campfire → hunt and cook → shelter → beacon → extraction
```

| Capability | Materials | Labor or effect |
|---|---:|---|
| Workbench | 3 wood, 1 stone | 20 labor |
| Axe | 2 wood, 1 stone | doubles wood yield |
| Spear | 2 wood, 1 stone | required to hunt deer |
| Campfire | 2 wood, 1 stone | 20 labor; cooks meat |
| Shelter | 4 wood, 2 stone | 40 labor; protects both occupants |
| Beacon | 4 wood, 2 stone | 60 labor; begins extraction |

Completing the beacon starts a 100-tick extraction window. Rescue requires both agents to remain
alive and at least the first night to have been survived.

One or two seeded stalkers spawn at night. They deal 25 damage per attack; sheltered agents cannot
be targeted. The public v1 profile deliberately excludes repair, revival, spoilage, markets,
communication, renewable resources, and the broader ten-agent sandbox mechanics.

### Actions

Each agent chooses one of 19 masked categorical actions:

```text
NOOP              MOVE_NORTH       MOVE_EAST
MOVE_SOUTH        MOVE_WEST        INTERACT
ATTACK            EAT              REST
DEPOSIT_ALL       WITHDRAW_FOOD    CRAFT_AXE
CRAFT_SPEAR       WORK_WORKBENCH   WORK_CAMPFIRE
WORK_SHELTER      WORK_BEACON      USE_CAMPFIRE
USE_SHELTER
```

The authoritative mask exposes legal actions. An invalid submitted action becomes no-op and incurs
an individual penalty. Contested movement and over-subscribed resource claims fail symmetrically;
arbitration is independent of dictionary order and permanent agent ID.

### Observation

The official actor receives structured local information. Its flattened representation contains
373 normalized values:

- a 7×7×7 egocentric tile tensor;
- health, hunger, energy, inventory capacity, shelter, and life state;
- food, wood, stone, raw meat, and cooked meat;
- axe and spear ownership;
- a two-value identity indicator;
- signed camp bearing and Manhattan distance; and
- public camp stock, structure progress, active population, time, and night state.

It does not expose hidden entity IDs, remote creature or resource locations, future spawns, food-lot
IDs, or privileged simulator state. Pixels are not an official v1 observation modality.

## Procedural generation and splits

Every accepted island contains reachable food, wood, stone, deer, camp, construction sites, and
stalker spawn candidates. The generator rejects disconnected or under-resourced maps.

| Split | Seeds | Purpose |
|---|---:|---|
| Training | `0–999` | policy optimization only |
| Development | `10000–10049` | milestone evaluation and checkpoint selection |
| Test | `20000–20099` | one-time official held-out evaluation |

The three manifests are disjoint, immutable, tracked in the repository, and packaged inside the
wheel. Matching algorithm runs receive matching deterministic training-island schedules.

## Reward and evaluation

### Public environment reward

- `+1.0` shared when an achievement is first unlocked.
- `+0.1` individual on a tick with health restoration.
- `-0.1` individual on a tick with health loss.
- `-1.0` shared for each agent death.
- `-0.05` individual for an invalid submission or symmetric conflict.

The included PPO runs use a separately versioned bounded shaping reward for causal delivery and
construction credit. That reward is never used as the public benchmark score.

### Achievement spectrum

The 15 evaluation achievements are:

1. Collect food.
2. Collect wood.
3. Collect stone.
4. Deposit wood.
5. Deposit stone.
6. Build the workbench.
7. Craft an axe.
8. Craft a spear.
9. Hunt a deer.
10. Build the campfire.
11. Cook meat.
12. Build the shelter.
13. Keep both agents alive through the first night.
14. Build the beacon.
15. Rescue both agents.

Every achievement success rate is reported. The headline score is the smoothed geometric mean:

```text
exp(mean(log(success_rate + 0.01))) - 0.01
```

Separate gathering, delivery, production, survival, and rescue group scores remain available for
interpretation. Evaluation also records achievement timing, joint survival, invalid actions,
remaining resources, and ledger-derived contributions.

## Official experiment protocol

The first baseline matrix contains:

- shared feed-forward PPO, training seeds 0, 1, and 2;
- shared recurrent PPO, training seeds 0, 1, and 2;
- one million agent transitions per run;
- checkpoints at 200,000-transition intervals;
- checkpoint selection by seeded-stochastic development achievement score;
- deterministic inference as a collapse diagnostic;
- legal random and a public-observation scripted oracle; and
- one final evaluation of all locked policies on 100 held-out islands.

Algorithm-family confidence intervals hierarchically resample both training seeds and matched test
islands. Comparisons against random preserve the matched-island pairing.

## Official results

| Policy | Score | 95% CI | Training-seed SD | Rescue |
|---|---:|---:|---:|---:|
| Legal random | 0.023 | 0.019–0.026 | — | 0% |
| Feed-forward PPO | 0.164 | 0.049–0.198 | 0.083 | 0% |
| Recurrent PPO | 0.072 | 0.032–0.106 | 0.040 | 0% |
| Scripted public-observation oracle | 0.986 | 0.973–0.997 | — | 96% |

Feed-forward PPO significantly exceeds random and reaches a middle-tree capability. Across its
three training seeds it averages approximately 94% first-night survival, 91–96% basic gathering,
65–72% resource deposits, 61% workbench construction, and 56% first-tool construction.

Neither learned family hunts, cooks, builds shelter, constructs the beacon, or rescues on a
held-out island. Recurrent PPO does not outperform feed-forward PPO under the frozen budget. The
oracle reaches at least 96% on every achievement and reconciles the conservation ledger.

### Predeclared release gates

| Gate | Outcome |
|---|---|
| Oracle solves at least 90% | Pass |
| Random never rescues | Pass |
| Random rarely builds the workbench | Pass |
| Learned baseline significantly exceeds random | Pass |
| Learned middle capability at least 25% | Pass |
| Learned rescue at least 25% | **Fail** |
| All ledgers reconcile | Pass |
| Large reproducibility artifacts published | Pass |

The overall preregistered release gate therefore fails. Voyager may be used as a difficult research
preview with a learned early economy and explicit late-game frontier, but the result does not support
a claim that PPO solves the benchmark.

## Measured compute

The six official runs were trained locally on a 15-inch 2023 MacBook Air with Apple M2 and 16 GB of
memory. Training-loop measurements, excluding the complete milestone and held-out evaluation cost,
were:

| Algorithm | Throughput | Time per 1M agent transitions |
|---|---:|---:|
| Feed-forward PPO | 402–516 transitions/s | 32–41 minutes |
| Recurrent PPO | 117–161 transitions/s | 1.7–2.4 hours |

These measurements describe the included TensorFlow baselines on one machine, not a hardware-normalized
benchmark score. Environment stepping represented roughly 8% of measured feed-forward training time;
network inference dominated that run.

## Determinism and auditability

- Reordering agent submissions produces identical state hashes and events.
- Identical seeds and joint actions reproduce byte-equivalent outcomes.
- Resources are tracked from source through holdings, transformations, and sinks.
- Replay reconstruction verifies recorded state hashes.
- Compact statistical exports are regenerated from raw episode records.
- Frozen references are protected by SHA-256 manifests and byte-comparison tests.
- Test seeds cannot be accessed by the official training command.

## Known limitations

1. **Cooperation is not isolated.** Agents share a goal and public economy, but they are symmetric
   and a single active producer can theoretically execute the technology chain. V1 should not be
   presented as a general social-intelligence benchmark.
2. **No learned rescue.** The reference algorithms establish early trainability but not full-task
   learning.
3. **Three training seeds.** Hierarchical intervals reflect this limited algorithm-level sample.
4. **Structured observations only.** Representation learning from pixels is outside the frozen v1
   claim.
5. **Shared-policy baselines.** Heterogeneous policies and unfamiliar-partner cross-play have not
   yet been evaluated.
6. **One generator family.** Generalization is across seeds from the same frozen procedural
   generator, not across unrelated island distributions.
7. **Large artifacts live outside Git.** Full checkpoints, raw episodes, histories, curves,
   configuration, curated replays, and checksums are distributed with the tagged GitHub release
   rather than versioned in the repository tree.

## Versioning policy

`VoyagerIsland-v1` is frozen after its test split was opened. Changes to mechanics, observations,
actions, rewards, achievements, generation, horizons, or evaluation seeds require a new version.
Bug fixes that can alter recorded outcomes also require a new contract unless they restore behavior
already specified and are accompanied by explicit compatibility evidence.

The older `VoyagerCivilization-v1` and `VoyagerCivilization-v2` environments remain compatibility
sandboxes over the same simulation family. Their ten-agent population and broader mechanics are not
part of this benchmark card.

## Artifacts and entry points

- Detailed contract: `docs/stage_7_island_benchmark_v1.md`
- Frozen compact report: `benchmarks/reference/voyager_island_v1_baselines_v1/`
- Seed manifests: `benchmarks/manifests/island_v1_*.json`
- Curated Replay 2.3 collection: `benchmarks/replays/stage7_curated_v1/`
- Procedural runner: `examples/run_stage7_island_procedural.py`
- Baseline evaluator: `examples/evaluate_stage7_island.py`
- Statistical exporter: `examples/export_stage7_island_results.py`
