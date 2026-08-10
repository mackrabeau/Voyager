# Voyager

[![CI](https://github.com/fawzmehfil/Voyager/actions/workflows/ci.yml/badge.svg)](https://github.com/fawzmehfil/Voyager/actions/workflows/ci.yml)
![Python 3.11 and 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Voyager is a deterministic two-agent reinforcement-learning benchmark set in a finite,
procedurally generated stranded-island economy. A decentralized pair must explore, gather and
return resources, invest in public infrastructure, survive repeated nights, and construct a
rescue beacon under a fixed interaction budget.

`VoyagerIsland-v1` is a research preview. Its environment contract and first official baseline
matrix are frozen. Feed-forward PPO learns a substantial early economy and significantly exceeds
legal random, but no learned baseline completes rescue. A public-observation scripted oracle
rescues on 96% of held-out islands, establishing that the task is mechanically solvable.

- [Benchmark and environment card](BENCHMARK.md)
- [Frozen result report](benchmarks/reference/voyager_island_v1_baselines_v1/report.md)
- [Detailed Stage 7 contract](docs/stage_7_island_benchmark_v1.md)
- [Contributing](CONTRIBUTING.md)

## Official baseline result

All learned policies use three independent training seeds, one million agent transitions per
seed, checkpoint selection on 50 development islands, and one final evaluation on 100 held-out
procedural islands.

| Policy | Achievement score | 95% CI | Rescue |
|---|---:|---:|---:|
| Legal random | 0.023 | 0.019–0.026 | 0% |
| Feed-forward PPO | 0.164 | 0.049–0.198 | 0% |
| Recurrent PPO | 0.072 | 0.032–0.106 | 0% |
| Scripted public-observation oracle | 0.986 | 0.973–0.997 | 96% |

The learned result is meaningful but incomplete: feed-forward PPO commonly gathers and deposits
resources, builds the workbench, crafts tools, and sometimes builds the campfire. It does not
hunt, cook, build shelter or the beacon, or rescue on the frozen test split. Voyager therefore
provides a learnable beginning and a measured late-game frontier; it is not claimed to be solved
by PPO.

## Install

The distribution name is `voyager-island` because `voyager` is already used by an unrelated PyPI
project. The Python import remains `voyager`.

Until the first PyPI release is published, install from source:

```bash
git clone https://github.com/fawzmehfil/Voyager.git
cd Voyager
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

TensorFlow is optional and only needed for the included PPO trainers:

```bash
python -m pip install -e ".[dev,train]"
```

## Five-minute environment check

The canonical interface is a PettingZoo `ParallelEnv`:

```python
from voyager.envs import VoyagerIslandEnv

env = VoyagerIslandEnv(procedural=True)
observations, infos = env.reset(seed=0)

while env.agents:
    actions = {
        agent_id: env.action_space(agent_id).sample(
            mask=infos[agent_id]["action_mask"]
        )
        for agent_id in env.agents
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)

print(env.metrics())
env.close()
```

`gym.make("VoyagerIsland-v1")` registers the same parallel environment with Gymnasium's order
wrapper disabled. `VoyagerIslandCentralized-v1` is an optional Gymnasium wrapper for algorithms
that emit both agents' actions jointly.

## Benchmark at a glance

| Contract | VoyagerIsland-v1 |
|---|---|
| Population | Two symmetric agents; shared-policy compatible |
| World | Procedural 48×48 island |
| Horizon | 1,200 ticks; four day/night cycles |
| Observation | Local structured observation; 373-value flattened actor input |
| Action space | 19 masked categorical actions |
| Economy | Finite food, wood, stone, deer, two tools, four public structures |
| Threat | One or two seeded night stalkers |
| Goal | Keep both agents alive and complete the rescue extraction window |
| Evaluation | 15 semantic achievements and a smoothed geometric-mean score |
| Splits | 1,000 train, 50 development, 100 frozen test islands |
| Replay | Deterministic Replay 2.3 with state hashes and conservation ledger |

The capability chain is:

```text
explore → gather → return → deposit → workbench → tools
        → hunt → campfire → cook → shelter → beacon → rescue
```

The public score is independent of the denser reward used by the included PPO baselines. Every
achievement success rate is reported individually before aggregation, preventing an algorithm
from hiding missing late-game capabilities behind repeated easy rewards.

## Evaluate reference policies

Evaluate legal random on ten development islands:

```bash
python examples/evaluate_stage7_island.py \
  --policy random \
  --split dev \
  --episodes 10 \
  --output results/random_dev.json
```

Run the public-observation solvability oracle on the development split:

```bash
python examples/evaluate_stage7_island.py \
  --policy oracle \
  --split dev \
  --output results/oracle_dev.json
```

Verify the tracked statistical reference byte-for-byte after restoring the corresponding raw
release artifacts:

```bash
python examples/export_stage7_island_results.py verify
```

## Train the included baselines

Validate an official run without loading TensorFlow or writing artifacts:

```bash
python examples/run_stage7_island_procedural.py train \
  --algorithm feed_forward_ppo \
  --training-seed 0 \
  --dry-run
```

Run a nonofficial 2,560-transition integration smoke test:

```bash
python examples/run_stage7_island_procedural.py train \
  --algorithm feed_forward_ppo \
  --training-seed 0 \
  --smoke
```

Official runs use `feed_forward_ppo` or `recurrent_ppo`, training seeds `0`, `1`, and `2`, and
exactly one million agent transitions each. Training consumes only the packaged training
manifest, development islands select checkpoints, and a separate finalization command is the
only path that opens the test split.

## Watch the benchmark

The viewer renders saved outcomes and never simulates or modifies them in the browser. Its default
showcase is a learned feed-forward PPO run. The separate **Perfect Run** is explicitly labeled as a
scripted public-observation oracle with 15/15 achievements and 2/2 agents rescued.

Build and run the viewer from a source checkout:

```bash
cd web
npm ci
npm run build
cd ..

VOYAGER_REPLAY_ROOTS=benchmarks/replays/stage7_curated_v1 voyager-web
```

Open `http://127.0.0.1:8000`. Useful routes include:

- `/` — learned-policy showcase and Perfect Run entry point.
- `/replays/{replay_id}` — inspection viewer with timeline and metrics.
- `/compare?left=island_legal_random_matched_v1&right=island_feed_forward_progression_v1`
  — seed-matched comparison.
- `/present/{replay_id}` — presentation mode.

## Reproducibility boundaries

- Agent submission order cannot change outcomes.
- Identical seeds and joint actions reproduce identical events and state hashes.
- Resource sources, transformations, holdings, and sinks reconcile every tick.
- Train, development, and test manifests are disjoint and packaged with the wheel.
- Public score, baseline training reward, environment, action, observation, generator, and replay
  contracts are versioned separately.
- Frozen `VoyagerIsland-v1` behavior will not be tuned after observing test results. Any mechanical
  change requires a new environment version and untouched evaluation split.

## Scope and limitations

Voyager currently measures decentralized team capability acquisition under shared resource
constraints. It does **not** yet establish emergent specialization, negotiation, human-like social
intelligence, or general cooperation with unfamiliar partners. Both agents are symmetric and each
can perform every action, so the benchmark offers coordination opportunities without isolating
social interdependence as a causal variable.

Other limitations:

- The official actor observation is structured rather than pixel-based.
- Only PPO and recurrent PPO are included as learned reference algorithms.
- Algorithm-family uncertainty is estimated from three training seeds.
- No learned baseline reaches the late-game rescue branch under the frozen one-million-transition
  budget.
- Large checkpoints, raw episodes, and curves must be distributed as release artifacts; the Git
  repository contains compact results and curated replays.
- The environment is not intended as an evaluation of pretrained language models or human social
  behavior.

These limitations are part of the published benchmark record rather than hidden release details.

## Development

```bash
ruff check voyager tests examples
mypy voyager
pytest -q

cd web
npm test
npm run build
```

The repository also retains the earlier `VoyagerCivilization-v1` and
`VoyagerCivilization-v2` sandboxes for compatibility. Their ten-agent population and broader
mechanics are not part of the official island benchmark.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). Voyager is released under the
[MIT License](LICENSE).
