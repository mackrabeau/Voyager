# Contributing to Voyager

Voyager welcomes focused fixes, environment integrations, baseline adapters, documentation,
and reproducibility improvements. Changes to the frozen `VoyagerIsland-v1` mechanics,
observations, actions, rewards, achievements, generator, or official seed manifests require a
new versioned contract and must not silently alter existing benchmark results.

## Local setup

Use Python 3.11 or 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the Python checks:

```bash
ruff check voyager tests examples
mypy voyager
pytest -q
```

Run the viewer checks when changing `web/`:

```bash
cd web
npm ci
npm test
npm run build
```

## Pull requests

- Keep changes narrowly scoped and explain their research or user impact.
- Add deterministic tests for simulation and benchmark-contract changes.
- Never select checkpoints or tune environment behavior against the frozen test split.
- Do not hand-edit generated benchmark summaries or curated replay outcomes.
- Include the commands used to validate the change.

Large checkpoints, raw experiments, and videos should be attached to releases rather than
committed directly to Git.
