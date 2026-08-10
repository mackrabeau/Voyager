# Voyager benchmark artifacts

The canonical benchmark is `VoyagerIsland-v1`, a two-agent procedural stranded-island task.
Its full contract and limitations are documented in [`BENCHMARK.md`](../BENCHMARK.md).

## Frozen seed manifests

The immutable generator splits are:

- `manifests/island_v1_train.json`: 1,000 training islands.
- `manifests/island_v1_dev.json`: 50 checkpoint-selection islands.
- `manifests/island_v1_test.json`: 100 held-out evaluation islands.

These files are also packaged in the `voyager-island` wheel. The official runner trains only on
the training manifest, selects checkpoints only on development, and exposes test through a separate
finalization command.

## Official Stage 7 reference

`reference/voyager_island_v1_baselines_v1/` freezes the compact first procedural evaluation:

- legal random;
- public-observation scripted oracle;
- feed-forward PPO with training seeds 0, 1, and 2; and
- recurrent PPO with training seeds 0, 1, and 2.

Each learned run uses one million agent transitions. Checkpoints were selected on all 50 development
islands before the 100-island test split was opened once. Seeded-stochastic inference is primary;
deterministic inference is a collapse diagnostic.

Regenerate the compact reference from restored raw release artifacts:

```bash
python examples/export_stage7_island_results.py export
```

Regenerate it in a temporary directory and require every output byte to match:

```bash
python examples/export_stage7_island_results.py verify
```

The directory contains:

- `summary.json`: complete machine-readable result and gate status;
- `report.md`: short human-readable interpretation;
- `policies.csv`: policy and algorithm-family aggregates;
- `achievements.csv`: per-achievement rates and intervals;
- `comparisons.csv`: paired comparisons against legal random; and
- `manifest.json`: input provenance and output hashes.

Large checkpoints, raw episodes, curves, and complete timing records are release artifacts rather
than Git contents.

## Curated Stage 7 replays

`replays/stage7_curated_v1/` contains five deterministic Replay 2.3 inspection artifacts:

- matched legal random;
- strongest feed-forward PPO progression;
- strongest recurrent PPO progression;
- representative learned failure; and
- scripted-oracle 15/15 Perfect Run with 2/2 rescued.

These episodes were selected after final evaluation for explanation and presentation. They are not
additional benchmark samples and support no quantitative claim. Their manifests pin the original
episode, checkpoint, result artifact, and terminal outcome.

## Legacy Stage 5.6 reference

`reference/stage5_6_final_v1/`, `manifests/stage5_6_*.json`, and the Stage 5.5 checkpoints retain the
earlier ten-agent, 32×32, 300-step benchmark for reproducibility. It is a historical compatibility
result, not the canonical island benchmark.
