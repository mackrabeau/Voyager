# Voyager Benchmarks

Stage 5.6 freezes the Stage 5.5 ten-agent, 32x32, 300-step scenario and evaluates
three independently trained PPO policies on explicit held-out seed suites.

Run the development suite:

```bash
python examples/run_benchmark.py \
  --manifest benchmarks/manifests/stage5_6_dev.json \
  --output results/benchmark/stage5_6_dev_v1
```

Run or resume the final suite:

```bash
python examples/run_benchmark.py \
  --manifest benchmarks/manifests/stage5_6_final.json \
  --output results/benchmark/stage5_6_final_v1 \
  --resume
```

Deterministic PPO is the official learned-policy result. Stochastic PPO is reported
separately as a diagnostic. Generated episode-level artifacts remain under `results/`;
the final manifest, summary, and CSV tables are copied into `benchmarks/reference/`.

## VoyagerIsland-v1 procedural baselines

The Stage 7 reference package freezes six one-million-transition runs: shared feed-forward
PPO and shared recurrent PPO, each trained with seeds 0/1/2. Checkpoints were selected on
the 50-island development split and evaluated once on the 100-island held-out test split.
Seeded-stochastic inference is primary; deterministic inference is diagnostic.

Generate the compact tracked reference from the completed local experiment:

```bash
python examples/export_stage7_island_results.py export
```

Regenerate it in a temporary directory and require every output byte to match:

```bash
python examples/export_stage7_island_results.py verify
```

The frozen report is under `benchmarks/reference/voyager_island_v1_baselines_v1/`. It
contains a JSON summary, policy and achievement CSVs, paired comparisons against legal
random, a human-readable report, source hashes, and output hashes. Raw episodes, curves,
checkpoints, and replays remain external release artifacts rather than Git contents.
