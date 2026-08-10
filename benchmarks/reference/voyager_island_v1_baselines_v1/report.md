# VoyagerIsland-v1 Official Baseline Results

This report freezes the first official procedural evaluation. Checkpoints were selected
only on the 50-island development split; the 100-island test split was opened once after
all six runs were locked. Seeded-stochastic inference is the primary learned-policy result.

## Headline results

| Policy | Public score | 95% CI | Training-seed SD | Rescue |
|---|---:|---:|---:|---:|
| Legal random | 0.023 | [0.019, 0.026] | — | 0.0% |
| Feed-forward PPO | 0.164 | [0.049, 0.198] | 0.083 | 0.0% |
| Recurrent PPO | 0.072 | [0.032, 0.106] | 0.040 | 0.0% |
| Scripted oracle | 0.986 | [0.973, 0.997] | — | 96.0% |

Legal random scored `0.023` and the scripted oracle scored `0.986`.
Feed-forward PPO learned gathering, delivery, the workbench, tools, and campfire construction
in multiple training seeds. Recurrent PPO did not improve on feed-forward PPO under the frozen
one-million-transition budget.

No learned policy built shelter, hunted, cooked meat, built the beacon, or rescued both agents
on a held-out island. The scripted oracle rescued both agents on 96% of islands, so the mechanics
are solvable even though the learned full task remains unsolved by these baselines.

## Frozen release-gate outcome

- PASS — `oracle_solvability_gate`
- PASS — `random_never_rescues`
- PASS — `random_rarely_builds_workbench`
- PASS — `learned_significantly_exceeds_random`
- PASS — `learned_middle_capability_at_least_25`
- FAIL — `learned_rescue_at_least_25`
- PASS — `all_ledgers_reconcile`
- FAIL — `large_reproducibility_artifacts_published`

Overall release gate: **FAIL**.

The environment is therefore supported as a difficult benchmark with demonstrable early-economy
learning and a clearly measured late-game frontier. It is not supported as a task solved by PPO.

## Statistical method

Per-policy intervals resample matched test islands. Algorithm-family intervals resample both the
three independent training seeds and the matched islands. Comparisons against legal random reuse
the same island resample on both sides. This preserves pairing and avoids treating 300 episodes as
300 independently trained policies. The public score is the frozen smoothed geometric mean over all
15 achievements. Deterministic argmax results are diagnostic and never replace the stochastic result.

See `summary.json` for complete statistics and provenance, and the CSV files for tables suitable
for analysis and plotting.
