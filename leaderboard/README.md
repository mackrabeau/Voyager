# Voyager Island leaderboard

This leaderboard uses the frozen `VoyagerIsland-v1` test split and public achievement score.
Submissions arrive as pull requests containing data, provenance, disclosures, and episode outcomes.
CI validates the JSON and recomputes the score; it never executes submitted policy or training code.

<!-- BEGIN GENERATED LEADERBOARD -->

### Learned agents under the official interaction budget

Ranked by the frozen public score. Three training seeds and 1M agent transitions are required.

| Method | Score (95% CI) | Rescue | Replay | Policy source |
|---|---:|---:|---|---|
| Feed-Forward PPO | 0.164 (0.049–0.198) | 0% | [replay](https://fawzmehfil.github.io/Voyager/replays/island_feed_forward_progression_v1) | [source@dc5458df](https://github.com/fawzmehfil/Voyager/tree/dc5458dfdab18e76790d74815625f0d372fbf3d6) |
| Recurrent PPO | 0.072 (0.032–0.106) | 0% | [replay](https://fawzmehfil.github.io/Voyager/replays/island_recurrent_progression_v1) | [source@dc5458df](https://github.com/fawzmehfil/Voyager/tree/dc5458dfdab18e76790d74815625f0d372fbf3d6) |

### External-knowledge / foundation-model agents

Reported separately because external data or pretrained models change the comparison.

_No accepted submissions yet._

### Scripted references and oracle (not ranked)

Diagnostic comparators only; these entries are excluded from learned-agent ranking.

| Method | Score (95% CI) | Rescue | Replay | Policy source |
|---|---:|---:|---|---|
| Legal Random | 0.023 (0.019–0.026) | 0% | [replay](https://fawzmehfil.github.io/Voyager/replays/island_legal_random_matched_v1) | [source@dc5458df](https://github.com/fawzmehfil/Voyager/tree/dc5458dfdab18e76790d74815625f0d372fbf3d6) |
| Scripted Public-Observation Oracle | 0.986 (0.973–0.997) | 96% | [replay](https://fawzmehfil.github.io/Voyager/replays/island_oracle_rescue_v1) | [source@dc5458df](https://github.com/fawzmehfil/Voyager/tree/dc5458dfdab18e76790d74815625f0d372fbf3d6) |

<!-- END GENERATED LEADERBOARD -->

## Submission contract

1. Install the package and evaluate an importable local policy:

   ```bash
   voyager-eval \
     --policy my_package.policy:create_policy \
     --split test \
     --output submission.json
   ```

   A factory for one fixed policy may accept the evaluation seed or no arguments. A learned-family
   factory evaluated with repeated `--training-seed` flags must accept the keyword arguments
   `training_seed` and `evaluation_seed` and return an independent policy instance.

2. Complete the generated draft: choose a stable `submission_id`, replace every disclosure and
   artifact placeholder, attach or link checkpoints and raw records, and add one representative
   replay. Learned official-budget entries require exactly three training seeds and one million
   agent transitions per seed.

3. Copy the file to `leaderboard/submissions/<submission_id>.json`, then run:

   ```bash
   python leaderboard/validate.py --write
   git diff --check
   ```

4. Open a pull request containing the submission JSON and regenerated `leaderboard/README.md`.

Every learned submission declares the benchmark/environment versions, method category, three
training seeds, interaction budget, exact policy repository commit, checkpoint links and hashes,
external data/pretrained-model/privileged-information disclosures, aggregate score and 95% interval,
all 15 achievement rates, rescue rate, raw episode records, and a representative replay.

## Categories

- `learned_official_budget`: ranked; three training seeds, 1M agent transitions each, no external
  knowledge beyond the declared benchmark training interface.
- `external_knowledge`: separate, non-comparable table for foundation models, external data,
  pretrained policies, privileged information, or a different interaction budget.
- `scripted_reference`: random/scripted/oracle diagnostics, excluded from ranking.

The validator accepts only the frozen seeds `20000–20099`, exactly once for every declared training
seed. It derives rates and the smoothed geometric-mean score directly from episode achievements.
Confidence intervals remain part of the scientific submission record and must contain the recomputed
point estimate.
