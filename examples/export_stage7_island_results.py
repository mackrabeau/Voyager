"""Export or verify the frozen VoyagerIsland-v1 official baseline report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voyager.benchmark.island_results import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
    export_island_results,
    verify_island_result_reference,
)

DEFAULT_EXPERIMENT_ROOT = Path("results/stage7/procedural_baselines_v1")
DEFAULT_OUTPUT_DIR = Path("benchmarks/reference/voyager_island_v1_baselines_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Create the immutable compact reference.")
    export.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    export.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    export.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    export.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    export.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)

    verify = subparsers.add_parser(
        "verify",
        help="Regenerate and byte-compare an existing reference without modifying it.",
    )
    verify.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    verify.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "export":
        output = export_island_results(
            args.experiment_root,
            args.output_dir,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            confidence=args.confidence,
        )
        print(f"Frozen Stage 7 result reference: {output.resolve()}")
        return 0
    result = verify_island_result_reference(args.experiment_root, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
