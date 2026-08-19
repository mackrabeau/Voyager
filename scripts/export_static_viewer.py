"""Export the curated replay API as static files for GitHub Pages."""

from __future__ import annotations

import argparse
from pathlib import Path

from voyager.replay.static_export import export_static_viewer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAYS = ROOT / "benchmarks/replays/stage7_curated_v1"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_static_viewer(args.replay_root, args.output)
    print(f"Static replay API: {args.output.resolve()}")


if __name__ == "__main__":
    main()
