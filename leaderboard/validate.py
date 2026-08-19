"""Validate submissions and check or regenerate the Markdown leaderboard."""

from __future__ import annotations

import argparse
from pathlib import Path

from voyager.benchmark.leaderboard import (
    load_submissions,
    render_generated_tables,
    update_readme,
)

ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD = ROOT / "leaderboard"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()

    submissions = load_submissions(
        LEADERBOARD / "submissions",
        LEADERBOARD / "schema.json",
    )
    readme_path = LEADERBOARD / "README.md"
    current = readme_path.read_text(encoding="utf-8")
    expected = update_readme(current, render_generated_tables(submissions))
    if args.check:
        if current != expected:
            raise SystemExit(
                "leaderboard/README.md is stale; run python leaderboard/validate.py --write"
            )
        print(f"Validated {len(submissions)} submissions; leaderboard is current.")
        return 0
    readme_path.write_text(expected, encoding="utf-8")
    print(f"Validated {len(submissions)} submissions and updated {readme_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
