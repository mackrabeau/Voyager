"""Synchronize generated viewer and curated replay files into the Python wheel."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    ROOT / "web/dist": ROOT / "voyager/server/static",
    ROOT / "benchmarks/replays/stage7_curated_v1": (
        ROOT / "voyager/data/replays/stage7_curated_v1"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when packaged assets differ from their source directories.",
    )
    args = parser.parse_args()
    for source, destination in SOURCES.items():
        if not source.is_dir():
            raise SystemExit(f"Missing source directory: {source}")
        if args.check:
            if _files(source) != _files(destination):
                raise SystemExit(f"Packaged assets are stale: {destination}")
            continue
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        print(f"Synchronized {source.relative_to(ROOT)} -> {destination.relative_to(ROOT)}")


def _files(root: Path) -> dict[str, bytes]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".DS_Store"
    }


if __name__ == "__main__":
    main()
