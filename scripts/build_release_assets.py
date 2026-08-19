"""Build deterministic v0.1.0 research-preview release artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import tarfile
from collections import deque
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "v0.1.0"
EXPERIMENT = ROOT / "results/stage7/procedural_baselines_v1"
ALGORITHMS = ("feed_forward_ppo", "recurrent_ppo")
SEEDS = (0, 1, 2)


def build_release_assets(output: Path) -> list[Path]:
    destination = output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _require_experiment()
    artifacts: list[Path] = []

    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            checkpoint = EXPERIMENT / "runs" / algorithm / f"seed_{seed}" / "checkpoints/best"
            name = f"voyager-island-{VERSION}-checkpoint-{algorithm.replace('_', '-')}-seed-{seed}.tar.gz"
            artifact = destination / name
            _archive_tree(
                artifact,
                checkpoint,
                Path("checkpoints") / algorithm / f"seed_{seed}" / "best",
            )
            artifacts.append(artifact)

    bundles = {
        f"voyager-island-{VERSION}-raw-episode-evaluations.tar.gz": [
            (path, Path("evaluations") / path.relative_to(EXPERIMENT / "final"))
            for path in sorted((EXPERIMENT / "final").rglob("*"))
            if path.is_file()
        ],
        f"voyager-island-{VERSION}-training-histories.tar.gz": _history_members(),
        f"voyager-island-{VERSION}-experiment-configuration.tar.gz": _configuration_members(),
        f"voyager-island-{VERSION}-curated-replays.tar.gz": [
            (path, Path("replays/stage7_curated_v1") / path.relative_to(ROOT / "benchmarks/replays/stage7_curated_v1"))
            for path in sorted((ROOT / "benchmarks/replays/stage7_curated_v1").rglob("*"))
            if path.is_file()
        ],
    }
    for name, members in bundles.items():
        artifact = destination / name
        _archive(artifact, members)
        artifacts.append(artifact)

    curves = destination / f"voyager-island-{VERSION}-training-curves.svg"
    curves.write_text(_training_curves_svg(), encoding="utf-8")
    artifacts.append(curves)

    manifest = destination / f"voyager-island-{VERSION}-artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "release": VERSION,
                "benchmark": "VoyagerIsland-v1",
                "experiment_contract": "voyager_island_procedural_baselines_v1",
                "artifacts": [
                    {"filename": path.name, "bytes": path.stat().st_size}
                    for path in artifacts
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.append(manifest)

    checksums = destination / "SHA256SUMS"
    checksums.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )
    artifacts.append(checksums)
    return artifacts


def _require_experiment() -> None:
    missing = [
        path
        for path in (EXPERIMENT / "protocol.json", EXPERIMENT / "final/final_evaluation.json")
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing official experiment artifacts: {missing}")


def _history_members() -> list[tuple[Path, Path]]:
    names = {
        "artifact_manifest.json",
        "development_evaluations.json",
        "environment_seed_history.json",
        "reward_diagnostics.json",
        "selection.json",
        "status.json",
        "training_history.csv",
        "training_history.jsonl",
    }
    return [
        (path, Path("training") / path.relative_to(EXPERIMENT / "runs"))
        for path in sorted((EXPERIMENT / "runs").rglob("*"))
        if path.is_file() and path.name in names
    ]


def _configuration_members() -> list[tuple[Path, Path]]:
    members = [(EXPERIMENT / "protocol.json", Path("configuration/protocol.json"))]
    members.extend(
        (
            EXPERIMENT / "runs" / algorithm / f"seed_{seed}" / "run_config.json",
            Path("configuration") / algorithm / f"seed_{seed}" / "run_config.json",
        )
        for algorithm in ALGORITHMS
        for seed in SEEDS
    )
    members.extend(
        (path, Path("configuration/manifests") / path.name)
        for path in sorted((ROOT / "benchmarks/manifests").glob("island_v1_*.json"))
    )
    return members


def _archive_tree(destination: Path, source: Path, prefix: Path) -> None:
    _archive(
        destination,
        [
            (path, prefix / path.relative_to(source))
            for path in sorted(source.rglob("*"))
            if path.is_file()
        ],
    )


def _archive(destination: Path, members: Iterable[tuple[Path, Path]]) -> None:
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for source, name in members:
            if not source.is_file():
                raise FileNotFoundError(source)
            info = archive.gettarinfo(str(source), arcname=name.as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            with source.open("rb") as handle:
                archive.addfile(info, handle)


def _training_curves_svg() -> str:
    series: list[tuple[str, str, list[tuple[float, float]], list[tuple[float, float]]]] = []
    colors = ["#edc967", "#f08a5d", "#62b6cb", "#7cc576", "#b388eb", "#ef6f9a"]
    color_index = 0
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            path = EXPERIMENT / "runs" / algorithm / f"seed_{seed}" / "training_history.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            returns = _rolling_points(rows, "mean_return", 100)
            throughput = _rolling_points(rows, "agent_steps_per_second", 100)
            label = f"{algorithm.replace('_ppo', '').replace('_', ' ').title()} seed {seed}"
            series.append((label, colors[color_index], returns, throughput))
            color_index += 1
    return _svg_document(series)


def _rolling_points(
    rows: list[dict[str, str]], field: str, window: int
) -> list[tuple[float, float]]:
    values: deque[float] = deque(maxlen=window)
    points: list[tuple[float, float]] = []
    stride = max(1, len(rows) // 500)
    for index, row in enumerate(rows):
        values.append(float(row[field]))
        if index % stride == 0 or index == len(rows) - 1:
            points.append((float(row["agent_steps"]), sum(values) / len(values)))
    return points


def _svg_document(
    series: list[tuple[str, str, list[tuple[float, float]], list[tuple[float, float]]]]
) -> str:
    width, height = 1200, 760
    left, right = 90, 1140
    panels = [
        (100, 360, "Mean rollout return (100-update moving average)", 2),
        (440, 700, "Training throughput, agent transitions/s (100-update moving average)", 3),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0f1714"/>',
        '<style>text{font-family:ui-monospace,monospace;fill:#e9efdf}.axis{stroke:#718078;stroke-width:1}.grid{stroke:#27352f;stroke-width:1}.curve{fill:none;stroke-width:2;opacity:.9}</style>',
        '<text x="60" y="45" font-size="24" font-weight="700">VoyagerIsland-v1 official baseline training curves</text>',
        '<text x="60" y="70" font-size="13" fill="#a9b7ae">Six runs · one million agent-transition budget · raw histories attached separately</text>',
    ]
    for top, bottom, title, data_index in panels:
        all_values = [point[1] for item in series for point in item[data_index]]
        minimum, maximum = min(all_values), max(all_values)
        padding = max((maximum - minimum) * 0.05, 1e-6)
        minimum -= padding
        maximum += padding
        parts.extend(
            [
                f'<text x="{left}" y="{top - 16}" font-size="15" font-weight="700">{html.escape(title)}</text>',
                f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
            ]
        )
        for tick in range(6):
            x = left + (right - left) * tick / 5
            steps = int(1_000_000 * tick / 5)
            parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>')
            parts.append(f'<text x="{x:.1f}" y="{bottom + 20}" font-size="11" text-anchor="middle">{steps // 1000}k</text>')
        for tick in range(5):
            y = bottom - (bottom - top) * tick / 4
            value = minimum + (maximum - minimum) * tick / 4
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
            parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" font-size="11" text-anchor="end">{value:.2f}</text>')
        for _label, color, *_data in series:
            points = _data[data_index - 2]
            coordinates = " ".join(
                f"{left + (right - left) * min(x, 1_000_000) / 1_000_000:.1f},{bottom - (bottom - top) * (y - minimum) / (maximum - minimum):.1f}"
                for x, y in points
            )
            parts.append(f'<polyline class="curve" stroke="{color}" points="{coordinates}"/>')
    for index, (label, color, *_data) in enumerate(series):
        x = 80 + (index % 3) * 370
        y = 735 + (index // 3) * 18
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 32}" y="{y + 4}" font-size="11">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = build_release_assets(args.output)
    for artifact in artifacts:
        print(f"{artifact.name}\t{artifact.stat().st_size}")


if __name__ == "__main__":
    main()
