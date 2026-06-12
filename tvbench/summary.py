from __future__ import annotations

from pathlib import Path
import json
import tarfile
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table


def load_jsons(directory: Path, pattern: str) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob(pattern))]


def write_final_table(final: pd.DataFrame, path: Path) -> None:
    table = Table(title="TurboVec Results", box=box.ASCII, show_lines=False)
    columns = [
        "count",
        "bit_width",
        "index_bytes",
        "bytes_per_vector",
        "compression_ratio",
        "p50_ms",
        "p95_ms",
        "median_qps",
        "recall_at_10",
        "peak_rss_bytes",
    ]
    headers = {
        "count": "Vectors",
        "bit_width": "Bits",
        "index_bytes": "Index",
        "bytes_per_vector": "B/vector",
        "compression_ratio": "Compress",
        "p50_ms": "p50 ms",
        "p95_ms": "p95 ms",
        "median_qps": "QPS",
        "recall_at_10": "Recall@10",
        "peak_rss_bytes": "Peak RSS",
    }

    for column in columns:
        justify = "right" if column != "bit_width" else "center"
        table.add_column(headers[column], justify=justify)

    for row in final[columns].itertuples(index=False, name=None):
        (
            count,
            bit_width,
            index_bytes,
            bytes_per_vector,
            compression_ratio,
            p50_ms,
            p95_ms,
            median_qps,
            recall_at_10,
            peak_rss_bytes,
        ) = row
        table.add_row(
            f"{count:,}",
            str(bit_width),
            f"{index_bytes / 1024 / 1024:.2f} MiB",
            f"{bytes_per_vector:.2f}",
            f"{compression_ratio:.2f}x",
            f"{p50_ms:.3f}",
            f"{p95_ms:.3f}",
            f"{median_qps:,.0f}",
            f"{recall_at_10:.3f}",
            f"{peak_rss_bytes / 1024 / 1024:.2f} MiB",
        )

    console = Console(record=True, width=120)
    console.print(table)
    path.write_text(console.export_text(styles=False))


def create_summary(config: dict) -> Path:
    run_dir: Path = config["paths"]["run_dir"]
    results_dir = run_dir / "results"
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    builds = load_jsons(results_dir, "build-full-*.json")
    latency = load_jsons(results_dir, "benchmark-full-*-latency.json")
    throughput = load_jsons(results_dir, "benchmark-full-*-throughput.json")

    if not builds or not latency or not throughput:
        raise RuntimeError("missing build or benchmark result files")

    build_df = pd.DataFrame(builds)
    latency_df = pd.DataFrame(latency)[
        ["bit_width", "count", "p50_ms", "p95_ms", "load_seconds",
         "prepare_seconds", "rss_after_prepare_bytes"]
    ]
    throughput_df = pd.DataFrame(throughput)[
        ["bit_width", "count", "median_qps", "recall_at_10"]
    ]

    final = (
        build_df.merge(latency_df, on=["bit_width", "count"])
        .merge(throughput_df, on=["bit_width", "count"])
        .sort_values(["count", "bit_width"])
    )

    csv_path = results_dir / "final-summary.csv"
    final.to_csv(csv_path, index=False)
    write_final_table(final, results_dir / "final-summary.txt")

    def save(fig, filename: str) -> None:
        fig.tight_layout()
        fig.savefig(charts_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for bits, group in final.groupby("bit_width"):
        group = group.sort_values("count")
        ax.plot(
            group["count"] / 1e6,
            group["index_bytes"] / 1e9,
            marker="o",
            label=f"{bits}-bit",
        )
    ax.set_xlabel("Vectors (millions)")
    ax.set_ylabel("Persisted index size (GB)")
    ax.set_title("TurboVec persisted index size")
    ax.legend()
    save(fig, "index-size.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for bits, group in final.groupby("bit_width"):
        group = group.sort_values("count")
        ax.plot(
            group["count"] / 1e6,
            group["p95_ms"],
            marker="o",
            label=f"{bits}-bit",
        )
    ax.set_xlabel("Vectors (millions)")
    ax.set_ylabel("Warm single-query p95 (ms)")
    ax.set_title("Search latency scaling")
    ax.legend()
    save(fig, "search-scaling.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for bits, group in final.groupby("bit_width"):
        group = group.sort_values("count")
        ax.plot(
            group["count"] / 1e6,
            group["recall_at_10"],
            marker="o",
            label=f"{bits}-bit",
        )
    ax.set_xlabel("Vectors (millions)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 1.02)
    ax.set_title("Recall against ground truth")
    ax.legend()
    save(fig, "recall.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for bits, group in final.groupby("bit_width"):
        group = group.sort_values("count")
        ax.plot(
            group["count"] / 1e6,
            group["cumulative_add_seconds"] / 60,
            marker="o",
            label=f"{bits}-bit",
        )
    ax.set_xlabel("Vectors (millions)")
    ax.set_ylabel("Cumulative indexing time (minutes)")
    ax.set_title("Online indexing time")
    ax.legend()
    save(fig, "indexing-time.png")

    return csv_path


def package_artifacts(config: dict) -> Path:
    run_dir: Path = config["paths"]["run_dir"]
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = artifact_dir / f"turbovec-100m-{stamp}.tar.gz"

    included = [
        run_dir / "results",
        run_dir / "charts",
        run_dir / "logs",
        run_dir / "state",
        run_dir / "status.json",
        Path(__file__).resolve().parents[1] / "config",
        Path(__file__).resolve().parents[1] / "Makefile",
        Path(__file__).resolve().parents[1] / "pyproject.toml",
        Path(__file__).resolve().parents[1] / "uv.lock",
    ]

    with tarfile.open(archive, "w:gz") as tar:
        for path in included:
            if path.exists():
                tar.add(path, arcname=path.name)

    return archive
