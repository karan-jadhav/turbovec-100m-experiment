#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import psutil

from tvbench.config import load_config
from tvbench.dataset import ground_truth_path
from tvbench.io import open_fbin, open_ibin


def recall_at_10(prediction: np.ndarray, truth: np.ndarray) -> float:
    values = []
    for predicted, expected in zip(
        prediction[:, :10], truth[:, :10], strict=True
    ):
        values.append(np.intersect1d(predicted, expected).size / 10)
    return float(np.mean(values))


def query_ids(config: dict, query_count: int) -> np.ndarray:
    run_dir: Path = config["paths"]["run_dir"]
    path = run_dir / "results" / "query-sample-ids.npy"
    wanted = min(int(config["benchmark"]["query_sample_size"]), query_count)

    if path.exists():
        ids = np.load(path)
        if ids.size == wanted and ids.max(initial=-1) < query_count:
            return ids

    rng = np.random.default_rng(int(config["benchmark"]["query_seed"]))
    ids = np.sort(
        rng.choice(query_count, size=wanted, replace=False)
    ).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, ids)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--bit-width", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--mode", choices=["latency", "throughput"], required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.profile)
    data_dir: Path = config["paths"]["data_dir"]

    from turbovec import TurboQuantIndex

    base = open_fbin(data_dir / "vectors.bin")
    full_count = int(base.shape[0])
    queries_all = open_fbin(data_dir / "query.bin")
    ids = query_ids(config, len(queries_all))
    queries = np.ascontiguousarray(queries_all[ids], dtype=np.float32)

    load_started = time.perf_counter()
    index = TurboQuantIndex.load(args.index)
    load_seconds = time.perf_counter() - load_started

    if len(index) != args.count:
        raise ValueError(f"index has {len(index)} vectors, expected {args.count}")

    prepare_started = time.perf_counter()
    index.prepare()
    prepare_seconds = time.perf_counter() - prepare_started
    rss = psutil.Process().memory_info().rss

    k = int(config["benchmark"]["k"])
    index.search(queries[: min(20, len(queries))], k=k)

    result = {
        "profile": args.profile,
        "label": args.label,
        "bit_width": args.bit_width,
        "count": args.count,
        "mode": args.mode,
        "rayon_threads": int(os.getenv("RAYON_NUM_THREADS", "0") or 0),
        "index_path": str(Path(args.index).resolve()),
        "index_bytes": Path(args.index).stat().st_size,
        "load_seconds": load_seconds,
        "prepare_seconds": prepare_seconds,
        "rss_after_prepare_bytes": rss,
    }

    if args.mode == "latency":
        wanted = min(int(config["benchmark"]["latency_queries"]), len(queries))
        repeats = int(config["benchmark"]["latency_repeats"])
        latencies = []

        for _ in range(repeats):
            for i in range(wanted):
                started = time.perf_counter_ns()
                _, prediction = index.search(queries[i : i + 1], k=k)
                latencies.append(
                    (time.perf_counter_ns() - started) / 1_000_000
                )
                if prediction.min() < 0 or prediction.max() >= args.count:
                    raise ValueError("out-of-range search result")

        values = np.asarray(latencies, dtype=np.float64)
        result.update(
            {
                "query_measurements": len(values),
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
            }
        )
    else:
        repeats = int(config["benchmark"]["throughput_repeats"])
        qps_values = []
        prediction = None

        for _ in range(repeats):
            started = time.perf_counter()
            _, prediction = index.search(queries, k=k)
            elapsed = time.perf_counter() - started
            qps_values.append(len(queries) / elapsed)

        assert prediction is not None
        if prediction.min() < 0 or prediction.max() >= args.count:
            raise ValueError("out-of-range search result")

        truth_all = open_ibin(
            ground_truth_path(config, args.count, full_count)
        )
        truth = np.asarray(truth_all[ids], dtype=np.int64)
        prediction = np.asarray(prediction, dtype=np.int64)

        result.update(
            {
                "queries_per_run": len(queries),
                "throughput_repeats": repeats,
                "median_qps": float(np.median(qps_values)),
                "recall_at_10": recall_at_10(prediction, truth),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
