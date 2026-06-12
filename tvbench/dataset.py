from __future__ import annotations

from pathlib import Path
import json
import subprocess
import numpy as np

from .io import open_fbin, open_ibin, write_fbin, write_ibin


def normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def exact_topk(
    base: np.ndarray,
    queries: np.ndarray,
    k: int,
    query_batch: int = 25,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    base_t = np.asarray(base, dtype=np.float32).T

    for start in range(0, len(queries), query_batch):
        end = min(start + query_batch, len(queries))
        scores = np.asarray(queries[start:end], dtype=np.float32) @ base_t
        candidates = np.argpartition(scores, -k, axis=1)[:, -k:]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(candidate_scores, axis=1)[:, ::-1]
        rows.append(np.take_along_axis(candidates, order, axis=1).astype(np.int32))

    return np.vstack(rows)


def generate_synthetic(config: dict) -> dict:
    data_dir: Path = config["paths"]["data_dir"]
    dataset = config["dataset"]
    count = int(dataset["vectors"])
    dimension = int(dataset["dimensions"])
    query_count = int(dataset["queries"])
    gt_k = int(dataset["ground_truth_k"])
    seed = int(dataset["seed"])
    checkpoints = [int(item) for item in dataset["checkpoints"]]

    rng = np.random.default_rng(seed)
    base = normalize(
        rng.normal(size=(count, dimension)).astype(np.float32)
    ).astype(np.float32)

    smallest_checkpoint = min(checkpoints)
    anchors = rng.choice(smallest_checkpoint, size=query_count, replace=False)
    queries = normalize(
        base[anchors]
        + rng.normal(scale=0.03, size=(query_count, dimension)).astype(np.float32)
    ).astype(np.float32)

    write_fbin(data_dir / "vectors.bin", base)
    write_fbin(data_dir / "query.bin", queries)

    for checkpoint in checkpoints:
        truth = exact_topk(base[:checkpoint], queries, gt_k)
        write_ibin(data_dir / f"gt-{checkpoint}.ibin", truth)

    return {
        "vectors": count,
        "dimensions": dimension,
        "queries": query_count,
        "checkpoints": checkpoints,
    }


def download_msmarco(config: dict, log) -> None:
    data_dir: Path = config["paths"]["data_dir"]
    urls: dict[str, str] = config["dataset"]["urls"]
    data_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in urls.items():
        output = data_dir / filename
        if output.exists() and output.stat().st_size > 8:
            log(f"Existing file will be checked/resumed: {output}")

        command = [
            "aria2c",
            "--continue=true",
            "--allow-overwrite=false",
            "--auto-file-renaming=false",
            "--file-allocation=none",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=16M",
            "--summary-interval=30",
            f"--dir={data_dir}",
            f"--out={filename}",
            url,
        ]
        log("$ " + " ".join(command))
        subprocess.run(command, check=True)


def ground_truth_path(config: dict, count: int, full_count: int) -> Path:
    data_dir: Path = config["paths"]["data_dir"]
    if config["dataset"]["source"] == "synthetic":
        return data_dir / f"gt-{count}.ibin"

    if count == 1_000_000:
        return data_dir / "msmarco-1M-gt100"
    if count == 10_000_000:
        return data_dir / "msmarco-10M-gt100"
    if count == full_count:
        return data_dir / "msmarco-100M-gt100"
    raise ValueError(f"no ground truth file for count={count}")


def resolve_checkpoints(config: dict, full_count: int) -> list[int]:
    resolved: list[int] = []
    for item in config["dataset"]["checkpoints"]:
        value = full_count if item == "full" else int(item)
        if value > full_count:
            raise ValueError(
                f"checkpoint {value:,} exceeds dataset count {full_count:,}"
            )
        resolved.append(value)
    return sorted(set(resolved))


def verify_dataset(config: dict) -> dict:
    data_dir: Path = config["paths"]["data_dir"]
    base = open_fbin(data_dir / "vectors.bin")
    queries = open_fbin(data_dir / "query.bin")
    count, dimension = map(int, base.shape)

    expected_dimension = int(config["dataset"]["dimensions"])
    if dimension != expected_dimension:
        raise ValueError(
            f"base dimension {dimension} != configured {expected_dimension}"
        )
    if queries.shape[1] != dimension:
        raise ValueError("base and query dimensions differ")

    sample_ids = np.linspace(0, count - 1, num=min(1024, count), dtype=np.int64)
    sample = np.asarray(base[sample_ids])
    query_sample = np.asarray(queries[: min(1024, len(queries))])

    if not np.isfinite(sample).all() or not np.isfinite(query_sample).all():
        raise ValueError("non-finite values detected")

    checkpoints = resolve_checkpoints(config, count)
    gt_summary = {}
    for checkpoint in checkpoints:
        path = ground_truth_path(config, checkpoint, count)
        truth = open_ibin(path)
        if truth.shape[0] != queries.shape[0]:
            raise ValueError(
                f"{path} has {truth.shape[0]} rows, expected {queries.shape[0]}"
            )
        if truth.shape[1] < int(config["benchmark"]["k"]):
            raise ValueError(f"{path} does not contain enough neighbors")
        if truth.min() < 0 or truth.max() >= checkpoint:
            raise ValueError(
                f"{path} contains IDs outside [0, {checkpoint})"
            )
        gt_summary[str(checkpoint)] = {
            "rows": int(truth.shape[0]),
            "neighbors": int(truth.shape[1]),
            "min_id": int(truth.min()),
            "max_id": int(truth.max()),
        }

    return {
        "vectors": count,
        "dimensions": dimension,
        "queries": int(queries.shape[0]),
        "base_file_bytes": int((data_dir / "vectors.bin").stat().st_size),
        "float32_payload_gb_decimal": count * dimension * 4 / 1e9,
        "sample_norm_p50": float(
            np.percentile(np.linalg.norm(sample, axis=1), 50)
        ),
        "query_norm_p50": float(
            np.percentile(np.linalg.norm(query_sample, axis=1), 50)
        ),
        "checkpoints": checkpoints,
        "ground_truth": gt_summary,
    }
