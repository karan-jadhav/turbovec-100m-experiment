from __future__ import annotations

from pathlib import Path
import json
import os
import threading
import time

import numpy as np
import psutil

from .dataset import resolve_checkpoints
from .io import open_fbin


class PeakRSS:
    def __init__(self) -> None:
        self.peak = 0
        self.stop = threading.Event()

    def run(self) -> None:
        proc = psutil.Process()
        while not self.stop.is_set():
            self.peak = max(self.peak, proc.memory_info().rss)
            self.stop.wait(0.5)


def build_indexes(
    config: dict,
    *,
    bit_width: int,
    label: str,
    max_vectors: int | None,
    status_callback,
    deadline_monotonic: float,
    log,
) -> list[dict]:
    from turbovec import TurboQuantIndex

    data_dir: Path = config["paths"]["data_dir"]
    index_dir: Path = config["paths"]["index_dir"]
    results_dir: Path = config["paths"]["run_dir"] / "results"
    index_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    base = open_fbin(data_dir / "vectors.bin")
    full_count, dimension = map(int, base.shape)
    target = min(max_vectors or full_count, full_count)

    if label == "smoke":
        checkpoints = [target]
    else:
        checkpoints = [
            value
            for value in resolve_checkpoints(config, full_count)
            if value <= target
        ]
        if target not in checkpoints:
            checkpoints.append(target)

    progress_path = results_dir / f"build-progress-{label}-{bit_width}bit.csv"
    if progress_path.exists():
        progress_path.unlink()

    index = TurboQuantIndex(dim=dimension, bit_width=bit_width)
    monitor = PeakRSS()
    monitor_thread = threading.Thread(target=monitor.run, daemon=True)
    monitor_thread.start()

    summaries: list[dict] = []
    total_add_seconds = 0.0
    run_started = time.perf_counter()
    current = 0
    proc = psutil.Process()

    try:
        with progress_path.open("w") as progress:
            progress.write(
                "bit_width,label,start,end,batch_vectors,batch_seconds,"
                "batch_vectors_per_second,cumulative_add_seconds,"
                "cumulative_vectors_per_second,rss_bytes\n"
            )

            for checkpoint in checkpoints:
                while current < checkpoint:
                    if time.monotonic() > deadline_monotonic:
                        raise TimeoutError("experiment exceeded configured runtime")

                    end = min(
                        current + int(config["build"]["batch_size"]),
                        checkpoint,
                    )
                    batch = np.ascontiguousarray(base[current:end], dtype=np.float32)

                    started = time.perf_counter()
                    index.add(batch)
                    elapsed = time.perf_counter() - started
                    total_add_seconds += elapsed
                    rss = proc.memory_info().rss
                    rate = end / max(total_add_seconds, 1e-9)

                    progress.write(
                        f"{bit_width},{label},{current},{end},{end-current},"
                        f"{elapsed},{(end-current)/elapsed},"
                        f"{total_add_seconds},{rate},{rss}\n"
                    )
                    progress.flush()

                    current = end
                    status_callback(
                        vectors_completed=current,
                        vectors_total=target,
                        current_rss_bytes=rss,
                        average_vectors_per_second=rate,
                    )

                index_path = index_dir / (
                    f"turbovec-{bit_width}bit-{label}-{checkpoint}.tv"
                )
                write_started = time.perf_counter()
                index.write(str(index_path))
                write_seconds = time.perf_counter() - write_started

                summary = {
                    "bit_width": bit_width,
                    "label": label,
                    "count": checkpoint,
                    "dimension": dimension,
                    "cumulative_add_seconds": total_add_seconds,
                    "wall_seconds_to_checkpoint": time.perf_counter() - run_started,
                    "average_vectors_per_second": (
                        checkpoint / max(total_add_seconds, 1e-9)
                    ),
                    "write_seconds": write_seconds,
                    "index_path": str(index_path),
                    "index_bytes": index_path.stat().st_size,
                    "bytes_per_vector": index_path.stat().st_size / checkpoint,
                    "raw_float32_bytes": checkpoint * dimension * 4,
                    "compression_ratio": (
                        checkpoint * dimension * 4 / index_path.stat().st_size
                    ),
                    "peak_rss_bytes": monitor.peak,
                }
                summary_path = results_dir / (
                    f"build-{label}-{bit_width}bit-{checkpoint}.json"
                )
                summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
                summaries.append(summary)
                log(
                    f"checkpoint={checkpoint:,} bit={bit_width} "
                    f"index_bytes={index_path.stat().st_size:,} "
                    f"add_seconds={total_add_seconds:.2f} "
                    f"peak_rss={monitor.peak:,}"
                )
    finally:
        monitor.stop.set()
        monitor_thread.join(timeout=2)

    return summaries
