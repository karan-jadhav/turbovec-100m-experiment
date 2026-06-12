from __future__ import annotations

from pathlib import Path
import numpy as np


def write_fbin(path: Path, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.asarray(array.shape, dtype=np.uint32).tofile(handle)
        array.tofile(handle)


def write_ibin(path: Path, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.asarray(array.shape, dtype=np.uint32).tofile(handle)
        array.tofile(handle)


def open_fbin(path: Path) -> np.memmap:
    header = np.fromfile(path, dtype=np.uint32, count=2)
    if header.size != 2:
        raise ValueError(f"invalid fbin header: {path}")
    count, dimension = map(int, header)
    expected = 8 + count * dimension * 4
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"{path}: header expects {expected:,} bytes, file has {actual:,}"
        )
    return np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        offset=8,
        shape=(count, dimension),
        order="C",
    )


def open_ibin(path: Path) -> np.memmap:
    header = np.fromfile(path, dtype=np.uint32, count=2)
    if header.size != 2:
        raise ValueError(f"invalid ibin header: {path}")
    rows, columns = map(int, header)
    expected = 8 + rows * columns * 4
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"{path}: header expects {expected:,} bytes, file has {actual:,}"
        )
    return np.memmap(
        path,
        dtype=np.int32,
        mode="r",
        offset=8,
        shape=(rows, columns),
        order="C",
    )


def human_bytes(value: int | float) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")
