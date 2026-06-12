#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Callable, Any

from telegram_notify import load_env, send as telegram_send
from tvbench.build import build_indexes
from tvbench.config import ROOT, ensure_run_dirs, load_config
from tvbench.dataset import (
    download_msmarco,
    generate_synthetic,
    resolve_checkpoints,
    verify_dataset,
)
from tvbench.io import human_bytes, open_fbin
from tvbench.summary import create_summary, package_artifacts


class Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class Runner:
    def __init__(self, profile: str, resume: bool, reset: bool) -> None:
        load_env()
        self.profile = profile
        self.config = load_config(profile)
        ensure_run_dirs(self.config)
        self.run_dir: Path = self.config["paths"]["run_dir"]
        self.state_dir = self.run_dir / "state"
        self.logs_dir = self.run_dir / "logs"
        self.results_dir = self.run_dir / "results"
        self.status_path = self.run_dir / "status.json"
        self.resume = resume
        self.reset_requested = reset
        self.run_started_monotonic = time.monotonic()
        self.run_started_wall = utc_now()
        self.deadline = self.run_started_monotonic + (
            float(self.config["runtime"]["max_hours"]) * 3600
        )
        self.current_phase = ""

        if reset:
            self.reset_runtime()
            ensure_run_dirs(self.config)

    def reset_runtime(self) -> None:
        names = ["state", "logs", "results", "charts", "artifacts"]
        if self.profile == "local":
            names.extend(["data", "indexes"])
        for name in names:
            path = self.run_dir / name
            if path.exists():
                shutil.rmtree(path)
        if self.status_path.exists():
            self.status_path.unlink()

    def notify(self, message: str) -> None:
        if bool(self.config["runtime"].get("telegram", False)):
            telegram_send(message)

    def update_status(self, **updates: Any) -> None:
        previous = {}
        if self.status_path.exists():
            try:
                previous = json.loads(self.status_path.read_text())
            except json.JSONDecodeError:
                previous = {}

        status = {
            **previous,
            "profile": self.profile,
            "run_started": previous.get(
                "run_started", self.run_started_wall
            ),
            "last_updated": utc_now(),
            "elapsed_seconds": time.monotonic() - self.run_started_monotonic,
            "current_phase": self.current_phase,
            **updates,
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2, sort_keys=True))
        temporary.replace(self.status_path)

    def phase_done(self, name: str) -> bool:
        return (self.state_dir / f"{name}.done").exists()

    def mark_done(self, name: str) -> None:
        marker = self.state_dir / f"{name}.done"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(utc_now())

    def check_deadline(self) -> None:
        if time.monotonic() > self.deadline:
            raise TimeoutError(
                f"run exceeded {self.config['runtime']['max_hours']} hours"
            )

    def log(self, message: str) -> None:
        print(f"[{utc_now()}] {message}", flush=True)

    def run_phase(
        self,
        name: str,
        function: Callable[[], Any],
        *,
        notification: str | None = None,
    ) -> Any:
        if self.resume and self.phase_done(name):
            self.current_phase = name
            self.update_status(status="skipped", message="already complete")
            print(f"SKIP {name}: already complete")
            return None

        self.check_deadline()
        self.current_phase = name
        phase_started = time.perf_counter()
        log_path = self.logs_dir / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self.update_status(
            status="running",
            phase_started=utc_now(),
            message="phase started",
            vectors_completed=None,
            vectors_total=None,
            current_rss_bytes=None,
            average_vectors_per_second=None,
        )

        with log_path.open("a") as file_log:
            tee_out = Tee(sys.stdout, file_log)
            tee_err = Tee(sys.stderr, file_log)
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                self.log(f"START {name}")
                try:
                    result = function()
                    self.mark_done(name)
                    elapsed = time.perf_counter() - phase_started
                    self.update_status(
                        status="phase_complete",
                        phase_seconds=elapsed,
                        message="phase completed",
                    )
                    self.log(f"DONE {name} in {format_duration(elapsed)}")
                    if notification:
                        self.notify(notification)
                    return result
                except Exception:
                    elapsed = time.perf_counter() - phase_started
                    self.update_status(
                        status="failed",
                        phase_seconds=elapsed,
                        message="phase failed",
                        error=traceback.format_exc(),
                    )
                    (self.state_dir / "FAILED").write_text(
                        f"{name}\n{utc_now()}\n{traceback.format_exc()}"
                    )
                    self.log(f"FAILED {name}")
                    traceback.print_exc()
                    self.notify(
                        f"❌ TurboVec 100M failed\n"
                        f"Profile: {self.profile}\n"
                        f"Phase: {name}\n"
                        f"Check: {log_path}"
                    )
                    raise

    def preflight(self) -> dict:
        data_dir: Path = self.config["paths"]["data_dir"]
        index_dir: Path = self.config["paths"]["index_dir"]

        if self.profile == "aws":
            for command in ("aria2c", "tmux", "nvme"):
                if shutil.which(command) is None:
                    raise RuntimeError(f"missing required command: {command}")

            for path in (Path("/mnt/tv-data"), Path("/mnt/tv-index")):
                if not path.is_mount():
                    raise RuntimeError(f"{path} is not a mount point")

            host_commands = {
                "lscpu.txt": ["lscpu"],
                "memory.txt": ["free", "-h"],
                "nvme.txt": ["nvme", "list"],
                "lsblk.txt": [
                    "lsblk", "-o",
                    "NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS",
                ],
                "uname.txt": ["uname", "-a"],
                "df.txt": ["df", "-hT"],
            }
            host_dir = self.results_dir / "host"
            host_dir.mkdir(parents=True, exist_ok=True)
            host_outputs = {}
            for filename, command in host_commands.items():
                output = subprocess.run(
                    command, check=True, text=True, capture_output=True
                ).stdout
                (host_dir / filename).write_text(output)
                host_outputs[filename] = output

            lscpu = host_outputs["lscpu.txt"]
            if "avx512bw" not in lscpu.lower():
                raise RuntimeError("AVX-512BW is not visible")

        disk_data = shutil.disk_usage(data_dir)
        disk_index = shutil.disk_usage(index_dir)
        summary = {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "data_dir": str(data_dir),
            "index_dir": str(index_dir),
            "data_free_bytes": disk_data.free,
            "index_free_bytes": disk_index.free,
        }
        (self.results_dir / "host-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        self.log(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    def generate_or_download(self) -> None:
        if self.config["dataset"]["source"] == "synthetic":
            summary = generate_synthetic(self.config)
            self.log(json.dumps(summary, indent=2, sort_keys=True))
        else:
            download_msmarco(self.config, self.log)

    def verify(self) -> dict:
        summary = verify_dataset(self.config)
        (self.results_dir / "dataset-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        self.log(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    def build(
        self,
        *,
        bit_width: int,
        label: str,
        max_vectors: int | None = None,
    ) -> list[dict]:
        return build_indexes(
            self.config,
            bit_width=bit_width,
            label=label,
            max_vectors=max_vectors,
            status_callback=self.update_status,
            deadline_monotonic=self.deadline,
            log=self.log,
        )

    def benchmark(
        self,
        *,
        bit_width: int,
        label: str,
        counts: list[int],
    ) -> None:
        threads = int(self.config["benchmark"]["throughput_threads"])

        for count in counts:
            index = self.config["paths"]["index_dir"] / (
                f"turbovec-{bit_width}bit-{label}-{count}.tv"
            )
            if not index.exists():
                raise FileNotFoundError(index)

            for mode, rayon_threads in (("latency", 1), ("throughput", threads)):
                self.check_deadline()
                output = self.results_dir / (
                    f"benchmark-{label}-{bit_width}bit-{count}-{mode}.json"
                )
                command = [
                    sys.executable,
                    str(ROOT / "benchmark_worker.py"),
                    "--profile",
                    self.profile,
                    "--label",
                    label,
                    "--bit-width",
                    str(bit_width),
                    "--count",
                    str(count),
                    "--mode",
                    mode,
                    "--index",
                    str(index),
                    "--output",
                    str(output),
                ]
                environment = os.environ.copy()
                environment.update(
                    {
                        "RAYON_NUM_THREADS": str(rayon_threads),
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    }
                )
                self.log(
                    f"benchmark bit={bit_width} count={count:,} "
                    f"mode={mode} threads={rayon_threads}"
                )
                subprocess.run(command, check=True, env=environment)

    def package_final_artifacts(self, name: str) -> Path | None:
        log_path = self.logs_dir / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with log_path.open("a") as file_log:
            tee_out = Tee(sys.stdout, file_log)
            tee_err = Tee(sys.stderr, file_log)
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                self.log(f"START {name}")
                try:
                    archive = package_artifacts(self.config)
                    elapsed = time.perf_counter() - started
                    self.mark_done(name)
                    self.update_status(
                        artifact_status="complete",
                        artifact_path=str(archive),
                        artifact_phase_seconds=elapsed,
                    )
                    self.log(f"DONE {name} in {format_duration(elapsed)}")
                    return archive
                except Exception:
                    elapsed = time.perf_counter() - started
                    self.update_status(
                        artifact_status="failed",
                        artifact_phase_seconds=elapsed,
                        artifact_error=traceback.format_exc(),
                    )
                    (self.state_dir / "ARTIFACT_FAILED").write_text(
                        f"{name}\n{utc_now()}\n{traceback.format_exc()}"
                    )
                    self.log(f"WARNING {name} failed; benchmark remains complete")
                    traceback.print_exc()
                    self.notify(
                        f"⚠️ TurboVec 100M artifact packaging failed\n"
                        f"Profile: {self.profile}\n"
                        f"Results are complete.\n"
                        f"Check: {log_path}"
                    )
                    return None

    def execute(self) -> None:
        if (self.state_dir / "FAILED").exists():
            (self.state_dir / "FAILED").unlink()
        if (self.state_dir / "SUCCESS").exists() and self.resume:
            print("Pipeline is already complete.")
            return

        self.update_status(status="starting", message="pipeline starting")
        self.notify(
            f"🚀 TurboVec 100M started\n"
            f"Profile: {self.profile}\n"
            f"Host: {platform.node()}"
        )

        try:
            self.run_phase("01_preflight", self.preflight)
            self.run_phase("02_dataset", self.generate_or_download)
            verified = self.run_phase("03_verify_dataset", self.verify)

            if verified is None:
                verified = json.loads(
                    (self.results_dir / "dataset-summary.json").read_text()
                )

            full_count = int(verified["vectors"])
            checkpoints = resolve_checkpoints(self.config, full_count)

            if self.profile == "aws" and self.config.get("smoke", {}).get(
                "enabled", False
            ):
                smoke_count = int(self.config["smoke"]["vectors"])
                smoke_bit = int(self.config["smoke"]["bit_width"])
                self.run_phase(
                    "04_smoke_build_4bit",
                    lambda: self.build(
                        bit_width=smoke_bit,
                        label="smoke",
                        max_vectors=smoke_count,
                    ),
                )
                self.run_phase(
                    "05_smoke_benchmark_4bit",
                    lambda: self.benchmark(
                        bit_width=smoke_bit,
                        label="smoke",
                        counts=[smoke_count],
                    ),
                    notification=(
                        "✅ TurboVec 100M real-data smoke test passed\n"
                        f"Vectors: {smoke_count:,}"
                    ),
                )
                phase_offset = 6
            else:
                phase_offset = 4

            bits = [int(item) for item in self.config["build"]["bit_widths"]]
            phase_number = phase_offset
            for bit in bits:
                build_name = f"{phase_number:02d}_build_{bit}bit"
                bench_name = f"{phase_number + 1:02d}_benchmark_{bit}bit"

                self.run_phase(
                    build_name,
                    lambda bit_width=bit: self.build(
                        bit_width=bit_width,
                        label="full",
                    ),
                    notification=(
                        f"✅ TurboVec {bit}-bit full build completed\n"
                        f"Profile: {self.profile}"
                    ),
                )
                self.run_phase(
                    bench_name,
                    lambda bit_width=bit: self.benchmark(
                        bit_width=bit_width,
                        label="full",
                        counts=checkpoints,
                    ),
                )
                phase_number += 2

            self.run_phase(
                f"{phase_number:02d}_summarize",
                lambda: create_summary(self.config),
            )

            (self.state_dir / "SUCCESS").write_text(utc_now())
            self.current_phase = "complete"
            self.update_status(
                status="complete",
                message="pipeline completed successfully",
            )
            self.package_final_artifacts(f"{phase_number + 1:02d}_package")

            final_table = (self.results_dir / "final-summary.txt").read_text()

            self.notify(
                "✅ TurboVec 100M completed\n"
                f"Profile: {self.profile}\n"
                f"Vectors: {full_count:,}\n"
                f"Results: {self.results_dir / 'final-summary.csv'}\n"
                "Terminate the EC2 instance after copying the artifact bundle."
            )
            print("\nPIPELINE PASSED")
            print(final_table)
        except Exception:
            raise


def show_status(profile: str) -> None:
    config = load_config(profile)
    path: Path = config["paths"]["run_dir"] / "status.json"
    if not path.exists():
        print(f"No status file exists for profile {profile}.")
        return

    status = json.loads(path.read_text())
    print(f"Profile: {status.get('profile', profile)}")
    print(f"Status: {status.get('status', '-')}")
    print(f"Phase: {status.get('current_phase', '-')}")
    completed = status.get("vectors_completed")
    total = status.get("vectors_total")
    if completed is not None and total is not None:
        print(f"Progress: {completed:,} / {total:,}")
    print(f"Elapsed: {format_duration(status.get('elapsed_seconds'))}")
    rss = status.get("current_rss_bytes")
    if rss is not None:
        print(f"Current RSS: {human_bytes(rss)}")
    rate = status.get("average_vectors_per_second")
    if rate is not None:
        print(f"Average rate: {rate:,.0f} vectors/s")
    print(f"Message: {status.get('message', '-')}")
    print(f"Last update: {status.get('last_updated', '-')}")
    if status.get("error"):
        print("\nError:")
        print(status["error"])


def follow_logs(profile: str, follow: bool) -> None:
    config = load_config(profile)
    run_dir: Path = config["paths"]["run_dir"]
    status_path = run_dir / "status.json"

    phase = None
    if status_path.exists():
        phase = json.loads(status_path.read_text()).get("current_phase")
    if not phase:
        print("No current phase is available.")
        return

    log_path = run_dir / "logs" / f"{phase}.log"
    log_path.touch(exist_ok=True)
    command = ["tail", "-n", "200"]
    if follow:
        command.append("-f")
    command.append(str(log_path))
    subprocess.run(command, check=False)


def reset_profile(profile: str) -> None:
    config = load_config(profile)
    run_dir: Path = config["paths"]["run_dir"]
    names = ["state", "logs", "results", "charts", "artifacts"]
    if profile == "local":
        names.extend(["data", "indexes"])
    for name in names:
        path = run_dir / name
        if path.exists():
            shutil.rmtree(path)
    status = run_dir / "status.json"
    if status.exists():
        status.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--profile", choices=["local", "aws"], required=True)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--reset", action="store_true")

    status_parser = sub.add_parser("status")
    status_parser.add_argument("--profile", choices=["local", "aws"], required=True)

    logs_parser = sub.add_parser("logs")
    logs_parser.add_argument("--profile", choices=["local", "aws"], required=True)
    logs_parser.add_argument("--follow", action="store_true")

    reset_parser = sub.add_parser("reset")
    reset_parser.add_argument("--profile", choices=["local", "aws"], required=True)

    args = parser.parse_args()

    if args.command == "run":
        Runner(args.profile, resume=args.resume, reset=args.reset).execute()
    elif args.command == "status":
        show_status(args.profile)
    elif args.command == "logs":
        follow_logs(args.profile, args.follow)
    else:
        reset_profile(args.profile)


if __name__ == "__main__":
    main()
