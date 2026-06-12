# TurboVec 100M automated experiment

This repository contains a reproducible benchmark pipeline for evaluating
TurboVec index construction, storage size, memory use, search latency,
throughput, and recall at increasing corpus sizes.

The experiment is motivated by a practical scaling question: if a 10 million
vector, 768-dimensional float32 corpus can be compressed substantially with
TurboVec, what happens when the same approach is evaluated at 1M, 10M, and full
100M MS MARCO Web Search scale?

The pipeline measures:

- online indexing time
- persisted index size
- peak RSS during indexing
- warm single-query p50 and p95 latency
- 32-thread batch throughput
- recall@10
- scaling at 1M, 10M, and the full corpus

It supports two profiles using the same pipeline:

- `local`: a generated 5,000-vector dataset for a complete end-to-end validation
- `aws`: the real MS MARCO Web Search corpus

The AWS run is launched in a detached tmux session and sends Telegram notifications.

## Commands

```bash
make help
make setup
make add-dep PKG='package-name'
make telegram-test
make local
make local-status

make aws-install
make aws-mount DATA_DEVICE=/dev/nvme1n1 INDEX_DEVICE=/dev/nvme2n1
make aws-start
make aws-status
make aws-logs
make aws-attach
make aws-resume
```

Python scripts are run through uv:

```bash
uv run python experiment.py status --profile local
uv run python telegram_notify.py send "test"
```

Dependencies are managed in `pyproject.toml` and `uv.lock`. Add or update dependencies with `uv add`, or through the Make helper:

```bash
uv add pandas
make add-dep PKG='pandas'
```

## Telegram setup

1. Create a bot using `@BotFather`.
2. Send any message to the new bot.
3. Copy `.env.example` to `.env`.
4. Put the bot token in `.env`.
5. Run `make telegram-chat-id`.
6. Put the returned chat ID in `.env`.
7. Run `make telegram-test`.

Telegram failure never aborts the benchmark. It is reported in the phase log.

## Local validation

```bash
make setup
make local
```

This executes the full pipeline:

```text
generate synthetic fbin/ibin data
→ verify it
→ build 4-bit
→ benchmark 4-bit
→ build 2-bit
→ benchmark 2-bit
→ create final CSV and charts
→ package artifacts
```

Expected output:

```text
runtime/local/
├── data/
├── indexes/
├── logs/
├── results/
├── charts/
├── state/
└── artifacts/
```

Run the local pipeline twice before AWS:

```bash
make local
make local
```

`make local` starts clean each time. Use `make local-resume` only when testing resume behavior.

## AWS preparation

Recommended instance:

- `i7i.8xlarge`
- Ubuntu Server 24.04 LTS, x86_64
- 100 GiB gp3 root disk
- both local 3.75 TB NVMe instance-store devices
- On-Demand
- detailed monitoring enabled
- termination protection enabled

Install system packages:

```bash
make aws-install
```

`make aws-install` also installs uv to `/usr/local/bin` when it is missing and installs the Python version from `.python-version`.

Identify the two local instance-store disks:

```bash
nvme list
lsblk
```

Mount them separately:

```bash
make aws-mount \
  DATA_DEVICE=/dev/nvme1n1 \
  INDEX_DEVICE=/dev/nvme2n1
```

This destroys existing contents on the two specified devices.

Paths:

```text
/mnt/tv-data     dataset
/mnt/tv-index    TurboVec indexes

runtime/aws      state, logs, results, charts and artifact bundle
```

The small evidence files stay on the root EBS disk. The large source dataset and indexes stay on instance-store NVMe.

## Start the unattended AWS run

```bash
make aws-start
```

This creates a detached tmux session named `turbovec100m`.

You can disconnect immediately.

Check progress:

```bash
make aws-status
```

Follow logs:

```bash
make aws-logs
```

Attach to tmux:

```bash
make aws-attach
```

Detach without stopping the run:

```text
Ctrl-b d
```

Resume after a failed or interrupted phase:

```bash
make aws-resume
```

Completed phases have `.done` markers and are skipped. A partially completed index build restarts that bit-width build from the beginning.

## AWS phases

```text
01_preflight
02_download
03_verify_dataset
04_smoke_build_4bit
05_smoke_benchmark_4bit
06_build_4bit
07_benchmark_4bit
08_build_2bit
09_benchmark_2bit
10_summarize
11_package
```

Notifications are sent for:

- run started
- real 1M smoke test passed
- full 4-bit build completed
- full 2-bit build completed
- run completed
- any failure

## Status

`make aws-status` reads `runtime/aws/status.json`.

During a build it shows:

```text
Profile: aws
Status: running
Phase: 06_build_4bit
Progress: 53,000,000 / 101,070,374
Elapsed: 00:47:21
Current RSS: 41.70 GiB
Average rate: 1,119,000 vectors/s
Last update: 2026-06-12T14:21:00+00:00
```

## Runtime guard

The AWS profile has a 20-hour maximum runtime. The runner checks it before every phase and during long builds.

A failure creates:

```text
runtime/aws/state/FAILED
runtime/aws/status.json
runtime/aws/logs/<phase>.log
runtime/aws/artifacts/
```

The instance is deliberately not stopped automatically. Review the Telegram completion message, copy the artifact bundle, and then terminate the instance.

## Final result files

```text
runtime/aws/results/final-summary.csv
runtime/aws/charts/index-size.png
runtime/aws/charts/search-scaling.png
runtime/aws/charts/recall.png
runtime/aws/charts/indexing-time.png
runtime/aws/artifacts/turbovec-100m-<timestamp>.tar.gz
```
