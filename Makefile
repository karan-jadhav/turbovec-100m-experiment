SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

UV ?= uv
UV_RUN := $(UV) run
UV_CACHE_DIR ?= .uv-cache
MPLCONFIGDIR ?= .matplotlib-cache
export UV_CACHE_DIR
export MPLCONFIGDIR
SESSION ?= turbovec100m
PROFILE ?= aws

.PHONY: help setup add-dep local local-resume local-status local-logs clean-local \
        aws-install aws-mount aws-start aws-resume aws-status aws-logs aws-attach \
        aws-stop aws-reset telegram-test telegram-chat-id _aws-run lint

help:
	@printf '%s\n' \
	  'TurboVec 100M experiment' \
	  '' \
	  'Local:' \
	  '  make setup             Sync uv environment and verify imports' \
	  '  make add-dep PKG=...   Add a dependency with uv add' \
	  '  make local             Clean and run the full tiny local pipeline' \
	  '  make local-resume      Resume the local pipeline' \
	  '  make local-status      Show local status' \
	  '  make local-logs        Follow the current local phase log' \
	  '  make clean-local       Delete runtime/local' \
	  '' \
	  'Telegram:' \
	  '  make telegram-chat-id  Read chat IDs from recent bot updates' \
	  '  make telegram-test     Send a test notification' \
	  '' \
	  'AWS:' \
	  '  make aws-install       Install Ubuntu packages, uv, and Python dependencies' \
	  '  make aws-mount DATA_DEVICE=/dev/... INDEX_DEVICE=/dev/...' \
	  '  make aws-start         Start/resume the AWS pipeline in detached tmux' \
	  '  make aws-status        Show progress' \
	  '  make aws-logs          Follow the main tmux log' \
	  '  make aws-attach        Attach to the tmux session' \
	  '  make aws-stop          Kill the tmux session' \
	  '  make aws-reset         Delete AWS state/results only, not data/indexes'

setup:
	@$(UV) sync
	@$(UV_RUN) python -c "import numpy, pandas, psutil, yaml, turbovec; print('uv environment ready')"

add-dep:
	@test -n "$(PKG)" || (echo 'PKG is required, for example: make add-dep PKG=numpy' >&2; exit 2)
	@$(UV) add '$(PKG)'

local: setup
	@$(UV_RUN) python experiment.py run --profile local --reset

local-resume: setup
	@$(UV_RUN) python experiment.py run --profile local --resume

local-status:
	@$(UV_RUN) python experiment.py status --profile local

local-logs:
	@$(UV_RUN) python experiment.py logs --profile local --follow

clean-local:
	@rm -rf runtime/local
	@echo "Deleted runtime/local"

aws-install:
	@sudo apt-get update
	@sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
		aria2 curl htop jq make nvme-cli sysstat tmux xfsprogs
	@if ! command -v "$(UV)" >/dev/null 2>&1; then \
		curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh; \
		sudo env UV_INSTALL_DIR=/usr/local/bin sh /tmp/uv-install.sh; \
	fi
	@$(UV) python install 3.13
	@$(MAKE) setup

aws-mount:
	@test -n "$(DATA_DEVICE)" || (echo 'DATA_DEVICE is required' >&2; exit 2)
	@test -n "$(INDEX_DEVICE)" || (echo 'INDEX_DEVICE is required' >&2; exit 2)
	@sudo DATA_DEVICE="$(DATA_DEVICE)" INDEX_DEVICE="$(INDEX_DEVICE)" \
		bash scripts/mount_nvme.sh

aws-start: setup
	@SESSION="$(SESSION)" PROFILE=aws bash scripts/start_tmux.sh

aws-resume: aws-start

_aws-run:
	@$(UV_RUN) python experiment.py run --profile aws --resume

aws-status:
	@$(UV_RUN) python experiment.py status --profile aws

aws-logs:
	@mkdir -p runtime/aws/logs
	@touch runtime/aws/logs/tmux.log
	@tail -n 200 -f runtime/aws/logs/tmux.log

aws-attach:
	@tmux attach-session -t "$(SESSION)"

aws-stop:
	@tmux kill-session -t "$(SESSION)"
	@echo "Stopped tmux session $(SESSION)"

aws-reset:
	@$(UV_RUN) python experiment.py reset --profile aws
	@echo "Deleted AWS state, logs, results, charts and artifacts."
	@echo "Dataset and indexes were left untouched."

telegram-chat-id:
	@$(UV_RUN) python telegram_notify.py get-chat-id

telegram-test:
	@$(UV_RUN) python telegram_notify.py send "TurboVec 100M notification test succeeded."

lint:
	@$(UV_RUN) python -m compileall -q experiment.py benchmark_worker.py telegram_notify.py tvbench
	@bash -n scripts/mount_nvme.sh
	@bash -n scripts/start_tmux.sh
	@echo "Syntax checks passed"
