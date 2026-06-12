#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-turbovec100m}"
PROFILE="${PROFILE:-aws}"
LOG="${ROOT}/runtime/${PROFILE}/logs/tmux.log"

mkdir -p "$(dirname "${LOG}")"
touch "${LOG}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session '${SESSION}' is already running."
  echo "Use: make aws-status"
  echo "Use: make aws-attach"
  exit 0
fi

COMMAND="cd '${ROOT}' && make _aws-run 2>&1 | tee -a '${LOG}'"
tmux new-session -d -s "${SESSION}" "${COMMAND}"

echo "Started tmux session: ${SESSION}"
echo "Status: make aws-status"
echo "Logs:   make aws-logs"
echo "Attach: make aws-attach"
