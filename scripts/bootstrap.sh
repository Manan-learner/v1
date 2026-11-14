#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-gdrive:AI-Agent-Memory}"
RESTORE="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3 -m pip install -e "${REPO_ROOT}"

STATE_HOME="${V1_MEMORY_HOME:-${HOME}/.v1-memory}"
mkdir -p "${STATE_HOME}"
if [ ! -f "${STATE_HOME}/context_finder_config.json" ]; then
  cp "${REPO_ROOT}/examples/context_finder_config.json" "${STATE_HOME}/context_finder_config.json"
fi

if [ "${RESTORE}" = "--restore" ]; then
  if ! command -v rclone >/dev/null 2>&1; then
    echo "ERROR: rclone is required for remote restore on Linux/macOS." >&2
    exit 1
  fi
  rclone copy "${REMOTE}/providers/antigravity" "${HOME}/.gemini" --update --progress || true
  rclone copy "${REMOTE}/providers/claude" "${HOME}/.claude" --update --progress || true
  rclone copy "${REMOTE}/providers/codex" "${HOME}/.codex" --update --progress || true
  rclone copy "${REMOTE}/providers/v1-memory" "${STATE_HOME}" --update --progress || true
fi

V1_SYNC_BIN="$(command -v v1-sync || true)"
if [ -n "${V1_SYNC_BIN}" ] && command -v crontab >/dev/null 2>&1; then
  CRON_LINE="0 21 * * * \"${V1_SYNC_BIN}\" --remote \"${REMOTE}\" >/dev/null 2>&1"
  (crontab -l 2>/dev/null | grep -v "v1-sync" || true; echo "${CRON_LINE}") | crontab -
  echo "Installed daily 9:00 PM cron job for v1-sync."
fi

v1-context --reindex || true
echo "Bootstrap complete! Run v1-status to inspect backup and provider state."
