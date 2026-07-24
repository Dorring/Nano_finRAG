#!/usr/bin/env bash
# Stop all Phase 7 services in reverse order (frontend -> backend -> model).
# Only stops processes we started: by tmux session name and by a recorded PID
# whose ownership is verified (start time + command marker) before any signal.
# Never uses pkill -f, killall, or killing by process name.
set -euo pipefail

__DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${__DIR}/load_env.sh"

# stop_service <label> <session> <pid_file> <status_file> <marker>
# (defined in load_env.sh)
stop_service "Frontend" "${TMUX_SESSION_FRONTEND}" "${PID_DIR}/frontend.pid" "${STATUS_DIR}/frontend.status" "npm"
stop_service "Backend"  "${TMUX_SESSION_BACKEND}"  "${PID_DIR}/backend.pid"  "${STATUS_DIR}/backend.status"  "src.main:app"
stop_service "Model"    "${TMUX_SESSION_MODEL}"    "${PID_DIR}/model.pid"    "${STATUS_DIR}/model.status"    "chat_openai_compat"

echo "[stop] Done."
