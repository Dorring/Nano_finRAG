#!/usr/bin/env bash
# Stop a single Phase 7 service by name (model | backend | frontend).
# Uses the same ownership-verified stop logic as stop_all.sh.
# Usage: bash scripts/deploy/stop_one.sh <model|backend|frontend>
set -euo pipefail

__DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${__DIR}/load_env.sh"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <model|backend|frontend>" >&2
    exit 2
fi

service="$1"

# stop_service <label> <session> <pid_file> <status_file> <marker>
# (defined in load_env.sh)
case "${service}" in
    model)
        stop_service "Model" "${TMUX_SESSION_MODEL}" "${PID_DIR}/model.pid" \
            "${STATUS_DIR}/model.status" "chat_openai_compat"
        ;;
    backend)
        stop_service "Backend" "${TMUX_SESSION_BACKEND}" "${PID_DIR}/backend.pid" \
            "${STATUS_DIR}/backend.status" "src.main:app"
        ;;
    frontend)
        stop_service "Frontend" "${TMUX_SESSION_FRONTEND}" "${PID_DIR}/frontend.pid" \
            "${STATUS_DIR}/frontend.status" "npm"
        ;;
    *)
        echo "Unknown service: '${service}'. Expected model|backend|frontend." >&2
        exit 2
        ;;
esac

echo "[stop_one] Done."
