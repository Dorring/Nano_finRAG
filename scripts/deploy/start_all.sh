#!/usr/bin/env bash
# Start all Phase 7 services (model -> backend -> frontend), failing fast on error.
# On any step failure, only services started by THIS run are rolled back;
# services that were already running before start_all.sh are left untouched.
set -euo pipefail

__DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${__DIR}/load_env.sh"

echo "=== Starting all services (model -> backend -> frontend) ==="

# Snapshot the set of tmux sessions that already exist, so the cleanup trap
# can distinguish "started by this run" from "pre-existing".
PRE_EXISTING_SESSIONS="$(tmux ls -F '#{session_name}' 2>/dev/null || true)"

# Returns 0 (true) if <session> exists now AND was NOT in the pre-existing
# snapshot (i.e. we created it during this run).
session_started_by_this_run() {
    local s="$1"
    tmux has-session -t "${s}" 2>/dev/null || return 1
    if grep -qxF "${s}" <<<"${PRE_EXISTING_SESSIONS}" 2>/dev/null; then
        return 1
    fi
    return 0
}

# Roll back services started by this run, in reverse order. Never touches
# pre-existing sessions.
cleanup_started() {
    local exit_code=$?
    if [[ "${exit_code}" -eq 0 ]]; then
        return 0
    fi
    echo "[start_all] Startup step failed (exit ${exit_code}); rolling back services started by this run..." >&2
    local session name
    for session in "${TMUX_SESSION_FRONTEND}" "${TMUX_SESSION_BACKEND}" "${TMUX_SESSION_MODEL}"; do
        if session_started_by_this_run "${session}"; then
            case "${session}" in
                "${TMUX_SESSION_MODEL}")    name="model" ;;
                "${TMUX_SESSION_BACKEND}")  name="backend" ;;
                "${TMUX_SESSION_FRONTEND}") name="frontend" ;;
                *) name="" ;;
            esac
            if [[ -n "${name}" ]]; then
                "${__DIR}/stop_one.sh" "${name}" || true
            fi
        fi
    done
}
trap cleanup_started EXIT

"${__DIR}/start_model.sh"
"${__DIR}/start_backend.sh"
"${__DIR}/start_frontend.sh"

trap - EXIT
echo "=== All services started ==="

read_status() {
    local f="$1"
    if [[ -f "$f" ]]; then
        cat "$f"
    else
        echo "UNKNOWN"
    fi
}

echo
printf 'Model service: %s\n'    "$(read_status "${STATUS_DIR}/model.status")"
printf 'Backend service: %s\n'  "$(read_status "${STATUS_DIR}/backend.status")"
printf 'Frontend service: %s\n' "$(read_status "${STATUS_DIR}/frontend.status")"
