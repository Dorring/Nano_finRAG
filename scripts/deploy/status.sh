#!/usr/bin/env bash
# Report status of all Phase 7 services.
set -euo pipefail

__DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${__DIR}/load_env.sh"

: "${MODEL_HOST:=127.0.0.1}";    : "${MODEL_PORT:=18001}"
: "${BACKEND_HOST:=127.0.0.1}";  : "${BACKEND_PORT:=18002}"
: "${FRONTEND_HOST:=127.0.0.1}"; : "${FRONTEND_PORT:=18003}"

# print_service <name> <session> <pid_file> <log_file> <health_url> <port>
print_service() {
    local name="$1" session="$2" pid_file="$3" log_file="$4" health_url="$5" port="$6"
    local tmux_ok="no" pid="-" pid_alive="no" health="DOWN" code state

    if tmux has-session -t "${session}" 2>/dev/null; then
        tmux_ok="yes"
    fi

    if [[ -f "${pid_file}" ]]; then
        pid="$(cat "${pid_file}" 2>/dev/null || echo "-")"
        if [[ "${pid}" != "-" ]] && kill -0 "${pid}" 2>/dev/null; then
            pid_alive="yes"
        fi
    fi

    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "${health_url}" 2>/dev/null || echo "000")"
    if [[ "${code}" == "200" ]]; then
        health="UP"
    fi

    state="STOPPED"
    if [[ "${tmux_ok}" == "yes" && "${pid_alive}" == "yes" && "${health}" == "UP" ]]; then
        state="RUNNING"
    elif [[ "${tmux_ok}" == "yes" || "${pid_alive}" == "yes" ]]; then
        state="PARTIAL"
    fi

    printf '%s: %s\n' "${name}" "${state}"
    printf '  PID: %s (alive: %s)\n' "${pid}" "${pid_alive}"
    printf '  Port: %s\n' "${port}"
    printf '  tmux: %s (%s)\n' "${tmux_ok}" "${session}"
    printf '  Health: %s (%s -> %s)\n' "${health}" "${health_url}" "${code}"
    if [[ -f "${log_file}" ]]; then
        printf '  Last log lines:\n'
        tail -n 3 "${log_file}" 2>/dev/null | sed 's/^/    /' || true
    fi
    printf '\n'
}

print_service "Model"    "${TMUX_SESSION_MODEL}"    "${PID_DIR}/model.pid"    "${LOG_DIR}/model.log"    "http://${MODEL_HOST}:${MODEL_PORT}/health"  "${MODEL_PORT}"
print_service "Backend"  "${TMUX_SESSION_BACKEND}"  "${PID_DIR}/backend.pid"  "${LOG_DIR}/backend.log"  "http://${BACKEND_HOST}:${BACKEND_PORT}/healthz" "${BACKEND_PORT}"
print_service "Frontend" "${TMUX_SESSION_FRONTEND}" "${PID_DIR}/frontend.pid" "${LOG_DIR}/frontend.log" "http://${FRONTEND_HOST}:${FRONTEND_PORT}/"      "${FRONTEND_PORT}"
