#!/usr/bin/env bash
# Shared environment loader for Phase 7 deployment scripts.
# Source this from other scripts: . "$(dirname "$0")/load_env.sh"
set -euo pipefail

# Resolve repo root relative to this script's location (scripts/deploy/).
__DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${__DEPLOY_DIR}/../.." && pwd)"

# Choose config file: prefer online.env, fall back to the committed example.
__ENV_FILE="${REPO_ROOT}/config/deployment/online.env"
if [[ ! -f "${__ENV_FILE}" ]]; then
    __ENV_FILE="${REPO_ROOT}/config/deployment/online.env.example"
    echo "[load_env] config/deployment/online.env not found; using online.env.example" >&2
fi

# Load and export all variables from the chosen env file.
set -a
# shellcheck disable=SC1090
. "${__ENV_FILE}"
set +a

# Runtime directory layout.
RUNTIME_DIR="${REPO_ROOT}/runtime/phase7"
LOG_DIR="${RUNTIME_DIR}/logs"
PID_DIR="${RUNTIME_DIR}/pids"
STATUS_DIR="${RUNTIME_DIR}/status"
mkdir -p "${LOG_DIR}" "${PID_DIR}" "${STATUS_DIR}"

# tmux session names (defaults if not provided by the env file).
: "${TMUX_SESSION_MODEL:=nano-finance-model}"
: "${TMUX_SESSION_BACKEND:=nano-finance-backend}"
: "${TMUX_SESSION_FRONTEND:=nano-finance-frontend}"

export REPO_ROOT RUNTIME_DIR LOG_DIR PID_DIR STATUS_DIR
export TMUX_SESSION_MODEL TMUX_SESSION_BACKEND TMUX_SESSION_FRONTEND

# ---------------------------------------------------------------------------
# Helpers usable by any script that sources this file.
# ---------------------------------------------------------------------------

# Print a single-quoted, safely-escaped form of a value for shell interpolation.
shell_squote() {
    local s="$1"
    s="${s//\'/\'\\\'\'}"
    printf "'%s'" "$s"
}

# Wait until an HTTP endpoint returns 200, or until timeout (seconds).
# Optionally fail fast if the PID recorded in a pid file dies.
# Usage: wait_for_http_checked <url> <timeout_seconds> [pid_file]
# Returns: 0 healthy, 1 timeout, 2 process died.
wait_for_http_checked() {
    local url="$1"
    local timeout="$2"
    local pid_file="${3:-}"
    local elapsed=0 code p
    while [[ "${elapsed}" -lt "${timeout}" ]]; do
        if [[ -n "${pid_file}" && -f "${pid_file}" ]]; then
            p="$(cat "${pid_file}" 2>/dev/null || true)"
            if [[ -n "${p}" ]] && ! kill -0 "${p}" 2>/dev/null; then
                return 2
            fi
        fi
        code="$(curl -s -o /dev/null -w '%{http_code}' "${url}" 2>/dev/null || true)"
        if [[ "${code}" == "200" ]]; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

# Return 0 if a TCP port is free, 1 if in use. Best-effort across ss/netstat.
port_is_free() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
            return 1
        fi
    elif command -v netstat >/dev/null 2>&1; then
        if netstat -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
            return 1
        fi
    fi
    return 0
}

# ---------------------------------------------------------------------------
# PID ownership verification (prevents killing a reused/unrelated PID)
# ---------------------------------------------------------------------------

# Read the process start time (field 22 of /proc/<pid>/stat, in clock ticks).
# Prints the value; prints empty string if unavailable.
# Usage: _read_start_time <pid>
_read_start_time() {
    local pid="$1"
    local stat after_comm
    stat="$(cat "/proc/${pid}/stat" 2>/dev/null || true)"
    [[ -n "${stat}" ]] || { printf ''; return; }
    # Field 2 (comm) may contain spaces inside parens; strip up to last ')'.
    after_comm="${stat##*) }"
    # After the comm field, starttime is field 20 (22 - 2 removed fields).
    printf '%s' "$(echo "${after_comm}" | awk '{print $20}')"
}

# Read /proc/<pid>/cmdline as a single space-joined string.
# Usage: _read_cmdline <pid>
_read_cmdline() {
    local pid="$1"
    tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || printf ''
}

# Write a metadata file (pid, start time, command marker, tmux session)
# alongside a PID file, so stop scripts can verify ownership before killing.
# Usage: write_pid_meta <pid_file> <pid> <marker> <session>
write_pid_meta() {
    local pid_file="$1" pid="$2" marker="$3" session="$4"
    local meta_file="${pid_file}.meta"
    local start_time
    start_time="$(_read_start_time "${pid}")"
    {
        printf 'pid=%s\n' "${pid}"
        printf 'process_start_time=%s\n' "${start_time}"
        printf 'expected_command_marker=%s\n' "${marker}"
        printf 'tmux_session=%s\n' "${session}"
    } > "${meta_file}"
}

# Verify that a recorded PID still belongs to the service that wrote the PID
# file. Returns 0 (true) if the PID is owned by us and safe to kill; returns
# 1 (false) if the PID is stale, reused, or the command marker is absent.
# Usage: verify_pid_owner <pid_file> <marker>
verify_pid_owner() {
    local pid_file="$1" marker="$2"
    local meta_file="${pid_file}.meta"
    local pid recorded_start cur_start cmdline

    [[ -f "${pid_file}" ]] || return 1
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    [[ -n "${pid}" ]] || return 1

    # PID must still be alive.
    kill -0 "${pid}" 2>/dev/null || return 1

    # If /proc is available (Linux), verify start time + command marker.
    if [[ -r "/proc/${pid}/stat" ]]; then
        # --- start-time check (detects PID reuse) ---
        recorded_start=""
        if [[ -f "${meta_file}" ]]; then
            while IFS='=' read -r k v; do
                [[ "${k}" == "process_start_time" ]] && recorded_start="${v}"
            done < "${meta_file}" 2>/dev/null || true
        fi
        cur_start="$(_read_start_time "${pid}")"
        if [[ -n "${recorded_start}" && -n "${cur_start}" \
              && "${recorded_start}" != "${cur_start}" ]]; then
            return 1   # start time changed -> PID was reused
        fi
        # --- command-marker check (detects unrelated process) ---
        cmdline="$(_read_cmdline "${pid}")"
        if [[ -n "${marker}" && -n "${cmdline}" ]]; then
            case "${cmdline}" in
                *"${marker}"*) ;;             # matches -> ok
                *) return 1 ;;                 # marker absent -> not ours
            esac
        fi
    fi
    return 0
}

# Stop a single service safely: kill its tmux session, then verify-and-kill
# the recorded PID. Never kills a PID that fails ownership verification.
# Usage: stop_service <label> <session> <pid_file> <status_file> <marker>
stop_service() {
    local label="$1" session="$2" pid_file="$3" status_file="$4" marker="$5"
    echo "[stop] ${label} ..."

    # Kill the tmux session first; this takes the whole process tree with it.
    if tmux has-session -t "${session}" 2>/dev/null; then
        tmux kill-session -t "${session}"
        echo "[stop] ${label}: killed tmux session '${session}'."
    else
        echo "[stop] ${label}: no tmux session '${session}'."
    fi

    # Kill by recorded PID only after ownership verification.
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            if verify_pid_owner "${pid_file}" "${marker}"; then
                echo "[stop] ${label}: verified PID ${pid}; sending TERM."
                kill -TERM "${pid}" 2>/dev/null || true
                local waited=0
                while [[ "${waited}" -lt 10 ]] && kill -0 "${pid}" 2>/dev/null; do
                    sleep 1
                    waited=$((waited + 1))
                done
                if kill -0 "${pid}" 2>/dev/null; then
                    echo "[stop] ${label}: still alive after 10s; sending KILL."
                    kill -9 "${pid}" 2>/dev/null || true
                fi
            else
                echo "[stop] ${label}: STALE_PID — PID ${pid} no longer belongs to this service; not killing."
            fi
        else
            echo "[stop] ${label}: PID ${pid:-<none>} not running."
        fi
        rm -f "${pid_file}" "${pid_file}.meta"
    else
        echo "[stop] ${label}: no PID file."
    fi

    rm -f "${status_file}"
}
