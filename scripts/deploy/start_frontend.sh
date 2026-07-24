#!/usr/bin/env bash
# Start the FinQuery frontend (Vite dev server) in a tmux session.
set -euo pipefail

__DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${__DIR}/load_env.sh"

SESSION="${TMUX_SESSION_FRONTEND}"
PID_FILE="${PID_DIR}/frontend.pid"
STATUS_FILE="${STATUS_DIR}/frontend.status"
LOG_FILE="${LOG_DIR}/frontend.log"
LAUNCHER="${RUNTIME_DIR}/.launch_frontend.sh"
FRONTEND_DIR="${REPO_ROOT}/finquery_rag/frontend"

: "${FRONTEND_HOST:=127.0.0.1}"
: "${FRONTEND_PORT:=18003}"
: "${BACKEND_HOST:=127.0.0.1}"
: "${BACKEND_PORT:=18002}"
: "${VITE_API_URL:=/api}"
: "${VITE_API_PROXY_TARGET:=http://${BACKEND_HOST}:${BACKEND_PORT}}"

write_status() { printf '%s\n' "$1" > "${STATUS_FILE}"; }

echo "[frontend] Session: ${SESSION}  Host: ${FRONTEND_HOST}  Port: ${FRONTEND_PORT}"

# Already running?
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}" 2>/dev/null || true)" 2>/dev/null; then
    echo "[frontend] Already running (PID $(cat "${PID_FILE}"))."
    write_status "READY"
    exit 0
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[frontend] tmux session '${SESSION}' already exists. Run stop_all.sh first." >&2
    write_status "FAILED"
    exit 1
fi

# Pre-flight: backend must be reachable.
__BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}/healthz"
__code="$(curl -s -o /dev/null -w '%{http_code}' "${__BACKEND_URL}" 2>/dev/null || true)"
if [[ "${__code}" != "200" ]]; then
    echo "[frontend] Backend not available at ${__BACKEND_URL} (status: ${__code}). Start the backend first." >&2
    write_status "FAILED"
    exit 1
fi

# Pre-flight: node/npm must exist.
if ! command -v node >/dev/null 2>&1; then
    echo "[frontend] node is not installed or not on PATH." >&2
    write_status "FAILED"
    exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
    echo "[frontend] npm is not installed or not on PATH." >&2
    write_status "FAILED"
    exit 1
fi

# Pre-flight: port checks.
if [[ "${FRONTEND_PORT}" -le 1024 ]]; then
    echo "[frontend] FRONTEND_PORT must be > 1024 (got ${FRONTEND_PORT})." >&2
    write_status "FAILED"
    exit 1
fi
if ! port_is_free "${FRONTEND_PORT}"; then
    echo "[frontend] Port ${FRONTEND_PORT} is already in use." >&2
    write_status "FAILED"
    exit 1
fi

# Install dependencies quietly if node_modules is missing.
if [[ ! -d "${FRONTEND_DIR}/node_modules" ]]; then
    echo "[frontend] node_modules missing; running npm install ..."
    (cd "${FRONTEND_DIR}" && npm install --no-audit --no-fund)
fi

# Generate a launcher script (avoids nested-quoting issues with tmux).
{
    printf '#!/usr/bin/env bash\n'
    printf 'set -e\n'
    printf 'echo $$ > %s\n' "$(shell_squote "${PID_FILE}")"
    printf 'cd %s\n' "$(shell_squote "${FRONTEND_DIR}")"
    printf 'export %s=%s\n' "VITE_API_URL" "$(shell_squote "${VITE_API_URL}")"
    printf 'export %s=%s\n' "VITE_API_PROXY_TARGET" "$(shell_squote "${VITE_API_PROXY_TARGET}")"
    printf 'exec npm run dev -- --host %s --port %s > %s 2>&1\n' \
        "$(shell_squote "${FRONTEND_HOST}")" \
        "$(shell_squote "${FRONTEND_PORT}")" \
        "$(shell_squote "${LOG_FILE}")"
} > "${LAUNCHER}"

: > "${LOG_FILE}"
tmux new-session -d -s "${SESSION}" "bash $(shell_squote "${LAUNCHER}")"

echo "[frontend] Started in tmux session '${SESSION}'. Waiting for HTTP 200 (up to 60s)..."

__URL="http://${FRONTEND_HOST}:${FRONTEND_PORT}/"
if wait_for_http_checked "${__URL}" 60 "${PID_FILE}"; then
    __pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${__pid}" ]]; then
        write_pid_meta "${PID_FILE}" "${__pid}" "npm" "${SESSION}"
    fi
    write_status "READY"
    echo "[frontend] READY (PID $(cat "${PID_FILE}" 2>/dev/null || echo "?"))."
    exit 0
fi

echo "[frontend] FAILED to become healthy within 60s. See ${LOG_FILE}." >&2
write_status "FAILED"
exit 1
