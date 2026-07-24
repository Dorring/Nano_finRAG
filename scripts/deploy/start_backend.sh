#!/usr/bin/env bash
# Start the FinQuery RAG backend (uvicorn) in a tmux session.
set -euo pipefail

__DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${__DIR}/load_env.sh"

SESSION="${TMUX_SESSION_BACKEND}"
PID_FILE="${PID_DIR}/backend.pid"
STATUS_FILE="${STATUS_DIR}/backend.status"
LOG_FILE="${LOG_DIR}/backend.log"
LAUNCHER="${RUNTIME_DIR}/.launch_backend.sh"
BACKEND_DIR="${REPO_ROOT}/finquery_rag/backend"

: "${BACKEND_HOST:=127.0.0.1}"
: "${BACKEND_PORT:=18002}"
: "${MODEL_HOST:=127.0.0.1}"
: "${MODEL_PORT:=18001}"

write_status() { printf '%s\n' "$1" > "${STATUS_FILE}"; }

echo "[backend] Session: ${SESSION}  Host: ${BACKEND_HOST}  Port: ${BACKEND_PORT}"

# Already running?
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}" 2>/dev/null || true)" 2>/dev/null; then
    echo "[backend] Already running (PID $(cat "${PID_FILE}"))."
    write_status "READY"
    exit 0
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[backend] tmux session '${SESSION}' already exists. Run stop_all.sh first." >&2
    write_status "FAILED"
    exit 1
fi

# Pre-flight: model service must be reachable.
__MODEL_URL="http://${MODEL_HOST}:${MODEL_PORT}/health"
__code="$(curl -s -o /dev/null -w '%{http_code}' "${__MODEL_URL}" 2>/dev/null || true)"
if [[ "${__code}" != "200" ]]; then
    echo "[backend] Model service not available at ${__MODEL_URL} (status: ${__code}). Start the model first." >&2
    write_status "FAILED"
    exit 1
fi

# Pre-flight: port checks.
if [[ "${BACKEND_PORT}" -le 1024 ]]; then
    echo "[backend] BACKEND_PORT must be > 1024 (got ${BACKEND_PORT})." >&2
    write_status "FAILED"
    exit 1
fi
if ! port_is_free "${BACKEND_PORT}"; then
    echo "[backend] Port ${BACKEND_PORT} is already in use." >&2
    write_status "FAILED"
    exit 1
fi

# Choose how to run uvicorn. Prefer a direct venv binary (clean PID), then uv run.
__UVICORN="uvicorn"
if [[ -n "${BACKEND_VENV_PATH:-}" && -x "${BACKEND_VENV_PATH}/bin/uvicorn" ]]; then
    __UVICORN="${BACKEND_VENV_PATH}/bin/uvicorn"
elif [[ -x "${BACKEND_DIR}/.venv/bin/uvicorn" ]]; then
    __UVICORN="${BACKEND_DIR}/.venv/bin/uvicorn"
elif command -v uv >/dev/null 2>&1; then
    __UVICORN="uv run uvicorn"
fi

# Environment variables the backend depends on (passed explicitly into the session).
__BACKEND_ENV_VARS=(
    LLM_API_BASE_URL LLM_API_KEY LLM_MODEL_NAME
    DATABASE_URL CHROMA_PATH BM25_DB_PATH
    DOCUMENT_REGISTRY_DB_PATH SESSIONS_DB_PATH TRACE_DB_PATH
    SECRET_KEY ALLOWED_ORIGINS
)

# Generate a launcher script (avoids nested-quoting issues with tmux).
{
    printf '#!/usr/bin/env bash\n'
    printf 'set -e\n'
    printf 'echo $$ > %s\n' "$(shell_squote "${PID_FILE}")"
    printf 'cd %s\n' "$(shell_squote "${BACKEND_DIR}")"
    __v=""
    for __v in "${__BACKEND_ENV_VARS[@]}"; do
        printf 'export %s=%s\n' "${__v}" "$(shell_squote "${!__v:-}")"
    done
    printf 'exec %s src.main:app --host %s --port %s --workers 1 > %s 2>&1\n' \
        "${__UVICORN}" \
        "$(shell_squote "${BACKEND_HOST}")" \
        "$(shell_squote "${BACKEND_PORT}")" \
        "$(shell_squote "${LOG_FILE}")"
} > "${LAUNCHER}"

: > "${LOG_FILE}"
tmux new-session -d -s "${SESSION}" "bash $(shell_squote "${LAUNCHER}")"

echo "[backend] Started in tmux session '${SESSION}'. Waiting for /healthz (up to 60s)..."

__URL="http://${BACKEND_HOST}:${BACKEND_PORT}/healthz"
if wait_for_http_checked "${__URL}" 60 "${PID_FILE}"; then
    __pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${__pid}" ]]; then
        write_pid_meta "${PID_FILE}" "${__pid}" "src.main:app" "${SESSION}"
    fi
    write_status "READY"
    echo "[backend] READY (PID $(cat "${PID_FILE}" 2>/dev/null || echo "?"))."
    exit 0
fi

echo "[backend] FAILED to become healthy within 60s. See ${LOG_FILE}." >&2
write_status "FAILED"
exit 1
