#!/usr/bin/env bash
# Start the NanoChat OpenAI-compatible model service in a tmux session.
set -euo pipefail

__DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${__DIR}/load_env.sh"

SESSION="${TMUX_SESSION_MODEL}"
PID_FILE="${PID_DIR}/model.pid"
STATUS_FILE="${STATUS_DIR}/model.status"
LOG_FILE="${LOG_DIR}/model.log"
LAUNCHER="${RUNTIME_DIR}/.launch_model.sh"

: "${MODEL_HOST:=127.0.0.1}"
: "${MODEL_PORT:=18001}"
: "${MODEL_NAME:=finquery-finance-sft1147}"
: "${MODEL_SOURCE:=sft}"
: "${MODEL_TAG:=d24_final_mixdata}"
: "${MODEL_STEP:=1147}"
: "${MODEL_TEMPERATURE:=0}"
: "${MODEL_MAX_TOKENS:=512}"
: "${CONDA_ENV_NAME:=nano}"
: "${MODEL_PYTHON:=}"
: "${HF_HUB_OFFLINE:=1}"
: "${HF_DATASETS_OFFLINE:=1}"

write_status() { printf '%s\n' "$1" > "${STATUS_FILE}"; }

echo "[model] Session: ${SESSION}  Host: ${MODEL_HOST}  Port: ${MODEL_PORT}"

# Already running?
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}" 2>/dev/null || true)" 2>/dev/null; then
    echo "[model] Already running (PID $(cat "${PID_FILE}"))."
    write_status "READY"
    exit 0
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[model] tmux session '${SESSION}' already exists. Run stop_all.sh first." >&2
    write_status "FAILED"
    exit 1
fi

# Pre-flight checks.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[model] CUDA_VISIBLE_DEVICES is not set." >&2
    write_status "FAILED"
    exit 1
fi
if [[ "${MODEL_PORT}" -le 1024 ]]; then
    echo "[model] MODEL_PORT must be > 1024 (got ${MODEL_PORT})." >&2
    write_status "FAILED"
    exit 1
fi
if ! port_is_free "${MODEL_PORT}"; then
    echo "[model] Port ${MODEL_PORT} is already in use." >&2
    write_status "FAILED"
    exit 1
fi

# Build conda activation prefix (best-effort; assume python is correct if absent).
# If MODEL_PYTHON is set to an executable, use it directly (venv) and skip conda.
__CONDA_PRE=""
__PYTHON="python"
if [[ -n "${MODEL_PYTHON}" && -x "${MODEL_PYTHON}" ]]; then
    __PYTHON="${MODEL_PYTHON}"
elif command -v conda >/dev/null 2>&1; then
    __CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${__CONDA_BASE}" && -f "${__CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        __CONDA_PRE="source $(shell_squote "${__CONDA_BASE}/etc/profile.d/conda.sh") && conda activate $(shell_squote "${CONDA_ENV_NAME}") && "
    fi
fi

# Generate a launcher script (avoids nested-quoting issues with tmux).
{
    printf '#!/usr/bin/env bash\n'
    printf 'set -e\n'
    printf 'echo $$ > %s\n' "$(shell_squote "${PID_FILE}")"
    printf 'cd %s\n' "$(shell_squote "${REPO_ROOT}")"
    printf 'export HF_HUB_OFFLINE=%s\n' "$(shell_squote "${HF_HUB_OFFLINE}")"
    printf 'export HF_DATASETS_OFFLINE=%s\n' "$(shell_squote "${HF_DATASETS_OFFLINE}")"
    printf '%sCUDA_VISIBLE_DEVICES=%s exec %s -m scripts.chat_openai_compat --source %s --model-tag %s --step %s --model-name %s --port %s --host %s --temperature %s --max-tokens %s > %s 2>&1\n' \
        "${__CONDA_PRE}" \
        "$(shell_squote "${CUDA_VISIBLE_DEVICES}")" \
        "$(shell_squote "${__PYTHON}")" \
        "$(shell_squote "${MODEL_SOURCE}")" \
        "$(shell_squote "${MODEL_TAG}")" \
        "$(shell_squote "${MODEL_STEP}")" \
        "$(shell_squote "${MODEL_NAME}")" \
        "$(shell_squote "${MODEL_PORT}")" \
        "$(shell_squote "${MODEL_HOST}")" \
        "$(shell_squote "${MODEL_TEMPERATURE}")" \
        "$(shell_squote "${MODEL_MAX_TOKENS}")" \
        "$(shell_squote "${LOG_FILE}")"
} > "${LAUNCHER}"

: > "${LOG_FILE}"
tmux new-session -d -s "${SESSION}" "bash $(shell_squote "${LAUNCHER}")"

echo "[model] Started in tmux session '${SESSION}'. Waiting for /health (up to 120s)..."

__URL="http://${MODEL_HOST}:${MODEL_PORT}/health"
if wait_for_http_checked "${__URL}" 120 "${PID_FILE}"; then
    __pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${__pid}" ]]; then
        write_pid_meta "${PID_FILE}" "${__pid}" "chat_openai_compat" "${SESSION}"
    fi
    write_status "READY"
    echo "[model] READY (PID $(cat "${PID_FILE}" 2>/dev/null || echo "?"))."
    exit 0
fi

echo "[model] FAILED to become healthy within 120s. See ${LOG_FILE}." >&2
write_status "FAILED"
exit 1
