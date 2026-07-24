# Troubleshooting

This document lists the most common problems encountered when running the Phase 7
three-service deployment, and how to diagnose and fix them.

The general debugging flow is:

1. Run `bash scripts/deploy/status.sh` to see which services are up.
2. Run `python scripts/deploy/healthcheck.py` to see which services answer HTTP.
3. Read the relevant log under `runtime/phase7/logs/` (see
   [How to Read Logs](#how-to-read-logs)).
4. Apply the fix below, then restart with `bash scripts/deploy/restart_all.sh`.

## Model Service Won't Start

The model service (`chat_openai_compat.py`, tmux `nano-finance-model`, port
`18001`) is the first service to start. If it fails, the backend and frontend
cannot start either.

### GPU not available

- **Symptom:** the log mentions `CUDA error`, `no CUDA-capable device`, or the
  process exits while loading the checkpoint.
- **Check:**
  ```bash
  nvidia-smi
  ```
  Confirm a GPU is visible and not fully occupied by another process.
- **Fix:**
  - Set `CUDA_VISIBLE_DEVICES` in `online.env` to a free GPU (e.g. `0`).
  - If the GPU is busy, free it or pick another device.
  - If you do not need GPU, confirm the model can run on CPU (depends on the
    checkpoint) before proceeding.

### Port 18001 already in use

- **Symptom:** log shows `Address already in use` or `bind failed`.
- **Check:**
  ```bash
  ss -lntp | grep 18001
  ```
- **Fix:**
  - If the existing process is a stale model service, kill it:
    ```bash
    bash scripts/deploy/stop_all.sh
    # or just the session:
    tmux kill-session -t nano-finance-model
    ```
  - If it is an unrelated process, either stop it or change `MODEL_PORT` in
    `online.env` (and update `LLM_API_BASE_URL` to match).

### Conda environment not found

- **Symptom:** log shows `conda activate: command not found`, `No such env`,
  or `ModuleNotFoundError` for the model dependencies.
- **Check:** the `CONDA_ENV_NAME` in `online.env` exists:
  ```bash
  conda env list
  ```
- **Fix:**
  - Correct `CONDA_ENV_NAME` to the actual environment name.
  - If the environment is missing, create/restore it from the project setup
    instructions.
  - Make sure `conda` is initialized in the shell that launches tmux
    (the start scripts source `load_env.sh`, which assumes `conda` is on
    `PATH`).

## Backend Won't Start

The backend (FastAPI/uvicorn, tmux `nano-finance-backend`, port `18002`) depends
on the model service being up.

### Model service is down

- **Symptom:** `start_all.sh` reports the model failed, or the backend log shows
  connection errors to `LLM_API_BASE_URL` (e.g. `Connection refused`).
- **Fix:** start the model service first (see
  [Model Service Won't Start](#model-service-wont-start)), then start the
  backend:
  ```bash
  bash scripts/deploy/start_backend.sh
  ```

### Port 18002 already in use

- **Symptom:** log shows `Address already in use`.
- **Check:**
  ```bash
  ss -lntp | grep 18002
  ```
- **Fix:** kill the stale backend session or process, or change `BACKEND_PORT`
  (and update `VITE_API_URL` and `ALLOWED_ORIGINS` to match):
  ```bash
  tmux kill-session -t nano-finance-backend
  ```

### uv / venv not found

- **Symptom:** log shows `ModuleNotFoundError: No module named 'fastapi'`,
  `uvicorn: command not found`, or similar.
- **Fix:**
  - Confirm the Python environment used by the backend is installed and
    activated. The backend typically runs in the conda env named by
    `CONDA_ENV_NAME` (or a dedicated venv).
  - Reinstall dependencies if the environment was partially created:
    ```bash
    conda activate "$CONDA_ENV_NAME"
    pip install -r requirements.txt   # use the project's actual requirements file
    ```
  - Verify `DATABASE_URL`, `CHROMA_PATH`, and `BM25_DB_PATH` point at existing
    resources; a missing database can also surface as an import/startup error.

## Frontend Won't Start

The frontend (Vite, tmux `nano-finance-frontend`, port `18003`) depends on the
backend being up.

### Backend is down

- **Symptom:** the frontend starts but the UI shows API errors, or
  `start_all.sh` reports the backend failed first.
- **Fix:** start the backend (see [Backend Won't Start](#backend-wont-start)),
  then start the frontend:
  ```bash
  bash scripts/deploy/start_frontend.sh
  ```

### `node_modules` missing

- **Symptom:** log shows `vite: command not found` or
  `Cannot find module 'vite'`.
- **Fix:** install dependencies from the frontend directory:
  ```bash
  npm install   # or pnpm install / yarn install, per the project setup
  ```
  Then restart the frontend.

### Port 18003 already in use

- **Symptom:** log shows `Address already in use` or `port is in use`.
- **Check:**
  ```bash
  ss -lntp | grep 18003
  ```
- **Fix:** kill the stale frontend session or change `FRONTEND_PORT` in
  `online.env` (and update your SSH tunnel `-L` and browser URL to match):
  ```bash
  tmux kill-session -t nano-finance-frontend
  ```

## Health Check Fails

`python scripts/deploy/healthcheck.py` exits non-zero when at least one service
does not answer HTTP.

### Service not ready yet

- **Cause:** the service is still starting (the model in particular can take a
  while to load the checkpoint).
- **Fix:** wait a few seconds and re-run the health check. If it still fails,
  check the service log for errors.

### Wrong port or host

- **Cause:** `online.env` has a port that differs from what the health check
  probes, or a service is bound to something other than `127.0.0.1`.
- **Fix:** verify `MODEL_PORT`, `BACKEND_PORT`, `FRONTEND_PORT` and the
  `*_HOST` values in `online.env`. All hosts should be `127.0.0.1`. Restart
  after fixing.

### Firewall / network

- **Cause:** a local firewall blocks loopback connections (rare) or the health
  check is being run from a different host than the server.
- **Fix:** run `healthcheck.py` on the server itself, where the loopback
  services live. Remote checks must go through the SSH tunnel (see
  [ssh-tunnel.md](ssh-tunnel.md)).

## SSH Tunnel Not Working

See [ssh-tunnel.md](ssh-tunnel.md) for the full guide. The most common issues:

### Wrong port forwarded

- You forwarded the wrong local or remote port. Re-run with the exact ports:
  ```bash
  ssh -N -L 18003:127.0.0.1:18003 -L 18002:127.0.0.1:18002 user@server
  ```

### Server unreachable

- `ssh: connect to host ... port 22: Connection refused` or a timeout means SSH
  itself is blocked or the server is down. Confirm plain SSH works first:
  ```bash
  ssh user@server echo ok
  ```

### Browser shows connection refused but tunnel is up

- The server-side service is not running. SSH in and run
  `bash scripts/deploy/status.sh`, then start the missing service.

## tmux Session Already Exists

- **Symptom:** start script reports `session already exists` or
  `duplicate session: nano-finance-backend`.
- **Cause:** a previous run left the tmux session alive (e.g. the process inside
  died but the session window remains).
- **Fix:** kill the old session first, then start again:
  ```bash
  tmux kill-session -t nano-finance-backend
  bash scripts/deploy/start_backend.sh
  ```
  Or clean everything at once:
  ```bash
  bash scripts/deploy/stop_all.sh
  bash scripts/deploy/start_all.sh
  ```

## Permission Denied

- **Symptom:** `Permission denied` when running a script, reading a file, or
  activating the conda environment.
- **Checks and fixes:**
  - **Scripts not executable:** run them with `bash scripts/deploy/...` (as
    documented) or `chmod +x` them.
  - **Config file unreadable:** if you `chmod 600`'d `online.env`, make sure
    the same user that runs the scripts owns it:
    ```bash
    ls -l config/deployment/online.env
    chown "$USER:$USER" config/deployment/online.env
    ```
  - **Conda env not owned:** the conda environment directory must be readable
    and executable by your user. If it lives under another user's home, copy or
    reinstall it under your own.
  - **Data paths:** `CHROMA_PATH` and `BM25_DB_PATH` must be readable/writable
    by your user. Fix ownership or move them:
    ```bash
    ls -ld "$CHROMA_PATH" "$BM25_DB_PATH"
    ```

## How to Read Logs

Each service writes to its own log file under `runtime/phase7/logs/`:

| Service   | Log file                           |
|-----------|------------------------------------|
| Model     | `runtime/phase7/logs/model.log`    |
| Backend   | `runtime/phase7/logs/backend.log`  |
| Frontend  | `runtime/phase7/logs/frontend.log` |

To follow a log live:

```bash
tail -f runtime/phase7/logs/backend.log
```

To see recent errors only:

```bash
grep -iE 'error|exception|traceback' runtime/phase7/logs/backend.log | tail -n 50
```

You can also attach to the tmux session to see live output and interact with the
process's shell:

```bash
tmux attach -t nano-finance-backend
```

Detach with `Ctrl-b` then `d` (the service keeps running).

## How to Debug Service Startup

1. **Check status and health first:**
   ```bash
   bash scripts/deploy/status.sh
   python scripts/deploy/healthcheck.py
   ```
2. **Read the failing service's log** from the top — the first error is usually
   the root cause; later errors are often downstream.
3. **Reproduce manually in the shell.** Start a shell with the same environment
   the script uses, then run the service command by hand to see the error
   directly:
   ```bash
   bash scripts/deploy/load_env.sh
   conda activate "$CONDA_ENV_NAME"
   # then run the model/backend/frontend command directly
   ```
4. **Confirm the prerequisites** for that service: GPU (model), model service
   reachability (backend), backend reachability (frontend), and that the
   configured ports are free.
5. **Fix and restart.** After changing `online.env` or fixing the environment,
   run `bash scripts/deploy/restart_all.sh` and re-run the smoke test:
   ```bash
   python scripts/deploy/smoke_test.py
   ```
