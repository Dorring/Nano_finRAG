# Configuration

The Phase 7 deployment is configured through a single environment file,
`config/deployment/online.env`. This document explains how to create it and what
each variable does.

## How to Configure

A template is shipped at `config/deployment/online.env.example`. To configure
your deployment, copy it and edit the copy:

```bash
cp config/deployment/online.env.example config/deployment/online.env
# then edit config/deployment/online.env
```

Only `online.env` is read by the deployment scripts (via `load_env.sh`). The
`online.env.example` file is a committed template and is never used at runtime.

After editing `online.env`, restart the services so they pick up the new values:

```bash
bash scripts/deploy/restart_all.sh
```

## Environment Variables

The variables are grouped by concern below. All values are strings unless noted.

### Model Service

| Variable                | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `MODEL_HOST`            | Host the model service binds to. Keep `127.0.0.1` (no public exposure).    |
| `MODEL_PORT`            | Port the model service listens on. Default `18001`. Must be `> 1024`.       |
| `MODEL_NAME`            | Display/logical name of the model served by the model service.              |
| `MODEL_SOURCE`          | Source/location of the model checkpoint (e.g. path or registry identifier). |
| `MODEL_TAG`             | Tag or label identifying the model variant.                                 |
| `MODEL_STEP`            | Training step of the checkpoint to load.                                    |
| `CUDA_VISIBLE_DEVICES`  | Comma-separated GPU IDs the model service may use (e.g. `0` or `0,1`).      |

### Backend Service

| Variable           | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `BACKEND_HOST`     | Host the backend binds to. Keep `127.0.0.1`.                               |
| `BACKEND_PORT`     | Port the backend listens on. Default `18002`. Must be `> 1024`.            |
| `BACKEND_WORKERS`  | Number of uvicorn workers. Phase 7 is single-worker (see limitations).     |
| `BACKEND_RELOAD`   | `true`/`false` — enable uvicorn auto-reload (dev only, off for stable run).|

### LLM Connection (Backend → Model)

These tell the backend how to reach the model service and how to authenticate.

| Variable          | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `LLM_API_BASE_URL`| Base URL of the model service, e.g. `http://127.0.0.1:18001/v1`.            |
| `LLM_MODEL_NAME`  | Model name the backend passes in model API requests.                        |
| `LLM_API_KEY`     | API key used when calling the model service. **Treat as a secret.**         |

### Data and Persistence

| Variable        | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `DATABASE_URL`  | Connection URL for the primary database. **Treat as a secret.**             |
| `CHROMA_PATH`   | Filesystem path to the Chroma vector store directory.                       |
| `BM25_DB_PATH`  | Filesystem path to the BM25 index/database.                                 |

### Security

| Variable          | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `SECRET_KEY`      | Secret used by the backend for signing sessions/tokens. **Treat as a secret.**|
| `ALLOWED_ORIGINS` | Comma-separated list of CORS origins the backend accepts (e.g. `http://127.0.0.1:18003`).|

### Frontend Service

| Variable         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `FRONTEND_HOST`  | Host the frontend binds to. Keep `127.0.0.1`.                              |
| `FRONTEND_PORT`  | Port the frontend listens on. Default `18003`. Must be `> 1024`.           |
| `VITE_API_URL`   | URL the frontend uses to call the backend, e.g. `http://127.0.0.1:18002`.   |

### Runtime Environment

| Variable                  | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `CONDA_ENV_NAME`          | Name of the conda environment used to run the model and backend services.   |
| `TMUX_SESSION_MODEL`      | tmux session name for the model service (default `nano-finance-model`).     |
| `TMUX_SESSION_BACKEND`    | tmux session name for the backend service (default `nano-finance-backend`). |
| `TMUX_SESSION_FRONTEND`   | tmux session name for the frontend service (default `nano-finance-frontend`).|

The `TMUX_SESSION_*` variables let you rename sessions if the defaults collide
with other work on the same server. The corresponding PID/status files under
`runtime/phase7/` track the sessions by these names.

## Security

- **Never commit `online.env`.** It contains secrets (`LLM_API_KEY`,
  `SECRET_KEY`, `DATABASE_URL`). Only `online.env.example` is tracked in
  version control. Ensure `online.env` is listed in `.gitignore`.
- **Do not put real IPs, hostnames, passwords, or keys in
  `online.env.example`.** The example file must contain only placeholder values
  (e.g. `changeme`, `http://127.0.0.1:18001/v1`).
- **Keep hosts on loopback.** `MODEL_HOST`, `BACKEND_HOST`, and
  `FRONTEND_HOST` should remain `127.0.0.1`. Binding to `0.0.0.0` or a public
  interface would expose the services and bypass the SSH-tunnel-only access
  model.
- **Restrict file permissions.** After creating `online.env`, tighten its
  permissions so only your user can read it:

  ```bash
  chmod 600 config/deployment/online.env
  ```

- **Rotate secrets off the repo.** If a secret is ever accidentally committed,
  rotate it immediately and remove it from history.

## Port Requirements

All service ports must satisfy the following:

- **Greater than 1024** — privileged ports (`< 1024`) require root, which this
  rootless deployment forbids.
- **Distinct** — `MODEL_PORT`, `BACKEND_PORT`, and `FRONTEND_PORT` must not
  collide with each other or with other processes on the server.
- **Loopback only** — services bind to `127.0.0.1`; the ports are reached
  remotely only via SSH tunnel (see [ssh-tunnel.md](ssh-tunnel.md)).

The defaults (`18001`, `18002`, `18003`) satisfy all of these. If you change
them, update `LLM_API_BASE_URL`, `VITE_API_URL`, `ALLOWED_ORIGINS`, and your SSH
tunnel `-L` arguments to match.

## Verifying Configuration

After editing `online.env`, you can sanity-check that the scripts can load it:

```bash
bash scripts/deploy/load_env.sh && echo "env loaded OK"
```

Then start the services and run the health check and smoke test to confirm the
values are correct end-to-end:

```bash
bash scripts/deploy/start_all.sh
python scripts/deploy/healthcheck.py
python scripts/deploy/smoke_test.py
```

See [start-stop.md](start-stop.md) and [troubleshooting.md](troubleshooting.md)
for more.
