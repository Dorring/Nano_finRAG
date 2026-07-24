# Online Deployment (Phase 7)

This document describes the rootless three-service online deployment for the
nano_finance project. The deployment runs entirely under a normal user account
(no root, no Docker, no systemd) and exposes no public network endpoints. Remote
access is provided through an SSH tunnel.

## Overview

Phase 7 introduces a rootless three-service online deployment that brings the
project online for interactive use. The deployment consists of three cooperating
services running in dedicated tmux sessions, all bound to the loopback interface
(`127.0.0.1`):

- **Model service** — serves the language model through an OpenAI-compatible
  HTTP API.
- **Backend service** — a FastAPI application (uvicorn) that hosts the business
  logic and proxies model calls.
- **Frontend service** — a Vite dev/preview server that serves the web UI.

The three services form a single request pipeline:

```
Browser  -->  Frontend (Vite)  -->  Backend (FastAPI)  -->  Model (OpenAI-compat)
            127.0.0.1:18003       127.0.0.1:18002         127.0.0.1:18001
```

Because everything listens on `127.0.0.1`, the deployment is not reachable from
the public internet. A developer connects over SSH and forwards the relevant
ports to their local machine (see [ssh-tunnel.md](ssh-tunnel.md)).

## Architecture Diagram

```
                  +--------------------------+
                  |   Developer Workstation  |
                  |   (browser / curl)       |
                  +------------+-------------+
                               |
                        SSH tunnel (-L ...)
                               |
                  +------------+-------------+
                  |     Remote Server        |
                  |   (user account, no root)|
                  |                          |
                  |  +--------------------+  |
                  |  | Frontend (Vite)    |  |   tmux: nano-finance-frontend
                  |  | 127.0.0.1:18003    |  |
                  |  +----------+---------+  |
                  |             |            |
                  |             v            |
                  |  +--------------------+  |
                  |  | Backend (FastAPI)  |  |   tmux: nano-finance-backend
                  |  | 127.0.0.1:18002    |  |
                  |  +----------+---------+  |
                  |             |            |
                  |             v            |
                  |  +--------------------+  |
                  |  | Model (OpenAI API) |  |   tmux: nano-finance-model
                  |  | 127.0.0.1:18001    |  |
                  |  +--------------------+  |
                  |                          |
                  +--------------------------+
```

## Constraints

The Phase 7 deployment is deliberately minimal and operates under the following
constraints:

- **No root access** — all services run under a normal user account. No
  `sudo`, no privileged ports, no system-level configuration changes.
- **No Docker / containerization** — services run directly in tmux sessions on
  the host. There is no container runtime.
- **No systemd / service manager** — lifecycle is managed by shell scripts and
  tmux sessions. Nothing is registered with the system init.
- **No public network exposure** — every service binds to `127.0.0.1` only.
  There is no reverse proxy, no public DNS, and no open firewall ports for the
  application.
- **No HTTPS / TLS** — traffic is plain HTTP on loopback. TLS termination, if
  ever needed, is outside the scope of this phase.
- **No auto-restart on reboot** — if the server reboots, services must be
  started manually (see [start-stop.md](start-stop.md)).

## Service Descriptions

### Model Service

- **Entry point:** `chat_openai_compat.py`
- **Port:** `18001` on `127.0.0.1`
- **tmux session:** `nano-finance-model`
- **Role:** Loads the language model checkpoint and exposes an
  OpenAI-compatible HTTP API (e.g. `/v1/chat/completions`) consumed by the
  backend. GPU selection is controlled via `CUDA_VISIBLE_DEVICES`.

### Backend Service

- **Entry point:** FastAPI application launched with `uvicorn`
- **Port:** `18002` on `127.0.0.1`
- **tmux session:** `nano-finance-backend`
- **Role:** Hosts business logic, persistence, and retrieval. Proxies model
  calls to the model service via `LLM_API_BASE_URL`. Serves the API that the
  frontend calls.

### Frontend Service

- **Entry point:** Vite dev/preview server
- **Port:** `18003` on `127.0.0.1`
- **tmux session:** `nano-finance-frontend`
- **Role:** Serves the web UI. Calls the backend through `VITE_API_URL`.

## Port Assignments

| Service   | Host        | Port  | tmux session             |
|-----------|-------------|-------|--------------------------|
| Model     | `127.0.0.1` | 18001 | `nano-finance-model`     |
| Backend   | `127.0.0.1` | 18002 | `nano-finance-backend`   |
| Frontend  | `127.0.0.1` | 18003 | `nano-finance-frontend`  |

All ports are greater than `1024` so no privileged access is required.

## tmux Session Names

Each service runs in its own tmux session so it can be inspected and managed
independently:

| tmux session              | Service   |
|---------------------------|-----------|
| `nano-finance-model`      | Model     |
| `nano-finance-backend`    | Backend   |
| `nano-finance-frontend`   | Frontend  |

To attach to a session interactively:

```bash
tmux attach -t nano-finance-backend
```

Detach with `Ctrl-b` then `d`. Attaching is optional; logs are also written to
disk (see [Runtime Directory Structure](#runtime-directory-structure)).

## Runtime Directory Structure

Runtime artifacts are written under `runtime/phase7/` and deployment reports
under `artifacts/deployment/phase7/`:

```
runtime/phase7/
├── logs/                 # Per-service stdout/stderr logs
│   ├── model.log
│   ├── backend.log
│   └── frontend.log
├── pids/                 # PID files for each service
│   ├── model.pid
│   ├── backend.pid
│   └── frontend.pid
└── status/               # Last-known status markers written by scripts
    ├── model.status
    ├── backend.status
    └── frontend.status

artifacts/deployment/phase7/
└── (deployment reports, smoke-test output, etc.)
```

The `logs/` directory is the primary source of truth when diagnosing problems.
The `pids/` files are used by `stop_all.sh` and `status.sh`. The `status/`
markers are written by the start scripts and consumed by `status.sh`.

## Related Documents

- [start-stop.md](start-stop.md) — starting, stopping, restarting, and checking
  service status.
- [ssh-tunnel.md](ssh-tunnel.md) — accessing the deployment from a local
  machine over SSH.
- [configuration.md](configuration.md) — environment configuration
  (`online.env`).
- [troubleshooting.md](troubleshooting.md) — diagnosing and fixing common
  problems.
- [known-limitations.md](known-limitations.md) — what this deployment does and
  does not provide.
