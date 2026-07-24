# Start, Stop, Restart, and Status

This document covers the day-to-day lifecycle of the Phase 7 three-service
deployment: starting, stopping, restarting, checking status, running health
checks and smoke tests, and locating logs.

All commands assume you are in the repository root (`y:\nanochat\repo` on the
server) and that `config/deployment/online.env` exists (see
[configuration.md](configuration.md)).

## Start Order and Stop Order

The services depend on each other, so order matters.

**Start order** (dependencies come first):

1. **Model** — `nano-finance-model` on `127.0.0.1:18001`
2. **Backend** — `nano-finance-backend` on `127.0.0.1:18002`
3. **Frontend** — `nano-finance-frontend` on `127.0.0.1:18003`

**Stop order** (dependents come first):

1. **Frontend**
2. **Backend**
3. **Model**

`start_all.sh` and `stop_all.sh` already apply the correct order. You only need
to remember the order when starting or stopping services individually.

## Start All Services

```bash
bash scripts/deploy/start_all.sh
```

This script:

1. Loads `config/deployment/online.env` (via `load_env.sh`).
2. Starts the model service, waits for it to be reachable.
3. Starts the backend service, waits for it to be reachable.
4. Starts the frontend service, waits for it to be reachable.
5. Writes PID files to `runtime/phase7/pids/` and status markers to
   `runtime/phase7/status/`.

Each service is launched inside its own tmux session, so it keeps running after
you detach or disconnect.

## Stop All Services

```bash
bash scripts/deploy/stop_all.sh
```

Stops services in reverse order (frontend → backend → model). The script reads
the PID files from `runtime/phase7/pids/`, sends a termination signal, and kills
the corresponding tmux session. Stale PID files are cleaned up.

## Restart All Services

```bash
bash scripts/deploy/restart_all.sh
```

Equivalent to `stop_all.sh` followed by `start_all.sh`. Useful after changing
`online.env` or pulling new code.

## Check Status

```bash
bash scripts/deploy/status.sh
```

Prints a summary of each service: tmux session state, PID file, process liveness,
and the status marker under `runtime/phase7/status/`. Use this as the first
check when something seems wrong.

## Health Check

```bash
python scripts/deploy/healthcheck.py
```

Probes each service's HTTP endpoint over loopback and reports whether it is
responding. A non-zero exit code indicates that at least one service is unhealthy.
This is a programmatic check suitable for scripts; for a human-friendly
end-to-end check, use the smoke test below.

## Smoke Test

```bash
python scripts/deploy/smoke_test.py
```

Runs an end-to-end functional test: it exercises the frontend, backend, and
model paths and writes a report under `artifacts/deployment/phase7/`. Use this
after a fresh start or after a restart to confirm the whole pipeline works.

To collect a consolidated deployment report (logs, status, and smoke-test
output) into the artifacts directory:

```bash
python scripts/deploy/collect_deployment_report.py
```

## Starting Individual Services

If you only need to start one service (for example, after editing backend code
while the model is still running), use the per-service scripts. They honor the
same start order constraints — make sure dependencies are already up.

```bash
# Model only (no dependencies)
bash scripts/deploy/start_model.sh

# Backend only (requires the model service to be running)
bash scripts/deploy/start_backend.sh

# Frontend only (requires the backend service to be running)
bash scripts/deploy/start_frontend.sh
```

There is no individual stop script in the standard set; to stop a single
service you can either run `stop_all.sh` or kill the specific tmux session:

```bash
tmux kill-session -t nano-finance-backend
```

## What Happens If a Service Fails to Start

`start_all.sh` waits for each service to become reachable before starting the
next one. If a service fails to come up:

1. The script reports the failure and halts the start sequence. Services that
   were already started remain running.
2. A status marker is written under `runtime/phase7/status/` indicating the
   failure.
3. Inspect the service log to find the cause (see [Log File Locations](#log-file-locations)).
4. After fixing the issue, either restart everything with `restart_all.sh` or
   start only the missing service with its individual script.

Common failure causes and their fixes are covered in
[troubleshooting.md](troubleshooting.md).

## Log File Locations

Each service writes its stdout and stderr to a log file under
`runtime/phase7/logs/`:

| Service   | Log file                          | tmux session             |
|-----------|-----------------------------------|--------------------------|
| Model     | `runtime/phase7/logs/model.log`   | `nano-finance-model`     |
| Backend   | `runtime/phase7/logs/backend.log` | `nano-finance-backend`   |
| Frontend  | `runtime/phase7/logs/frontend.log`| `nano-finance-frontend`  |

To follow a log in real time:

```bash
tail -f runtime/phase7/logs/backend.log
```

You can also attach to the tmux session to see live output:

```bash
tmux attach -t nano-finance-backend
```

Detach with `Ctrl-b` then `d` (the service keeps running).

## Quick Reference

| Task                  | Command                                        |
|-----------------------|------------------------------------------------|
| Start everything      | `bash scripts/deploy/start_all.sh`             |
| Stop everything       | `bash scripts/deploy/stop_all.sh`              |
| Restart everything    | `bash scripts/deploy/restart_all.sh`           |
| Status summary        | `bash scripts/deploy/status.sh`                |
| Health check          | `python scripts/deploy/healthcheck.py`         |
| Smoke test            | `python scripts/deploy/smoke_test.py`          |
| Deployment report     | `python scripts/deploy/collect_deployment_report.py` |
| Start model only      | `bash scripts/deploy/start_model.sh`           |
| Start backend only    | `bash scripts/deploy/start_backend.sh`         |
| Start frontend only   | `bash scripts/deploy/start_frontend.sh`        |
