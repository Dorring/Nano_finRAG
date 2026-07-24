# Known Limitations

The Phase 7 deployment is intentionally minimal: it brings the three services
online under a normal user account with no infrastructure dependencies. This
document lists what the deployment does **not** provide, so expectations are
set correctly and future phases have a clear list of gaps to close.

## Operational Limitations

### No root access

The deployment runs entirely under a normal user account. There is no `sudo`,
no privileged port binding, and no system-level configuration. Anything
requiring root (system packages, kernel parameters, system services) is out of
scope.

### No Docker / containerization

Services run directly on the host inside tmux sessions. There is no container
runtime, no image, and no isolation beyond the user account. Environment
reproducibility depends on the conda environment and `node_modules` being set up
manually on the host.

### No auto-restart on server reboot

There is no systemd unit, supervisor, or watchdog. If the server reboots, all
three services stop and stay down. After a reboot you must start them manually:

```bash
bash scripts/deploy/start_all.sh
```

There is also no process-level auto-restart if a service crashes on its own —
only the manual scripts in `scripts/deploy/` are available.

### No process supervision beyond tmux

tmux keeps a service running after you disconnect, but it does not restart a
crashed process. A service that exits stays exited until you restart it.

## Network and Security Limitations

### No public network access

All services bind to `127.0.0.1` only. The deployment is not reachable from the
public internet, and there is no reverse proxy or public DNS. Remote access
requires an SSH tunnel (see [ssh-tunnel.md](ssh-tunnel.md)).

### No HTTPS / TLS

Traffic between the browser and the frontend, and between the services, is plain
HTTP over loopback. SSH encrypts the tunnel to your workstation, but there is no
TLS termination at the application layer. Do not transmit sensitive data under
the assumption of end-to-end TLS.

## Scalability and Reliability Limitations

### No load balancing or auto-scaling

There is a single instance of each service. There is no load balancer, no
request distribution, and no mechanism to add replicas. A single service
failure takes down the whole pipeline.

### Single worker (no multi-process)

The backend runs with a single uvicorn worker (`BACKEND_WORKERS` is 1). There is
no multi-process concurrency, so throughput is bounded by one process. The model
service likewise serves from a single process.

### No multi-machine deployment

All three services run on one server. There is no support for splitting services
across hosts, shared storage, or cross-machine service discovery.

## Observability Limitations

### No external monitoring

There is no Prometheus, OpenTelemetry, Grafana, or other metrics/tracing
pipeline. Observability is limited to the log files under
`runtime/phase7/logs/`, the status markers under `runtime/phase7/status/`, and
the `healthcheck.py` / `smoke_test.py` scripts. There is no alerting.

## Data and Verification Limitations

### Historical checkpoint not verifiable (Phase 6 limitation)

Carried over from Phase 6: historical model checkpoints are not independently
verifiable. The deployment uses the checkpoint configured in `online.env`
(`MODEL_SOURCE`, `MODEL_TAG`, `MODEL_STEP`) but cannot prove its provenance or
integrity. Treat model outputs accordingly.

### Performance is baseline only (not optimized)

The deployment is configured for correctness and accessibility, not performance.
No effort has been spent on inference optimization, batching, caching, or
tuning. Latency and throughput represent a baseline; do not treat them as
production-grade targets.

## Summary Table

| Limitation                              | Impact / Workaround                                   |
|-----------------------------------------|-------------------------------------------------------|
| No root access                          | User-level only; privileged ops unavailable.          |
| No Docker/containerization              | Host-installed conda env + `node_modules`.            |
| No public network access                | SSH tunnel required for remote use.                   |
| No HTTPS/TLS                            | Plain HTTP on loopback; tunnel-only encryption.       |
| No auto-restart on reboot               | Run `bash scripts/deploy/start_all.sh` after reboot.  |
| No load balancing / auto-scaling        | Single instance per service; no replicas.             |
| Single worker (no multi-process)        | One backend worker; throughput bounded.               |
| No external monitoring                  | Logs + `healthcheck.py`/`smoke_test.py` only.         |
| No multi-machine deployment             | All services on one host.                             |
| Historical checkpoint not verifiable    | Phase 6 limitation; provenance unproven.              |
| Performance is baseline only            | Not optimized; baseline metrics only.                 |

For day-to-day operation, see [start-stop.md](start-stop.md) and
[troubleshooting.md](troubleshooting.md). For the overall design and
constraints, see [online-deployment.md](online-deployment.md).
