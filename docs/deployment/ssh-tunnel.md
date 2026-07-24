# SSH Tunnel for Remote Access

The Phase 7 deployment binds every service to `127.0.0.1` on the server, so
none of the ports (`18001`, `18002`, `18003`) are reachable from the public
internet. To use the deployment from your local workstation, you forward the
server's loopback ports to your machine over an SSH tunnel. Your browser then
talks to `127.0.0.1` on your workstation, and SSH carries the traffic to the
server.

## Why a Tunnel Is Required

Because the services listen only on the loopback interface:

- They cannot be opened in a browser by visiting the server's public IP.
- They are not exposed by any firewall rule or reverse proxy.
- The only supported remote access path is an authenticated SSH connection.

This keeps the deployment private: only someone with SSH access to the server
can reach the services.

## Setting Up the SSH Tunnel

From your **local workstation**, run:

```bash
ssh -N -L 18003:127.0.0.1:18003 -L 18002:127.0.0.1:18002 user@server
```

Where:

- `user@server` is the SSH user and host you normally use to log in.
- `-N` tells SSH not to open a remote shell (forwarding only).
- `-L 18003:127.0.0.1:18003` forwards local port `18003` to the server's
  `127.0.0.1:18003` (frontend).
- `-L 18002:127.0.0.1:18002` forwards local port `18002` to the server's
  `127.0.0.1:18002` (backend).

You generally do **not** need to forward the model port (`18001`) — the backend
talks to the model server-side. Forward it only if you intend to call the model
API directly from your workstation:

```bash
ssh -N -L 18003:127.0.0.1:18003 -L 18002:127.0.0.1:18002 -L 18001:127.0.0.1:18001 user@server
```

Leave this SSH command running in a terminal for as long as you need access.
Closing the terminal (or pressing `Ctrl-C`) tears the tunnel down.

### Running the Tunnel in the Background

To free up your terminal, add `-f` (background after authentication) and keep
`-N`:

```bash
ssh -f -N -L 18003:127.0.0.1:18003 -L 18002:127.0.0.1:18002 user@server
```

To tear down a backgrounded tunnel later, find and kill the SSH process:

```bash
pkill -f "ssh -f -N -L 18003"
```

## Browser Access

Once the tunnel is up, open a browser on your local workstation and visit:

```
http://127.0.0.1:18003
```

This is the frontend. It calls the backend at the forwarded port `18002`, which
in turn calls the model on the server.

To call the backend API directly from your workstation:

```bash
curl http://127.0.0.1:18002/docs
```

Note the `http://` scheme — the tunnel carries plain HTTP (see
[Security Notes](#security-notes)).

## Verifying the Tunnel Is Working

1. **Check that SSH is still connected.** If you ran it in the foreground, the
   terminal should still be open with no error. For a backgrounded tunnel:

   ```bash
   pgrep -af "ssh.*-L 18003"
   ```

   A running process means the tunnel is up.

2. **Check the frontend responds locally.**

   ```bash
   curl -I http://127.0.0.1:18003
   ```

   An HTTP `200`/`301`/`304` response means the tunnel and the frontend are both
   working.

3. **Check the backend responds locally.**

   ```bash
   curl -I http://127.0.0.1:18002
   ```

4. **If a port refuses to connect** while SSH is up, the corresponding service
   on the server is probably not running. SSH into the server and check with:

   ```bash
   bash scripts/deploy/status.sh
   ```

## Troubleshooting Tunnel Issues

### `bind: Address already in use`

Another local process is using the forwarded port (often a previous tunnel that
was not closed). Find and stop it:

```bash
# Linux/macOS
lsof -i :18003
# or
ss -lntp | grep 18003

# then kill the offending process, or pick a different local port:
ssh -N -L 18083:127.0.0.1:18003 user@server
```

If you remap the local port (e.g. `18083`), point your browser at the remapped
port: `http://127.0.0.1:18083`.

### `Connection refused` from the browser

- The SSH tunnel is up but the **server-side** service is not running. SSH in
  and run `bash scripts/deploy/status.sh`, then start the missing service (see
  [start-stop.md](start-stop.md)).
- You forwarded the wrong port, or the service is bound to a different host than
  `127.0.0.1` on the server.

### `ssh: connect to host ... port 22: Connection refused` / timeouts

- The server is unreachable, or SSH is firewalled. Confirm basic SSH access
  works first: `ssh user@server echo ok`.
- You are on a network that blocks outbound SSH (port 22). Try a different
  network or a jump host.

### Tunnel works briefly then drops

- An idle SSH connection may be killed by a stateful firewall or NAT. Add
  keep-alive options:

  ```bash
  ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -L 18003:127.0.0.1:18003 -L 18002:127.0.0.1:18002 user@server
  ```

### Port forwarding silently does nothing

- Make sure you included `-L` and that SSH did not error on connect. Without
  `-N`, SSH would open a shell; with `-N` and a typo, it may exit immediately.
  Run without `-f` first to see any errors inline.

## Security Notes

- **No public access.** The services are never exposed on the server's public
  interfaces. The SSH tunnel is the *only* remote access path.
- **Authenticated transit.** All remote traffic traverses SSH, so it inherits
  SSH's authentication and encryption. Anyone reaching the services must already
  have valid SSH credentials for the server.
- **Plain HTTP on loopback.** Between SSH and the services the traffic is plain
  HTTP, but it never leaves the server's loopback interface (or your local
  machine after decryption by SSH). There is no TLS termination in this phase.
- **Do not weaken the deployment.** Do not re-bind services to `0.0.0.0`, do not
  open firewall ports for `18001`–`18003`, and do not commit any server IPs or
  credentials into the repository (see [configuration.md](configuration.md)).
- **Local port conflicts.** Forwarding to a privileged local port (<1024) would
  require root on your workstation; the default high ports avoid that and keep
  the workflow rootless end-to-end.
