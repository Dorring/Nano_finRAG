"""Tests verifying no root/privileged commands in deployment scripts.

Scan all scripts in ``scripts/deploy/`` for forbidden commands
(sudo, docker, podman, apt/yum, pkill -f, killall, systemctl, nginx),
ports <= 1024, ``--reload`` in production startup, and verify
backend workers default to 1.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "deploy"

FORBIDDEN_COMMANDS = [
    r"\bsudo\b",
    r"\bdocker\b",
    r"\bpodman\b",
    r"\bapt(-get)?\b",
    r"\byum\b",
    r"\bpkill\s+-f\b",
    r"\bkillall\b",
    r"\bsystemctl\b",
    r"\bnginx\b",
]


def _list_shell_scripts() -> list[Path]:
    if not SCRIPTS_DIR.is_dir():
        return []
    return sorted(SCRIPTS_DIR.glob("*.sh"))


def _strip_comments(text: str) -> str:
    """Remove shell comments (full-line and inline) for pattern scanning."""
    result: list[str] = []
    for raw_line in text.splitlines():
        # Remove inline comments (naive: respects single # outside quotes)
        stripped = raw_line.lstrip()
        if stripped.startswith("#"):
            continue
        # Strip inline comment after first unquoted #
        result.append(raw_line)
    return "\n".join(result)


@pytest.fixture
def shell_scripts() -> list[Path]:
    scripts = _list_shell_scripts()
    if not scripts:
        pytest.skip("No shell scripts found in scripts/deploy/")
    return scripts


@pytest.mark.parametrize("pattern", FORBIDDEN_COMMANDS)
def test_no_forbidden_commands(shell_scripts, pattern):
    # Verify no script contains forbidden root/privileged commands (outside comments)
    regex = re.compile(pattern)
    for script in shell_scripts:
        text = _strip_comments(script.read_text(encoding="utf-8"))
        m = regex.search(text)
        assert not m, \
            f"Forbidden command '{pattern}' found in {script.name}: ...{m.group()}..."


def test_no_port_binding_below_1024(shell_scripts):
    # Verify no script binds to ports <= 1024
    port_pattern = re.compile(r"--port\s+(\d+)")
    for script in shell_scripts:
        text = _strip_comments(script.read_text(encoding="utf-8"))
        for m in port_pattern.finditer(text):
            port = int(m.group(1))
            assert port > 1024, \
                f"{script.name} binds to port {port} which is <= 1024"
    # Also check for explicit port guard logic (PORT <= 1024 check)
    for script in shell_scripts:
        text = script.read_text(encoding="utf-8")
        assert re.search(r"-le\s+1024|>\s*1024|>=?\s*1024", text) is not None \
            or "--port" not in text, \
            f"{script.name} uses --port but has no > 1024 guard"


def test_no_reload_in_production_startup(shell_scripts):
    # Verify no script uses --reload in production startup
    for script in shell_scripts:
        text = _strip_comments(script.read_text(encoding="utf-8"))
        assert "--reload" not in text, \
            f"{script.name} contains '--reload' in production startup"


def test_backend_workers_default_to_one():
    # Verify backend workers default to 1
    backend_script = SCRIPTS_DIR / "start_backend.sh"
    if not backend_script.is_file():
        pytest.skip("start_backend.sh not found")
    text = backend_script.read_text(encoding="utf-8")
    # Check for --workers 1 in the uvicorn command
    assert re.search(r"--workers\s+1\b", text) or "--workers 1" in text, \
        "start_backend.sh does not default to 1 worker"
    # Also check the example env defaults BACKEND_WORKERS=1
    env_example = REPO_ROOT / "config" / "deployment" / "online.env.example"
    if env_example.is_file():
        env_text = env_example.read_text(encoding="utf-8")
        assert "BACKEND_WORKERS=1" in env_text, \
            "online.env.example does not set BACKEND_WORKERS=1"
