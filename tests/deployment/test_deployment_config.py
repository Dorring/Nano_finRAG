"""Tests for config/deployment/online.env.example.

Verify Phase 7 deployment configuration: file existence, required
variables, port ranges (> 1024), host binding (127.0.0.1), no real
secrets in the example, and that the active online.env is gitignored.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config" / "deployment"
EXAMPLE_ENV = CONFIG_DIR / "online.env.example"
ACTIVE_ENV = CONFIG_DIR / "online.env"
GITIGNORE = REPO_ROOT / ".gitignore"

REQUIRED_VARS = [
    "MODEL_HOST",
    "MODEL_PORT",
    "MODEL_NAME",
    "CUDA_VISIBLE_DEVICES",
    "BACKEND_HOST",
    "BACKEND_PORT",
    "BACKEND_WORKERS",
    "BACKEND_RELOAD",
    "FRONTEND_HOST",
    "FRONTEND_PORT",
]

PORT_VARS = ["MODEL_PORT", "BACKEND_PORT", "FRONTEND_PORT"]
HOST_VARS = ["MODEL_HOST", "BACKEND_HOST", "FRONTEND_HOST"]

SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
    re.compile(r"gho_[a-zA-Z0-9]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def _parse_env(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE pairs from an env file, skipping comments/blanks."""
    config: dict[str, str] = {}
    if not path.is_file():
        return config
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        config[key.strip()] = value.strip()
    return config


@pytest.fixture
def env_config() -> dict[str, str]:
    if not EXAMPLE_ENV.is_file():
        pytest.skip(f"Example env not found: {EXAMPLE_ENV}")
    return _parse_env(EXAMPLE_ENV)


def test_example_env_exists():
    # Verify config/deployment/online.env.example exists
    assert EXAMPLE_ENV.is_file(), f"Expected {EXAMPLE_ENV} to exist"


@pytest.mark.parametrize("var_name", REQUIRED_VARS)
def test_required_variable_present(env_config, var_name):
    # Verify each required variable is defined in the example
    assert var_name in env_config, f"Missing required variable: {var_name}"


@pytest.mark.parametrize("port_var", PORT_VARS)
def test_ports_above_1024(env_config, port_var):
    # Verify all ports are > 1024
    value = env_config.get(port_var)
    assert value is not None, f"{port_var} not set"
    assert value.isdigit(), f"{port_var}='{value}' is not numeric"
    port = int(value)
    assert port > 1024, f"{port_var}={port} must be > 1024"


@pytest.mark.parametrize("host_var", HOST_VARS)
def test_hosts_are_loopback(env_config, host_var):
    # Verify all hosts bind to 127.0.0.1
    value = env_config.get(host_var)
    assert value is not None, f"{host_var} not set"
    assert value == "127.0.0.1", f"{host_var}='{value}' should be 127.0.0.1"


def test_no_real_secrets_in_example():
    # Verify the example file contains no real secrets/tokens/passwords
    if not EXAMPLE_ENV.is_file():
        pytest.skip(f"Example env not found: {EXAMPLE_ENV}")
    text = EXAMPLE_ENV.read_text(encoding="utf-8")
    for pat in SECRET_PATTERNS:
        m = pat.search(text)
        assert not m, f"Possible secret in {EXAMPLE_ENV.name}: {m.group()}"


def test_example_uses_placeholders_not_real_values():
    # Verify SECRET_KEY and LLM_API_KEY are placeholders, not real values
    if not EXAMPLE_ENV.is_file():
        pytest.skip(f"Example env not found: {EXAMPLE_ENV}")
    config = _parse_env(EXAMPLE_ENV)
    secret_key = config.get("SECRET_KEY", "")
    api_key = config.get("LLM_API_KEY", "")
    # Placeholders should be short, human-readable markers — not long random strings.
    assert len(secret_key) < 60, "SECRET_KEY looks like a real secret value"
    assert "change" in secret_key.lower() or "placeholder" in secret_key.lower() \
        or "example" in secret_key.lower() or secret_key == "", \
        f"SECRET_KEY='{secret_key}' does not look like a placeholder"
    assert "not-needed" in api_key.lower() or "placeholder" in api_key.lower() \
        or "example" in api_key.lower() or "change" in api_key.lower() \
        or api_key == "", \
        f"LLM_API_KEY='{api_key}' does not look like a placeholder"


def test_active_env_is_gitignored():
    # Verify config/deployment/online.env is in .gitignore
    if not GITIGNORE.is_file():
        pytest.skip(".gitignore not found")
    text = GITIGNORE.read_text(encoding="utf-8")
    assert "config/deployment/online.env" in text, \
        "config/deployment/online.env should be listed in .gitignore"
