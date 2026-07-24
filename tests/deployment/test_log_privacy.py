"""Tests verifying log/secret privacy across deployment files.

Verify .gitignore excludes runtime/ and config/deployment/online.env,
no committed file contains real server IPs or hardcoded passwords/tokens,
healthcheck.py and smoke_test.py sanitize output, and artifacts (if
present) don't contain absolute paths or server IPs.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "deploy"
CONFIG_DIR = REPO_ROOT / "config" / "deployment"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "deployment" / "phase7"
GITIGNORE = REPO_ROOT / ".gitignore"

# Patterns that indicate real server IPs (non-loopback)
SERVER_IP_PATTERNS = [
    re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b43\.139\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b10\.157\.\d{1,3}\.\d{1,3}\b"),
]

# Patterns indicating hardcoded passwords/tokens
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
    re.compile(r"gho_[a-zA-Z0-9]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]

# Password assignment pattern — only flag non-test/placeholder passwords
PASSWORD_PATTERN = re.compile(
    r"(?:password|passwd|pwd)\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE,
)

# Keywords that indicate a test/placeholder password (not a real secret)
PASSWORD_PLACEHOLDER_KEYWORDS = (
    "smoke", "test", "phase7", "example", "change", "placeholder",
    "not-needed", "dummy", "your-", "xxx", "todo", "fixme",
)

# Absolute path patterns
ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:\\(?:Users|home|root|var|opt|data|tmp|mnt)"),
    re.compile(r"/(?:home|Users|root|var|opt|data|tmp|mnt|srv)/[^\s\"']+"),
]


def _read_gitignore() -> str:
    if not GITIGNORE.is_file():
        pytest.skip(".gitignore not found")
    return GITIGNORE.read_text(encoding="utf-8")


def test_gitignore_excludes_runtime_dir():
    # Verify .gitignore excludes runtime/ directory
    text = _read_gitignore()
    assert "runtime/" in text, ".gitignore should exclude runtime/"


def test_gitignore_excludes_online_env():
    # Verify .gitignore excludes config/deployment/online.env
    text = _read_gitignore()
    assert "config/deployment/online.env" in text, \
        ".gitignore should exclude config/deployment/online.env"


def _collect_deploy_files() -> list[Path]:
    """Collect all committed files in scripts/deploy/."""
    if not SCRIPTS_DIR.is_dir():
        return []
    return sorted(SCRIPTS_DIR.iterdir())


def test_no_real_server_ips_in_committed_files():
    # Scan committed files for real server IPs (non-loopback)
    deploy_files = _collect_deploy_files()
    assert len(deploy_files) > 0, "No files in scripts/deploy/"
    for f in deploy_files:
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        for pat in SERVER_IP_PATTERNS:
            m = pat.search(text)
            assert not m, \
                f"Real server IP '{m.group()}' found in {f.name}"


def test_no_hardcoded_passwords_in_deploy_scripts():
    # Verify no committed file in scripts/deploy/ contains hardcoded passwords/tokens
    deploy_files = _collect_deploy_files()
    assert len(deploy_files) > 0, "No files in scripts/deploy/"
    for f in deploy_files:
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        # Check for API key / token patterns (always flagged)
        for pat in SECRET_PATTERNS:
            m = pat.search(text)
            assert not m, \
                f"Hardcoded secret/token found in {f.name}: {m.group()}"
        # Check for password assignments (skip test/placeholder passwords)
        for m in PASSWORD_PATTERN.finditer(text):
            pw_value = m.group(1).lower()
            is_placeholder = any(kw in pw_value for kw in PASSWORD_PLACEHOLDER_KEYWORDS)
            assert is_placeholder, \
                f"Hardcoded password in {f.name}: {m.group(0)}"


def test_healthcheck_sanitizes_output():
    # Verify healthcheck.py sanitizes output (uses sanitize_url or masking)
    healthcheck = SCRIPTS_DIR / "healthcheck.py"
    if not healthcheck.is_file():
        pytest.skip("healthcheck.py not found")
    text = healthcheck.read_text(encoding="utf-8")
    assert "sanitize" in text.lower(), \
        "healthcheck.py should sanitize URLs/hosts in output"


def test_smoke_test_sanitizes_output():
    # Verify smoke_test.py sanitizes output (contains_path_leak checks, no raw paths)
    smoke = SCRIPTS_DIR / "smoke_test.py"
    if not smoke.is_file():
        pytest.skip("smoke_test.py not found")
    text = smoke.read_text(encoding="utf-8")
    assert "path_leak" in text.lower() or "sanitize" in text.lower(), \
        "smoke_test.py should have path leak detection / sanitization"


def test_artifacts_no_absolute_paths():
    # Verify artifacts (if they exist) don't contain absolute paths
    if not ARTIFACTS_DIR.is_dir():
        pytest.skip("Artifacts directory not found")
    artifact_files = list(ARTIFACTS_DIR.glob("*.json"))
    if not artifact_files:
        pytest.skip("No JSON artifacts found")
    for af in artifact_files:
        text = af.read_text(encoding="utf-8", errors="replace")
        for pat in ABSOLUTE_PATH_PATTERNS:
            m = pat.search(text)
            assert not m, \
                f"Absolute path '{m.group()}' found in artifact {af.name}"


def test_artifacts_no_server_ips():
    # Verify artifacts (if they exist) don't contain real server IPs
    if not ARTIFACTS_DIR.is_dir():
        pytest.skip("Artifacts directory not found")
    artifact_files = list(ARTIFACTS_DIR.glob("*.json"))
    if not artifact_files:
        pytest.skip("No JSON artifacts found")
    for af in artifact_files:
        text = af.read_text(encoding="utf-8", errors="replace")
        for pat in SERVER_IP_PATTERNS:
            m = pat.search(text)
            assert not m, \
                f"Server IP '{m.group()}' found in artifact {af.name}"


def test_deploy_scripts_no_absolute_paths_in_output():
    # Verify deploy scripts don't echo absolute paths to stdout (in printf/echo)
    deploy_files = _collect_deploy_files()
    assert len(deploy_files) > 0, "No files in scripts/deploy/"
    for f in deploy_files:
        if not f.is_file() or f.suffix != ".sh":
            continue
        text = f.read_text(encoding="utf-8")
        # Check for echo/printf statements that include absolute paths
        echo_lines = re.findall(
            r'(?:echo|printf)\s+["\'].*(?:/home/|/Users/|/root/|C:\\\\Users).*["\']',
            text,
        )
        assert not echo_lines, \
            f"{f.name} echoes absolute paths: {echo_lines[:3]}"
