"""Tests for demo asset privacy and security.

Verify README files don't contain server IPs or credentials, SVG files
don't contain IP addresses or email addresses, and example data is
explicitly marked as synthetic.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
README_ZH = REPO_ROOT / "README.zh-CN.md"
DEMO_DIR = REPO_ROOT / "assets" / "demo"
EXAMPLES_DIR = REPO_ROOT / "examples" / "demo"
EXAMPLES_README = EXAMPLES_DIR / "README.md"

# IP address patterns (IPv4)
IP_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)

# Email patterns
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# Credential/token patterns
CREDENTIAL_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
    re.compile(r"gho_[a-zA-Z0-9]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*['\"](?!smoke|test|example|placeholder|change|not-needed|dummy|xxx|todo)[^'\"]{8,}['\"]", re.IGNORECASE),
]

# IP addresses that are acceptable (loopback, example ranges)
SAFE_IPS = {
    "127.0.0.1", "0.0.0.0", "255.255.255.255",
    "10.0.0.0", "172.16.0.0", "192.168.0.0",
}


def _read_if_exists(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def test_no_server_ip_in_readme():
    # README.md doesn't contain real server IP patterns
    text = _read_if_exists(README)
    # Allow loopback, but flag private/real IPs
    ips = IP_PATTERN.findall(text)
    for ip in ips:
        if ip == "127.0.0.1":
            continue
        # Check if it's a non-safe IP
        parts = ip.split(".")
        if len(parts) == 4:
            octets = [int(p) for p in parts]
            # Allow documentation-style IPs like <user>@<server> placeholders
            # but flag anything that looks like a real server IP
            if octets[0] in (10,):
                assert False, f"Private IP address found in README.md: {ip}"
            if octets[0] == 172 and 16 <= octets[1] <= 31:
                assert False, f"Private IP address found in README.md: {ip}"
            if octets[0] == 192 and octets[1] == 168:
                assert False, f"Private IP address found in README.md: {ip}"
            # Also flag public IPs in specific known ranges (e.g., 43.139.x.x)
            if octets[0] == 43 and octets[1] == 139:
                assert False, f"Server IP address found in README.md: {ip}"


def test_no_credentials_in_readme():
    # README.md doesn't contain passwords/tokens
    text = _read_if_exists(README)
    for pat in CREDENTIAL_PATTERNS:
        m = pat.search(text)
        assert not m, f"Potential credential found in README.md: {m.group() if m else ''}"


def test_svg_no_ip_addresses():
    # Demo SVGs don't contain real IP addresses
    if not DEMO_DIR.is_dir():
        pytest.skip("assets/demo/ directory not found")
    svg_files = sorted(DEMO_DIR.glob("*.svg"))
    if not svg_files:
        pytest.skip("No SVG files in assets/demo/")
    for svg_path in svg_files:
        text = svg_path.read_text(encoding="utf-8", errors="replace")
        ips = IP_PATTERN.findall(text)
        for ip in ips:
            parts = ip.split(".")
            if len(parts) == 4:
                octets = [int(p) for p in parts]
                # Allow loopback
                if ip == "127.0.0.1":
                    continue
                # Allow private ranges in context of documentation (e.g., config text)
                if octets[0] == 10 or (octets[0] == 172 and 16 <= octets[1] <= 31) or \
                   (octets[0] == 192 and octets[1] == 168):
                    continue
                assert False, \
                    f"Public IP address found in {svg_path.name}: {ip}"


def test_svg_no_email_addresses():
    # Demo SVGs don't contain email addresses
    if not DEMO_DIR.is_dir():
        pytest.skip("assets/demo/ directory not found")
    svg_files = sorted(DEMO_DIR.glob("*.svg"))
    if not svg_files:
        pytest.skip("No SVG files in assets/demo/")
    for svg_path in svg_files:
        text = svg_path.read_text(encoding="utf-8", errors="replace")
        emails = EMAIL_PATTERN.findall(text)
        assert not emails, \
            f"Email address(es) found in {svg_path.name}: {emails}"


def test_examples_not_real_data():
    # examples/demo/README.md exists and says the data is synthetic
    if not EXAMPLES_README.is_file():
        pytest.skip(f"Not found: {EXAMPLES_README}")
    text = EXAMPLES_README.read_text(encoding="utf-8")
    synthetic_markers = ["synthetic", "synthetically", "generated", "not real", "simulated", "dummy", "test data"]
    found = any(marker in text.lower() for marker in synthetic_markers)
    assert found, \
        "examples/demo/README.md should state that the data is synthetic"
