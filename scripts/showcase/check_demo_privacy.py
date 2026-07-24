#!/usr/bin/env python3
"""Check for privacy issues in demo assets and documentation.

Output: artifacts/showcase/phase8/privacy-report.json
Exit: 0 on all pass, 1 on any failure.
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "artifacts" / "showcase" / "phase8"
OUTPUT_FILE = OUTPUT_DIR / "privacy-report.json"

# --- Regex patterns ---

# IPv4 addresses (including 10.x, 192.168.x, 172.16-31.x)
IP_PATTERN = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b"
)

# Email addresses
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Common username patterns in demo content
USERNAME_PATTERNS = [
    re.compile(r"\b(?:username|user)\s*[=:]\s*['\"]?(\w+)['\"]?", re.IGNORECASE),
    re.compile(r"\b(?:login\s*as|logged\s*in\s*as)\s+['\"]?(\w+)['\"]?", re.IGNORECASE),
    re.compile(r"\buser_id\s*[=:]\s*['\"]?(\w+)['\"]?", re.IGNORECASE),
    re.compile(r"\b(?:author|created\s*by)\s*[=:]\s*['\"]?(\w+)['\"]?", re.IGNORECASE),
]

# Credential patterns
PASSWORD_PATTERN = re.compile(
    r"(?:password|passwd|pwd|token|secret|api_key|apikey|access_key)\s*[=:]\s*['\"]?([^\s'\"]{3,})['\"]?",
    re.IGNORECASE,
)

# Server IP patterns specifically in READMEs
SERVER_IP_PATTERN = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_ip_addresses(text: str) -> list[str]:
    """Find IP addresses. Exclude 127.0.0.1 (localhost)."""
    matches = IP_PATTERN.findall(text)
    return [m for m in matches if m != "127.0.0.1" and not m.startswith("127.0.")]


def find_emails(text: str) -> list[str]:
    return re.findall(EMAIL_PATTERN, text)


def find_usernames(text: str) -> list[str]:
    found: list[str] = []
    for pattern in USERNAME_PATTERNS:
        for match in pattern.finditer(text):
            val = match.group(1)
            if val and len(val) > 1:
                found.append(f"{match.group(0)}  (pattern: {pattern.pattern})")
    return found


def find_credentials(text: str) -> list[str]:
    matches: list[str] = []
    for match in PASSWORD_PATTERN.finditer(text):
        full = match.group(0)
        val = match.group(1)
        # Exclude placeholder patterns like <user>, <token>, <password>, YOUR_TOKEN, etc.
        if re.match(r"^[<\[(].*[>)\]]$", val):
            continue
        if val.upper() in ("TOKEN", "PASSWORD", "SECRET", "API_KEY", "KEY"):
            continue
        if re.match(r"^YOUR_", val, re.IGNORECASE):
            continue
        matches.append(full)
    return matches


def find_server_ips(text: str) -> list[str]:
    """Find private network IPs in the text."""
    matches = SERVER_IP_PATTERN.findall(text)
    return [m for m in matches if not m.startswith("127.")]


def check_svg_files() -> list[dict]:
    """Check all .svg files in assets/ and assets/demo/ for private data."""
    results: list[dict] = []
    svg_dirs = [
        REPO_ROOT / "assets" / "demo",
        REPO_ROOT / "assets",
    ]

    svg_files: list[Path] = []
    for d in svg_dirs:
        if d.exists() and d.is_dir():
            svg_files.extend(d.rglob("*.svg"))

    if not svg_files:
        results.append({
            "check": "SVG privacy: IP addresses, emails, usernames",
            "pass": True,
            "detail": "No .svg files found in assets/demo/ or assets/",
        })
        return results

    for svg_path in svg_files:
        rel_path = svg_path.relative_to(REPO_ROOT)
        text = read_text(svg_path)

        ips = find_ip_addresses(text)
        emails = find_emails(text)
        usernames = find_usernames(text)

        issues: list[str] = []
        if ips:
            issues.append(f"IP addresses: {ips}")
        if emails:
            issues.append(f"emails: {emails}")
        if usernames:
            issues.append(f"usernames: {usernames}")

        passed = len(issues) == 0
        results.append({
            "check": f"SVG privacy: {rel_path}",
            "pass": passed,
            "detail": "OK" if passed else f"Issues: {'; '.join(issues)}",
            "file": str(rel_path),
        })

    return results


def check_readme_server_ips() -> dict:
    """Check README.md for real server IPs."""
    readme_path = REPO_ROOT / "README.md"
    if not readme_path.exists():
        return {
            "check": "README.md server IPs",
            "pass": True,
            "detail": "README.md not found",
        }
    text = read_text(readme_path)
    ips = find_server_ips(text)
    passed = len(ips) == 0
    return {
        "check": "README.md: no real server IPs (10.x, 192.168.x.x)",
        "pass": passed,
        "detail": "OK" if passed else f"Found server IPs: {ips}",
    }


def check_readme_credentials() -> dict:
    """Check README.md for credentials."""
    readme_path = REPO_ROOT / "README.md"
    if not readme_path.exists():
        return {
            "check": "README.md credentials",
            "pass": True,
            "detail": "README.md not found",
        }
    text = read_text(readme_path)
    creds = find_credentials(text)
    passed = len(creds) == 0
    return {
        "check": "README.md: no credentials (password/token with real values)",
        "pass": passed,
        "detail": "OK" if passed else f"Found credentials: {creds}",
    }


def check_examples_demo() -> dict:
    """Check examples/demo/ files don't claim to be real company data."""
    examples_demo_dir = REPO_ROOT / "examples" / "demo"
    if not examples_demo_dir.exists() or not examples_demo_dir.is_dir():
        return {
            "check": "examples/demo/: no real company data claims",
            "pass": True,
            "detail": "examples/demo/ directory does not exist",
        }

    real_company_patterns = [
        re.compile(r"\breal\s+(?:company|data|financial)\b", re.IGNORECASE),
        re.compile(r"\bactual\s+(?:company|data|financial|earnings)\b", re.IGNORECASE),
        re.compile(r"\b(?:true|genuine)\s+(?:company|corporate)\s+(?:data|financials|reports)\b", re.IGNORECASE),
    ]

    issues: list[dict] = []
    for file_path in examples_demo_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(REPO_ROOT)
        try:
            text = read_text(file_path)
        except Exception:
            continue
        for pattern in real_company_patterns:
            matches = pattern.findall(text)
            if matches:
                issues.append({
                    "file": str(rel_path),
                    "matches": matches,
                })

    passed = len(issues) == 0
    detail = "OK" if passed else f"Issues: {json.dumps(issues)}"
    return {
        "check": "examples/demo/: no real company data claims",
        "pass": passed,
        "detail": detail,
    }


def main() -> int:
    results: list[dict] = []

    # Check 1: SVG files
    results.extend(check_svg_files())

    # Check 2: README.md server IPs
    results.append(check_readme_server_ips())

    # Check 3: README.md credentials
    results.append(check_readme_credentials())

    # Check 4: examples/demo/ files
    results.append(check_examples_demo())

    # Summary
    all_pass = all(r["pass"] for r in results)
    failures = [r for r in results if not r["pass"]]

    report = {
        "title": "Privacy Report",
        "overall_pass": all_pass,
        "total_checks": len(results),
        "passed": sum(1 for r in results if r["pass"]),
        "failed": len(failures),
        "checks": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to {OUTPUT_FILE}")

    if all_pass:
        print(f"ALL {len(results)} CHECKS PASSED")
        return 0
    else:
        print(f"{len(failures)}/{len(results)} CHECKS FAILED:")
        for f in failures:
            print(f"  FAIL: {f['check']} — {f['detail']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
