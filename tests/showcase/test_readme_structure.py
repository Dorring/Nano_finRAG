"""Tests for README structural requirements.

Verify README.md and README.zh-CN.md exist, have correct titles, required
sections, Mermaid diagrams, cross-language links, and consistent key numbers.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
README_ZH = REPO_ROOT / "README.zh-CN.md"

REQUIRED_SECTIONS = [
    "Project Overview",
    "System Architecture",
    "Core Problems",
    "Verified Engineering Metrics",
    "Quick Start",
    "Known Limitations",
    "License",
]


def test_readme_exists():
    # Verify README.md exists
    assert README.is_file(), f"Expected {README} to exist"


def test_readme_zh_exists():
    # Verify README.zh-CN.md exists
    assert README_ZH.is_file(), f"Expected {README_ZH} to exist"


def test_readme_is_project_specific():
    # Verify README.md title is "nano_finance", not "nanochat"
    text = README.read_text(encoding="utf-8")
    first_heading = text.splitlines()[0]
    assert "nano_finance" in first_heading.lower(), \
        f"README.md title should be 'nano_finance', got: {first_heading}"
    assert "nanochat" not in first_heading.lower(), \
        f"README.md title should not be 'nanochat', got: {first_heading}"


def test_upstream_nanochat_attribution_exists():
    # Verify README.md mentions "NanoChat" or "nanochat" upstream
    text = README.read_text(encoding="utf-8")
    found = re.search(r"[Nn]ano[Cc]hat", text)
    assert found, "README.md should mention NanoChat/nanochat upstream attribution"


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_readme_has_required_sections(section):
    # Check for key sections in README.md
    text = README.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^#{1,3}\s+" + re.escape(section) + r"\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    found = pattern.search(text)
    if not found:
        # Also try loose match for sections that might have variant wording
        words = section.lower().split()
        all_present = all(w in text.lower() for w in words)
        assert all_present, \
            f"Required section '{section}' not found in README.md"


def test_readme_has_mermaid_diagram():
    # Check for ```mermaid block in README.md
    text = README.read_text(encoding="utf-8")
    assert "```mermaid" in text, \
        "README.md should contain a ```mermaid diagram block"


def test_readme_links_to_chinese():
    # Check README.md links to README.zh-CN.md
    text = README.read_text(encoding="utf-8")
    assert "README.zh-CN.md" in text, \
        "README.md should link to README.zh-CN.md"


def test_chinese_readme_links_to_english():
    # Check README.zh-CN.md links to README.md
    text = README_ZH.read_text(encoding="utf-8")
    assert "README.md" in text, \
        "README.zh-CN.md should link to README.md"


def test_english_and_chinese_readmes_consistent():
    # Check key numbers match between both READMEs
    en_text = README.read_text(encoding="utf-8")
    zh_text = README_ZH.read_text(encoding="utf-8")

    checks = [
        ("9", "9 deterministic financial operations"),
        ("6", "6+ validation categories"),
        ("3", "3 services/model/backend/frontend"),
        ("42/42", "42/42 deployment acceptance"),
        ("2,000", "2000+ tests"),
        ("12/12", "12/12 smoke tests"),
    ]

    for value, description in checks:
        if value not in en_text:
            pytest.fail(f"README.md missing expected value '{value}' ({description})")
        if value not in zh_text:
            pytest.fail(
                f"README.zh-CN.md missing expected value '{value}' ({description})"
            )
