"""Tests for the verified-metrics documentation.

Verify docs/showcase/verified-metrics.md exists and contains required
entries for financial operations, validation categories, services,
deployment acceptance, and tests. Also verify it does NOT contain
prohibited items presented as quality metrics.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIED_METRICS = REPO_ROOT / "docs" / "showcase" / "verified-metrics.md"

REQUIRED_ENTRIES = [
    "9",
    "financial operations",
    "validation categories",
    "3",
    "services",
    "42/42",
    "deployment acceptance",
    ("2,000", "2000"),
    "tests",
]

# Items that should NOT appear as quality metrics in this doc
PROHIBITED_CONTENT = [
    "0/54",
    "59.5%",
    "17.68B",
]


def _read_doc() -> str:
    if not VERIFIED_METRICS.is_file():
        pytest.skip(f"Not found: {VERIFIED_METRICS}")
    return VERIFIED_METRICS.read_text(encoding="utf-8")


def test_verified_metrics_doc_exists():
    # docs/showcase/verified-metrics.md exists
    assert VERIFIED_METRICS.is_file(), \
        f"Expected {VERIFIED_METRICS} to exist"


def test_verified_metrics_has_required_entries():
    # Contains entries for: financial operations (9), validation categories,
    # services (3), deployment acceptance (42/42), tests (2000+)
    text = _read_doc().lower()

    assert "9" in text, "verified-metrics.md should mention 9 financial operations"
    assert "financial" in text, "verified-metrics.md should discuss financial operations"
    assert "validation" in text, "verified-metrics.md should discuss validation categories"
    assert "3" in text, "verified-metrics.md should mention 3 services"
    assert "service" in text, "verified-metrics.md should discuss services"
    assert "42/42" in text, "verified-metrics.md should mention 42/42 deployment acceptance"
    assert "test" in text, "verified-metrics.md should mention tests"
    assert ("2000" in text or "2,000" in text), \
        "verified-metrics.md should mention 2000+ tests"


def test_verified_metrics_does_not_contain_prohibited():
    # Does NOT contain: 0/54, 59.5%, 17.68B as quality metrics
    text = _read_doc()

    # These terms may appear in a "not used as quality claims" section
    # but should not appear in a positive/metrics context
    for term in PROHIBITED_CONTENT:
        if term in text:
            # Check that it appears in a "not used" or "prohibited" context
            # Find surrounding context
            idx = text.find(term)
            context_start = max(0, idx - 200)
            context_end = min(len(text), idx + 200)
            surrounding = text[context_start:context_end].lower()
            negating_phrases = [
                "not", "excluded", "prohibited", "should not",
                "不应", "不得", "不可",
                "unavailable", "unverifiable", "historical",
            ]
            is_in_negating_context = any(p in surrounding for p in negating_phrases)
            assert is_in_negating_context, \
                f"'{term}' found in verified-metrics.md outside of disclaimer context"
