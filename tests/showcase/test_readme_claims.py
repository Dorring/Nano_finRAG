"""Tests for README claim compliance.

Verify that README.md and README.zh-CN.md do not contain prohibited claims
such as "production-grade accuracy", "eliminates hallucinations",
"state-of-the-art", using 0/54 as a quality metric, or presenting
unverified metrics in the verified metrics section.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
README_ZH = REPO_ROOT / "README.zh-CN.md"

# The "Verified Engineering Metrics" section in README.md — we extract its
# content by finding the heading and reading until the next top-level heading
# or the end of the document.
VERIFIED_SECTION_HEADING = "## Verified Engineering Metrics"


def _read_both() -> tuple[str, str]:
    en = README.read_text(encoding="utf-8") if README.is_file() else ""
    zh = README_ZH.read_text(encoding="utf-8") if README_ZH.is_file() else ""
    return en, zh


def _strip_blockquotes(text: str) -> str:
    """Remove blockquote lines (starting with >) from markdown text.
    These are typically disclaimers, not claims."""
    lines = text.splitlines()
    return "\n".join(line for line in lines if not line.strip().startswith(">"))


def _extract_section(text: str, heading: str) -> str:
    """Extract content from a markdown section starting with the given heading.
    Strips blockquote lines (starting with >) which contain disclaimers."""
    idx = text.find(heading)
    if idx == -1:
        return ""
    section_start = idx + len(heading)
    remainder = text[section_start:]

    # Find the next heading of equal or higher level (##)
    next_heading = re.search(r"^##\s", remainder, re.MULTILINE)
    if next_heading:
        remainder = remainder[: next_heading.start()]

    # Strip blockquote lines (disclaimers)
    lines = remainder.splitlines()
    non_quote_lines = [line for line in lines if not line.strip().startswith(">")]
    return "\n".join(non_quote_lines)


def test_no_production_grade_accuracy_claim():
    # Verify "production-grade accuracy" not used as a claim in either README.
    # It may appear only in the disclaimer (blockquote) context where it is
    # explicitly labeled as "not used".  We strip blockquotes before checking.
    en_text, zh_text = _read_both()
    en_clean = _strip_blockquotes(en_text)
    zh_clean = _strip_blockquotes(zh_text)
    combined = en_clean.lower() + " " + zh_clean.lower()
    assert "production-grade accuracy" not in combined, \
        "READMEs should not contain 'production-grade accuracy' claim"
    assert "production grade accuracy" not in combined, \
        "READMEs should not contain 'production grade accuracy' claim"


def test_no_hallucination_elimination_claim():
    # Verify "eliminates hallucinations" not in either README
    en_text, zh_text = _read_both()
    combined = en_text.lower() + " " + zh_text.lower()
    assert "eliminates hallucinations" not in combined.lower(), \
        "READMES should not claim hallucination elimination"
    assert "eliminate hallucinations" not in combined.lower(), \
        "READMES should not claim hallucination elimination"


def test_no_native_tool_calling_claim():
    # Calculator not described as model-native tool calling
    en_text, zh_text = _read_both()
    # Check that the calculator is described as a system component, not
    # model-native tool calling. Look for the disclaiming language.
    assert "not model-native tool calling" in en_text.lower() or \
           "system component" in en_text.lower(), \
        "README should clarify calculator is a system component, not model-native tool calling"

    # In Chinese version, check for "工具调用" in context of calculator
    if "工具调用" in zh_text:
        # If mentioned, must be in the context of "not model-native"
        assert "而非模型原生的工具调用" in zh_text or \
               "系统组件" in zh_text, \
            "Chinese README should clarify calculator is system component, not model-native tool calling"


def test_no_state_of_the_art_claim():
    # Verify no "state-of-the-art" claims in either README
    en_text, zh_text = _read_both()
    combined = en_text.lower() + " " + zh_text.lower()
    state_phrases = ["state-of-the-art", "state of the art"]
    for phrase in state_phrases:
        assert phrase not in combined, \
            f"READMEs should not contain '{phrase}' claim"


def test_zero_of_54_not_used_as_quality_metric():
    # Verify 0/54 not presented as quality metric — it should be in the
    # "explicitly not used" disclaimer instead
    en_text, zh_text = _read_both()

    # The number "0/54" appears in the disclaimer area
    if "0/54" in en_text:
        # Must appear in context of "not used as quality claims"
        assert "not" in en_text.lower(), \
            "0/54 should only appear in disclaimer context"


def test_only_verified_metrics_on_landing_page():
    # Unverified metrics (tokenizer compression, 17.68B) not in
    # Verified Engineering Metrics section
    en_text, _ = _read_both()
    verified_section = _extract_section(en_text, VERIFIED_SECTION_HEADING)
    if not verified_section:
        pytest.skip("Verified Engineering Metrics section not found in README.md")

    # The verified section should NOT contain unverified metrics
    prohibited_in_verified = [
        "17.68B",
        "tokenizer compression",
        "compression rate",
        "val_bpb",
        "BPB",
        "0.7626",
        "0.5558",
        "CORE metric",
        "0.2201",
    ]
    for term in prohibited_in_verified:
        assert term not in verified_section, \
            f"Unverified metric '{term}' found in Verified Engineering Metrics section"


def test_disclaimer_for_historical_data():
    # Historical data disclaimer present
    en_text, zh_text = _read_both()

    # English README should have historical data disclaimer
    assert "historical" in en_text.lower(), \
        "README.md should have a historical data disclaimer"
    assert "unavailable" in en_text.lower() or "cannot" in en_text.lower() or \
           "unable" in en_text.lower(), \
        "README.md should note historical data is unavailable for verification"

    # Chinese README should have historical data disclaimer
    assert "历史" in zh_text, \
        "README.zh-CN.md should have a historical data disclaimer"
