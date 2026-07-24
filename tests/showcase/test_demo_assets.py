"""Tests for demo assets existence and validity.

Verify demo SVG files exist and are valid SVG, example data files exist
and contain valid JSON, and the SVG files are properly structured.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "assets" / "demo"
EXAMPLES_DIR = REPO_ROOT / "examples" / "demo"

EXPECTED_SVGS = [
    "demo-01-financial-qa.svg",
    "demo-02-deterministic-calculation.svg",
    "demo-03-unit-period-ambiguity.svg",
    "demo-04-unanswerable-safe-refusal.svg",
    "demo-05-online-three-service.svg",
    "demo-06-ssh-tunnel.svg",
]

EXPECTED_JSON_FILES = [
    "demo-questions.json",
    "expected-behaviors.json",
]


def test_demo_assets_exist():
    # All 6 demo SVG files exist in assets/demo/
    missing = []
    for svg_name in EXPECTED_SVGS:
        svg_path = DEMO_DIR / svg_name
        if not svg_path.is_file():
            missing.append(svg_name)
    assert not missing, \
        "Missing demo SVG files:\n" + "\n".join(missing)


@pytest.mark.parametrize("svg_name", EXPECTED_SVGS)
def test_demo_assets_are_valid_svg(svg_name):
    # Each SVG file starts with <svg tag
    svg_path = DEMO_DIR / svg_name
    if not svg_path.is_file():
        pytest.skip(f"SVG file not found: {svg_name}")
    content = svg_path.read_text(encoding="utf-8").strip()
    # SVG files may start with <?xml declaration, but should contain <svg
    assert "<svg" in content.lower(), \
        f"{svg_name} does not contain an <svg> tag"


def test_example_data_exists():
    # examples/demo/demo-questions.json and expected-behaviors.json exist
    missing = []
    for json_name in EXPECTED_JSON_FILES:
        json_path = EXAMPLES_DIR / json_name
        if not json_path.is_file():
            missing.append(json_name)
    assert not missing, \
        "Missing example data files:\n" + "\n".join(missing)


@pytest.mark.parametrize("json_name", EXPECTED_JSON_FILES)
def test_example_data_valid_json(json_name):
    # Both JSON files are valid JSON
    json_path = EXAMPLES_DIR / json_name
    if not json_path.is_file():
        pytest.skip(f"JSON file not found: {json_name}")
    text = json_path.read_text(encoding="utf-8")
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        pytest.fail(f"{json_name} is not valid JSON: {e}")
