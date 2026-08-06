from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, filename: str):
    path = ROOT / "scripts" / "evaluation" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


shadow = _load("gate04c", "run_pdf_v4_gate_04c_shadow.py")


def _features(**overrides):
    base = {
        "same_section": True,
        "same_statement": True,
        "column_count_compatible": True,
        "column_band_similarity": 0.95,
        "scale_compatible": True,
        "currency_compatible": True,
        "period_set_compatible": True,
        "left_near_page_bottom": True,
        "right_near_page_top": True,
        "row_label_style_compatible": True,
        "header_fingerprint_equal": False,
        "left_row_count": 25,
        "right_row_count": 9,
        "right_first_row_role": "metric",
        "right_first_row_label": "Stockholders equity",
    }
    base.update(overrides)
    return {"features": base, "hard_blockers": []}


def test_moderate_structural_evidence_creates_soft_link_only():
    accepted, confidence, signals = shadow._soft_link(_features())
    assert accepted
    assert confidence == 1.0
    assert "fragment_size_asymmetry" in signals


def test_unit_only_first_row_is_not_soft_linked():
    accepted, _, _ = shadow._soft_link(_features(right_first_row_role="category", right_first_row_label="(In millions)"))
    assert not accepted


def test_hard_blocker_fail_closed():
    item = _features()
    item["hard_blockers"] = ["conflicting_statement"]
    accepted, confidence, reasons = shadow._soft_link(item)
    assert not accepted
    assert confidence == 0.0
    assert reasons == ["blocked:conflicting_statement"]
