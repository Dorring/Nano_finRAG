"""Tests for Gate 08 R1.1 Coverage Denominator Closure.

Covers:
- classify_gold_source: four-way mutually-exclusive classification (A > B > C > D)
- _check_view_page_presence: V4 view-based page presence check
- Artifact contracts: acceptance, integrity, classification structure
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SCRIPT_PATH = ROOT / "scripts" / "evaluation" / "rescore_pdf_v4_gate_08_r1_1.py"


def _load_script_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("rescore_r1_1", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# classify_gold_source: four-way mutually-exclusive classification
# ---------------------------------------------------------------------------


class TestClassifyGoldSource:
    """Verify priority A > B > C > D and mutual exclusivity."""

    def test_class_a_recovered_takes_precedence_over_strict_mapped(self) -> None:
        mod = _load_script_module()
        # In pool AND strict-mapped → A (not B)
        result = mod.classify_gold_source(
            in_combined_pool=True, strict_mapped=True
        )
        assert result == mod.COVERAGE_RECOVERED

    def test_class_a_recovered_takes_precedence_over_absent(self) -> None:
        mod = _load_script_module()
        # In pool but not strict-mapped → A (not D)
        result = mod.classify_gold_source(
            in_combined_pool=True, strict_mapped=False
        )
        assert result == mod.COVERAGE_RECOVERED

    def test_class_b_strict_mapped_not_retrieved(self) -> None:
        mod = _load_script_module()
        # Not in pool but strict-mapped → B
        result = mod.classify_gold_source(
            in_combined_pool=False, strict_mapped=True
        )
        assert result == mod.COVERAGE_STRICT_MAPPED_NOT_RETRIEVED

    def test_class_d_structurally_absent(self) -> None:
        mod = _load_script_module()
        # Not in pool and not strict-mapped → D
        result = mod.classify_gold_source(
            in_combined_pool=False, strict_mapped=False
        )
        assert result == mod.COVERAGE_STRUCTURALLY_ABSENT

    def test_class_c_is_empty_by_construction(self) -> None:
        """C is always empty because structural_present = strict_mapped OR in_pool.

        Not-strict-mapped-and-not-in-pool is always D (absent).
        There is no input combination that produces C.
        """
        mod = _load_script_module()
        for in_pool in (True, False):
            for strict in (True, False):
                result = mod.classify_gold_source(
                    in_combined_pool=in_pool, strict_mapped=strict
                )
                assert result != mod.COVERAGE_STRUCTURAL_PRESENT_STRICT_UNMAPPED, (
                    f"C should be empty by construction, but got C for "
                    f"in_pool={in_pool}, strict={strict}"
                )

    def test_all_results_are_known_classes(self) -> None:
        mod = _load_script_module()
        for in_pool in (True, False):
            for strict in (True, False):
                result = mod.classify_gold_source(
                    in_combined_pool=in_pool, strict_mapped=strict
                )
                assert result in mod.ALL_COVERAGE_CLASSES

    def test_stage_mapping_complete(self) -> None:
        mod = _load_script_module()
        for cls in mod.ALL_COVERAGE_CLASSES:
            assert cls in mod.STAGE_BY_CLASS


# ---------------------------------------------------------------------------
# _check_view_page_presence
# ---------------------------------------------------------------------------


class TestViewPagePresence:
    def test_present_returns_true_and_view_ids(self) -> None:
        mod = _load_script_module()
        index = {("doc-a", 12): {"view-1", "view-2"}}
        present, view_ids = mod._check_view_page_presence(index, "doc-a", 12)
        assert present is True
        assert view_ids == ["view-1", "view-2"]

    def test_absent_returns_false(self) -> None:
        mod = _load_script_module()
        index = {("doc-a", 12): {"view-1"}}
        present, view_ids = mod._check_view_page_presence(index, "doc-a", 99)
        assert present is False
        assert view_ids == []

    def test_empty_document_id_returns_false(self) -> None:
        mod = _load_script_module()
        index = {("doc-a", 12): {"view-1"}}
        present, _ = mod._check_view_page_presence(index, "", 12)
        assert present is False

    def test_none_page_returns_false(self) -> None:
        mod = _load_script_module()
        index = {("doc-a", 12): {"view-1"}}
        present, _ = mod._check_view_page_presence(index, "doc-a", None)
        assert present is False

    def test_zero_page_returns_false(self) -> None:
        mod = _load_script_module()
        index = {("doc-a", 0): {"view-1"}}
        present, _ = mod._check_view_page_presence(index, "doc-a", 0)
        assert present is False


# ---------------------------------------------------------------------------
# Artifact contract tests (run against generated artifacts if present)
# ---------------------------------------------------------------------------


class TestR11ArtifactsContract:
    """Validate the structure of generated R1.1 artifacts."""

    R11_DIR = ROOT / "artifacts" / "evaluation" / "pdf-retrieval-v4-gate-08-r1-1"

    def _skip_if_missing(self) -> None:
        if not self.R11_DIR.is_dir():
            pytest.skip("R1.1 artifacts not generated")

    def test_acceptance_json_fields(self) -> None:
        self._skip_if_missing()
        data = json.loads((self.R11_DIR / "acceptance.json").read_text(encoding="utf-8"))
        required = {
            "gate",
            "gold_classification",
            "classification_mutually_exclusive",
            "classification_total",
            "recovered",
            "structural_universe",
            "strict_mapped_universe",
            "coverage_class_counts",
            "prediction_rerun",
            "retriever_runs",
            "runtime_gold_reads",
            "decision",
        }
        assert required.issubset(data.keys())
        assert data["prediction_rerun"] is False
        assert data["retriever_runs"] == 0
        assert data["runtime_gold_reads"] == 0

    def test_acceptance_decision_closed(self) -> None:
        self._skip_if_missing()
        data = json.loads((self.R11_DIR / "acceptance.json").read_text(encoding="utf-8"))
        assert data["decision"] == "coverage_denominator_contract_closed"

    def test_acceptance_key_metrics(self) -> None:
        self._skip_if_missing()
        data = json.loads((self.R11_DIR / "acceptance.json").read_text(encoding="utf-8"))
        assert data["recovered"] == 42
        assert data["structural_universe"] == 64
        assert data["strict_mapped_universe"] == 55

    def test_classification_integrity(self) -> None:
        self._skip_if_missing()
        data = json.loads(
            (self.R11_DIR / "classification-integrity.json").read_text(encoding="utf-8")
        )
        assert data["is_exhaustive"] is True
        assert data["is_mutually_exclusive"] is True
        assert data["all_classes_known"] is True
        assert data["sum_equals_total"] is True
        assert data["sum_check"] == 80
        assert data["total_gold"] == 80

    def test_gold_coverage_classification_rows(self) -> None:
        self._skip_if_missing()
        data = json.loads(
            (self.R11_DIR / "gold-coverage-classification.json").read_text(encoding="utf-8")
        )
        assert data["total_gold"] == 80
        assert data["mutually_exclusive"] is True
        assert data["exhaustive"] is True
        assert len(data["rows"]) == 80
        # Each row has required fields
        required_fields = {
            "gold_source_identity",
            "case_id",
            "source_index",
            "gold_candidate_key",
            "document_id",
            "pdf_page",
            "structural_present",
            "strict_mapping_available",
            "retrieved",
            "in_combined_pool",
            "coverage_class",
            "first_failure_stage",
        }
        for row in data["rows"]:
            assert required_fields.issubset(row.keys())

    def test_coverage_class_counts_sum_to_80(self) -> None:
        self._skip_if_missing()
        data = json.loads(
            (self.R11_DIR / "gold-coverage-classification.json").read_text(encoding="utf-8")
        )
        counts = data["coverage_class_counts"]
        total = sum(counts.values())
        assert total == 80

    def test_next_gate(self) -> None:
        self._skip_if_missing()
        data = json.loads((self.R11_DIR / "next-gate.json").read_text(encoding="utf-8"))
        assert data["next_gate"] == "gate_05_r5a_strict_candidate_bridge_recovery"
        assert data["decision"] == "coverage_denominator_contract_closed"

    def test_structural_universe_metrics(self) -> None:
        self._skip_if_missing()
        data = json.loads(
            (self.R11_DIR / "structural-universe-metrics.json").read_text(encoding="utf-8")
        )
        assert data["structural_universe_count"] == 64
        assert data["structurally_absent_count"] == 16

    def test_strict_mapping_universe_metrics(self) -> None:
        self._skip_if_missing()
        data = json.loads(
            (self.R11_DIR / "strict-mapping-universe-metrics.json").read_text(encoding="utf-8")
        )
        assert data["strict_mapped_universe_count"] == 55
        assert data["strict_unmapped_count"] == 25

    def test_retrieval_gap_metrics(self) -> None:
        self._skip_if_missing()
        data = json.loads(
            (self.R11_DIR / "retrieval-gap-metrics.json").read_text(encoding="utf-8")
        )
        assert data["recovered_strict"] == 42
        assert data["structurally_absent"] == 16
        assert data["universe_expansion_target"] == 16
