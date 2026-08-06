"""Tests for Gate 08 R1 Evaluation Contract Repair.

Covers:
- stage_attribution.classify_first_failure: 9-class A-I unique classification
- structural_gold_mapper.GoldStructuralMatch: in_structured_universe property
- structural_gold_mapper.StructuralGoldMapper: priority matching (with fake mapper)
- rescore_pdf_v4_gate_08_r1._match_slots_semantic: Role→Period→Metric matching
- rescore_pdf_v4_gate_08_r1._norm_period / _norm_metric normalization
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.stage_attribution import (  # noqa: E402
    FirstFailureStage,
    StageAttributionInput,
    classify_first_failure,
)
from src.pdf_retrieval_v4.structural_gold_mapper import (  # noqa: E402
    GoldStructuralMatch,
    StructuralGoldMapper,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_sets() -> dict[str, set[str]]:
    return {
        "retrieved_table_view_ids": set(),
        "retrieved_table_candidate_keys": set(),
        "retrieved_row_view_ids": set(),
        "retrieved_row_candidate_keys": set(),
        "retrieved_fact_view_ids": set(),
        "retrieved_fact_candidate_keys": set(),
        "structured_pool_candidate_keys": set(),
        "combined_pool_candidate_keys": set(),
    }


def _input(case_id: str = "c1", gold_key: str = "g1", **overrides: Any) -> StageAttributionInput:
    base: dict[str, Any] = {
        "case_id": case_id,
        "gold_candidate_key": gold_key,
        "in_structured_universe": True,
        "universe_mapping_status": "unique",
        "gold_view_id": "view-g1",
        "structured_ambiguous_mapping_count": 0,
        **_empty_sets(),
    }
    base.update(overrides)
    return StageAttributionInput(**base)


# ---------------------------------------------------------------------------
# classify_first_failure: each Gold gets exactly one category
# ---------------------------------------------------------------------------


class TestClassifyFirstFailure:
    def test_recovered_when_in_combined_pool(self) -> None:
        result = classify_first_failure(
            _input(combined_pool_candidate_keys={"g1"})
        )
        assert result.first_failure_stage is FirstFailureStage.RECOVERED
        assert result.in_combined_pool is True

    def test_not_in_structured_universe(self) -> None:
        result = classify_first_failure(
            _input(in_structured_universe=False)
        )
        assert result.first_failure_stage is FirstFailureStage.NOT_IN_STRUCTURED_UNIVERSE
        assert result.in_combined_pool is False

    def test_structured_view_unmapped(self) -> None:
        result = classify_first_failure(
            _input(universe_mapping_status="unmapped")
        )
        assert result.first_failure_stage is FirstFailureStage.STRUCTURED_VIEW_UNMAPPED

    def test_structured_mapping_ambiguous(self) -> None:
        result = classify_first_failure(
            _input(universe_mapping_status="ambiguous")
        )
        assert result.first_failure_stage is FirstFailureStage.STRUCTURED_MAPPING_AMBIGUOUS

    def test_gold_table_not_retrieved(self) -> None:
        result = classify_first_failure(_input())
        assert result.first_failure_stage is FirstFailureStage.GOLD_TABLE_NOT_RETRIEVED
        assert result.recoverable_by_larger_k is True
        assert result.in_retrieved_table is False

    def test_gold_row_not_retrieved(self) -> None:
        result = classify_first_failure(
            _input(
                retrieved_table_candidate_keys={"g1"},
                retrieved_table_view_ids={"view-g1"},
            )
        )
        assert result.first_failure_stage is FirstFailureStage.GOLD_ROW_NOT_RETRIEVED
        assert result.in_retrieved_table is True
        assert result.in_retrieved_row is False

    def test_gold_fact_not_retrieved(self) -> None:
        result = classify_first_failure(
            _input(
                retrieved_table_candidate_keys={"g1"},
                retrieved_table_view_ids={"view-g1"},
                retrieved_row_candidate_keys={"g1"},
                retrieved_row_view_ids={"view-g1"},
            )
        )
        assert result.first_failure_stage is FirstFailureStage.GOLD_FACT_NOT_RETRIEVED
        assert result.in_retrieved_row is True
        assert result.in_retrieved_fact is False

    def test_fact_retrieved_mapping_ambiguous(self) -> None:
        result = classify_first_failure(
            _input(
                retrieved_table_candidate_keys={"g1"},
                retrieved_table_view_ids={"view-g1"},
                retrieved_row_candidate_keys={"g1"},
                retrieved_row_view_ids={"view-g1"},
                retrieved_fact_candidate_keys={"g1"},
                retrieved_fact_view_ids={"view-g1"},
                structured_ambiguous_mapping_count=2,
            )
        )
        assert result.first_failure_stage is FirstFailureStage.FACT_RETRIEVED_MAPPING_AMBIGUOUS

    def test_structured_budget_truncated(self) -> None:
        result = classify_first_failure(
            _input(
                retrieved_table_candidate_keys={"g1"},
                retrieved_table_view_ids={"view-g1"},
                retrieved_row_candidate_keys={"g1"},
                retrieved_row_view_ids={"view-g1"},
                retrieved_fact_candidate_keys={"g1"},
                retrieved_fact_view_ids={"view-g1"},
                structured_pool_candidate_keys={"g1"},
            )
        )
        assert result.first_failure_stage is FirstFailureStage.STRUCTURED_BUDGET_TRUNCATED

    def test_fact_retrieved_mapping_failed_fallback(self) -> None:
        result = classify_first_failure(
            _input(
                retrieved_table_candidate_keys={"g1"},
                retrieved_table_view_ids={"view-g1"},
                retrieved_row_candidate_keys={"g1"},
                retrieved_row_view_ids={"view-g1"},
                retrieved_fact_candidate_keys={"g1"},
                retrieved_fact_view_ids={"view-g1"},
            )
        )
        assert result.first_failure_stage is FirstFailureStage.FACT_RETRIEVED_MAPPING_FAILED

    def test_recovered_takes_precedence_over_universe(self) -> None:
        # Even if not in universe, combined pool recovery wins (stage 0 check)
        result = classify_first_failure(
            _input(
                in_structured_universe=False,
                combined_pool_candidate_keys={"g1"},
            )
        )
        assert result.first_failure_stage is FirstFailureStage.RECOVERED

    def test_to_dict_roundtrip(self) -> None:
        result = classify_first_failure(_input())
        d = result.to_dict()
        assert d["first_failure_stage"] == "gold_table_not_retrieved"
        assert d["candidate_identity"] == "g1"
        assert d["case_id"] == "c1"


# ---------------------------------------------------------------------------
# GoldStructuralMatch dataclass
# ---------------------------------------------------------------------------


class TestGoldStructuralMatch:
    def test_in_structured_universe_true_when_view_id_present(self) -> None:
        match = GoldStructuralMatch(
            gold_candidate_key="k1",
            case_id="c1",
            source_index=0,
            matched_retrieval_view_id="view-1",
        )
        assert match.in_structured_universe is True

    def test_in_structured_universe_false_when_view_id_none(self) -> None:
        match = GoldStructuralMatch(
            gold_candidate_key="k1",
            case_id="c1",
            source_index=0,
        )
        assert match.in_structured_universe is False
        assert match.mapping_method == "unresolved"

    def test_to_dict_contains_in_structured_universe(self) -> None:
        match = GoldStructuralMatch(
            gold_candidate_key="k1",
            case_id="c1",
            source_index=0,
            matched_retrieval_view_id="view-1",
            mapping_method="direct_candidate_key",
        )
        d = match.to_dict()
        assert d["in_structured_universe"] is True
        assert d["mapping_method"] == "direct_candidate_key"


# ---------------------------------------------------------------------------
# StructuralGoldMapper with a fake in-memory metadata DB
# ---------------------------------------------------------------------------


class _FakeMapper:
    """Fake ProductionCandidateMapper that maps views by a fixed candidate_key."""

    def __init__(self, key_map: dict[str, str | None]) -> None:
        # key_map: retrieval_view_id -> candidate_key (or None for unmapped)
        self._key_map = key_map

    def map_view(self, view: dict[str, Any]) -> dict[str, Any]:
        view_id = str(view.get("retrieval_view_id", ""))
        key = self._key_map.get(view_id)
        if key is None:
            return {"strict_candidate_status": "unmapped", "candidate_key": None}
        return {"strict_candidate_status": "unique", "candidate_key": key, "mapping_score": 1.0}


def _build_metadata_db(db_path: Path, views: list[dict[str, Any]]) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE retrieval_views ("
        "retrieval_view_id TEXT, evidence_unit_id TEXT, unit_type TEXT, "
        "retrieval_text TEXT, metadata_json TEXT)"
    )
    for view in views:
        conn.execute(
            "INSERT INTO retrieval_views VALUES (?, ?, ?, ?, ?)",
            (
                view["retrieval_view_id"],
                view.get("evidence_unit_id", "e-1"),
                view["unit_type"],
                view.get("retrieval_text", ""),
                json.dumps(view.get("metadata", {})),
            ),
        )
    conn.commit()
    conn.close()


class TestStructuralGoldMapper:
    @pytest.fixture()
    def metadata_db(self, tmp_path: Path) -> Path:
        db = tmp_path / "metadata.sqlite"
        views = [
            {
                "retrieval_view_id": "view-fact-1",
                "unit_type": "atomic_fact",
                "metadata": {
                    "document_id": "doc-a",
                    "pdf_pages": [12],
                    "logical_table_id": "tbl-1",
                    "row_id": "row-1",
                    "fact_id": "fact-1",
                    "metric_path": "total revenue",
                    "periods": ["FY2024"],
                },
            },
            {
                "retrieval_view_id": "view-row-1",
                "unit_type": "row",
                "metadata": {
                    "document_id": "doc-a",
                    "pdf_pages": [12],
                    "logical_table_id": "tbl-1",
                    "row_id": "row-1",
                    "metric_path": "net income",
                },
            },
            {
                "retrieval_view_id": "view-section-1",
                "unit_type": "section",
                "metadata": {"document_id": "doc-a", "pdf_pages": [1]},
            },
        ]
        _build_metadata_db(db, views)
        return db

    def test_priority1_direct_candidate_key_match(self, metadata_db: Path) -> None:
        fake_mapper = _FakeMapper({"view-fact-1": "gold-key-1"})
        with StructuralGoldMapper(metadata_db, fake_mapper) as mapper:  # type: ignore[arg-type]
            match = mapper.map_gold_source(
                case_id="c1",
                source_index=0,
                gold_candidate_key="gold-key-1",
            )
        assert match.in_structured_universe is True
        assert match.mapping_method == "direct_candidate_key"
        assert match.matched_retrieval_view_id == "view-fact-1"
        assert match.fact_id == "fact-1"

    def test_priority2_document_page_row_text(self, metadata_db: Path) -> None:
        # No direct key match; row view matches by doc+page+row text
        fake_mapper = _FakeMapper({"view-fact-1": "other-key"})
        with StructuralGoldMapper(metadata_db, fake_mapper) as mapper:  # type: ignore[arg-type]
            match = mapper.map_gold_source(
                case_id="c1",
                source_index=0,
                gold_candidate_key="gold-key-2",
                gold_document_id="doc-a",
                gold_page=12,
                gold_row_label="net income",
            )
        assert match.in_structured_universe is True
        assert match.mapping_method == "document_page_row_text"
        assert match.matched_unit_type == "row"

    def test_priority3_document_page_metric_period(self, metadata_db: Path) -> None:
        fake_mapper = _FakeMapper({})
        with StructuralGoldMapper(metadata_db, fake_mapper) as mapper:  # type: ignore[arg-type]
            match = mapper.map_gold_source(
                case_id="c1",
                source_index=0,
                gold_candidate_key="gold-key-3",
                gold_document_id="doc-a",
                gold_page=12,
                gold_metric="revenue",
                gold_period="FY2024",
            )
        assert match.in_structured_universe is True
        assert match.mapping_method == "document_page_metric_period"
        assert match.matched_unit_type == "atomic_fact"

    def test_priority5_unresolved(self, metadata_db: Path) -> None:
        fake_mapper = _FakeMapper({})
        with StructuralGoldMapper(metadata_db, fake_mapper) as mapper:  # type: ignore[arg-type]
            match = mapper.map_gold_source(
                case_id="c1",
                source_index=0,
                gold_candidate_key="gold-key-x",
                gold_document_id="doc-zzz",
                gold_page=999,
            )
        assert match.in_structured_universe is False
        assert match.mapping_method == "unresolved"

    def test_universe_candidate_map_records(self, metadata_db: Path) -> None:
        fake_mapper = _FakeMapper({"view-fact-1": "key-1", "view-row-1": "key-2"})
        with StructuralGoldMapper(metadata_db, fake_mapper) as mapper:  # type: ignore[arg-type]
            records = mapper.universe_candidate_map_records()
        assert len(records) == 3
        fact_rec = next(r for r in records if r["unit_type"] == "atomic_fact")
        assert fact_rec["bridge_status"] == "unique"
        assert fact_rec["direct_original_candidate_identities"] == ["key-1"]
        section_rec = next(r for r in records if r["unit_type"] == "section")
        assert section_rec["bridge_status"] == "unmapped"

    def test_view_counts_and_total(self, metadata_db: Path) -> None:
        fake_mapper = _FakeMapper({})
        with StructuralGoldMapper(metadata_db, fake_mapper) as mapper:  # type: ignore[arg-type]
            assert mapper.total_view_count == 3
            assert mapper.view_counts["atomic_fact"] == 1
            assert mapper.view_counts["row"] == 1


# ---------------------------------------------------------------------------
# rescore_pdf_v4_gate_08_r1: semantic slot matching helpers
# ---------------------------------------------------------------------------

# Import the script module (script path is not a package, so import by file)
_SCRIPT_PATH = ROOT / "scripts" / "evaluation" / "rescore_pdf_v4_gate_08_r1.py"


def _load_script_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("rescore_r1", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSlotSemanticMatching:
    def test_role_period_metric_match(self) -> None:
        mod = _load_script_module()
        gov_slots = [
            {"slot_id": "s1", "role": "current_period", "period": "FY2024", "metric": "revenue"},
        ]
        gold_sources = [
            {"candidate_key": "gold-1", "role": "current_period", "period": "FY2024", "row_label": "revenue"},
        ]
        records = mod._match_slots_semantic(gov_slots, ["s1"], gold_sources)
        assert records[0]["gold_identity"] == "gold-1"
        assert records[0]["match_method"] == "role_period_metric"

    def test_role_only_fallback(self) -> None:
        mod = _load_script_module()
        gov_slots = [
            {"slot_id": "s1", "role": "current_period", "period": "FY2024", "metric": "revenue"},
        ]
        gold_sources = [
            {"candidate_key": "gold-1", "role": "current_period", "period": "FY2023", "row_label": "income"},
        ]
        records = mod._match_slots_semantic(gov_slots, ["s1"], gold_sources)
        # Role matches but period/metric differ -> phase 3 role-only match
        assert records[0]["gold_identity"] == "gold-1"

    def test_index_fallback_when_no_semantic_match(self) -> None:
        mod = _load_script_module()
        gov_slots = [
            {"slot_id": "s1", "role": "left", "period": "FY2024", "metric": "revenue"},
        ]
        gold_sources = [
            {"candidate_key": "gold-1", "role": "right", "period": "FY2023", "row_label": "cost"},
        ]
        records = mod._match_slots_semantic(gov_slots, ["s1"], gold_sources)
        # No semantic overlap → index fallback returns gold-1
        assert records[0]["gold_identity"] == "gold-1"

    def test_unmatched_when_no_slots_and_no_gold(self) -> None:
        mod = _load_script_module()
        records = mod._match_slots_semantic([], [], [])
        assert records == []


class TestNormalization:
    def test_norm_period_strips_separators(self) -> None:
        mod = _load_script_module()
        # "FY 2024" has a space, so \bFY(\d{4})\b does not match after uppercase
        # -> only non-alphanumeric stripped -> "FY2024"
        assert mod._norm_period("FY 2024") == "FY2024"
        # "fy2024" -> uppercase "FY2024" -> regex strips FY prefix -> "2024"
        assert mod._norm_period("fy2024") == "2024"
        assert mod._norm_period("FY2024") == "2024"
        assert mod._norm_period("2024") == "2024"
        assert mod._norm_period(None) == ""

    def test_norm_metric_lowercases_and_strips(self) -> None:
        mod = _load_script_module()
        assert mod._norm_metric("Total Revenue!") == "total revenue"
        assert mod._norm_metric(None) == ""

    def test_canonical_slot_role_maps_known_roles(self) -> None:
        mod = _load_script_module()
        assert mod._canonical_slot_role({"role": "current_period"}) == "current"
        assert mod._canonical_slot_role({"role": "base_period"}) == "previous"
        assert mod._canonical_slot_role({"role": "previous_period"}) == "previous"
        assert mod._canonical_slot_role({"slot_id": "metric_left"}) == "left"
        assert mod._canonical_slot_role({"role": "value"}) == "value"


# ---------------------------------------------------------------------------
# Acceptance artifact contract (run against generated artifacts if present)
# ---------------------------------------------------------------------------


class TestR1ArtifactsContract:
    """Validate the structure of generated R1 artifacts.

    These tests run against the sealed artifact directory. They are skipped if
    the artifacts have not been generated yet (no network/server dependency).
    """

    R1_DIR = ROOT / "artifacts" / "evaluation" / "pdf-retrieval-v4-gate-08-r1"

    def _skip_if_missing(self) -> None:
        if not self.R1_DIR.is_dir():
            pytest.skip("R1 artifacts not generated")

    def test_acceptance_json_fields(self) -> None:
        self._skip_if_missing()
        data = json.loads((self.R1_DIR / "acceptance.json").read_text(encoding="utf-8"))
        required = {
            "gate",
            "prediction_seal_verified",
            "prediction_rerun",
            "gold_source_count",
            "strict_candidate_universe_coverage",
            "raw_full_pool_recall",
            "structured_strict_source_recall",
            "combined_raw_protected_pool_recall",
            "failure_attribution_counts",
            "decision",
            "next_gate",
            "production_switch_allowed",
            "universe_retrieved_pool_separated",
            "failure_attribution_reclassified",
            "slot_semantic_mapping_used",
        }
        assert required.issubset(data.keys())
        assert data["prediction_rerun"] is False
        assert data["production_switch_allowed"] is False
        assert data["universe_retrieved_pool_separated"] is True

    def test_failure_attribution_only_known_stages(self) -> None:
        self._skip_if_missing()
        data = json.loads(
            (self.R1_DIR / "corrected-failure-attribution.json").read_text(encoding="utf-8")
        )
        known = {stage.value for stage in FirstFailureStage}
        for stage_name in data["counts"]:
            assert stage_name in known, f"unknown failure stage: {stage_name}"

    def test_gold_structural_map_method_counts(self) -> None:
        self._skip_if_missing()
        data = json.loads(
            (self.R1_DIR / "gold-structural-map.json").read_text(encoding="utf-8")
        )
        assert data["total_gold"] == len(data["matches"])
        assert data["in_structured_universe"] + data["unresolved"] == data["total_gold"]

    def test_universe_coverage_consistency(self) -> None:
        self._skip_if_missing()
        cov = json.loads(
            (self.R1_DIR / "structured-universe-coverage.json").read_text(encoding="utf-8")
        )
        assert cov["gold_in_structured_universe"] + cov["gold_not_in_structured_universe"] == cov[
            "total_gold_sources"
        ]
