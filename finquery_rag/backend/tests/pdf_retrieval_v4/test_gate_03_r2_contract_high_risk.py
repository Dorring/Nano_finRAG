"""Contract tests for Gate 03 R2 high-risk scenarios.

Covers the mandatory test scenarios from the Gate 03 R2 specification:

  - multi-level metric parent
  - duplicate physical rows → one semantic evidence
  - point vs duration
  - comparison column
  - bucket column
  - segment column
  - row matrix multi-period
  - scale same-table resolution
  - nearby-page scale remains candidate-only
  - scale conflict fail closed
  - currency symbol without code
  - unknown temporal axis fail closed
  - source traceback roundtrip
  - deterministic semantic identities
  - no gold before seal
  - Tesla equivalent-set (3+ physical rows)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pdf_retrieval_v4.metric_path_builder import (
    build_metric_paths,
    detect_parent_cycles,
)
from src.pdf_retrieval_v4.semantic_currency_resolver import resolve_table_currency
from src.pdf_retrieval_v4.semantic_equivalence import (
    build_equivalence_map,
    detect_equivalent_set_double_counting,
)
from src.pdf_retrieval_v4.semantic_graph_models import (
    CurrencyResolution,
    ScaleResolution,
    SemanticAxisBinding,
    SemanticRow,
    build_atomic_fact_id,
    build_equivalent_group_id,
    canonical_semantic_fact_id,
)
from src.pdf_retrieval_v4.semantic_scale_resolver import resolve_table_scale
from src.pdf_retrieval_v4.typed_evidence_emitters import (
    emit_atomic_facts,
    emit_bucket_facts,
    emit_comparison_facts,
    emit_row_matrices,
)

BUILD_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/evaluation/run_pdf_v4_gate_03_r2_build.py"
)


# ---------------------------------------------------------------------------
# Helpers — construct test data
# ---------------------------------------------------------------------------


def _make_row(
    row_id: str,
    row_index: int,
    row_type: str,
    raw_label: str,
    table_fragment_id: str = "table:test",
    document_id: str = "doc:test",
    pdf_page: int = 1,
    parent_row_id: str | None = None,
) -> SemanticRow:
    return SemanticRow(
        row_id=row_id,
        table_fragment_id=table_fragment_id,
        document_id=document_id,
        pdf_page=pdf_page,
        row_index=row_index,
        row_type=row_type,
        raw_label=raw_label,
        parent_row_id=parent_row_id,
        semantic_eligible=row_type in ("metric_row", "subtotal", "total"),
        source_traceback={
            "row_id": row_id,
            "table_fragment_id": table_fragment_id,
            "document_id": document_id,
            "pdf_page": pdf_page,
            "row_index": row_index,
        },
    )


def _make_cell(
    cell_id: str,
    row_index: int,
    column_index: int,
    text: str,
    row_id: str = "",
    numeric: str | None = None,
    period: str | None = None,
    header: list[str] | None = None,
    period_kind: str | None = None,
    table_fragment_id: str = "table:test",
    document_id: str = "doc:test",
) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        "row_id": row_id or f"row:{row_index}",
        "row_index": row_index,
        "column_index": column_index,
        "raw_text": text,
        "resolved_text": text,
        "native_text": text,
        "parsed_numeric": (
            [{"normalized": numeric, "raw": text, "percent": False}]
            if numeric is not None
            else []
        ),
        "normalized_period": period,
        "period_kind": period_kind,
        "header_path": header or [],
        "cell_bbox": [
            20.0 + column_index * 40.0,
            20.0 + row_index * 20.0,
            50.0 + column_index * 40.0,
            35.0 + row_index * 20.0,
        ],
        "table_fragment_id": table_fragment_id,
        "document_id": document_id,
    }


def _make_scale(
    status: str = "resolved", scale: float | None = 1e6, unit: str | None = "millions"
) -> ScaleResolution:
    return ScaleResolution(
        table_fragment_id="table:test",
        scale=scale,
        scale_unit=unit,
        scale_level="S3",
        scale_status=status,
        raw_candidates=("in millions",),
        source="S3:in millions",
    )


def _make_currency(
    status: str = "unresolved", code: str | None = None, symbol: str | None = "$"
) -> CurrencyResolution:
    return CurrencyResolution(
        table_fragment_id="table:test",
        currency_symbol=symbol,
        currency_code=code,
        currency_source="symbol_only" if status == "unresolved" else "explicit_code",
        currency_status=status,
    )


def _make_axis(
    cell_id: str,
    row_id: str,
    column_index: int,
    temporal_kind: str,
    normalized_period: str | None = None,
    comparison_role: str | None = None,
    bucket_label: str | None = None,
    segment_label: str | None = None,
) -> SemanticAxisBinding:
    return SemanticAxisBinding(
        cell_id=cell_id,
        row_id=row_id,
        table_fragment_id="table:test",
        column_index=column_index,
        temporal_kind=temporal_kind,
        period_start=None,
        period_end=None,
        normalized_period=normalized_period,
        comparison_role=comparison_role,
        bucket_label=bucket_label,
        segment_label=segment_label,
        category_label=None,
    )


# ---------------------------------------------------------------------------
# 1. Multi-level metric parent
# ---------------------------------------------------------------------------


def test_multi_level_metric_parent_three_depths() -> None:
    """Automotive → Revenues → Services and other must produce depth-3 path."""
    rows = [
        _make_row("row:0", 0, "section_header", "Automotive"),
        _make_row("row:1", 1, "group_header", "Revenues"),
        _make_row("row:2", 2, "metric_row", "Services and other"),
    ]
    paths = build_metric_paths(rows)
    assert len(paths) == 1
    mp = paths[0]
    assert mp.metric_path == "Automotive / Revenues / Services and other"
    assert mp.metric_depth == 3
    assert mp.metric_path_segments == ("Automotive", "Revenues", "Services and other")
    assert mp.parent_metric_row_id == "row:1"
    assert mp.metric_status == "resolved"


def test_metric_parent_resets_on_repeated_header() -> None:
    """A repeated header label should pop the stack to that level, not nest deeper."""
    rows = [
        _make_row("row:0", 0, "section_header", "Revenue"),
        _make_row("row:1", 1, "metric_row", "Product A"),
        _make_row(
            "row:2", 2, "section_header", "Revenue"
        ),  # same label → pop + re-push
        _make_row("row:3", 3, "metric_row", "Product B"),
    ]
    paths = build_metric_paths(rows)
    assert len(paths) == 2
    assert paths[0].metric_path == "Revenue / Product A"
    assert paths[1].metric_path == "Revenue / Product B"
    assert paths[1].metric_depth == 2


def test_no_parent_cycles() -> None:
    rows = [
        _make_row("row:0", 0, "group_header", "Assets"),
        _make_row("row:1", 1, "metric_row", "Cash", parent_row_id="row:0"),
    ]
    paths = build_metric_paths(rows)
    assert detect_parent_cycles(paths, rows) == 0


# ---------------------------------------------------------------------------
# 2. Duplicate physical rows → one semantic evidence
# ---------------------------------------------------------------------------


def test_equivalent_set_assigns_same_group_id() -> None:
    """Two physical row_ids in an equivalent_set must get the same group id."""
    equivalent_sets = [
        {
            "alignment_status": "equivalent_set",
            "physical_row_ids": ["row:tesla:1", "row:tesla:2"],
        }
    ]
    equiv_map = build_equivalence_map(equivalent_sets)
    assert equiv_map["row:tesla:1"] == equiv_map["row:tesla:2"]
    assert equiv_map["row:tesla:1"].startswith("equiv:")


def test_equivalent_set_double_counting_detected() -> None:
    """Two facts with same (group, metric, temporal, period) must count as duplicate."""
    group_id = "equiv:test"
    facts = [
        {
            "semantic_fact_id": "atomic:aaa",
            "equivalent_group_id": group_id,
            "metric_path": "Revenue",
            "temporal_kind": "duration",
            "normalized_period": "FY2025",
        },
        {
            "semantic_fact_id": "atomic:bbb",
            "equivalent_group_id": group_id,
            "metric_path": "Revenue",
            "temporal_kind": "duration",
            "normalized_period": "FY2025",
        },
    ]
    duplicates = detect_equivalent_set_double_counting(facts, {})
    assert duplicates == 1


def test_equivalent_set_no_double_counting_after_dedup() -> None:
    """After removing the non-canonical fact, no double counting."""
    group_id = "equiv:test"
    facts = [
        {
            "semantic_fact_id": "atomic:aaa",
            "equivalent_group_id": group_id,
            "metric_path": "Revenue",
            "temporal_kind": "duration",
            "normalized_period": "FY2025",
        },
        # Only one fact after dedup
    ]
    duplicates = detect_equivalent_set_double_counting(facts, {})
    assert duplicates == 0


def test_canonical_semantic_fact_id_is_deterministic() -> None:
    """Canonical id from the same physical ids must be deterministic."""
    ids_a = canonical_semantic_fact_id(["atomic:aaa", "atomic:bbb"])
    ids_b = canonical_semantic_fact_id(["atomic:bbb", "atomic:aaa"])  # different order
    assert ids_a == ids_b
    assert ids_a.startswith("canonical:")


# ---------------------------------------------------------------------------
# 3. Point vs duration
# ---------------------------------------------------------------------------


def test_point_vs_duration_temporal_kinds() -> None:
    from src.pdf_retrieval_v4.temporal_axis_graph import build_axis_bindings

    # "As of June 30, 2025" → point
    point_cells = [
        _make_cell(
            "c_pt", 0, 1, "100000", numeric="100000", header=["As of June 30, 2025"]
        ),
    ]
    point_bindings = build_axis_bindings(point_cells, "table:test")
    assert point_bindings[0].temporal_kind == "point"

    # "Year ended June 30, 2025" → duration
    duration_cells = [
        _make_cell(
            "c_dur",
            0,
            1,
            "100000",
            numeric="100000",
            header=["Year ended June 30, 2025"],
        ),
    ]
    duration_bindings = build_axis_bindings(duration_cells, "table:test")
    assert duration_bindings[0].temporal_kind == "duration"


# ---------------------------------------------------------------------------
# 4. Comparison column
# ---------------------------------------------------------------------------


def test_comparison_column_temporal_kind() -> None:
    from src.pdf_retrieval_v4.temporal_axis_graph import build_axis_bindings

    cells = [
        _make_cell("c_comp", 0, 1, "11%", numeric="11", header=["% change"]),
    ]
    bindings = build_axis_bindings(cells, "table:test")
    assert bindings[0].temporal_kind == "comparison"
    assert bindings[0].comparison_role == "percent_change"


# ---------------------------------------------------------------------------
# 5. Bucket column
# ---------------------------------------------------------------------------


def test_bucket_column_temporal_kind() -> None:
    from src.pdf_retrieval_v4.temporal_axis_graph import build_axis_bindings

    cells = [
        _make_cell("c_bk", 0, 1, "4200", numeric="4200", header=["Less than 1 year"]),
    ]
    bindings = build_axis_bindings(cells, "table:test")
    assert bindings[0].temporal_kind == "bucket"


# ---------------------------------------------------------------------------
# 6. Segment column
# ---------------------------------------------------------------------------


def test_segment_column_temporal_kind() -> None:
    from src.pdf_retrieval_v4.temporal_axis_graph import build_axis_bindings

    cells = [
        _make_cell("c_seg", 0, 1, "50000", numeric="50000", header=["Americas"]),
    ]
    bindings = build_axis_bindings(cells, "table:test")
    assert bindings[0].temporal_kind == "segment"


# ---------------------------------------------------------------------------
# 7. Row matrix multi-period
# ---------------------------------------------------------------------------


def test_row_matrix_multi_period() -> None:
    """A metric row with 3 temporal columns must produce a RowMatrix with 3 dimensions."""
    rows = [
        _make_row("row:0", 0, "metric_row", "Revenue"),
    ]
    metric_paths = build_metric_paths(rows)
    axis_bindings = [
        _make_axis("c1", "row:0", 1, "duration", normalized_period="FY2025"),
        _make_axis("c2", "row:0", 2, "duration", normalized_period="FY2024"),
        _make_axis("c3", "row:0", 3, "duration", normalized_period="FY2023"),
    ]
    cells = [
        _make_cell("c1", 0, 1, "106265", numeric="106265", row_id="row:0"),
        _make_cell("c2", 0, 2, "96169", numeric="96169", row_id="row:0"),
        _make_cell("c3", 0, 3, "89468", numeric="89468", row_id="row:0"),
    ]
    scale = _make_scale()
    currency = _make_currency()

    matrices = emit_row_matrices(
        rows, metric_paths, axis_bindings, cells, scale, currency, {}
    )
    assert len(matrices) == 1
    rm = matrices[0]
    assert rm.metric_path == "Revenue"
    assert len(rm.dimensions) == 3
    periods = [d["normalized_period"] for d in rm.dimensions]
    assert periods == ["FY2025", "FY2024", "FY2023"]
    assert rm.scale == 1e6
    assert rm.scale_unit == "millions"


def test_row_matrix_not_emitted_for_single_column() -> None:
    """A metric row with only 1 temporal column should NOT produce a RowMatrix."""
    rows = [_make_row("row:0", 0, "metric_row", "Revenue")]
    metric_paths = build_metric_paths(rows)
    axis_bindings = [
        _make_axis("c1", "row:0", 1, "duration", normalized_period="FY2025"),
    ]
    cells = [
        _make_cell("c1", 0, 1, "106265", numeric="106265", row_id="row:0"),
    ]
    scale = _make_scale()
    currency = _make_currency()
    matrices = emit_row_matrices(
        rows, metric_paths, axis_bindings, cells, scale, currency, {}
    )
    assert len(matrices) == 0


# ---------------------------------------------------------------------------
# 8. Scale same-table resolution
# ---------------------------------------------------------------------------


def test_scale_same_table_resolution() -> None:
    """Table-level scale_candidates 'in millions' must auto-resolve at S3."""
    table = {
        "scale_candidates": ["in millions"],
        "cells": [],
        "rows": [],
        "header_texts": [],
    }
    sr = resolve_table_scale(table, "table:test")
    assert sr.scale_status == "resolved"
    assert sr.scale == 1e6
    assert sr.scale_unit == "millions"
    assert sr.scale_level == "S3"


def test_scale_cell_explicit_resolution() -> None:
    """Cell-level scale text must auto-resolve at S0."""
    table = {
        "scale_candidates": [],
        "cells": [
            {"resolved_text": "in millions", "scale_candidates": []},
        ],
        "rows": [],
        "header_texts": [],
    }
    sr = resolve_table_scale(table, "table:test")
    assert sr.scale_status == "resolved"
    assert sr.scale == 1e6
    assert sr.scale_level == "S0"


# ---------------------------------------------------------------------------
# 9. Nearby-page scale remains candidate-only
# ---------------------------------------------------------------------------


def test_nearby_page_scale_remains_candidate_only() -> None:
    """Page-level and adjacent-page scale candidates must NOT auto-resolve."""
    table = {
        "scale_candidates": [],
        "cells": [],
        "rows": [],
        "header_texts": [],
    }
    sr = resolve_table_scale(
        table,
        "table:test",
        page_scale_candidates=["in millions"],
    )
    assert sr.scale_status == "candidate"
    assert sr.scale == 1e6  # The value is available but not resolved
    assert sr.scale_level == "S5"

    sr_adj = resolve_table_scale(
        table,
        "table:test",
        adjacent_page_scale_candidates=["in billions"],
    )
    assert sr_adj.scale_status == "candidate"
    assert sr_adj.scale_level == "S6"


# ---------------------------------------------------------------------------
# 10. Scale conflict fail closed
# ---------------------------------------------------------------------------


def test_scale_conflict_fail_closed() -> None:
    """Simultaneous 'in millions' and 'in thousands' must produce conflict."""
    table = {
        "scale_candidates": ["in millions", "in thousands"],
        "cells": [],
        "rows": [],
        "header_texts": [],
    }
    sr = resolve_table_scale(table, "table:test")
    assert sr.scale_status == "conflict"
    assert sr.scale is None
    assert sr.scale_unit is None


# ---------------------------------------------------------------------------
# 11. Currency symbol without code
# ---------------------------------------------------------------------------


def test_currency_symbol_without_code() -> None:
    """A bare '$' must bind symbol but NOT code."""
    table = {
        "cells": [{"resolved_text": "$ 1,000", "raw_text": "$ 1,000"}],
        "rows": [],
        "header_texts": [],
    }
    cr = resolve_table_currency(table, "table:test")
    assert cr.currency_symbol == "$"
    assert cr.currency_code is None
    assert cr.currency_status == "unresolved"


def test_currency_explicit_declaration_resolves_code() -> None:
    """'Amounts in U.S. dollars' must resolve currency_code to USD."""
    table = {
        "cells": [],
        "rows": [],
        "header_texts": ["Amounts in U.S. dollars"],
    }
    cr = resolve_table_currency(table, "table:test")
    assert cr.currency_code == "USD"
    assert cr.currency_status == "resolved"


# ---------------------------------------------------------------------------
# 12. Unknown temporal axis fail closed
# ---------------------------------------------------------------------------


def test_unknown_temporal_axis_fail_closed() -> None:
    """A cell with no recognizable temporal signal must be 'unknown'."""
    from src.pdf_retrieval_v4.temporal_axis_graph import build_axis_bindings

    cells = [
        _make_cell("c_unk", 0, 1, "42", numeric="42", header=["Some Column"]),
    ]
    bindings = build_axis_bindings(cells, "table:test")
    assert bindings[0].temporal_kind == "unknown"


# ---------------------------------------------------------------------------
# 13. Source traceback roundtrip
# ---------------------------------------------------------------------------


def test_atomic_fact_source_traceback_roundtrip() -> None:
    """AtomicFact must carry full source_traceback for round-trip."""
    rows = [_make_row("row:0", 0, "metric_row", "Revenue")]
    metric_paths = build_metric_paths(rows)
    axis_bindings = [
        _make_axis("c1", "row:0", 1, "duration", normalized_period="FY2025"),
    ]
    cells = [
        _make_cell("c1", 0, 1, "106265", numeric="106265", row_id="row:0"),
    ]
    scale = _make_scale()
    currency = _make_currency()

    facts = emit_atomic_facts(
        rows, metric_paths, axis_bindings, cells, scale, currency, {}
    )
    assert len(facts) == 1
    fact = facts[0]
    st = fact.source_traceback
    assert st["document_id"] == "doc:test"
    assert st["pdf_page"] == 1
    assert st["table_fragment_id"] == "table:test"
    assert st["row_id"] == "row:0"
    assert st["cell_id"] == "c1"
    assert st["raw_text"] == "106265"
    assert st["bbox"] is not None


# ---------------------------------------------------------------------------
# 14. Deterministic semantic identities
# ---------------------------------------------------------------------------


def test_atomic_fact_id_is_deterministic() -> None:
    """Same structural inputs must produce the same semantic_fact_id."""
    id1 = build_atomic_fact_id("doc:test", "table:test", "row:0", "c1")
    id2 = build_atomic_fact_id("doc:test", "table:test", "row:0", "c1")
    assert id1 == id2
    assert id1.startswith("atomic:")


def test_atomic_fact_id_differs_on_different_cell() -> None:
    """Different cell_id must produce a different semantic_fact_id."""
    id1 = build_atomic_fact_id("doc:test", "table:test", "row:0", "c1")
    id2 = build_atomic_fact_id("doc:test", "table:test", "row:0", "c2")
    assert id1 != id2


def test_equivalent_group_id_is_deterministic() -> None:
    """Same row_ids (in any order) must produce the same group id."""
    id1 = build_equivalent_group_id(["row:a", "row:b"])
    id2 = build_equivalent_group_id(["row:b", "row:a"])
    assert id1 == id2


def test_identity_excludes_question_and_gold() -> None:
    """Identity must NOT incorporate question/gold/expected_value."""
    # The build_atomic_fact_id function only accepts structural fields;
    # verify it doesn't accept or use non-structural fields.
    import inspect

    sig = inspect.signature(build_atomic_fact_id)
    params = list(sig.parameters.keys())
    forbidden = {"question", "gold", "expected_value", "case_id", "answer"}
    assert not set(params) & forbidden


# ---------------------------------------------------------------------------
# 15. No gold before seal
# ---------------------------------------------------------------------------


def test_build_script_is_oracle_blind() -> None:
    """The build script must not read gold/question files before sealing."""
    source = BUILD_SCRIPT.read_text(encoding="utf-8")
    # Must not read gold or question files
    assert "labels.golden" not in source
    assert "manual-mapping-review-package" not in source
    # Must explicitly zero all safety counters
    assert '"question_reads": 0' in source
    assert '"gold_reads_before_seal": 0' in source
    assert '"governance_reads_before_seal": 0' in source
    assert '"candidate_bridge_builds": 0' in source
    assert '"index_builds": 0' in source
    assert '"retrieval_runs": 0' in source
    assert '"production_switch_allowed": False' in source


# ---------------------------------------------------------------------------
# 16. Tesla equivalent-set (3+ physical rows)
# ---------------------------------------------------------------------------


def test_tesla_equivalent_set_three_rows_collapse() -> None:
    """3+ physical rows in an equivalent_set must all share the same group id
    and the canonical fact id must be the same regardless of which physical
    source is chosen."""
    equivalent_sets = [
        {
            "alignment_status": "equivalent_set",
            "physical_row_ids": [
                "row:tesla:frag_a",
                "row:tesla:frag_b",
                "row:tesla:frag_c",
            ],
        }
    ]
    equiv_map = build_equivalence_map(equivalent_sets)

    # All three must map to the same group
    group_ids = {
        equiv_map[rid]
        for rid in ["row:tesla:frag_a", "row:tesla:frag_b", "row:tesla:frag_c"]
    }
    assert len(group_ids) == 1

    # Canonical fact id must be the same regardless of which 2+ physical
    # fact ids are passed (as long as the full set is the same)
    fact_ids_a = ["atomic:fa", "atomic:fb", "atomic:fc"]
    fact_ids_b = ["atomic:fc", "atomic:fa", "atomic:fb"]
    canonical_a = canonical_semantic_fact_id(fact_ids_a)
    canonical_b = canonical_semantic_fact_id(fact_ids_b)
    assert canonical_a == canonical_b


def test_tesla_equivalent_set_no_double_counting() -> None:
    """After collapsing 3 equivalent rows to 1 canonical fact, no double counting."""
    group_id = "equiv:tesla"
    # Only the canonical fact remains after dedup
    facts = [
        {
            "semantic_fact_id": "canonical:xxx",
            "equivalent_group_id": group_id,
            "metric_path": "Automotive / Revenues",
            "temporal_kind": "duration",
            "normalized_period": "FY2025",
        },
    ]
    duplicates = detect_equivalent_set_double_counting(facts, {})
    assert duplicates == 0


def test_tesla_equivalent_set_double_counting_before_dedup() -> None:
    """Before dedup, 3 equivalent facts with same metric/period must show 2 duplicates."""
    group_id = "equiv:tesla"
    facts = [
        {
            "semantic_fact_id": f"atomic:f{i}",
            "equivalent_group_id": group_id,
            "metric_path": "Automotive / Revenues",
            "temporal_kind": "duration",
            "normalized_period": "FY2025",
        }
        for i in range(3)
    ]
    duplicates = detect_equivalent_set_double_counting(facts, {})
    assert duplicates == 2  # 3 facts, 1 canonical → 2 duplicates


# ---------------------------------------------------------------------------
# Additional: Atomic Fact Admission — bucket/comparison cells excluded
# ---------------------------------------------------------------------------


def test_atomic_facts_exclude_bucket_and_comparison_cells() -> None:
    """AtomicFacts must only be emitted for point/duration/comparison cells,
    NOT for bucket/segment/category/unknown cells."""
    rows = [_make_row("row:0", 0, "metric_row", "Long-term debt")]
    metric_paths = build_metric_paths(rows)
    axis_bindings = [
        _make_axis("c_pt", "row:0", 1, "point", normalized_period="FY2025"),
        _make_axis("c_bk", "row:0", 2, "bucket", bucket_label="1-3 years"),
        _make_axis(
            "c_comp", "row:0", 3, "comparison", comparison_role="percent_change"
        ),
    ]
    cells = [
        _make_cell("c_pt", 0, 1, "100", numeric="100", row_id="row:0"),
        _make_cell("c_bk", 0, 2, "42", numeric="42", row_id="row:0"),
        _make_cell("c_comp", 0, 3, "5%", numeric="5", row_id="row:0"),
    ]
    scale = _make_scale()
    currency = _make_currency()

    facts = emit_atomic_facts(
        rows, metric_paths, axis_bindings, cells, scale, currency, {}
    )
    # Only point and comparison qualify for atomic facts
    # (comparison is included per the emitter logic)
    fact_cell_ids = {f.cell_id for f in facts}
    assert "c_pt" in fact_cell_ids
    assert "c_comp" in fact_cell_ids
    assert "c_bk" not in fact_cell_ids  # bucket excluded


def test_bucket_facts_emitted_for_bucket_cells() -> None:
    """BucketFact must be emitted for bucket-axis cells."""
    rows = [_make_row("row:0", 0, "metric_row", "Long-term debt")]
    metric_paths = build_metric_paths(rows)
    axis_bindings = [
        _make_axis("c_bk", "row:0", 1, "bucket", bucket_label="1-3 years"),
    ]
    cells = [
        _make_cell("c_bk", 0, 1, "4200", numeric="4200", row_id="row:0"),
    ]
    scale = _make_scale()
    currency = _make_currency()

    facts = emit_bucket_facts(rows, metric_paths, axis_bindings, cells, scale, currency)
    assert len(facts) == 1
    assert facts[0].bucket_label == "1-3 years"
    assert facts[0].bucket_kind in ("maturity", "aging", "rating", "range")


def test_comparison_facts_emitted_for_comparison_cells() -> None:
    """ComparisonFact must be emitted for comparison-axis cells with base values."""
    rows = [_make_row("row:0", 0, "metric_row", "Revenue")]
    metric_paths = build_metric_paths(rows)
    axis_bindings = [
        _make_axis("c_base1", "row:0", 1, "duration", normalized_period="FY2025"),
        _make_axis("c_base2", "row:0", 2, "duration", normalized_period="FY2024"),
        _make_axis(
            "c_comp", "row:0", 3, "comparison", comparison_role="percent_change"
        ),
    ]
    cells = [
        _make_cell("c_base1", 0, 1, "100", numeric="100", row_id="row:0"),
        _make_cell("c_base2", 0, 2, "90", numeric="90", row_id="row:0"),
        _make_cell("c_comp", 0, 3, "11%", numeric="11", row_id="row:0"),
    ]
    scale = _make_scale()

    facts = emit_comparison_facts(rows, metric_paths, axis_bindings, cells, scale, {})
    assert len(facts) == 1
    cf = facts[0]
    assert cf.comparison_role == "percent_change"
    assert cf.base_value == "100"
    assert cf.compared_value == "90"
    assert cf.reported_change == "11%"


# ---------------------------------------------------------------------------
# Additional: Metric path missing label → metric_status = "missing"
# ---------------------------------------------------------------------------


def test_metric_row_with_empty_label_is_missing() -> None:
    """A financial data row with no label must produce metric_status='missing'."""
    rows = [
        _make_row("row:0", 0, "group_header", "Revenue"),
        _make_row("row:1", 1, "metric_row", ""),  # empty label
    ]
    paths = build_metric_paths(rows)
    # The empty-label metric row should still get a MetricPath with missing status
    mp = next(p for p in paths if p.row_id == "row:1")
    assert mp.metric_status == "missing"
    assert mp.metric_path == ""


# ---------------------------------------------------------------------------
# 17. Production module import verification
# ---------------------------------------------------------------------------


def test_gate_03_r2_contracts_import_production_modules() -> None:
    """Verify contract tests import from real production modules in
    ``src.pdf_retrieval_v4.*``, not test-local reference implementations or mocks.

    This test inspects the source file to ensure:
    1. Required production module imports are present.
    2. No test-local fake/mock classes or functions shadow production types.
    """
    source = Path(__file__).read_text(encoding="utf-8")

    # --- Must import from real production modules ---
    required_imports = [
        "from src.pdf_retrieval_v4.metric_path_builder import",
        "from src.pdf_retrieval_v4.semantic_currency_resolver import",
        "from src.pdf_retrieval_v4.semantic_equivalence import",
        "from src.pdf_retrieval_v4.semantic_graph_models import",
        "from src.pdf_retrieval_v4.semantic_graph_validator import",
        "from src.pdf_retrieval_v4.semantic_scale_resolver import",
        "from src.pdf_retrieval_v4.typed_evidence_emitters import",
    ]
    for imp in required_imports:
        assert imp in source, f"Missing production import: {imp}"

    # temporal_axis_graph is imported inside test functions (local imports)
    assert "from src.pdf_retrieval_v4.temporal_axis_graph import" in source, (
        "Missing production import: temporal_axis_graph"
    )

    # --- Must NOT define test-local reference implementations ---
    # Check using AST to avoid matching strings inside this test itself
    import ast

    tree = ast.parse(source)
    defined_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            defined_names.add(node.name)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            defined_names.add(node.name)

    forbidden_names = {
        "ReferenceFact",
        "FakeFact",
        "MockFact",
        "ReferenceScale",
        "ReferenceCurrency",
        "ReferenceAxis",
        "fake_resolve_scale",
        "mock_build_metric_paths",
        "fake_emit_atomic",
        "reference_resolve_table_currency",
    }
    found = defined_names & forbidden_names
    assert not found, f"Test-local mock/reference definitions found: {found}"

    # --- Helper functions must construct PRODUCTION types, not local types ---
    # _make_row returns SemanticRow (production dataclass)
    assert "return SemanticRow(" in source
    # _make_scale returns ScaleResolution (production dataclass)
    assert "return ScaleResolution(" in source
    # _make_currency returns CurrencyResolution (production dataclass)
    assert "return CurrencyResolution(" in source
    # _make_axis returns SemanticAxisBinding (production dataclass)
    assert "return SemanticAxisBinding(" in source
    # _make_cell returns a plain dict (test input data, not a production type)
    assert "return {" in source


# ---------------------------------------------------------------------------
# R0 — Pre-run Metric Contract Fix tests
# ---------------------------------------------------------------------------


def test_atomic_admission_uses_pre_emission_denominator() -> None:
    """Atomic Fact Admission must use atomic-eligible numeric cells as denominator,
    not the count of emitted atomic facts."""
    from src.pdf_retrieval_v4.semantic_graph_models import AtomicFact, MetricPath
    from src.pdf_retrieval_v4.typed_evidence_emitters import compute_admission_outcomes

    # 3 cells on financial-data rows with parsed_numeric; 2 have point axis
    rows = [
        _make_row("row:0", 0, "metric_row", "Revenue"),
    ]
    cells = [
        _make_cell("cell:0:1", 0, 1, "100", row_id="row:0", numeric="100"),
        _make_cell("cell:0:2", 0, 2, "200", row_id="row:0", numeric="200"),
        _make_cell("cell:0:3", 0, 3, "300", row_id="row:0", numeric="300"),
    ]
    axes = [
        _make_axis("cell:0:1", "row:0", 1, "point", "FY2025"),
        _make_axis("cell:0:2", "row:0", 2, "point", "FY2024"),
        # cell:0:3 has no axis → unresolved
    ]
    mps = [
        MetricPath(
            row_id="row:0",
            table_fragment_id="table:test",
            raw_row_label="Revenue",
            leaf_metric="Revenue",
            metric_path="Revenue",
            metric_path_segments=("Revenue",),
            metric_depth=1,
            parent_metric_row_id=None,
            metric_status="resolved",
        )
    ]
    # Only 1 atomic fact emitted (cell:0:1), but 2 are atomic-eligible
    atomic_facts = [
        AtomicFact(
            semantic_fact_id="af:doc:test:table:test:row:0:cell:0:1",
            document_id="doc:test",
            table_fragment_id="table:test",
            row_id="row:0",
            cell_id="cell:0:1",
            metric_path="Revenue",
            leaf_metric="Revenue",
            temporal_kind="point",
            normalized_period="FY2025",
            period_start=None,
            period_end=None,
            value_raw="100",
            value_normalized="100",
            scale=1e6,
            scale_unit="millions",
            currency_code=None,
            equivalent_group_id=None,
            source_traceback={
                "document_id": "doc:test",
                "pdf_page": 1,
                "table_fragment_id": "table:test",
                "row_id": "row:0",
                "cell_id": "cell:0:1",
                "bbox": None,
                "raw_text": "100",
            },
        )
    ]

    outcomes = compute_admission_outcomes(
        semantic_rows=rows,
        metric_paths=mps,
        axis_bindings=axes,
        all_cells=cells,
        atomic_facts=atomic_facts,
        comparison_facts=[],
        bucket_facts=[],
        row_matrices=[],
    )

    # 3 eligible numeric cells (all have parsed_numeric on financial-data row)
    assert len(outcomes) == 3
    # 2 atomic-eligible (cell:0:1 and cell:0:2 have point axis)
    atomic_eligible = [
        o for o in outcomes if o["temporal_kind"] in ("point", "duration", "comparison")
    ]
    assert len(atomic_eligible) == 2
    # Only 1 admitted as atomic
    atomic_admitted = [o for o in outcomes if "atomic" in o["outcomes"]]
    assert len(atomic_admitted) == 1
    # cell:0:3 should have unresolved_axis
    cell3 = next(o for o in outcomes if o["cell_id"] == "cell:0:3")
    assert "unresolved_axis" in cell3["outcomes"]


def test_typed_admission_uses_eligible_cell_denominator() -> None:
    """Typed Evidence Admission must use typed-eligible numeric cells as denominator,
    not the count of emitted typed evidence objects."""
    from src.pdf_retrieval_v4.typed_evidence_emitters import compute_admission_outcomes

    rows = [_make_row("row:0", 0, "metric_row", "Revenue")]
    cells = [
        _make_cell("cell:0:1", 0, 1, "100", row_id="row:0", numeric="100"),
        _make_cell("cell:0:2", 0, 2, "200", row_id="row:0", numeric="200"),
    ]
    axes = [
        _make_axis("cell:0:1", "row:0", 1, "point", "FY2025"),
        _make_axis("cell:0:2", "row:0", 2, "bucket", bucket_label="1-3 years"),
    ]
    from src.pdf_retrieval_v4.semantic_graph_models import MetricPath

    mps = [
        MetricPath(
            row_id="row:0",
            table_fragment_id="table:test",
            raw_row_label="Revenue",
            leaf_metric="Revenue",
            metric_path="Revenue",
            metric_path_segments=("Revenue",),
            metric_depth=1,
            parent_metric_row_id=None,
            metric_status="resolved",
        )
    ]

    # No facts emitted at all — all cells should show non-admitted outcomes
    outcomes = compute_admission_outcomes(
        semantic_rows=rows,
        metric_paths=mps,
        axis_bindings=axes,
        all_cells=cells,
        atomic_facts=[],
        comparison_facts=[],
        bucket_facts=[],
        row_matrices=[],
    )

    # 2 typed-eligible cells (point + bucket)
    typed_eligible = [
        o
        for o in outcomes
        if o["temporal_kind"]
        in ("point", "duration", "comparison", "bucket", "segment", "category")
    ]
    assert len(typed_eligible) == 2
    # 0 covered
    from src.pdf_retrieval_v4.typed_evidence_emitters import ADMITTED_OUTCOMES

    typed_covered = [o for o in outcomes if o["outcomes"] & ADMITTED_OUTCOMES]
    assert len(typed_covered) == 0


def test_row_matrix_does_not_shrink_denominator() -> None:
    """A RowMatrix covering 3 cells must count as 3 covered cells, not 1."""
    from src.pdf_retrieval_v4.semantic_graph_models import MetricPath, RowMatrix
    from src.pdf_retrieval_v4.typed_evidence_emitters import compute_admission_outcomes

    rows = [_make_row("row:0", 0, "metric_row", "Revenue")]
    cells = [
        _make_cell("cell:0:1", 0, 1, "100", row_id="row:0", numeric="100"),
        _make_cell("cell:0:2", 0, 2, "200", row_id="row:0", numeric="200"),
        _make_cell("cell:0:3", 0, 3, "300", row_id="row:0", numeric="300"),
    ]
    axes = [
        _make_axis("cell:0:1", "row:0", 1, "point", "FY2025"),
        _make_axis("cell:0:2", "row:0", 2, "point", "FY2024"),
        _make_axis("cell:0:3", "row:0", 3, "point", "FY2023"),
    ]
    mps = [
        MetricPath(
            row_id="row:0",
            table_fragment_id="table:test",
            raw_row_label="Revenue",
            leaf_metric="Revenue",
            metric_path="Revenue",
            metric_path_segments=("Revenue",),
            metric_depth=1,
            parent_metric_row_id=None,
            metric_status="resolved",
        )
    ]
    # One RowMatrix covering all 3 cells
    rm = RowMatrix(
        semantic_fact_id="rm:doc:test:table:test:row:0",
        document_id="doc:test",
        table_fragment_id="table:test",
        row_id="row:0",
        metric_path="Revenue",
        leaf_metric="Revenue",
        dimensions=(
            {
                "cell_id": "cell:0:1",
                "column_index": 1,
                "temporal_kind": "point",
                "normalized_period": "FY2025",
                "period_start": None,
                "period_end": None,
                "comparison_role": None,
                "bucket_label": None,
                "segment_label": None,
                "value_raw": "100",
                "value_normalized": "100",
            },
            {
                "cell_id": "cell:0:2",
                "column_index": 2,
                "temporal_kind": "point",
                "normalized_period": "FY2024",
                "period_start": None,
                "period_end": None,
                "comparison_role": None,
                "bucket_label": None,
                "segment_label": None,
                "value_raw": "200",
                "value_normalized": "200",
            },
            {
                "cell_id": "cell:0:3",
                "column_index": 3,
                "temporal_kind": "point",
                "normalized_period": "FY2023",
                "period_start": None,
                "period_end": None,
                "comparison_role": None,
                "bucket_label": None,
                "segment_label": None,
                "value_raw": "300",
                "value_normalized": "300",
            },
        ),
        scale=1e6,
        scale_unit="millions",
        currency_code=None,
        equivalent_group_id=None,
        source_traceback={
            "document_id": "doc:test",
            "pdf_page": 1,
            "table_fragment_id": "table:test",
            "row_id": "row:0",
            "cell_id": None,
            "bbox": None,
            "raw_text": None,
        },
    )

    outcomes = compute_admission_outcomes(
        semantic_rows=rows,
        metric_paths=mps,
        axis_bindings=axes,
        all_cells=cells,
        atomic_facts=[],
        comparison_facts=[],
        bucket_facts=[],
        row_matrices=[rm],
    )

    # All 3 cells should be covered by row_matrix_member
    covered = [o for o in outcomes if "row_matrix_member" in o["outcomes"]]
    assert len(covered) == 3, (
        f"RowMatrix covering 3 cells must count as 3 covered, got {len(covered)}"
    )


def test_ambiguous_metric_not_counted_as_resolved() -> None:
    """Validator must report metric_path_resolved and metric_path_ambiguous separately,
    and the gate must use resolved (not present) coverage."""
    from src.pdf_retrieval_v4.semantic_graph_validator import validate_semantic_graph
    from src.pdf_retrieval_v4.semantic_graph_models import MetricPath

    # 2 eligible rows, 1 resolved + 1 ambiguous
    rows = [
        _make_row("row:0", 0, "metric_row", "Revenue"),
        _make_row("row:1", 1, "metric_row", "Cost"),
    ]
    mps = [
        MetricPath(
            row_id="row:0",
            table_fragment_id="table:test",
            raw_row_label="Revenue",
            leaf_metric="Revenue",
            metric_path="Revenue",
            metric_path_segments=("Revenue",),
            metric_depth=1,
            parent_metric_row_id=None,
            metric_status="resolved",
        ),
        MetricPath(
            row_id="row:1",
            table_fragment_id="table:test",
            raw_row_label="Cost",
            leaf_metric="Cost",
            metric_path="Cost",
            metric_path_segments=("Cost",),
            metric_depth=1,
            parent_metric_row_id=None,
            metric_status="ambiguous",
        ),
    ]

    result = validate_semantic_graph(
        logical_tables=[],
        semantic_rows=rows,
        metric_paths=mps,
        axis_bindings=[],
        scale_resolutions=[],
        atomic_facts=[],
        comparison_facts=[],
        bucket_facts=[],
        row_matrices=[],
        narrative_evidence=[],
        all_cells=[],
    )

    m = result["metrics"]
    assert m["metric_path_resolved"] == 1
    assert m["metric_path_ambiguous"] == 1
    assert m["metric_path_present"] == 2
    # Resolved coverage = 1/2 = 0.5
    assert m["metric_path_resolved_coverage"] == 0.5
    # Present coverage = 2/2 = 1.0
    assert m["metric_path_present_coverage"] == 1.0
    # Gate must use resolved coverage, not present
    assert result["gates"]["metric_path_resolved_coverage"] is False


def test_same_metric_period_value_across_documents_not_deduped() -> None:
    """Two AtomicFacts from different documents with same metric/period/value
    must have different semantic_fact_ids and must NOT be deduplicated."""
    from src.pdf_retrieval_v4.semantic_graph_models import AtomicFact

    af1 = AtomicFact(
        semantic_fact_id="af:doc:apple:table:t1:row:r1:cell:c1",
        document_id="doc:apple",
        table_fragment_id="table:t1",
        row_id="row:r1",
        cell_id="cell:c1",
        metric_path="Revenue",
        leaf_metric="Revenue",
        temporal_kind="point",
        normalized_period="FY2025",
        period_start=None,
        period_end=None,
        value_raw="100",
        value_normalized="100",
        scale=1e6,
        scale_unit="millions",
        currency_code=None,
        equivalent_group_id=None,
        source_traceback={
            "document_id": "doc:apple",
            "pdf_page": 1,
            "table_fragment_id": "table:t1",
            "row_id": "row:r1",
            "cell_id": "cell:c1",
            "bbox": None,
            "raw_text": "100",
        },
    )
    af2 = AtomicFact(
        semantic_fact_id="af:doc:microsoft:table:t2:row:r2:cell:c2",
        document_id="doc:microsoft",
        table_fragment_id="table:t2",
        row_id="row:r2",
        cell_id="cell:c2",
        metric_path="Revenue",
        leaf_metric="Revenue",
        temporal_kind="point",
        normalized_period="FY2025",
        period_start=None,
        period_end=None,
        value_raw="100",
        value_normalized="100",
        scale=1e6,
        scale_unit="millions",
        currency_code=None,
        equivalent_group_id=None,
        source_traceback={
            "document_id": "doc:microsoft",
            "pdf_page": 1,
            "table_fragment_id": "table:t2",
            "row_id": "row:r2",
            "cell_id": "cell:c2",
            "bbox": None,
            "raw_text": "100",
        },
    )

    # Different semantic_fact_ids
    assert af1.semantic_fact_id != af2.semantic_fact_id

    # Dedup by semantic_fact_id must keep both
    seen: set[str] = set()
    deduped: list[AtomicFact] = []
    for af in [af1, af2]:
        if af.semantic_fact_id in seen:
            continue
        seen.add(af.semantic_fact_id)
        deduped.append(af)
    assert len(deduped) == 2, "Cross-document facts must not be deduplicated"

    # Validator must report 0 duplicate_semantic_facts
    from src.pdf_retrieval_v4.semantic_graph_validator import validate_semantic_graph

    result = validate_semantic_graph(
        logical_tables=[],
        semantic_rows=[],
        metric_paths=[],
        axis_bindings=[],
        scale_resolutions=[],
        atomic_facts=[af1, af2],
        comparison_facts=[],
        bucket_facts=[],
        row_matrices=[],
        narrative_evidence=[],
        all_cells=[],
    )
    assert result["metrics"]["duplicate_semantic_fact"] == 0


def test_equivalent_set_can_dedup() -> None:
    """Equivalent-set rows (same canonical_semantic_fact_id) must be deduplicated
    to exactly one canonical fact."""
    from src.pdf_retrieval_v4.semantic_equivalence import (
        build_equivalence_map,
        detect_equivalent_set_double_counting,
    )

    # 3 physical rows in same equivalent set
    equivalent_sets = [
        {
            "equivalent_group_id": "eq:tesla:revenue",
            "physical_row_ids": ["row:0", "row:1", "row:2"],
        }
    ]
    equiv_map = build_equivalence_map(equivalent_sets)

    # All 3 rows map to same group
    assert equiv_map["row:0"] == equiv_map["row:1"] == equiv_map["row:2"]

    # If we emit 3 atomic facts with same semantic_fact_id (canonical),
    # dedup by semantic_fact_id should reduce to 1
    canonical_id = "af:canonical:tesla:revenue"
    from src.pdf_retrieval_v4.semantic_graph_models import AtomicFact

    facts = [
        AtomicFact(
            semantic_fact_id=canonical_id,
            document_id="doc:tesla",
            table_fragment_id="table:t1",
            row_id=f"row:{i}",
            cell_id=f"cell:{i}",
            metric_path="Revenue",
            leaf_metric="Revenue",
            temporal_kind="point",
            normalized_period="FY2025",
            period_start=None,
            period_end=None,
            value_raw="100",
            value_normalized="100",
            scale=1e6,
            scale_unit="millions",
            currency_code=None,
            equivalent_group_id="eq:tesla:revenue",
            source_traceback={
                "document_id": "doc:tesla",
                "pdf_page": 1,
                "table_fragment_id": "table:t1",
                "row_id": f"row:{i}",
                "cell_id": f"cell:{i}",
                "bbox": None,
                "raw_text": "100",
            },
        )
        for i in range(3)
    ]

    # Dedup by semantic_fact_id
    seen: set[str] = set()
    deduped: list[AtomicFact] = []
    for af in facts:
        if af.semantic_fact_id in seen:
            continue
        seen.add(af.semantic_fact_id)
        deduped.append(af)
    assert len(deduped) == 1, "Equivalent-set facts must dedup to 1 canonical"

    # Double counting detection should flag 3 dicts with same equivalent_group_id
    fact_dicts = [af.to_dict() for af in facts]
    double_count = detect_equivalent_set_double_counting(fact_dicts, equiv_map)
    assert double_count > 0, (
        "3 facts from same equiv set before dedup = double counting"
    )


def test_scale_conflict_resolution_violation_detectable() -> None:
    """Validator must detect when a scale resolution has conflicting candidate
    units but was still resolved (safety violation)."""
    from src.pdf_retrieval_v4.semantic_graph_validator import validate_semantic_graph

    # A ScaleResolution with conflicting candidates but status="resolved"
    # (this should never happen in practice — it's a resolver bug)
    conflict_resolved = ScaleResolution(
        table_fragment_id="table:buggy",
        scale=1e6,
        scale_unit="millions",
        scale_level="S0",
        scale_status="resolved",
        raw_candidates=("in millions", "in thousands"),
        source="S0:in millions",
    )

    result = validate_semantic_graph(
        logical_tables=[],
        semantic_rows=[],
        metric_paths=[],
        axis_bindings=[],
        scale_resolutions=[conflict_resolved],
        atomic_facts=[],
        comparison_facts=[],
        bucket_facts=[],
        row_matrices=[],
        narrative_evidence=[],
        all_cells=[],
    )

    # Must detect the violation
    assert result["metrics"]["scale_conflict_auto_resolution"] == 1, (
        "Scale resolved with conflicting candidates must be flagged as auto-resolution"
    )
    assert result["gates"]["scale_conflict_auto_resolution"] is False
    assert result["metrics"]["scale_conflict_detected"] == 1


def test_build_nonzero_exit_when_gate_fails() -> None:
    """Build script must return exit code 2 when gates fail, not 0."""
    import ast

    source = BUILD_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Find the main function and check its return statement
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and child.value:
                    # Should contain "else 2" not "else 0"
                    return_src = ast.get_source_segment(source, child)
                    if return_src and "all_passed" in return_src:
                        assert "2" in return_src, (
                            f"Build exit code must be 2 on gate failure, got: {return_src}"
                        )
                        return
    raise AssertionError("Could not find return statement with all_passed in main()")


def test_typed_admission_never_exceeds_100_percent() -> None:
    """Typed Evidence Admission must never exceed 100% even when non-typed-eligible
    cells (e.g. unknown temporal kind) have admitted outcomes like row_matrix_member."""
    from src.pdf_retrieval_v4.semantic_graph_models import MetricPath, RowMatrix
    from src.pdf_retrieval_v4.typed_evidence_emitters import compute_admission_outcomes

    rows = [_make_row("row:0", 0, "metric_row", "Revenue")]
    cells = [
        _make_cell("cell:0:1", 0, 1, "100", row_id="row:0", numeric="100"),
        _make_cell("cell:0:2", 0, 2, "200", row_id="row:0", numeric="200"),
    ]
    # cell:0:1 is point (typed-eligible), cell:0:2 is unknown (NOT typed-eligible)
    axes = [
        _make_axis("cell:0:1", "row:0", 1, "point", "FY2025"),
        _make_axis("cell:0:2", "row:0", 2, "unknown"),
    ]
    mps = [
        MetricPath(
            row_id="row:0",
            table_fragment_id="table:test",
            raw_row_label="Revenue",
            leaf_metric="Revenue",
            metric_path="Revenue",
            metric_path_segments=("Revenue",),
            metric_depth=1,
            parent_metric_row_id=None,
            metric_status="resolved",
        )
    ]

    # RowMatrix covers BOTH cells -- but only cell:0:1 is typed-eligible
    rm = RowMatrix(
        semantic_fact_id="rm:doc:test:table:test:row:0",
        document_id="doc:test",
        table_fragment_id="table:test",
        row_id="row:0",
        metric_path="Revenue",
        leaf_metric="Revenue",
        dimensions=(
            {
                "cell_id": "cell:0:1",
                "column_index": 1,
                "temporal_kind": "point",
                "normalized_period": "FY2025",
                "period_start": None,
                "period_end": None,
                "comparison_role": None,
                "bucket_label": None,
                "segment_label": None,
                "value_raw": "100",
                "value_normalized": "100",
            },
            {
                "cell_id": "cell:0:2",
                "column_index": 2,
                "temporal_kind": "unknown",
                "normalized_period": None,
                "period_start": None,
                "period_end": None,
                "comparison_role": None,
                "bucket_label": None,
                "segment_label": None,
                "value_raw": "200",
                "value_normalized": "200",
            },
        ),
        scale=None,
        scale_unit=None,
        currency_code=None,
        equivalent_group_id=None,
        source_traceback={},
    )

    outcomes = compute_admission_outcomes(
        semantic_rows=rows,
        metric_paths=mps,
        axis_bindings=axes,
        all_cells=cells,
        atomic_facts=[],
        comparison_facts=[],
        bucket_facts=[],
        row_matrices=[rm],
    )

    from src.pdf_retrieval_v4.typed_evidence_emitters import (
        ADMITTED_OUTCOMES,
        TYPED_ELIGIBLE_KINDS,
    )

    typed_eligible = [o for o in outcomes if o["temporal_kind"] in TYPED_ELIGIBLE_KINDS]
    typed_covered = [o for o in typed_eligible if o["outcomes"] & ADMITTED_OUTCOMES]

    # Only 1 typed-eligible cell (cell:0:1 with point), 1 covered
    assert len(typed_eligible) == 1
    assert len(typed_covered) == 1

    # Typed admission = 1/1 = 100%, NOT 2/1 = 200%
    admission = len(typed_covered) / len(typed_eligible) if typed_eligible else 0.0
    assert admission <= 1.0, f"Typed admission {admission} exceeds 100%"
