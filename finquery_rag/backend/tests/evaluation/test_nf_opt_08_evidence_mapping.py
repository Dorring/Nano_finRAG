from src.evaluation.nf_opt_08 import ShadowEvidenceMapping, mapping_is_manually_verified

def _mapping(**changes):
    values = dict(legacy_candidate_key="old", shadow_table_id="table", shadow_row_id="row", shadow_cell_ids=("cell",), relation="same_table_row", document_match=True, page_match=True, metric_match=True, period_match=True, value_match=True, scale_match=True, reviewer="human", reviewed_at="2026-08-04", verified=True)
    values.update(changes)
    return ShadowEvidenceMapping(**values)

def test_same_page_is_not_evidence_mapping():
    assert not mapping_is_manually_verified(_mapping(metric_match=False))

def test_mapping_requires_metric_period_value_match():
    assert not mapping_is_manually_verified(_mapping(value_match=False))

def test_parser_top1_is_not_auto_verified():
    assert not mapping_is_manually_verified(_mapping(reviewer=None, reviewed_at=None, verified=False))


def test_shadow_identity_does_not_depend_on_case_id():
    from src.evaluation.nf_opt_08 import stable_shadow_id
    assert stable_shadow_id("doc", 1, "parser", [0, 0, 1, 1], [["x"]]) == stable_shadow_id("doc", 1, "parser", [0, 0, 1, 1], [["x"]])


def _option(**changes):
    value = {
        "metric_score": 1.0,
        "normalized_period": "FY2025",
        "parsed_numeric_value": "100",
        "parsed_scale": "million",
        "scale_context_source": "table_header",
        "cell_geometry_valid": True,
        "header_path": ["FY2025"],
        "normalized_base_value": "100000000",
        "strict": True,
        "shadow_table_id": "table",
        "shadow_row_id": "row",
        "shadow_cell_ids": ["cell"],
        "row_index": 0,
        "column_index": 0,
    }
    value.update(changes)
    return value


def _contract(**changes):
    value = {"expected_period": "FY2025", "expected_value": "100000000"}
    value.update(changes)
    return value


def test_missing_period_column_is_not_candidate_pending():
    from scripts.evaluation.run_nf_opt_08_r2_mapping_package import classify

    status, _, proposed, _ = classify(
        _contract(), [_option(normalized_period=None, strict=False)]
    )

    assert status == "missing_period_column"
    assert proposed is None


def test_missing_scale_is_not_candidate_pending():
    from scripts.evaluation.run_nf_opt_08_r2_mapping_package import classify

    status, _, proposed, _ = classify(
        _contract(), [_option(parsed_scale=None, normalized_base_value=None, strict=False)]
    )

    assert status == "missing_scale"
    assert proposed is None


def test_wrong_value_candidate_is_not_candidate_pending():
    from scripts.evaluation.run_nf_opt_08_r2_mapping_package import classify

    status, _, proposed, _ = classify(
        _contract(), [_option(normalized_base_value="120810000", strict=False)]
    )

    assert status == "wrong_table_candidate"
    assert proposed is None


def test_ambiguous_mapping_exposes_strict_competitors():
    from scripts.evaluation.run_nf_opt_08_r2_mapping_package import classify

    first = _option(shadow_cell_ids=["cell-1"])
    second = _option(shadow_cell_ids=["cell-2"])
    status, _, proposed, competitors = classify(_contract(), [first, second])

    assert status == "ambiguous"
    assert proposed is None
    assert {item["shadow_cell_ids"][0] for item in competitors} == {"cell-1", "cell-2"}


def test_wrong_metric_row_cannot_be_pending():
    from scripts.evaluation.run_nf_opt_08_r2_mapping_package import classify

    status, _, proposed, _ = classify(
        _contract(), [_option(metric_score=0.5, strict=False)]
    )

    assert status == "missing_row"
    assert proposed is None


def test_tail_currency_symbol_is_parsed_safely():
    from decimal import Decimal
    from scripts.evaluation.run_nf_opt_08_r2_mapping_package import number

    assert number("47,061  $") == Decimal("47061")
    assert number("($47,061)") == Decimal("-47061")
    assert number("47,061 estimated") is None
    assert number("47,061 / 48,000") is None


def test_table_header_scale_context_is_recovered():
    from scripts.evaluation.run_nf_opt_08_r2_mapping_package import table_context

    context, currency, scale, source = table_context(
        [["December 31, (in millions, except share data)", "", "2025", "2024"]],
        "",
    )

    assert "millions" in context.casefold()
    assert currency is None
    assert scale == "million"
    assert source == "table_header"


def test_invalid_parser_header_falls_back_to_matrix_period_header():
    from scripts.evaluation.run_nf_opt_08_r2_mapping_package import Table, resolve_header

    table = Table(
        "pymupdf", "1", None, "doc", 1, 0, None,
        [["Year ended", "", "2025", "2024"], ["Revenue", "$", "100", "90"]],
        [[None] * 4, [None] * 4], ["Revenues"], None, None, None, None, None, None,
    )
    path, raw, period, resolution = resolve_header(table, 2)

    assert path
    assert raw == "2025"
    assert period == "FY2025"
    assert resolution == "matrix_multilevel"


def test_cross_table_row_cell_reference_fails_hierarchy_validation():
    from scripts.evaluation.run_nf_opt_08_r2_mapping_package import references_are_hierarchical

    reference = {"shadow_table_id": "table-a", "shadow_row_id": "row-b", "shadow_cell_ids": ["cell-b"]}
    assert not references_are_hierarchical(
        [reference],
        {"table-a": {"row-a"}, "table-b": {"row-b"}},
        {"row-a": {"cell-a"}, "row-b": {"cell-b"}},
    )


def test_acceptance_fields_fail_closed():
    from scripts.evaluation.run_nf_opt_08_r2_mapping_package import acceptance_is_valid

    valid = {
        "source_count": 22,
        "case_source_unique": True,
        "sorted_by_case_source": True,
        "all_review_status_pending": True,
        "reviewer_non_null_count": 0,
        "reviewed_at_non_null_count": 0,
        "automatic_verified_count": 0,
        "input_hashes_verified": True,
        "production_switch_allowed": False,
        "manual_review_allowed": False,
        "status_fields_consistent": True,
        "hierarchical_identity_references_valid": True,
        "zero_execution_counts": True,
    }
    assert acceptance_is_valid(valid)
    for key, bad_value in (
        ("source_count", 21),
        ("reviewer_non_null_count", 1),
        ("input_hashes_verified", False),
        ("production_switch_allowed", True),
        ("manual_review_allowed", True),
        ("hierarchical_identity_references_valid", False),
        ("zero_execution_counts", False),
    ):
        candidate = dict(valid)
        candidate[key] = bad_value
        assert not acceptance_is_valid(candidate)


def test_target_excerpt_contains_a_late_target_column():
    from scripts.evaluation.run_nf_opt_08_r2_mapping_package import target_excerpt

    excerpt = target_excerpt(["Metric", "1", "2", "3", "4", "5", "6", "7", "47,941"], "Metric", 8)
    assert excerpt["target_cell_excerpt"]["raw_text"] == "47,941"
    assert any(cell["column_index"] == 8 for cell in excerpt["neighbor_cells"])


def test_single_period_many_value_columns_is_not_unique_candidate():
    from scripts.evaluation.run_nf_opt_08_r2_mapping_package import classify

    options = [
        _option(
            column_index=index,
            parsed_scale="million",
            scale_context_source="table_header",
            cell_geometry_valid=True,
            header_path=["FY2025"],
            strict=True,
        )
        for index in (2, 4, 6)
    ]
    status, _, proposed, _ = classify(_contract(), options)
    assert status == "missing_period_column"
    assert proposed is None


def test_cell_geometry_rejects_bbox_outside_table():
    from scripts.evaluation.run_nf_opt_08_r2_mapping_package import Table, cell_geometry_valid

    table = Table(
        "pymupdf", "1", None, "doc", 1, 0, (0, 0, 100, 100),
        [["Revenue", "100"]], [[(200, 0, 210, 10), (10, 0, 20, 10)]],
        [], None, None, None, None, None, None,
    )
    assert not cell_geometry_valid(table, 0, 0)
    assert cell_geometry_valid(table, 0, 1)
