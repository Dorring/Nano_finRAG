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
        "normalized_base_value": "100000000",
        "strict": True,
        "shadow_table_id": "table",
        "shadow_row_id": "row",
        "shadow_cell_ids": ["cell"],
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
