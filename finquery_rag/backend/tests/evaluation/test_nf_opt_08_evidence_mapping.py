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
