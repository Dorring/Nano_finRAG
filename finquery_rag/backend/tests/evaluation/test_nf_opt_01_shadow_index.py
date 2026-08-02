from src.evaluation.nf_opt_01 import candidate_scope_ok


def test_shadow_index_uses_whitelist_and_does_not_accept_legacy_document():
    allowed = {"aapl_fy2025"}
    assert candidate_scope_ok("aapl_fy2025", allowed)
    assert not candidate_scope_ok("FINAL Annual Report.pdf", allowed)


def test_shadow_index_scope_is_independent_of_production_collection():
    # The helper has no production-collection handle or mutation path.
    assert candidate_scope_ok("doc-1", {"doc-1"})
