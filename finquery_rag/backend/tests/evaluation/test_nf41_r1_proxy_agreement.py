from src.evaluation.nf41_production_attribution import proxy_production_relation


def test_proxy_fact_is_not_labeled_production_fact():
    assert proxy_production_relation(proxy_failure="production_fact_not_extracted", production_failure="production_fact_not_extracted") == "agreement"


def test_incomplete_production_trace_overrides_proxy_claim():
    assert proxy_production_relation(proxy_failure="production_fact_available_not_selected", production_failure="production_trace_insufficient") == "trace_insufficient"
