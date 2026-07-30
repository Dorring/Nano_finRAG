from src.evaluation.nf41_production_attribution import next_step_for_observed_failures


def test_trace_insufficiency_blocks_nf42_direction():
    assert next_step_for_observed_failures({"production_trace_insufficient": 1, "production_fact_not_extracted": 6}) == ("none", "expand_observer_coverage")
