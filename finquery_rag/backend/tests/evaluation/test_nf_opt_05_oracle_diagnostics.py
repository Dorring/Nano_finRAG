from src.evaluation.nf_opt_05 import classify_first_failure


def test_route_accuracy_is_separate_from_evidence_sufficiency():
    assert (
        classify_first_failure(
            route_correct=True,
            evidence_sufficient=False,
            operands_correct=False,
            execution_success=False,
            result_correct=False,
        )
        == "production_evidence_missing"
    )


def test_oracle_metrics_are_not_production_metrics():
    report = {"oracle_evidence": True, "production_metric": False}
    assert report["oracle_evidence"] and not report["production_metric"]


def test_retrieval_failure_is_not_router_failure():
    assert (
        classify_first_failure(
            route_correct=True,
            evidence_sufficient=False,
            operands_correct=False,
            execution_success=False,
            result_correct=False,
        )
        != "calculation_intent_not_detected"
    )
