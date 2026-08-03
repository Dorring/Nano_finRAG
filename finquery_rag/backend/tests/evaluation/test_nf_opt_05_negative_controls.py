from src.domain.calculation import CalculationStatus
from src.finance.operation_router import route_calculation
from src.services.intent import classify_query_intent


def _route(question: str):
    return route_calculation(
        question, classify_query_intent(question), allow_derived_document_qa=True
    )


def test_direct_percentage_fact_stays_fact():
    assert (
        _route("What was the gross margin?").status is CalculationStatus.NOT_APPLICABLE
    )


def test_disclosed_margin_stays_fact():
    assert (
        _route("What was the reported net margin in FY2025?").status
        is CalculationStatus.NOT_APPLICABLE
    )


def test_report_both_stays_fact():
    assert (
        _route("Report revenue for FY2024 and FY2025.").status
        is CalculationStatus.NOT_APPLICABLE
    )


def test_qualitative_comparison_stays_fact():
    assert (
        _route("Which segment was larger in FY2025?").status
        is CalculationStatus.NOT_APPLICABLE
    )


def test_no_answer_does_not_route_to_calculation():
    assert (
        _route("What is the internal employee performance accuracy?").status
        is CalculationStatus.NOT_APPLICABLE
    )
