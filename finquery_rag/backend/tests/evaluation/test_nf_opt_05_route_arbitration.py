from src.domain.calculation import CalculationStatus
from src.finance.operation_router import route_calculation
from src.services.intent import classify_query_intent


def _route(question: str):
    return route_calculation(
        question, classify_query_intent(question), allow_derived_document_qa=True
    )


def test_calculation_route_precedes_fact_only_for_derived_value():
    decision = _route(
        "What was the year-over-year growth rate of revenue from FY2024 to FY2025?"
    )
    assert decision.status is CalculationStatus.READY


def test_missing_operation_does_not_route_to_calculation():
    decision = _route("What was revenue in FY2025?")
    assert decision.status is CalculationStatus.NOT_APPLICABLE


def test_missing_operand_structure_does_not_route_to_calculation():
    decision = _route("What was the growth rate of revenue?")
    assert decision.status is CalculationStatus.NOT_APPLICABLE


def test_shadow_router_is_disabled_by_default():
    question = (
        "What was the year-over-year growth rate of revenue from FY2024 to FY2025?"
    )
    decision = route_calculation(question, classify_query_intent(question))
    assert decision.status is CalculationStatus.NOT_APPLICABLE
