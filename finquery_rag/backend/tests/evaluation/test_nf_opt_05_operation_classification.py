from src.domain.calculation import CalculationOperation, CalculationStatus
from src.finance.operation_router import route_calculation
from src.services.intent import classify_query_intent


def test_operation_accuracy_is_reported_by_router():
    question = "What percentage of total revenue came from Services?"
    decision = route_calculation(
        question, classify_query_intent(question), allow_derived_document_qa=True
    )
    assert decision.status is CalculationStatus.READY
    assert decision.operation is CalculationOperation.PERCENTAGE_SHARE
