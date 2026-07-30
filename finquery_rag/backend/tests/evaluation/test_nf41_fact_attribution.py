from decimal import Decimal

from src.evaluation.evaluation import EvaluationCase, ExpectedSource
from src.evaluation.nf41_fact_selection import (
    AnswerExecutionMode,
    DeterministicFactFailure,
    StructuredFactCandidate,
    build_constraints,
    classify_fact_failure,
)


def _case(*, number: str = "72", page: int = 3) -> EvaluationCase:
    return EvaluationCase(
        case_id="case-1",
        question="What was gross margin in 2025?",
        expected_sources=(ExpectedSource(filename="report.pdf", page=page),),
        expected_numbers=(number,),
        expected_no_answer=False,
    )


def _fact(*, value: str = "72", page: int = 3, key: str = "candidate-1", period: str | None = "2025"):
    return StructuredFactCandidate(
        fact_id=f"{key}:{value}", candidate_key=key, candidate_rank=1,
        document_id="report.pdf", page=page, text=f"Gross margin was {value}% in 2025.",
        value=Decimal(value), source_expression=f"{value}%", unit="percentage",
        period=period, metric_text="Gross margin", extraction_source="numeric_span",
        extraction_confidence=1.0,
    )


def test_deterministic_error_is_not_labeled_llm_generation_error():
    assert AnswerExecutionMode.DETERMINISTIC_FACT.value != "llm_generation"


def test_gold_source_without_fact_is_factization_failure():
    case = _case()
    fact = _fact(page=4)
    assert classify_fact_failure(case=case, facts=[fact], selected=fact, rendered_answer="72%") == DeterministicFactFailure.GOLD_SOURCE_NOT_FACTIZED


def test_correct_fact_available_wrong_selected_is_selector_failure():
    case = _case()
    correct = _fact(value="72", key="candidate-1")
    selected = _fact(value="70", key="candidate-2")
    assert classify_fact_failure(case=case, facts=[correct, selected], selected=selected, rendered_answer="70%") == DeterministicFactFailure.WRONG_CANDIDATE_SELECTED


def test_correct_fact_selected_wrong_rendered_is_renderer_failure():
    case = _case()
    fact = _fact()
    assert classify_fact_failure(case=case, facts=[fact], selected=fact, rendered_answer="70%") == DeterministicFactFailure.FACT_CORRECT_RENDERING_WRONG


def test_wrong_period_is_distinct_from_wrong_metric():
    case = _case()
    correct = _fact(value="72", key="candidate-1")
    selected = _fact(value="70", key="candidate-2", period="2024")
    assert classify_fact_failure(case=case, facts=[correct, selected], selected=selected, rendered_answer="70%") == DeterministicFactFailure.WRONG_PERIOD_SELECTED


def test_constraints_are_generic_and_preserve_requested_period():
    constraints = build_constraints("What percentage of revenue was reported in 2025?")
    assert constraints.periods == ("2025",)
    assert constraints.expected_unit_family == "percentage"
