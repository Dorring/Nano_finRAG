from src.evaluation.nf40_frozen_context import FrozenCaseContext, FrozenContextCandidate
from src.evaluation.nf42_structured_fact_extraction import extract_structured_facts


def _context(content: str) -> FrozenCaseContext:
    candidate = FrozenContextCandidate(
        case_id="case", rank=1, candidate_key="candidate:v1:test", content_hash="hash",
        rendered_content="[report.pdf, p3]\n" + content, document_id="report.pdf",
        source_id="source", page=3, block_type="table",
    )
    return FrozenCaseContext("case", (candidate,), "context")


def test_table_columns_keep_distinct_header_periods():
    facts = extract_structured_facts(_context(
        "Cash and cash equivalents | 143,540 | 206,031\n"
        "Table column context: 2020 | 2019"
    ))
    numeric = [fact for fact in facts if fact.value_expression]
    assert [(fact.value_expression, fact.period) for fact in numeric] == [
        ("143,540", "2020"), ("206,031", "2019"),
    ]


def test_narrative_with_multiple_years_does_not_bind_to_first_year():
    facts = extract_structured_facts(_context(
        "Revenue was $42.2 million in 2025 compared with $31.0 million in 2024."
    ))
    numeric = [fact for fact in facts if fact.value_expression]
    assert all(fact.period is None for fact in numeric)


def test_fact_identity_is_candidate_based_not_text_or_rank_based():
    first = extract_structured_facts(_context("Revenue | $42.2 million"))[0]
    second = extract_structured_facts(_context("Revenue | $42.2 million"))[0]
    assert first.fact_id == second.fact_id
