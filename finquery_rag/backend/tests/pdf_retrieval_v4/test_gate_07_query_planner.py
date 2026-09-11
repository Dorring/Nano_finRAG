from src.pdf_retrieval_v4.planner import ConceptResolver, build_query_plan
from src.pdf_retrieval_v4.query_plan_models import QueryPlan
from src.pdf_retrieval_v4.query_plan_validator import validate_query_plan


def plan(question: str, scope=("issuer",)):
    return build_query_plan(question, tuple(scope), ConceptResolver())


def test_table_single_fact_plan():
    result = plan("What was revenue in FY2025?")
    assert result.task_type == "table_single_fact"
    assert {route.index_type for route in result.retrieval_routes} >= {"raw_production", "table", "row", "atomic_fact"}


def test_iso_filing_question_has_clean_metric_and_period_slot():
    result = plan("What does the AMZN filing for 2023-12-31 report for the 'Uncertain Tax Positions' row?")
    assert result.metric_phrases == ("Uncertain Tax Positions",)
    assert result.periods == ("FY2023",)
    assert [(slot.raw_metric_phrase, slot.period) for slot in result.operand_slots] == [("Uncertain Tax Positions", "FY2023")]


def test_comparison_plan_has_clean_metric_and_two_periods():
    result = plan("Compare Apple FY2024 vs FY2023 Revenue.")
    assert result.metric_phrases == ("Revenue",)
    assert result.periods == ("FY2024", "FY2023")
    assert [(slot.raw_metric_phrase, slot.period) for slot in result.operand_slots] == [
        ("Revenue", "FY2024"),
        ("Revenue", "FY2023"),
    ]


def test_reported_values_their_sum_builds_calculation_slots():
    result = plan(
        "Using the reported values for 'Deferred tax assets' and 'Operating leases' "
        "in the filing for 2025-12-31, what is their sum?"
    )
    assert result.task_type == "calculation_multi_operand"
    assert result.operation == "sum"
    assert [slot.raw_metric_phrase for slot in result.operand_slots] == ["Deferred tax assets", "Operating leases"]


def test_reported_values_their_difference_builds_calculation_slots():
    result = plan(
        "Using the reported values for 'Current' and 'Deferred' "
        "in the filing for 2024-12-31, what is their difference?"
    )
    assert result.task_type == "calculation_multi_operand"
    assert result.operation == "difference"
    assert [slot.raw_metric_phrase for slot in result.operand_slots] == ["Current", "Deferred"]


def test_date_like_row_labels_survive_calculation_operand_cleanup():
    result = plan(
        "Using the reported values for 'December 31, 2023' and "
        "'Weighted-average discount rate - operating leases' in the filing "
        "for 2023-12-31, what is their sum?"
    )
    assert result.task_type == "calculation_multi_operand"
    assert [slot.raw_metric_phrase for slot in result.operand_slots] == [
        "December 31 2023",
        "Weighted-average discount rate - operating leases",
    ]


def test_growth_rate_roles():
    result = plan("Calculate the year-over-year growth rate of revenue from FY2024 to FY2025?")
    assert result.operation == "growth_rate"
    assert [slot.role for slot in result.operand_slots] == ["current_period", "base_period"]
    assert {slot.period for slot in result.operand_slots} == {"FY2024", "FY2025"}


def test_percentage_share_roles():
    result = plan("What percentage of total revenue came from Services in FY2025?")
    assert result.operation == "percentage_share"
    assert [slot.role for slot in result.operand_slots] == ["numerator", "denominator"]


def test_difference_roles():
    result = plan("Calculate the difference between automotive revenue and energy revenue in FY2025?")
    assert result.operation == "difference"
    assert [slot.role for slot in result.operand_slots] == ["minuend", "subtrahend"]


def test_multi_period_and_raw_protection():
    result = plan("What were revenue values in FY2024 and FY2025?")
    assert result.task_type == "single_metric_multi_period"
    assert result.requires_multiple_sources
    assert result.raw_protection_required
    assert "multi_operand_set" in result.evidence_shapes


def test_bucket_route():
    result = plan("What amount matures in one to three years?")
    assert any(slot.bucket_label for slot in result.operand_slots)
    assert any(route.index_type == "bucket_fact" and route.required for route in result.retrieval_routes)


def test_narrative_section_only():
    result = plan("Why did operating expenses increase?")
    assert result.task_type == "narrative_or_note"
    assert {route.index_type for route in result.retrieval_routes} == {"raw_production", "section"}


def test_unsupported_raw_only():
    result = plan("Explain an unsupported quantum battery disclosure.")
    if result.task_type == "unsupported":
        assert [route.index_type for route in result.retrieval_routes] == ["raw_production"]


def test_no_answer_is_not_runtime_type():
    result = plan("Does the report disclose a customer-level contract renewal rate in FY2025?")
    assert result.task_type != "no_answer"
    assert result.answerability_check_required


def test_soft_continuation_disabled_and_plan_deterministic():
    first = plan("What was revenue in FY2025?")
    second = plan("What was revenue in FY2025?")
    assert first.plan_version == "pdf-v4-query-plan-v2"
    assert first.plan_id == second.plan_id
    assert not first.constraints.soft_continuation_expansion
    assert not first.constraints.follow_soft_link
    assert not first.constraints.merge_neighbor_table
    assert not first.constraints.inherit_previous_header


def test_validator_blocks_missing_raw_route():
    result = plan("What was revenue in FY2025?")
    bad = QueryPlan(**{**result.__dict__, "retrieval_routes": tuple(route for route in result.retrieval_routes if route.index_type != "raw_production")})
    assert "raw_route_missing" in validate_query_plan(bad)
