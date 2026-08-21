from rag_v2.adaptive import (
    AdaptiveRAGBudgetV1,
    AdaptiveRAGStateV1,
    BoundedAdaptiveRAGV1,
    ConsistencyDecision,
    EvidenceConsistencyGateV1,
    PeriodSemantics,
    ReasonCode,
    ReplanActionV1,
    TemporalEvidenceV1,
    ToolCapability,
)


def _packet(evidence_id: str, metric: str, value: str, period: str, *, source: str = "10-K") -> dict:
    temporal = TemporalEvidenceV1(
        entity="Acme", document_id=evidence_id, fiscal_year=period[-4:],
        period_semantics=PeriodSemantics.ANNUAL, source=source,
        metric=metric, value=value,
    )
    return {
        "evidence_id": evidence_id, "metric": metric, "value": value,
        "period": period, "entity": "Acme", "scope": "consolidated",
        "source": source, "document_id": evidence_id, "temporal": temporal.to_dict(),
    }


def _state() -> AdaptiveRAGStateV1:
    return AdaptiveRAGStateV1.new(
        "q1", "What was Revenue in FY2024?",
        required_slots=[{"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"}],
    )


def test_budget_is_explicit_and_bounded() -> None:
    budget = AdaptiveRAGBudgetV1()
    assert budget.max_replan_rounds == 2
    assert budget.max_total_tool_calls == 5
    assert budget.max_identical_query_retry == 0


def test_recoverable_observation_replans_then_ready() -> None:
    state = _state()
    calls = {"n": 0}

    def tool(query, current):
        calls["n"] += 1
        return [] if calls["n"] == 1 else [_packet("e24", "Revenue", "120", "FY2024")]

    action = ReplanActionV1(ToolCapability.SEMANTIC_RETRIEVAL, state.normalized_query, ReasonCode.MISSING_SLOT)
    result = BoundedAdaptiveRAGV1().run(state, {ToolCapability.SEMANTIC_RETRIEVAL: tool}, initial_action=action)
    assert result.state.status == "READY_TO_GENERATE"
    assert result.state.replan_rounds == 1
    assert result.state.tool_calls == 2


def test_identical_evidence_fails_closed_no_progress() -> None:
    state = _state()
    item = _packet("wrong", "Cost", "40", "FY2024")

    def tool(query, current):
        return [item]

    result = BoundedAdaptiveRAGV1().run(
        state,
        {ToolCapability.SEMANTIC_RETRIEVAL: tool},
        initial_action=ReplanActionV1(ToolCapability.SEMANTIC_RETRIEVAL, state.normalized_query, ReasonCode.MISSING_SLOT),
    )
    assert result.state.status == "FAIL_CLOSED"
    assert result.state.stop_reason == "NO_PROGRESS"


def test_same_scope_conflict_is_unresolved() -> None:
    gate = EvidenceConsistencyGateV1()
    result = gate.evaluate([
        _packet("a", "Debt", "10", "FY2024"),
        _packet("b", "Debt", "20", "FY2024"),
    ])
    assert result.decision is ConsistencyDecision.UNRESOLVED_CONFLICT


def test_different_sources_are_not_automatically_conflict() -> None:
    gate = EvidenceConsistencyGateV1()
    result = gate.evaluate([
        _packet("a", "Rating", "BUY", "FY2024", source="broker-a"),
        _packet("b", "Rating", "HOLD", "FY2024", source="broker-b"),
    ])
    assert result.decision is ConsistencyDecision.MULTI_SOURCE_COMPATIBLE
