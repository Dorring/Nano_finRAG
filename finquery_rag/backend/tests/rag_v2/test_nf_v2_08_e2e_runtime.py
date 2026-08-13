from __future__ import annotations

from rag_v2.contracts.plan import Action, Intent, SupervisorPlan
from rag_v2.generation import MockGeneratorProviderV1, ProviderRegistryV1
from rag_v2.runtime import (
    GeneratorRouteConfigV1,
    GeneratorRoutingPolicyV1,
    RuntimeMetricAggregatorV1,
    RuntimeRouteV1,
    TerminalReason,
    TrustedRAGQueryV2,
    TrustedRAGRuntimeV2,
    V2FinalEvaluationRunner,
)


def packet(route: str = "DIRECT", query_id: str = "q1") -> dict:
    return {"query_id": query_id, "route": route, "validation_status": "VERIFIED",
            "allowed_citation_ids": ["EV-1"], "evidence_items": [{
                "citation_id": "EV-1", "fact_id": "fact-1", "source_id": "source-1",
                "metric": "revenue", "period": "FY2025", "value": "100", "unit": "USD",
                "currency": "USD", "scale": "1", "provenance": {"physical_source_id": "physical-1"}}]}


def plan(intent: Intent = Intent.DIRECT_FACT) -> SupervisorPlan:
    return SupervisorPlan.from_dict({"intent": intent.value, "operation": "difference" if intent is Intent.CALCULATION else None,
        "next_action": Action.GENERATE.value, "required_slots": [{"slot_id": "slot_1", "metric": "revenue",
            "period": "FY2025", "role": "fact", "value_type": "number", "unit": "USD"}]})


def good(qid: str = "q1") -> dict:
    return {"query_id": qid, "route": "DIRECT", "answer_text": "100 USD in FY2025 [EV-1].",
            "citation_ids": ["EV-1"], "generation_status": "complete", "generator_model": "mock"}


def bad(qid: str = "q1") -> dict:
    value = good(qid)
    value["answer_text"] = "101 USD in FY2025 [EV-1]."
    return value


def runtime(primary, fallback=None) -> TrustedRAGRuntimeV2:
    providers = {"primary": primary}
    if fallback:
        providers["fallback"] = fallback
    return TrustedRAGRuntimeV2(ProviderRegistryV1(providers), GeneratorRoutingPolicyV1({
        RuntimeRouteV1.DIRECT_FACT: GeneratorRouteConfigV1("primary", "fallback" if fallback else None),
        RuntimeRouteV1.CALCULATION: GeneratorRouteConfigV1(None),
        RuntimeRouteV1.MULTI_EVIDENCE: GeneratorRouteConfigV1(None),
    }))


def query(packet_value=None, plan_value=None, no_answer=False) -> TrustedRAGQueryV2:
    return TrustedRAGQueryV2("q1", "question", plan_value or plan(), packet_value, no_answer)


def test_direct_primary_pass_and_trace() -> None:
    provider = MockGeneratorProviderV1("primary", "mock", good())
    response = runtime(provider).handle(query(packet()))
    assert response.released and response.terminal_reason is TerminalReason.TR0_RELEASED_PRIMARY
    assert response.attempt_count == 1 and response.trace.primary_provider == "primary"


def test_direct_fail_no_fallback_and_fallback_pass_or_fail() -> None:
    assert runtime(MockGeneratorProviderV1("primary", "mock", bad())).handle(query(packet())).terminal_reason is TerminalReason.TR3_PRIMARY_VALIDATION_FAIL_NO_FALLBACK
    released = runtime(MockGeneratorProviderV1("primary", "mock", bad()), MockGeneratorProviderV1("fallback", "mock", good())).handle(query(packet()))
    assert released.released and released.terminal_reason is TerminalReason.TR1_RELEASED_FALLBACK
    abstained = runtime(MockGeneratorProviderV1("primary", "mock", bad()), MockGeneratorProviderV1("fallback", "mock", bad())).handle(query(packet()))
    assert not abstained.released and abstained.terminal_reason is TerminalReason.TR4_FALLBACK_VALIDATION_FAIL


def test_no_trusted_evidence_no_answer_and_not_ready_routes_do_not_generate() -> None:
    provider = MockGeneratorProviderV1("primary", "mock", good())
    rt = runtime(provider)
    assert rt.handle(query(None)).terminal_reason is TerminalReason.TR2_NO_TRUSTED_EVIDENCE
    assert rt.handle(query(None, None, True)).terminal_reason is TerminalReason.TR7_NO_ANSWER
    assert rt.handle(query(None, plan(Intent.CALCULATION))).terminal_reason is TerminalReason.TR8_CALCULATION_NOT_READY
    assert rt.handle(query(None, plan(Intent.MULTI_EVIDENCE))).terminal_reason is TerminalReason.TR9_MULTI_NOT_READY
    assert provider.calls == 0


def test_provider_error_and_budget_invariants() -> None:
    class Broken(MockGeneratorProviderV1):
        def generate(self, generation_input, generation_context):
            self.calls += 1
            raise RuntimeError("broken")
    response = runtime(Broken("primary", "broken"), MockGeneratorProviderV1("fallback", "mock", good())).handle(query(packet()))
    assert response.released and response.attempt_count == 2 and response.used_fallback
    assert response.attempt_count <= 2
    assert runtime(Broken("primary", "broken")).handle(query(packet())).terminal_reason is TerminalReason.TR5_PROVIDER_ERROR


def test_config_metrics_and_seal_order() -> None:
    config = {"generation": {"routing": {"DIRECT_FACT": {"primary": "primary", "fallback": None},
        "CALCULATION": {"primary": None, "fallback": None}, "MULTI_EVIDENCE": {"primary": None, "fallback": None}}}}
    policy = GeneratorRoutingPolicyV1.from_config(config)
    response = runtime(MockGeneratorProviderV1("primary", "mock", good())).handle(query(packet()))
    metrics = RuntimeMetricAggregatorV1()
    metrics.observe(response)
    assert policy.for_route(RuntimeRouteV1.DIRECT_FACT).primary == "primary"
    assert metrics.snapshot()["queries_total"] == 1
    events = []
    runner = V2FinalEvaluationRunner(runtime(MockGeneratorProviderV1("primary", "mock", good())))
    result = runner.run([query(packet())], post_seal_evaluator=lambda _: events.append("after_seal"))
    assert result.prediction_seal and result.gold_loaded_after_seal and events == ["after_seal"]


def test_soft_fail_policy_can_abstain_without_fallback() -> None:
    soft = good()
    soft["citation_ids"] = []
    soft["answer_text"] = "100 USD in FY2025."
    rt = TrustedRAGRuntimeV2(ProviderRegistryV1({
        "primary": MockGeneratorProviderV1("primary", "mock", soft),
        "fallback": MockGeneratorProviderV1("fallback", "mock", good()),
    }), GeneratorRoutingPolicyV1({RuntimeRouteV1.DIRECT_FACT: GeneratorRouteConfigV1(
        "primary", "fallback", fallback_on_soft_fail=False)}))
    response = rt.handle(query(packet()))
    assert not response.released and response.terminal_reason is TerminalReason.TR3_PRIMARY_VALIDATION_FAIL_NO_FALLBACK
