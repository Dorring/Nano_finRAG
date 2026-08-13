from __future__ import annotations

from pathlib import Path

from rag_v2.contracts.plan import Action, Intent, SupervisorPlan
from rag_v2.generation import (
    FinancialGenerationViewRendererV1,
    FinancialGenerationViewV1,
    MockGeneratorProviderV1,
    ProviderRegistryV1,
)
from rag_v2.runtime import (
    GeneratorRouteConfigV1,
    GeneratorRoutingPolicyV1,
    RuntimeRouteV1,
    TerminalReason,
    TrustedRAGQueryV2,
    TrustedRAGRuntimeV2,
    V2FinalEvaluationRunner,
)


def packet(route: str = "DIRECT", query_id: str = "q1") -> dict:
    return {"query_id": query_id, "question": "What was revenue?", "route": route,
            "validation_status": "VERIFIED", "allowed_citation_ids": ["EV-1"],
            "evidence_items": [{"citation_id": "EV-1", "fact_id": "f1", "source_id": "s1", "metric": "revenue",
                                "period": "FY2025", "scope": "total", "value": "100", "unit": "USD",
                                "currency": "USD", "scale": "1", "source_text": "Revenue was 100 USD.",
                                "provenance": {"physical_source_id": "physical-1"}}]}


def plan(intent: Intent = Intent.DIRECT_FACT) -> SupervisorPlan:
    return SupervisorPlan.from_dict({"intent": intent.value, "operation": "difference" if intent is Intent.CALCULATION else None,
        "next_action": Action.GENERATE.value, "required_slots": [{"slot_id": "slot_1", "metric": "revenue",
            "period": "FY2025", "role": "fact", "value_type": "number", "unit": "USD"}]})


def test_renderer_is_deterministic_and_preserves_fields() -> None:
    value = FinancialGenerationViewV1.from_verified_packet(packet())
    repeated = FinancialGenerationViewV1.from_verified_packet(packet())
    assert value.rendered_text == repeated.rendered_text and value.view_sha256 == repeated.view_sha256
    assert "[E1]" in value.rendered_text and "Metric: revenue" in value.rendered_text
    assert "Gold" not in value.rendered_text and "oracle_training_evidence" not in value.rendered_text
    generation_input = value.to_generation_input(packet())
    assert generation_input.packet["allowed_citation_ids"] == ["E1"]
    assert generation_input.packet["evidence_items"][0]["value"] == "100"


def test_mock_provider_receives_view_not_raw_packet() -> None:
    captured = []

    def response(generation_input, _context):
        captured.append(generation_input)
        return {"query_id": generation_input.query_id, "route": generation_input.route,
                "answer_text": "The supplied evidence is available [E1].", "citation_ids": ["E1"],
                "generator_model": "mock"}

    renderer = FinancialGenerationViewRendererV1()
    provider = MockGeneratorProviderV1("local_financial_grounded", "mock", response=response)
    runtime = TrustedRAGRuntimeV2(ProviderRegistryV1({"local_financial_grounded": provider}),
        GeneratorRoutingPolicyV1({RuntimeRouteV1.DIRECT_FACT: GeneratorRouteConfigV1("local_financial_grounded")}),
        renderer=renderer)
    result = runtime.handle(TrustedRAGQueryV2("q1", "What was revenue?", plan(), packet()))
    assert result.released and len(captured) == 1
    assert captured[0].renderer_id == "financial_generation_view_v1"
    assert captured[0].rendered_text.startswith("[QUESTION]")
    assert "evidence_items" not in captured[0].rendered_text


def test_calculation_rendering_preserves_canonical_result() -> None:
    value = packet("CALCULATION", "calc")
    value["calculation_result"] = {"operation": "difference", "value": "20", "period": "FY2025",
                                    "unit": "USD", "currency": "USD", "scale": "1",
                                    "allowed_citation_ids": ["EV-1"]}
    view = FinancialGenerationViewV1.from_verified_packet(value)
    assert view.calculation_ids == ("C1",)
    assert "Canonical Result: 20" in view.rendered_text
    assert view.to_generation_input(value).packet["calculation_result"]["value"] == "20"


def test_no_answer_calculation_and_multi_never_render() -> None:
    class CountingRenderer(FinancialGenerationViewRendererV1):
        calls = 0

        def render(self, value):
            self.calls += 1
            return super().render(value)

    renderer = CountingRenderer()
    provider = MockGeneratorProviderV1("provider", "mock")
    runtime = TrustedRAGRuntimeV2(ProviderRegistryV1({"provider": provider}),
        GeneratorRoutingPolicyV1({RuntimeRouteV1.DIRECT_FACT: GeneratorRouteConfigV1("provider")}),
        renderer=renderer)
    assert runtime.handle(TrustedRAGQueryV2("none", "", None, None, True)).terminal_reason is TerminalReason.TR7_NO_ANSWER
    assert runtime.handle(TrustedRAGQueryV2("calc", "", plan(Intent.CALCULATION), None)).terminal_reason is TerminalReason.TR8_CALCULATION_NOT_READY
    assert runtime.handle(TrustedRAGQueryV2("multi", "", plan(Intent.MULTI_EVIDENCE), None)).terminal_reason is TerminalReason.TR9_MULTI_NOT_READY
    assert renderer.calls == 0 and provider.calls == 0


def test_train_runtime_contract_parity_and_token_bound() -> None:
    view = FinancialGenerationViewV1.from_verified_packet(packet())
    md = Path("data/grounding_alignment/v1/financial-generation-view-v1.md").read_text(encoding="utf-8")
    assert all(section in md and section in view.rendered_text for section in ("[QUESTION]", "[VERIFIED EVIDENCE]", "[ANSWER RULES]"))
    assert "[E1]" in md and "[E1]" in view.rendered_text
    assert len(view.rendered_text) < 4096 * 4


def test_runner_seals_before_post_evaluation_callback_and_future_config() -> None:
    provider = MockGeneratorProviderV1("provider", "mock", response={"query_id": "q1", "route": "DIRECT",
        "answer_text": "Evidence [E1].", "citation_ids": ["E1"], "generator_model": "mock"})
    runtime = TrustedRAGRuntimeV2(ProviderRegistryV1({"provider": provider}),
        GeneratorRoutingPolicyV1({RuntimeRouteV1.DIRECT_FACT: GeneratorRouteConfigV1("provider")}),
        renderer=FinancialGenerationViewRendererV1())
    seen = []
    result = V2FinalEvaluationRunner(runtime).run([TrustedRAGQueryV2("q1", "What was revenue?", plan(), packet())],
                                                  post_seal_evaluator=lambda _: seen.append("sealed"))
    assert result.prediction_seal and result.gold_loaded_after_seal and seen == ["sealed"]
    policy = GeneratorRoutingPolicyV1.from_config({"generation": {"routing": {
        "DIRECT_FACT": {"primary": "local_financial_grounded", "fallback": "general"},
        "CALCULATION": {"primary": "local_financial_grounded", "fallback": "general"},
        "MULTI_EVIDENCE": {"primary": "general", "fallback": None}}}})
    assert policy.for_route(RuntimeRouteV1.DIRECT_FACT).primary == "local_financial_grounded"
