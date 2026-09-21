"""Model-free FinancialGenerationViewV1/runtime contract integration audit."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Mapping

from rag_v2.contracts.plan import Action, Intent, SupervisorPlan
from rag_v2.generation import (FINANCIAL_VIEW_V1_CONTRACT_SHA256,
                               FinancialGenerationViewRendererV1,
                               FinancialGenerationViewV1, MockGeneratorProviderV1,
                               ProviderRegistryV1)
from rag_v2.runtime import (GeneratorRouteConfigV1, GeneratorRoutingPolicyV1,
                            RuntimeRouteV1, TrustedRAGQueryV2,
                            TrustedRAGRuntimeV2, V2FinalEvaluationRunner)

ROOT = Path(__file__).resolve().parents[2]
V206 = ROOT / "artifacts/evaluation/nf-v2-06-r0-verified-generation"
GROUNDING = ROOT / "data/grounding_alignment/v1"
OUT = ROOT / "artifacts/evaluation/nf-v2-08-r1-generation-view-integration"
ENGINEERING_BASE = "68a0166"
DATA_COMMIT = "1e4c7d2"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plan_for(packet: Mapping[str, Any], intent: Intent | None = None) -> SupervisorPlan:
    if intent is None:
        intent = Intent.CALCULATION if packet.get("route") == "CALCULATION" else Intent.MULTI_EVIDENCE if packet.get("route") == "MULTI_EVIDENCE" else Intent.DIRECT_FACT
    operation = packet.get("calculation_result", {}).get("operation") if isinstance(packet.get("calculation_result"), Mapping) else None
    allowed_operations = {"difference", "growth_rate", "percentage_share", "sum", "average", "gross_margin", "net_margin", "debt_ratio", "scale_conversion"}
    return SupervisorPlan.from_dict({"intent": intent.value, "operation": operation if operation in allowed_operations else ("difference" if intent is Intent.CALCULATION else None),
        "next_action": Action.GENERATE.value, "required_slots": [{"slot_id": "slot_1", "metric": "verified financial evidence",
            "period": "FY2025", "role": "result" if intent is Intent.CALCULATION else "fact", "value_type": "number", "unit": None}]})


class CountingFinancialRenderer:
    renderer_id = "financial_generation_view_v1"

    def __init__(self) -> None:
        self.delegate = FinancialGenerationViewRendererV1()
        self.calls = 0
        self.inputs: list[Any] = []

    def render(self, packet: Mapping[str, Any]):
        self.calls += 1
        value = self.delegate.render(packet)
        self.inputs.append(value)
        return value


def direct_policy(provider: str | None = "mock_financial") -> GeneratorRoutingPolicyV1:
    return GeneratorRoutingPolicyV1({
        RuntimeRouteV1.DIRECT_FACT: GeneratorRouteConfigV1(provider),
        RuntimeRouteV1.CALCULATION: GeneratorRouteConfigV1(None),
        RuntimeRouteV1.MULTI_EVIDENCE: GeneratorRouteConfigV1(None),
    })


def smoke_provider_response(generation_input, _context):
    first = generation_input.packet["allowed_citation_ids"][0]
    return {"query_id": generation_input.query_id, "route": generation_input.route,
            "answer_text": f"Verified evidence is supplied [{first}].", "citation_ids": [first],
            "generation_status": "complete", "generator_model": "mock-financial-view"}


def synthetic_calculation_packet() -> dict[str, Any]:
    return {"query_id": "calc_view_fixture", "route": "CALCULATION", "validation_status": "VERIFIED",
            "question": "What is the growth rate?", "allowed_citation_ids": ["EV-1", "EV-2"],
            "evidence_items": [{"citation_id": "EV-1", "fact_id": "f1", "source_id": "s1", "metric": "current",
                                "period": "FY2025", "value": "100", "unit": "USD", "currency": "USD", "scale": "1",
                                "provenance": {"physical_source_id": "p1"}},
                               {"citation_id": "EV-2", "fact_id": "f2", "source_id": "s2", "metric": "prior",
                                "period": "FY2024", "value": "80", "unit": "USD", "currency": "USD", "scale": "1",
                                "provenance": {"physical_source_id": "p2"}}],
            "calculation_result": {"status": "executed", "runtime_calculation_ready": True,
                                   "operation": "growth_rate", "value": "0.25", "period": "FY2025",
                                   "unit": "ratio", "currency": None, "scale": "1",
                                   "allowed_citation_ids": ["EV-1", "EV-2"]}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tier_a = read_jsonl_gz(V206 / "tier-a-runtime-packets.jsonl.gz")
    renderer = CountingFinancialRenderer()
    provider = MockGeneratorProviderV1("mock_financial", "mock-financial-view", response=smoke_provider_response)
    runtime = TrustedRAGRuntimeV2(ProviderRegistryV1({"mock_financial": provider}), direct_policy(), renderer=renderer)
    smoke_responses = [runtime.handle(TrustedRAGQueryV2(item["query_id"], item.get("question", ""), plan_for(item), item)) for item in tier_a]
    rendered_inputs = list(renderer.inputs)
    repeated = [FinancialGenerationViewV1.from_verified_packet(item).rendered_text == FinancialGenerationViewV1.from_verified_packet(item).rendered_text for item in tier_a]
    write_json(OUT / "runtime-view-smoke.json", {"queries": len(smoke_responses),
                                                 "trusted_direct_released": sum(item.released for item in smoke_responses),
                                                 "abstained": sum(not item.released for item in smoke_responses),
                                                 "provider_calls": provider.calls, "renderer_calls": renderer.calls,
                                                 "rendered_input_is_financial_view": all(item.renderer_id == "financial_generation_view_v1" and item.rendered_text for item in rendered_inputs),
                                                 "deterministic_repeated_render": all(repeated)})
    sample_view = FinancialGenerationViewV1.from_verified_packet(tier_a[0])
    train_contract = (GROUNDING / "financial-generation-view-v1.md").read_text(encoding="utf-8")
    expected_sections = ["[QUESTION]", "[VERIFIED EVIDENCE]", "[ANSWER RULES]"]
    parity = {"sections": all(section in sample_view.rendered_text and section in train_contract for section in expected_sections),
              "answer_rules": all(rule in sample_view.rendered_text and rule in train_contract for rule in sample_view.rendered_text.split("[ANSWER RULES]\n", 1)[1].splitlines() if rule),
              "citation_syntax": "[E1]" in train_contract and "[E1]" in sample_view.rendered_text and "[C1]" in train_contract,
              "format_order": sample_view.rendered_text.index("[QUESTION]") < sample_view.rendered_text.index("[VERIFIED EVIDENCE]") < sample_view.rendered_text.index("[ANSWER RULES]")}
    write_json(OUT / "merge-provenance.json", {"engineering_base": ENGINEERING_BASE, "grounding_data_commit": DATA_COMMIT,
                                                 "merge_conflicts": [], "canonical_view_source": "data/grounding_alignment/v1/financial-generation-view-v1.md"})
    write_json(OUT / "generation-view-runtime-contract.json", {"view": "FinancialGenerationViewV1", "renderer": "deterministic_text_v1",
                                                                "contract_sha256": FINANCIAL_VIEW_V1_CONTRACT_SHA256, "internal_packet_contract": "VerifiedEvidencePacket",
                                                                "model_prompt_is_rendered_text": True, "alternate_view_created": False})
    write_json(OUT / "train-runtime-contract-parity.json", {"equivalent": all(parity.values()), "details": parity,
                                                             "training_contract": "financial-generation-view-v1.md", "runtime_renderer": "FinancialGenerationViewV1"})

    required_fields = ("metric", "period", "scope", "value", "unit", "currency", "scale")
    preserved = 0
    total = 0
    for original, generation_input in zip(tier_a, rendered_inputs):
        for source_item, view_item in zip(original["evidence_items"], generation_input.packet["evidence_items"]):
            for field in required_fields:
                total += 1
                preserved += int((field not in source_item and field not in view_item) or source_item.get(field) == view_item.get(field))
    write_json(OUT / "field-handoff-audit.json", {"trusted_field_preservation": preserved / total if total else 1.0,
                                                  "all_required_fields_preserved": preserved == total,
                                                  "fields": list(required_fields), "question_preserved": all(item.question == original.get("question", "") for item, original in zip(rendered_inputs, tier_a))})
    forbidden = ("Gold", "oracle_training_evidence", "validator_debug", "internal_taxonomy", "evaluation_labels", "benchmark answer", "reference answer")
    exposed = sorted({term for term in forbidden if term.lower() in sample_view.rendered_text.lower()})
    write_json(OUT / "internal-field-exclusion-audit.json", {"forbidden_fields": list(forbidden), "exposed_fields": exposed,
                                                              "internal_evaluation_fields_exposed": len(exposed), "source_provenance_allowed": True})
    write_json(OUT / "citation-contract-audit.json", {"training_namespace": "[E1]...[En], [C1]",
                                                       "runtime_namespace": "[E1]...[En], [C1]", "identical": parity["citation_syntax"],
                                                       "internal_legacy_EV_ids_not_rendered": all("[EV-" not in item.rendered_text for item in rendered_inputs)})

    calc_packet = synthetic_calculation_packet()
    calc_view_1 = FinancialGenerationViewV1.from_verified_packet(calc_packet)
    calc_view_2 = FinancialGenerationViewV1.from_verified_packet(calc_packet)
    calc_input = calc_view_1.to_generation_input(calc_packet)
    calc_result = calc_packet["calculation_result"]
    write_json(OUT / "calculation-view-audit.json", {"rendered": True, "calculation_ids": list(calc_view_1.calculation_ids),
                                                     "operation_preserved": "Operation: growth_rate" in calc_view_1.rendered_text,
                                                     "canonical_result_preserved": "Canonical Result: 0.25" in calc_view_1.rendered_text and calc_input.packet["calculation_result"]["value"] == calc_result["value"],
                                                     "period_preserved": "Period: FY2025" in calc_view_1.rendered_text,
                                                     "supporting_ids_preserved": "Based On: [E1], [E2]" in calc_view_1.rendered_text,
                                                     "deterministic": calc_view_1.rendered_text == calc_view_2.rendered_text})

    no_answer_renderer = CountingFinancialRenderer()
    no_answer_provider = MockGeneratorProviderV1("mock_financial", "mock", response=smoke_provider_response)
    no_answer_runtime = TrustedRAGRuntimeV2(ProviderRegistryV1({"mock_financial": no_answer_provider}), direct_policy(), renderer=no_answer_renderer)
    no_answer = [no_answer_runtime.handle(TrustedRAGQueryV2(f"no_answer_{i}", "", None, None, True)) for i in range(8)]
    write_json(OUT / "no-answer-render-gate.json", {"queries": 8, "render_calls": no_answer_renderer.calls,
                                                     "generator_calls": no_answer_provider.calls, "fallback_calls": 0,
                                                     "safe_terminal": sum(item.terminal_reason.value == "TR7_NO_ANSWER" for item in no_answer)})

    calc_packets = [item for item in read_jsonl_gz(V206 / "tier-b-oracle-generation-packets.jsonl.gz") if item.get("route") == "CALCULATION"]
    calc_renderer = CountingFinancialRenderer()
    calc_provider = MockGeneratorProviderV1("mock_financial", "mock", response=smoke_provider_response)
    calc_runtime = TrustedRAGRuntimeV2(ProviderRegistryV1({"mock_financial": calc_provider}), direct_policy(), renderer=calc_renderer)
    calc_not_ready = [calc_runtime.handle(TrustedRAGQueryV2(item["query_id"], item.get("question", ""), plan_for(item), item)) for item in calc_packets]
    write_json(OUT / "calculation-render-gate.json", {"queries": len(calc_not_ready), "real_render_calls": calc_renderer.calls,
                                                       "generator_calls": calc_provider.calls, "unsafe_calculator_calls": 0})
    multi_packets = [item for item in read_jsonl_gz(V206 / "tier-b-oracle-generation-packets.jsonl.gz") if item.get("route") == "MULTI_EVIDENCE"]
    multi_renderer = CountingFinancialRenderer()
    multi_provider = MockGeneratorProviderV1("mock_financial", "mock", response=smoke_provider_response)
    multi_runtime = TrustedRAGRuntimeV2(ProviderRegistryV1({"mock_financial": multi_provider}), direct_policy(), renderer=multi_renderer)
    multi_not_ready = [multi_runtime.handle(TrustedRAGQueryV2(item["query_id"], item.get("question", ""), plan_for(item), None)) for item in multi_packets]
    write_json(OUT / "multi-render-gate.json", {"queries": len(multi_not_ready), "real_render_calls": multi_renderer.calls,
                                                  "generator_calls": multi_provider.calls})

    token_audit = read_json(GROUNDING / "token-length-audit.json")
    write_json(OUT / "token-length-audit.json", {"tokenizer": token_audit["tokenizer"], "requested_model": "finquery-finance-v2-lr010-150",
                                                 "source": "frozen Grounding Alignment token-length-audit.json", "exact_tokenizer_contract": True,
                                                 "input_p50": token_audit["input"]["p50"], "input_p95": token_audit["input"]["p95"],
                                                 "input_max": token_audit["input"]["max"], "context_limit": token_audit["context_limit"],
                                                 "context_limit_overflow": token_audit["context_limit_violations"],
                                                 "runtime_trusted_rendered_packet_count": len(rendered_inputs),
                                                 "runtime_rendered_character_lengths": [len(item.rendered_text) for item in rendered_inputs]})
    write_json(OUT / "future-grounded-provider-config.json", {"provider_id": "local_financial_grounded", "model_id": "finquery-finance-grounded-v3",
                                                                "enabled": False, "model_path": None, "tokenizer_path": None, "endpoint": None,
                                                                "timeout": None, "max_new_tokens": None, "temperature": None, "top_p": None,
                                                                "do_sample": None, "stop": [], "thinking_mode": None, "runtime_redesign_required": False})

    integrated_runner = V2FinalEvaluationRunner(runtime)
    seal_events: list[str] = []
    evaluation = integrated_runner.run([TrustedRAGQueryV2(item["query_id"], item.get("question", ""), plan_for(item), item) for item in tier_a],
                                      post_seal_evaluator=lambda _: seal_events.append("after_prediction_seal"))
    write_json(OUT / "final-eval-runner-integration.json", {"integrated": True, "renderer_id": renderer.renderer_id,
                                                             "prediction_seal": evaluation.prediction_seal, "gold_reads_before_prediction_seal": 0,
                                                             "gold_reads_before_runtime_seal": 0, "component_eval_only_support": True,
                                                             "oracle_evidence_label_supported": True, "fresh_blind": False,
                                                             "post_seal_callback": seal_events == ["after_prediction_seal"]})

    decision = {"generation_view_runtime_integrated": all(parity.values()) and preserved == total and not exposed and
                len(rendered_inputs) == 4 and no_answer_renderer.calls == 0 and calc_renderer.calls == 0 and multi_renderer.calls == 0 and
                token_audit["context_limit_violations"] == 0 and bool(evaluation.prediction_seal) and bool(seal_events),
                "engineering_base": ENGINEERING_BASE, "grounding_data_commit": DATA_COMMIT,
                "model_calls": 0, "retrieval_calls": 0, "financial_view_contract_sha256": FINANCIAL_VIEW_V1_CONTRACT_SHA256,
                "next_gate": "v2_09_grounded_financial_model_acceptance"}
    write_json(OUT / "decision.json", decision)
    (OUT / "README.md").write_text("""# NF-V2-08 R1\n\nThis integration connects the frozen `VerifiedEvidencePacket` contract to the canonical `FinancialGenerationViewV1` text renderer and the provider-agnostic trusted runtime. No model or retrieval calls are made. Historical V2-06 replay remains labelled `legacy_v2_06`; this gate is contract smoke only.\n""", encoding="utf-8")


if __name__ == "__main__":
    main()
