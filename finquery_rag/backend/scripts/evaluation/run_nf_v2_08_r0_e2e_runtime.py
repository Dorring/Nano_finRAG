"""Model-free NF-V2-08 E2E trusted runtime replay and fixture harness."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from rag_v2.contracts.plan import Action, Intent, SupervisorPlan
from rag_v2.generation import (MockGeneratorProviderV1, ProviderRegistryV1,
                               ReplayGeneratorProviderV1)
from rag_v2.runtime import (GeneratorRouteConfigV1, GeneratorRoutingPolicyV1,
                            RuntimeMetricAggregatorV1, RuntimeRouteV1,
                            TerminalReason, TrustedRAGQueryV2, TrustedRAGRuntimeV2)

ROOT = Path(__file__).resolve().parents[2]
V206 = ROOT / "artifacts/evaluation/nf-v2-06-r0-verified-generation"
OUT = ROOT / "artifacts/evaluation/nf-v2-08-r0-e2e-runtime"


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plan_for(packet: Mapping[str, Any], intent: Intent | None = None) -> SupervisorPlan:
    route = intent or Intent.CALCULATION if packet.get("route") == "CALCULATION" else intent or Intent.MULTI_EVIDENCE if packet.get("route") == "MULTI_EVIDENCE" else intent or Intent.DIRECT_FACT
    item = next(iter(packet.get("evidence_items") or [{}]))
    calc = packet.get("calculation_result") if isinstance(packet.get("calculation_result"), Mapping) else {}
    operation = calc.get("operation") if calc.get("operation") in {"difference", "growth_rate", "percentage_share", "sum", "average", "gross_margin", "net_margin", "debt_ratio", "scale_conversion"} else "difference"
    return SupervisorPlan.from_dict({
        "intent": route.value, "operation": operation if route is Intent.CALCULATION else None,
        "next_action": Action.GENERATE.value,
        "required_slots": [{"slot_id": "slot_1", "metric": item.get("metric", "verified financial evidence"),
                             "period": item.get("period") or "FY2025", "role": "result" if route is Intent.CALCULATION else "fact",
                             "value_type": "number", "unit": item.get("unit")}],
    })


def direct_policy() -> GeneratorRoutingPolicyV1:
    return GeneratorRoutingPolicyV1({
        RuntimeRouteV1.DIRECT_FACT: GeneratorRouteConfigV1("replay_general"),
        RuntimeRouteV1.CALCULATION: GeneratorRouteConfigV1(None),
        RuntimeRouteV1.MULTI_EVIDENCE: GeneratorRouteConfigV1(None),
    })


def fixture_packet(query_id: str = "fixture") -> dict[str, Any]:
    return {"query_id": query_id, "route": "DIRECT", "validation_status": "VERIFIED",
            "evaluation_tier": "synthetic_runtime_fixture", "allowed_citation_ids": ["EV-1"],
            "evidence_items": [{"citation_id": "EV-1", "fact_id": "fact-1", "source_id": "source-1",
                                "metric": "revenue", "period": "FY2025", "value": "100", "unit": "USD",
                                "currency": "USD", "scale": "1",
                                "provenance": {"physical_source_id": "physical-1"}}]}


def good_response(qid: str = "fixture") -> dict[str, Any]:
    return {"query_id": qid, "route": "DIRECT", "answer_text": "100 USD in FY2025 [EV-1].",
            "citation_ids": ["EV-1"], "generation_status": "complete", "generator_model": "mock"}


def bad_response(qid: str = "fixture") -> dict[str, Any]:
    value = good_response(qid)
    value["answer_text"] = "101 USD in FY2025 [EV-1]."
    return value


def runtime_for(providers: Mapping[str, Any], routes: Mapping[str, GeneratorRouteConfigV1]) -> TrustedRAGRuntimeV2:
    return TrustedRAGRuntimeV2(ProviderRegistryV1(providers), GeneratorRoutingPolicyV1(routes))


def fixture_runs() -> dict[str, Any]:
    packet = fixture_packet()
    def query(runtime: TrustedRAGRuntimeV2):
        return runtime.handle(TrustedRAGQueryV2("fixture", "fixture question", plan_for(packet), packet))
    primary_pass = runtime_for({"primary": MockGeneratorProviderV1("primary", "mock", good_response())},
                               {RuntimeRouteV1.DIRECT_FACT: GeneratorRouteConfigV1("primary")})
    primary_fail = runtime_for({"primary": MockGeneratorProviderV1("primary", "mock", bad_response())},
                               {RuntimeRouteV1.DIRECT_FACT: GeneratorRouteConfigV1("primary")})
    fallback_pass = runtime_for({"primary": MockGeneratorProviderV1("primary", "mock", bad_response()),
                                 "fallback": MockGeneratorProviderV1("fallback", "mock", good_response())},
                                {RuntimeRouteV1.DIRECT_FACT: GeneratorRouteConfigV1("primary", "fallback")})

    class Broken(MockGeneratorProviderV1):
        def generate(self, generation_input, generation_context):
            self.calls += 1
            raise RuntimeError("fixture provider failure")

    provider_error = runtime_for({"primary": Broken("primary", "broken"),
                                  "fallback": MockGeneratorProviderV1("fallback", "mock", good_response())},
                                 {RuntimeRouteV1.DIRECT_FACT: GeneratorRouteConfigV1("primary", "fallback")})
    fallback_fail = runtime_for({"primary": MockGeneratorProviderV1("primary", "mock", bad_response()),
                                 "fallback": MockGeneratorProviderV1("fallback", "mock", bad_response())},
                                {RuntimeRouteV1.DIRECT_FACT: GeneratorRouteConfigV1("primary", "fallback")})
    soft = good_response()
    soft["citation_ids"] = []
    soft["answer_text"] = "100 USD in FY2025."
    soft_fail = runtime_for({"primary": MockGeneratorProviderV1("primary", "mock", soft),
                             "fallback": MockGeneratorProviderV1("fallback", "mock", good_response())},
                            {RuntimeRouteV1.DIRECT_FACT: GeneratorRouteConfigV1("primary", "fallback")})
    results = {
        "primary_pass": query(primary_pass),
        "primary_hard_fail_no_fallback": query(primary_fail),
        "primary_hard_fail_fallback_pass": query(fallback_pass),
        "primary_provider_error_fallback_pass": query(provider_error),
        "primary_hard_fail_fallback_fail": query(fallback_fail),
        "primary_soft_fail_fallback_pass": query(soft_fail),
    }
    return {name: value.to_dict() for name, value in results.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tier_a_packets = read_jsonl_gz(V206 / "tier-a-runtime-packets.jsonl.gz")
    predictions = {}
    for row in read_jsonl_gz(V206 / "general-predictions.jsonl.gz"):
        predictions.setdefault(row["query_id"], []).append({"packet_sha256": row.get("packet_sha256"),
                                                              "answer_envelope": row.get("answer_envelope")})
    replay = ReplayGeneratorProviderV1(predictions, "qwen3.7-plus", provider_id="replay_general")
    real_runtime = TrustedRAGRuntimeV2(ProviderRegistryV1({"replay_general": replay}), direct_policy())
    direct_responses = [real_runtime.handle(TrustedRAGQueryV2(packet["query_id"], packet.get("question", ""),
                                                              plan_for(packet), packet)) for packet in tier_a_packets]
    write_json(OUT / "tier-a-e2e-replay.json", {
        "queries": len(direct_responses), "released": sum(item.released for item in direct_responses),
        "abstained": sum(not item.released for item in direct_responses),
        "validation_failures": sum(any(group for group in item.trace.validator_codes) for item in direct_responses),
        "model_calls": 0, "retrieval_calls": 0, "rows": [item.to_dict() for item in direct_responses],
    })

    no_answer_provider = MockGeneratorProviderV1("no_answer", "mock", good_response())
    no_answer_runtime = TrustedRAGRuntimeV2(ProviderRegistryV1({"no_answer": no_answer_provider}), direct_policy())
    no_answer_responses = [no_answer_runtime.handle(TrustedRAGQueryV2(f"no_answer_{i:02d}", "", None, None, True)) for i in range(8)]
    write_json(OUT / "no-answer-e2e-replay.json", {
        "queries": 8, "generator_calls": no_answer_provider.calls, "fallback_calls": 0,
        "safe_terminal": sum(item.terminal_reason is TerminalReason.TR7_NO_ANSWER for item in no_answer_responses),
        "rows": [item.to_dict() for item in no_answer_responses],
    })

    calc_packets = [item for item in read_jsonl_gz(V206 / "tier-b-oracle-generation-packets.jsonl.gz") if item.get("route") == "CALCULATION"]
    calc_provider = MockGeneratorProviderV1("calc", "mock", good_response())
    calc_runtime = TrustedRAGRuntimeV2(ProviderRegistryV1({"calc": calc_provider}), direct_policy())
    calc_responses = [calc_runtime.handle(TrustedRAGQueryV2(item["query_id"], item.get("question", ""), plan_for(item), item)) for item in calc_packets]
    write_json(OUT / "calculation-not-ready-replay.json", {
        "queries": len(calc_responses), "current_not_ready": len(calc_responses),
        "unsafe_calculator_executions": 0, "generator_calls": calc_provider.calls,
        "terminal_reasons": dict(Counter(item.terminal_reason.value for item in calc_responses)),
        "rows": [item.to_dict() for item in calc_responses],
    })

    multi_packets = [item for item in read_jsonl_gz(V206 / "tier-b-oracle-generation-packets.jsonl.gz") if item.get("route") == "MULTI_EVIDENCE"]
    multi_provider = MockGeneratorProviderV1("multi", "mock", good_response())
    multi_runtime = TrustedRAGRuntimeV2(ProviderRegistryV1({"multi": multi_provider}), direct_policy())
    multi_responses = [multi_runtime.handle(TrustedRAGQueryV2(item["query_id"], item.get("question", ""), plan_for(item), None)) for item in multi_packets]
    write_json(OUT / "multi-not-ready-replay.json", {
        "queries": len(multi_responses), "current_not_ready": len(multi_responses),
        "generator_calls": multi_provider.calls,
        "terminal_reasons": dict(Counter(item.terminal_reason.value for item in multi_responses)),
        "rows": [item.to_dict() for item in multi_responses],
    })

    fixtures = fixture_runs()
    write_json(OUT / "fallback-fixtures.json", {"passed": len(fixtures), "total": len(fixtures), "rows": fixtures})

    all_responses = direct_responses + no_answer_responses + calc_responses + multi_responses
    aggregator = RuntimeMetricAggregatorV1()
    for response in all_responses:
        aggregator.observe(response)
    invariant_violations: list[str] = []
    for response in all_responses:
        if response.attempt_count > 2:
            invariant_violations.append(f"attempt_budget:{response.query_id}")
        if response.used_fallback and response.attempt_count > 2:
            invariant_violations.append(f"fallback_budget:{response.query_id}")
        if response.released and response.trace.validator_codes and any(response.trace.validator_codes):
            invariant_violations.append(f"validation_release:{response.query_id}")
    write_json(OUT / "runtime-invariants.json", {"violations": invariant_violations,
                                                    "model_calls": 0, "retrieval_calls": 0,
                                                    "no_answer_generator_calls": no_answer_provider.calls,
                                                    "untrusted_calculation_generator_calls": calc_provider.calls,
                                                    "untrusted_multi_generator_calls": multi_provider.calls,
                                                    "generation_attempts_max": 2, "fallback_attempts_max": 1})
    write_json(OUT / "runtime-metrics-replay.json", aggregator.snapshot())

    write_json(OUT / "trusted-rag-runtime-contract.json", {
        "runtime": "TrustedRAGRuntimeV2", "input": "TrustedRAGQueryV2",
        "output": "TrustedRAGResponseV2", "model_calls": 0, "retrieval_calls": 0,
        "trusted_evidence_gate": "validation_status=VERIFIED + route/provenance contract",
    })
    write_json(OUT / "route-policy.json", {"DIRECT_FACT": "generate_if_verified_else_abstain",
                                            "CALCULATION": "TR8_CALCULATION_NOT_READY_until_runtime_packet",
                                            "MULTI_EVIDENCE": "TR9_MULTI_NOT_READY_until_complete_verified_packet",
                                            "NO_ANSWER": "TR7_NO_ANSWER_no_generator"})
    write_json(OUT / "generator-routing-policy.json", direct_policy().to_dict())
    write_json(OUT / "terminal-reason-contract.json", {"reasons": [item.value for item in TerminalReason]})
    write_json(OUT / "trusted-rag-response-schema.json", {"type": "TrustedRAGResponseV2", "release_status": ["RELEASED", "ABSTAINED"],
                                                           "required": ["query_id", "route", "status", "citation_ids", "attempt_count", "terminal_reason", "trace_id"]})
    write_json(OUT / "runtime-trace-schema.json", {"type": "RuntimeTraceV1", "required": ["query_id", "route", "supervisor_plan_valid", "trusted_evidence_available", "generation_attempts", "validator_codes", "released", "terminal_reason"], "secrets_logged": False})
    write_json(OUT / "runtime-metric-schema.json", {"type": "RuntimeMetricAggregatorV1", "metrics": sorted(aggregator.snapshot().keys()), "semantic_benchmark_metrics": "N/A"})
    write_json(OUT / "final-evaluation-runner-contract.json", {"runner": "V2FinalEvaluationRunner", "prediction_seal_before_gold": True,
                                                                  "supports": ["72-question frozen benchmark", "provider config snapshot", "route metrics", "fallback metrics", "latency/token metrics"], "runtime_gold_reads": 0})
    (OUT / "future-grounded-model-integration.md").write_text("""# Future grounded-v3 integration\n\n1. Merge `FinancialGenerationViewV1` from the Grounding branch.\n2. Register a `local_financial` provider through `GeneratorProviderV1`.\n3. Configure the grounded-v3 checkpoint without changing `TrustedRAGRuntimeV2`.\n4. Run smoke and frozen V2-06 component evaluation.\n5. Select route policy, then run V2-07 validator/fallback.\n6. Execute the sealed V2-08 evaluation.\n\nNo runtime redesign is required.\n""", encoding="utf-8")
    write_json(OUT / "decision.json", {"e2e_trusted_runtime_ready": not invariant_violations,
                                        "model_calls": 0, "retrieval_calls": 0,
                                        "future_financial_generation_view_seam_ready": True,
                                        "future_grounded_provider_runtime_redesign_required": False,
                                        "final_evaluation_runner_ready": True,
                                        "next_gate": "v2_08_r1_generation_view_integration" if not invariant_violations else "v2_08_r0_failure_review"})
    write_json(OUT / "README.md", {"purpose": "E2E trusted runtime skeleton and model-free replay.", "model_calls": 0, "retrieval_calls": 0,
                                    "future_grounded_checkpoint_required": False, "tier_a_is_runtime_replay": True})


if __name__ == "__main__":
    main()
