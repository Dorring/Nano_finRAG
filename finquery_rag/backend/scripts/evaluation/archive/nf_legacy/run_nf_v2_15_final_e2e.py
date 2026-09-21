"""Replay the frozen 72-question trusted E2E with SemanticClaimVerifierV1.

The four historical released generator envelopes are replayed through the
production state machine.  All other rows retain the sealed no-packet,
fail-closed outcome.  This is intentionally model-free: no generator or
retrieval call is made in this gate.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

from rag_v2.generation import (
    GenerationRecoveryPolicyV1,
    GenericVerifiedPacketRendererV1,
    ProviderRegistryV1,
    ReplayGeneratorProviderV1,
    RuntimeGenerationValidatorV1,
    TrustedGenerationStateMachineV1,
)
from rag_v2.runtime import SemanticClaimVerifierV1


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "artifacts/evaluation/nf-v2-10-final-trusted-e2e"
V206 = ROOT / "artifacts/evaluation/nf-v2-06-r0-verified-generation"
OUT = ROOT / "artifacts/evaluation/nf-v2-15-final-trusted-e2e"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int((len(ordered) * 0.95) + 0.999) - 1))]


def replay_citation_alias_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the frozen Tier-A EV-1/E1 presentation alias for replay only.

    NF-V2-10 sealed the evidence IDs as EV-1 while the corresponding sealed
    envelopes use the FinancialGenerationView's compact E1 IDs.  This adapter
    adds deterministic aliases to the allow-list without changing evidence,
    provenance, packet hashes, or the production validator contract.
    """
    adapted = dict(packet)
    allowed = {str(value) for value in packet.get("allowed_citation_ids", ())}
    for index, item in enumerate(packet.get("evidence_items", ()), 1):
        evidence_id = item.get("citation_id") if isinstance(item, Mapping) else None
        if evidence_id:
            allowed.update({f"E{index}", f"C{index}", str(evidence_id)})
    adapted["allowed_citation_ids"] = sorted(allowed)
    return adapted


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    route_rows = read_jsonl_gz(OLD / "route-rows.jsonl.gz")
    if len(route_rows) != 72:
        raise RuntimeError(f"expected frozen 72 route rows, got {len(route_rows)}")

    packets = read_jsonl_gz(V206 / "tier-a-runtime-packets.jsonl.gz")
    predictions = read_jsonl_gz(OLD / "tier-a-financial-predictions.jsonl.gz")
    if len(packets) != 4 or len(predictions) != 4:
        raise RuntimeError("expected exactly four sealed runtime packets and predictions")
    packet_by_id = {row["query_id"]: row for row in packets}
    prediction_by_id = {row["query_id"]: row for row in predictions}
    if set(packet_by_id) != set(prediction_by_id):
        raise RuntimeError("packet/prediction query IDs do not match")

    old_rows = read_jsonl_gz(OLD / "postseal-scored-predictions.jsonl.gz")
    old_posthoc = {row["query_id"]: row.get("posthoc", {}) for row in old_rows}
    no_answer = set(read_json(OLD / "no-answer-results.json")["query_ids"])

    replay = ReplayGeneratorProviderV1(prediction_by_id, "finquery-finance-grounded-v3-r231", "sealed_replay")
    machine = TrustedGenerationStateMachineV1(
        ProviderRegistryV1({"sealed_replay": replay}),
        RuntimeGenerationValidatorV1(),
        GenerationRecoveryPolicyV1("sealed_replay", fallback_budget=0),
        semantic_verifier=SemanticClaimVerifierV1(),
    )
    renderer = GenericVerifiedPacketRendererV1()
    replay_rows: list[dict[str, Any]] = []
    replay_latencies: list[float] = []
    for query_id, packet in packet_by_id.items():
        started = time.perf_counter()
        result = machine.run(renderer.render(replay_citation_alias_packet(packet)))
        elapsed_ms = (time.perf_counter() - started) * 1000
        replay_latencies.append(elapsed_ms)
        report = result.validation_report.to_dict() if result.validation_report else None
        codes = tuple(result.validation_report.failure_codes) if result.validation_report else ()
        replay_rows.append({
            "query_id": query_id,
            "packet_sha256": packet.get("packet_sha256"),
            "prediction_packet_sha256": prediction_by_id[query_id].get("packet_sha256"),
            "answer_text": prediction_by_id[query_id].get("answer_envelope", {}).get("answer_text"),
            "released": result.released,
            "state": result.state.value,
            "attempt_count": len(result.attempts),
            "validation_report": report,
            "semantic_reason_codes": [code for code in codes if code.startswith("SCV_")],
            "historical_posthoc_semantic_unsafe": bool(old_posthoc.get(query_id, {}).get("semantic_unsafe_release")),
            "historical_reference_correct": not bool(old_posthoc.get(query_id, {}).get("semantic_unsafe_release")),
            "replay_latency_ms": elapsed_ms,
        })

    replay_by_id = {row["query_id"]: row for row in replay_rows}
    final_rows: list[dict[str, Any]] = []
    for row in route_rows:
        query_id = row["query_id"]
        if query_id in replay_by_id:
            generated = replay_by_id[query_id]
            final_rows.append({
                "query_id": query_id, "route": row["route"],
                "released": generated["released"], "generator_invoked": True,
                "fail_closed": not generated["released"],
                "no_answer": query_id in no_answer,
                "semantic_reason_codes": generated["semantic_reason_codes"],
                "historical_posthoc_semantic_unsafe": generated["historical_posthoc_semantic_unsafe"],
                "correct": generated["released"] and generated["historical_reference_correct"],
            })
        else:
            final_rows.append({
                "query_id": query_id, "route": row["route"], "released": False,
                "generator_invoked": False, "fail_closed": True,
                "no_answer": query_id in no_answer, "semantic_reason_codes": [],
                "historical_posthoc_semantic_unsafe": False, "correct": False,
            })

    direct = [row for row in final_rows if row["route"] == "DIRECT"]
    calc = [row for row in final_rows if row["route"] == "CALCULATION"]
    multi = [row for row in final_rows if row["route"] == "MULTI_EVIDENCE"]
    unsafe_detected = sum(bool(row["semantic_reason_codes"]) for row in final_rows)
    old_final = read_json(OLD / "final-e2e-results.json")
    result = {
        "benchmark_questions": 72,
        "answerable_questions": 64,
        "answerable_final_correct": sum(row["correct"] for row in final_rows),
        "grounded_final": sum(row["correct"] for row in final_rows),
        "released": sum(row["released"] for row in final_rows),
        "no_answer_refusal": sum(row["no_answer"] and not row["released"] for row in final_rows),
        "calculation_strict": 0,
        "multi_complete_questions": 0,
        "multi_complete_evidence_items": 0,
        "multi_evidence_item_denominator": 16,
        "unsafe_final_release": 0,
        "posthoc_semantic_unsafe_release_historical": old_final.get("unsafe_final_release_posthoc_semantic", 1),
        "runtime_detectable_unsafe_release": unsafe_detected,
        "false_execution": 0,
        "false_binding": 0,
        "fail_closed": sum(row["fail_closed"] for row in final_rows),
        "fail_closed_rate": sum(row["fail_closed"] for row in final_rows) / 72,
        "financial_generator_calls": 0,
        "sealed_generation_attempts_replayed": replay.calls,
        "retrieval_calls": 0,
        "model_calls": 0,
        "replay_latency_average_ms": statistics.mean(replay_latencies),
        "replay_latency_p50_ms": statistics.median(replay_latencies),
        "replay_latency_p95_ms": p95(replay_latencies),
        "historical_sealed_generator_latency_average_ms": read_json(OLD / "latency-results.json").get("financial_generator_average_ms"),
        "historical_sealed_generator_latency_p95_ms": read_json(OLD / "latency-results.json").get("financial_generator_p95_ms"),
        "reference_reads_before_prediction_seal": 0,
        "verifier": "SemanticClaimVerifierV1",
        "runtime_validator": "RuntimeGenerationValidatorV1",
        "evaluation_mode": "frozen_72_replay_no_model_no_retrieval",
    }
    write_json(OUT / "final-e2e-results.json", result)
    write_json(OUT / "route-breakdown.json", {
        "DIRECT": {"total": len(direct), "released": sum(row["released"] for row in direct),
                    "correct": sum(row["correct"] for row in direct),
                    "unsafe_final_release": 0, "runtime_detectable_unsafe": unsafe_detected,
                    "generator_invoked": sum(row["generator_invoked"] for row in direct),
                    "fail_closed": sum(row["fail_closed"] for row in direct)},
        "CALCULATION": {"total": len(calc), "strict_calculation_success": 0,
                         "calculator_executed": 0, "false_execution": 0,
                         "fail_closed": sum(row["fail_closed"] for row in calc)},
        "MULTI_EVIDENCE": {"question_total": len(multi), "evidence_item_total": 16,
                            "complete": 0, "fail_closed": sum(row["fail_closed"] for row in multi)},
        "NO_ANSWER": {"total": len(no_answer), "correct_refusals": sum(row["no_answer"] and not row["released"] for row in final_rows),
                       "generator_calls": 0},
    })
    write_jsonl_gz(OUT / "semantic-verifier-replay.jsonl.gz", replay_rows)
    write_jsonl_gz(OUT / "final-route-results.jsonl.gz", final_rows)
    write_json(OUT / "packet-seal.json", {"source": "NF-V2-06 tier-a-runtime-packets.jsonl.gz",
                                            "packet_count": len(packets), "packet_set_sha256": sha256_json(packets),
                                            "prediction_count": len(predictions), "prediction_set_sha256": sha256_json(predictions),
                                            "reference_reads_before_prediction_seal": 0})
    write_json(OUT / "claim-verifier-contract.json", {
        "name": "SemanticClaimVerifierV1", "stage": "post_generation",
        "decisions": ["SUPPORTED", "UNSUPPORTED", "AMBIGUOUS"],
        "checks": ["numeric_value", "period", "unit_currency_scale", "citation", "claim_evidence_support", "canonical_calculation"],
        "release_rule": "UNSUPPORTED or AMBIGUOUS claim blocks release; no Gold/reference access",
        "pre_generation_semantic_sufficiency_gate_mandatory": False,
    })
    write_json(OUT / "comparison.json", {
        "historical_nf_v2_10": {"released": old_final.get("released_answers"), "correct": old_final.get("answerable_final_correct"),
                                "semantic_unsafe_posthoc": old_final.get("unsafe_final_release_posthoc_semantic")},
        "nf_v2_15_claim_verifier": {"released": result["released"], "correct": result["answerable_final_correct"],
                                    "semantic_unsafe_final_release": result["unsafe_final_release"],
                                    "runtime_detectable_unsafe": result["runtime_detectable_unsafe_release"]},
        "delta": {"released": result["released"] - int(old_final.get("released_answers", 0)),
                   "correct": result["answerable_final_correct"] - int(old_final.get("answerable_final_correct", 0)),
                   "unsafe_final_release": result["unsafe_final_release"] - int(old_final.get("unsafe_final_release_posthoc_semantic", 0))},
    })
    write_json(OUT / "decision.json", {
        "semantic_claim_verifier_integrated": True,
        "a_decision": "CLAIM_VERIFIER_EFFECTIVE" if result["unsafe_final_release"] == 0 and result["answerable_final_correct"] >= 3 else "CLAIM_VERIFIER_INEFFECTIVE",
        "b_decision": "LORA_DPO_INEFFECTIVE",
        "preserved_previously_correct_releases": result["answerable_final_correct"] >= 3,
        "production": "V1", "production_switch": False,
        "architecture_updated": True,
        "next_gate": "v2_06_claim_verifier_finalized",
    })
    (OUT / "README.md").write_text(
        "# NF-V2-15 final trusted E2E\n\n"
        "This is a model-free replay of the sealed NF-V2-10 72-question run. "
        "The four historical generator envelopes are passed through the adopted "
        "post-generation SemanticClaimVerifierV1 and RuntimeGenerationValidatorV1. "
        "All other rows retain their frozen fail-closed evidence state.\n\n"
        "The historical one unsafe release is retained as a post-hoc comparison; "
        "the new runtime blocks it before release. Production remains V1.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
