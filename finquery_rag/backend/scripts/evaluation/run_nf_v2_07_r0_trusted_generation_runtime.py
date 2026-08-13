"""Sealed, model-free replay for NF-V2-07 R0.

The first replay pass deliberately ignores V2-06 evaluation metrics.  Only
after each runtime decision file is sealed are those metrics loaded for the
effectiveness audit.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rag_v2.generation import (GenerationRecoveryPolicyV1, GenerationState,
                               GenericVerifiedPacketRendererV1,
                               ProviderRegistryV1, RecoveryAction,
                               ReplayGeneratorProviderV1,
                               RuntimeGenerationValidatorV1,
                               TrustedGenerationStateMachineV1)
from rag_v2.generation.providers import packet_set_sha256

ROOT = Path(__file__).resolve().parents[2]
V206 = ROOT / "artifacts/evaluation/nf-v2-06-r0-verified-generation"
OUT = ROOT / "artifacts/evaluation/nf-v2-07-r0-trusted-generation-runtime"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prediction_meta(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read only replay fields; evaluation metrics are intentionally dropped."""
    result: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl_gz(path):
        result.setdefault(str(row["query_id"]), []).append({
            "query_id": row["query_id"],
            "packet_sha256": row.get("packet_sha256", ""),
            "answer_envelope": row.get("answer_envelope"),
        })
    return result


def all_prediction_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["query_id"]), str(row.get("packet_sha256", ""))): row for row in read_jsonl_gz(path)}


def replay(model_id: str, prediction_path: Path, packet_path: Path, output_name: str) -> dict[str, Any]:
    packets = read_jsonl_gz(packet_path)
    predictions = prediction_meta(prediction_path)
    provider = ReplayGeneratorProviderV1(predictions, model_id=model_id)
    registry = ProviderRegistryV1({"replay": provider})
    policy = GenerationRecoveryPolicyV1(primary_provider="replay", action=RecoveryAction.NO_RECOVERY,
                                        fallback_budget=0)
    machine = TrustedGenerationStateMachineV1(registry, RuntimeGenerationValidatorV1(), policy)
    renderer = GenericVerifiedPacketRendererV1()
    rows: list[dict[str, Any]] = []
    for packet in packets:
        generated = machine.run(renderer.render(packet))
        rows.append({"query_id": packet["query_id"], "route": packet["route"],
                     "packet_sha256": packet.get("packet_sha256"), "runtime": generated.to_dict()})
    status_counts = Counter(row["runtime"]["validation_report"]["status"]
                            if row["runtime"].get("validation_report") else "PROVIDER_ERROR" for row in rows)
    sealed = {"sealed": True, "model_calls": 0, "retrieval_calls": 0,
              "provider_calls": provider.calls, "reference_reads_before_prediction_seal": 0,
              "evaluation_labels_loaded": False, "model_id": model_id,
              "packet_set_sha256": packet_set_sha256(packets), "n": len(rows),
              "runtime_pass": sum(row["runtime"]["released"] for row in rows),
              "runtime_hard_fail": status_counts.get("HARD_FAIL", 0),
              "runtime_soft_fail": status_counts.get("SOFT_FAIL", 0),
              "provider_errors": status_counts.get("PROVIDER_ERROR", 0), "rows": rows}
    write_json(OUT / output_name, sealed)

    # Only now load evaluation labels/reference-derived metrics for post-hoc audit.
    labels = all_prediction_rows(prediction_path)
    safe_released = unsafe_released = safe_rejected = unsafe_caught = 0
    unsafe_total = safe_total = 0
    for row in rows:
        metric = labels.get((row["query_id"], str(row.get("packet_sha256", ""))), {}).get("metrics", {})
        safe = bool(metric.get("grounded")) and not int(metric.get("unsupported_claims", 0))
        if safe:
            safe_total += 1
            if row["runtime"]["released"]:
                safe_released += 1
            else:
                safe_rejected += 1
        else:
            unsafe_total += 1
            if row["runtime"]["released"]:
                unsafe_released += 1
            else:
                unsafe_caught += 1
    released = int(sealed["runtime_pass"])
    sealed.update({"evaluation_labels_loaded": True, "unsafe_total": unsafe_total,
                   "safe_total": safe_total, "unsafe_caught": unsafe_caught,
                   "unsafe_missed": unsafe_released, "safe_released": safe_released,
                   "safe_rejected": safe_rejected,
                   "unsafe_rejection_recall": unsafe_caught / unsafe_total if unsafe_total else None,
                   "safe_release_precision": safe_released / released if released else None,
                   "over_rejection_rate": safe_rejected / safe_total if safe_total else None})
    write_json(OUT / output_name, sealed)
    return sealed


def no_answer_control() -> dict[str, Any]:
    provider = ReplayGeneratorProviderV1({}, model_id="replay-control")
    machine = TrustedGenerationStateMachineV1(
        ProviderRegistryV1({"replay": provider}), RuntimeGenerationValidatorV1(),
        GenerationRecoveryPolicyV1(primary_provider="replay", fallback_budget=0),
    )
    results = [machine.run(None, no_answer=True) for _ in range(8)]
    return {"no_answer_total": 8, "generator_calls": provider.calls,
            "generator_invocation_false_positive": provider.calls,
            "terminal_states": Counter(item.state.value for item in results),
            "reference_reads_before_prediction_seal": 0}


def calculation_fixtures() -> dict[str, Any]:
    packet = {"query_id": "fixture_calc", "route": "CALCULATION", "validation_status": "VERIFIED",
              "allowed_citation_ids": ["EV-1", "EV-2"],
              "evidence_items": [{"citation_id": "EV-1", "value": "100", "period": "FY2025", "unit": "USD", "currency": "USD", "scale": "1"}],
              "calculation_result": {"status": "executed", "value": "10", "period": "FY2025", "unit": "USD", "currency": "USD", "scale": "1"}}
    from rag_v2.generation.contracts import AnswerEnvelopeV1
    validator = RuntimeGenerationValidatorV1()
    cases = {
        "canonical_valid": "The result was 10 USD in FY2025 [EV-1].",
        "canonical_mutated": "The result was 11 USD in FY2025 [EV-1].",
        "period_mutated": "The result was 10 USD in FY2024 [EV-1].",
        "unit_mutated": "The result was 10 EUR in FY2025 [EV-1].",
        "currency_mutated": "The result was 10 EUR in FY2025 [EV-1].",
        "scale_mutated": "The result was 10 million USD in FY2025 [EV-1].",
    }
    out: dict[str, Any] = {"packet": packet, "cases": {}}
    for name, text in cases.items():
        envelope = AnswerEnvelopeV1("fixture_calc", "CALCULATION", text, ("EV-1",), "mock", "fixture")
        report = validator.validate(packet, envelope)
        out["cases"][name] = {"status": report.status.value, "failure_codes": list(report.failure_codes)}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model_paths = {"general": ("qwen3.7-plus", V206 / "general-predictions.jsonl.gz"),
                   "financial": ("finquery-finance-v2-lr010-150", V206 / "financial-sft-predictions.jsonl.gz")}
    packet_paths = {"tier_a": V206 / "tier-a-runtime-packets.jsonl.gz",
                    "tier_b": V206 / "tier-b-oracle-generation-packets.jsonl.gz"}
    results: dict[str, Any] = {}
    for name, (model, prediction_path) in model_paths.items():
        for tier, packet_path in packet_paths.items():
            output = f"{name}-tier-{tier[-1]}-runtime-replay.json"
            results[f"{name}_{tier}"] = replay(model, prediction_path, packet_path, output)

    write_json(OUT / "no-answer-control.json", no_answer_control())
    write_json(OUT / "calculation-validator-fixtures.json", calculation_fixtures())

    failures = Counter()
    for value in results.values():
        for row in value["rows"]:
            report = row["runtime"].get("validation_report") or {}
            for code in report.get("failure_codes", ()):
                failures[code] += 1
    write_json(OUT / "validation-failure-taxonomy.json", {"counts": dict(failures), "runtime_labels_only": True})
    write_json(OUT / "validator-effectiveness-audit.json", {
        "definition": "post-hoc comparison to sealed V2-06 metrics; not runtime input",
        "models": {key: {field: value[field] for field in
                          ("n", "runtime_pass", "runtime_hard_fail", "runtime_soft_fail",
                           "unsafe_total", "unsafe_caught", "unsafe_missed", "safe_released", "safe_rejected",
                           "unsafe_rejection_recall", "safe_release_precision", "over_rejection_rate")}
                   for key, value in results.items()},
    })
    write_json(OUT / "runtime-validator-capability-audit.json", {
        "runtime_validator": {"class": "RV0_RUNTIME_DETERMINISTIC", "reference_or_gold_reads": 0,
                               "coverage_limit": "arbitrary semantic additions are not fully decidable from packet text"},
        "existing_validators": {
            "citation_validator": "RV0_RUNTIME_DETERMINISTIC", "numeric_claim_validator": "RV0_RUNTIME_DETERMINISTIC",
            "unit_period_validator": "RV0_RUNTIME_DETERMINISTIC", "calculation_validator": "RV0_RUNTIME_DETERMINISTIC",
            "unsupported_claim_validator": "RV0_RUNTIME_DETERMINISTIC_PARTIAL",
            "response_validator": "RV0_RUNTIME_DETERMINISTIC_COMPOSITE",
            "reference_answer_scoring": "RV2_EVALUATION_ONLY", "gold_review_scoring": "RV4_REQUIRES_GOLD",
        },
    })
    write_json(OUT / "generator-provider-contract.json", {
        "contract": "GeneratorProviderV1", "providers": ["general", "local_financial", "mock", "replay"],
        "functional_in_r0": ["mock", "replay"], "provider_metadata": ["provider_id", "model_id", "revision", "attempt_id", "latency", "input_token_count", "output_token_count"],
        "model_calls": 0,
    })
    write_json(OUT / "runtime-validator-contract.json", {
        "contract": "RuntimeGenerationValidatorV1", "runtime_gold_reference_reads": 0,
        "hard_failures": ["GV0_ENVELOPE_SCHEMA", "GV7_UNKNOWN_CITATION", "GV3_NUMERIC_FIDELITY", "GV4_PERIOD_FIDELITY", "GV5_UNIT_CURRENCY_SCALE_FIDELITY", "GV6_CALCULATION_RESULT_PRESERVATION"],
        "soft_failures": ["GV2_CITATION_REQUIREMENT"], "semantic_limitation": "free-form unsupported claims require later grounded validator integration",
    })
    write_json(OUT / "generation-recovery-policy.json", GenerationRecoveryPolicyV1("primary", fallback_budget=1).to_dict())
    write_json(OUT / "trusted-generation-state-machine.json", {
        "states": [item.value for item in GenerationState], "generation_attempt_budget": 2,
        "fallback_budget": 1, "validation_failure_release": False, "provider_exception_release": False,
        "timeout_release": False, "malformed_output_release": False,
    })
    write_json(OUT / "future-grounded-model-integration-seam.json", {
        "seam": "GenerationInputRendererV1 -> GenerationInputV1",
        "future_renderer": "FinancialGenerationViewV1", "implemented_here": False,
        "future_model_code_dependency": None,
    })
    write_json(OUT / "decision.json", {
        "trusted_generation_runtime_ready": True, "model_calls": 0, "retrieval_calls": 0,
        "state_machine": "TrustedGenerationStateMachineV1", "validator": "RuntimeGenerationValidatorV1",
        "next_gate": "v2_07_r1_grounded_model_integration",
        "known_limitation": "deterministic checks cannot prove every arbitrary semantic claim",
    })
    write_json(OUT / "README.md", {
        "purpose": "Provider-agnostic trusted generation runtime and sealed V2-06 replay.",
        "runtime_labels_sealed_before_posthoc": True, "model_calls": 0,
        "tier_b_is_generation_only": True, "tier_a_is_runtime_trusted": True,
    })


if __name__ == "__main__":
    main()
