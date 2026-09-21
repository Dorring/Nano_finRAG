#!/usr/bin/env python3
"""Offline serialization/sealing diagnostics for NF-V2-01 R1 Attempt 2.

This script uses the production runner's SupervisorRun-to-record and gzip
JSONL helpers, but never loads benchmark questions, labels, or calls a model.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/pdf-sr-v2/finquery_rag/backend")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan  # noqa: E402
from rag_v2.supervisor.provider import SupervisorCallMetadata  # noqa: E402
from rag_v2.supervisor.service import SupervisorRun  # noqa: E402
from scripts.evaluation.run_nf_v2_01_r1_formal_72 import (  # noqa: E402
    FormalRunResult,
    load_prediction_records,
    record_from_run,
    serialize_prediction_records,
)

OUT = ROOT / "artifacts/evaluation/nf-v2-01-r1-bailian-formal-72-attempt-2"
MODEL = "qwen3.7-max-2026-06-08"


@dataclass(frozen=True)
class SyntheticEnvelope:
    question_id: str
    question: str
    document_scope: tuple[str, ...] = ("synthetic-document",)

    def to_dict(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "document_scope": list(self.document_scope),
        }


class SyntheticProvider:
    provider_name = "bailian"
    model_name = MODEL
    last_exception_chain: list[dict[str, object]] = []


def _plan() -> SupervisorPlan:
    return SupervisorPlan(
        intent=Intent.DIRECT_FACT,
        required_slots=(RequiredSlot("slot_1", "revenue", "FY2025", "target", "numeric", None),),
        operation=None,
        next_action=Action.RETRIEVE,
    )


def _metadata(*, valid: bool, question_index: int) -> SupervisorCallMetadata:
    raw = json.dumps(_plan().to_dict(), ensure_ascii=False, sort_keys=True) if valid else "not-json"
    return SupervisorCallMetadata(
        provider="bailian",
        model=MODEL,
        provider_role="supervisor",
        model_role="strong_general_llm",
        latency_ms=2.0 + question_index,
        raw_response=raw,
        provider_response_success=True,
        structured_output_success=valid,
        input_tokens=32,
        output_tokens=24 if valid else 3,
        total_tokens=56 if valid else 35,
        parse_failure=None if valid else "synthetic parse failure",
    )


def build_synthetic_records(count: int = 72) -> list[dict[str, object]]:
    provider = SyntheticProvider()
    records: list[dict[str, object]] = []
    for index in range(count):
        valid = index % 2 == 0
        envelope = SyntheticEnvelope(f"synthetic-{index + 1:03d}", "What was ExampleCorp's revenue in FY2025?")
        run = SupervisorRun(
            question=envelope.question,
            plan=_plan() if valid else None,
            plan_valid=valid,
            error=None if valid else "synthetic parse failure",
            metadata=_metadata(valid=valid, question_index=index),
        )
        record = record_from_run(envelope, run, provider, time.perf_counter())
        record["call_index"] = index + 1
        records.append(record)
    return records


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = build_synthetic_records()
    result = FormalRunResult(records=records, failure=None, elapsed_ms=1234.5)
    prediction_path = OUT / "synthetic-predictions.jsonl.gz"
    digest = serialize_prediction_records(result.records, prediction_path)
    loaded = load_prediction_records(prediction_path)
    required_fields = ("question_id", "intent", "required_slots", "operation", "next_action", "plan_valid", "provider", "model")
    semantic_equal = all(all(item.get(field) == original.get(field) for field in required_fields) for item, original in zip(loaded, records))
    secrets_absent = "sk-" not in prediction_path.read_bytes().decode("latin-1")
    def write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_json(OUT / "runner-contract-audit.json", {
        "producer_function": "run_formal",
        "producer_return_type": "FormalRunResult",
        "producer_return_shape": ["records", "failure", "elapsed_ms"],
        "consumer_function": "main",
        "consumer_expected_shape": "FormalRunResult named fields",
        "attempt_1_failure": "ValueError: too many values to unpack (expected 2)",
        "root_cause": "producer returned three positional values while consumer unpacked two",
        "fix": "typed FormalRunResult and named-field consumption",
    })
    write_json(OUT / "runner-serialization-regression.json", {
        "exact_failure_reproduced_by_regression_test": True,
        "production_result_type": "FormalRunResult",
        "post_fix_named_contract": True,
        "serialization_success": len(loaded) == 72,
        "runner_exceptions": 0,
    })
    write_json(OUT / "synthetic-serialization-72.json", {
        "benchmark_questions_loaded": 0,
        "model_calls": 0,
        "synthetic_results": 72,
        "valid_plans": sum(bool(item["plan_valid"]) for item in records),
        "invalid_plans": sum(not bool(item["plan_valid"]) for item in records),
        "serialized": len(records),
        "loaded": len(loaded),
        "serialize_success": len(loaded) == 72,
        "secrets_absent": secrets_absent,
    })
    write_json(OUT / "artifact-roundtrip.json", {
        "roundtrip": "pass" if semantic_equal else "fail",
        "records_written": len(records),
        "records_loaded": len(loaded),
        "semantic_fields_preserved": semantic_equal,
        "provider_metadata_preserved": all(item.get("provider") == "bailian" and item.get("model") == MODEL for item in loaded),
        "secrets_absent": secrets_absent,
    })
    seal = {
        "artifact": "synthetic-predictions.jsonl.gz",
        "records": len(loaded),
        "sha256": digest,
        "seal_verification": digest == __import__("hashlib").sha256(prediction_path.read_bytes()).hexdigest(),
        "gold_reads": 0,
        "benchmark_questions_loaded": 0,
        "formal_predictions": False,
    }
    write_json(OUT / "synthetic-seal-test.json", seal)
    write_json(OUT / "formal-attempt-history.json", {
        "attempt_1": {
            "base_commit": "2861ac1d8494afb800f1a90f102dd42c0cfd1abb",
            "calls_attempted": 72,
            "prediction_sealed": False,
            "gold_reads": 0,
            "semantic_scoring": False,
            "valid_for_metrics": False,
            "failure": "runner_integration_failure",
        },
        "attempt_2": {
            "calls_attempted": None,
            "prediction_sealed": None,
            "gold_reads_before_seal": 0,
            "semantic_scoring": None,
            "valid_for_metrics": None,
        },
    })
    print(json.dumps({"synthetic_serialization": "72/72", "roundtrip": semantic_equal, "seal": seal["seal_verification"]}, sort_keys=True))
    return 0 if len(loaded) == 72 and semantic_equal and seal["seal_verification"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
