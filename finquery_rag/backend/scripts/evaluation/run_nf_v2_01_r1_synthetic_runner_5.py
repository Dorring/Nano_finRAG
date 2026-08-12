#!/usr/bin/env python3
"""Five-question, non-benchmark end-to-end runner/serialization smoke."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/pdf-sr-v2/finquery_rag/backend")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.supervisor.bailian_provider import BailianProvider  # noqa: E402
from rag_v2.supervisor.service import SupervisorService  # noqa: E402
from scripts.evaluation.run_nf_v2_01_r1_bailian_strong_general_supervisor import load_env_config  # noqa: E402
from scripts.evaluation.run_nf_v2_01_r1_formal_72 import (  # noqa: E402
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


def main() -> int:
    config, error = load_env_config()
    if error or config is None:
        raise SystemExit(f"infrastructure_blocked: {error}")
    questions = (
        "What was ExampleCorp's revenue in FY2025?",
        "What was ExampleCorp's operating income in FY2024?",
        "Which period did ExampleCorp report its net sales?",
        "What was ExampleCorp's total assets in FY2023?",
        "What was ExampleCorp's cash flow from operations in FY2022?",
    )
    envelopes = tuple(SyntheticEnvelope(f"synthetic-runner-{index + 1}", question) for index, question in enumerate(questions))
    provider = BailianProvider(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_name=config["model"],
        enable_thinking=False,
        temperature=0.0,
        max_retries=0,
    )
    service = SupervisorService(provider)
    records = []
    failures = []
    started_all = time.perf_counter()
    try:
        for index, envelope in enumerate(envelopes, start=1):
            started = time.perf_counter()
            run = service.plan(envelope.question)
            record = record_from_run(envelope, run, provider, started)
            record["call_index"] = index
            records.append(record)
            if not record["provider_response_success"] or not record["structured_output_success"] or not record["plan_valid"]:
                failures.append({"question_id": envelope.question_id, "error": record.get("error"), "exception_chain": record.get("exception_chain")})
    finally:
        provider.close()
    path = OUT / "synthetic-runner-5-plans.jsonl.gz"
    digest = serialize_prediction_records(records, path)
    loaded = load_prediction_records(path)
    seal = {
        "artifact": path.name,
        "records": len(loaded),
        "sha256": digest,
        "seal_verification": digest == hashlib.sha256(path.read_bytes()).hexdigest(),
        "provider": "bailian",
        "model": MODEL,
        "model_calls": len(records),
        "provider_success": sum(bool(record.get("provider_response_success")) for record in records),
        "structured_output_success": sum(bool(record.get("structured_output_success")) for record in records),
        "plan_validator_pass": sum(bool(record.get("plan_valid")) for record in records),
        "runner_exceptions": len(failures),
        "api_connection_errors": sum(any(item.get("type") == "APIConnectionError" for item in record.get("exception_chain") or []) for record in records),
        "gold_reads": 0,
        "benchmark_questions_loaded": 0,
        "total_wall_time_ms": (time.perf_counter() - started_all) * 1000,
    }
    (OUT / "synthetic-runner-5.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(seal, ensure_ascii=False, sort_keys=True))
    return 0 if len(records) == 5 and len(loaded) == 5 and not failures and seal["seal_verification"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
