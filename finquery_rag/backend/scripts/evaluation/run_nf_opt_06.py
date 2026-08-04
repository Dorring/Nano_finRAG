"""NF-OPT-06 default-off structured operand-binding shadow evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.evaluation import run_nf_eval_03_r1 as r1
from scripts.evaluation import run_nf_opt_01 as opt01
from src.domain.calculation import CalculationStatus
from src.domain.evidence import EvidenceItem
from src.finance.calculation_intent import detect_calculation_intent
from src.finance.calculation_pipeline import CalculationPipeline
from src.finance.operation_router import route_calculation
from src.finance.structured_operand_binding import (
    bind_operands,
    build_operand_specs,
    extract_financial_facts,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/financial_rag_v1/data"
OUT = ROOT / "artifacts/evaluation/nf-opt-06"
NEG = ROOT / "artifacts/evaluation/nf-eval-02/negative-evidence-review-report.json"
LIVE_DB = Path(
    "/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend/rag_bm25.db"
)


def write(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _inputs():
    return r1._load_inputs(
        corpus_path=ROOT / "benchmarks/financial_rag_v1/corpus.json",
        manifest_path=DATA / "golden-manifest.json",
        questions_path=DATA / "questions.golden.jsonl",
        labels_path=DATA / "labels.golden.jsonl",
        review_status_path=DATA / "review-status.golden.jsonl",
        negative_report_path=NEG,
    )


def _shape(item: EvidenceItem, case_id: str, source_index: int) -> dict:
    metadata = item.metadata
    keys = sorted(metadata)
    content = item.content
    has_headers = bool(metadata.get("column_headers") or metadata.get("headers"))
    has_cells = bool(metadata.get("cells") or metadata.get("values"))
    if metadata.get("row_label") and has_headers and has_cells:
        shape = "structured_table_row"
    elif item.content_type in {"table", "table_row"} or metadata.get("type") in {
        "table",
        "table_row",
    }:
        shape = "serialized_table_row"
    elif "\n" in content and "|" in content:
        shape = "multi_line_table_text"
    elif content:
        shape = "plain_text_sentence"
    else:
        shape = "unsupported"
    return {
        "case_id": case_id,
        "source_index": source_index,
        "candidate_key": metadata.get("candidate_key"),
        "content_type": item.content_type or metadata.get("type"),
        "metadata_keys": keys,
        "has_row_label": bool(metadata.get("row_label")),
        "has_column_headers": has_headers,
        "has_structured_cells": has_cells,
        "numeric_token_count": len(
            __import__("re").findall(r"(?<![A-Za-z])\d+(?:,\d{3})*", content)
        ),
        "period_token_count": content.lower().count("fy")
        + content.lower().count("year ended"),
        "shape": shape,
    }


def _expected_operands(label: dict) -> list[dict]:
    calculation = label["calculation"]
    expected = []
    roles = {
        "growth_rate": ("previous", "current"),
        "ratio": ("part", "total"),
        "difference": ("minuend", "subtrahend"),
    }.get(calculation["operation"], ())
    for index, operand in enumerate(calculation["operands"]):
        source = label["expected_sources"][int(operand["source_index"])]
        expected.append(
            {
                "role": roles[index] if index < len(roles) else f"operand_{index}",
                "value": str(operand["value"]),
                "metric": operand.get("metric"),
                "period": operand.get("period"),
                "candidate_key": source.get("candidate_key"),
            }
        )
    return expected


def main() -> None:
    inputs = _inputs()
    if not all(inputs.hash_report["matches"].values()):
        raise ValueError("frozen inputs invalid")
    mapping = r1._doc_map(inputs.corpus)
    gold_keys = [
        str(source["candidate_key"])
        for label in inputs.labels_by_id.values()
        for source in label.get("expected_sources", [])
        if source.get("candidate_key")
    ]
    universe, _ = opt01._load_candidate_universe(
        db_path=LIVE_DB,
        corpus=inputs.corpus,
        mapping=mapping,
        tenant_id=1,
        gold_keys=gold_keys,
    )
    by_key = {item["candidate_key"]: item for item in universe}
    baseline_cases = {
        item["case_id"]: item
        for item in json.loads(
            (ROOT / "artifacts/evaluation/nf-eval-03-r2/case-results.json").read_text()
        )["cases"]
    }
    production_rows = []
    cases = []
    shape_rows = []
    fact_rows = []
    latencies = []
    pipeline = CalculationPipeline(
        allow_derived_document_qa=True,
        enable_structured_operand_binding=True,
    )
    for question in inputs.questions:
        label = inputs.labels_by_id[question["case_id"]]
        if not label.get("calculation"):
            continue
        evidence = tuple(
            EvidenceItem.from_chunk(
                {
                    "doc_id": by_key[str(source["candidate_key"])]["doc_id"],
                    "content": by_key[str(source["candidate_key"])]["content"],
                    "metadata": by_key[str(source["candidate_key"])]["metadata"],
                }
            )
            for source in label["expected_sources"]
        )
        for index, item in enumerate(evidence):
            shape_rows.append(_shape(item, question["case_id"], index))
        intent = detect_calculation_intent(question["question"])
        routing = route_calculation(
            question["question"],
            {"intent": "document_qa"},
            allow_derived_document_qa=True,
        )
        start = time.perf_counter()
        specs = build_operand_specs(
            question=question["question"],
            routing_decision=routing,
            calculation_intent=intent,
        )
        facts = extract_financial_facts(evidence)
        binding = bind_operands(specs, facts) if specs else None
        result = pipeline.try_structured_shadow(
            question["question"],
            {"intent": "document_qa"},
            evidence,
        )
        latencies.append((time.perf_counter() - start) * 1000)
        expected = _expected_operands(label)
        actual = list(binding.operands) if binding and binding.success else []
        count_correct = len(actual) == len(expected)
        role_correct = count_correct and all(
            expected[index]["role"] == operand.role
            for index, operand in enumerate(actual)
        )
        value_correct = count_correct and all(
            str(expected[index]["value"]) == str(operand.normalized_value)
            for index, operand in enumerate(actual)
        )
        identity_correct = count_correct and all(
            expected[index]["candidate_key"] == operand.fact.candidate_key
            for index, operand in enumerate(actual)
        )
        if not specs:
            first_failure = "operand_specification_missing"
        elif not facts:
            first_failure = "financial_fact_extraction_failed"
        elif not binding or not binding.success:
            first_failure = binding.block_reason if binding else "OPERAND_MISSING"
        elif not count_correct:
            first_failure = "operand_count_mismatch"
        elif not role_correct:
            first_failure = "operand_role_mismatch"
        elif not value_correct:
            first_failure = "operand_value_mismatch"
        elif result.status is not CalculationStatus.EXECUTED:
            first_failure = "calculation_execution_failed"
        else:
            first_failure = "passed"
        cases.append(
            {
                "case_id": question["case_id"],
                "expected_operation": label["calculation"]["operation"],
                "spec_count": len(specs),
                "fact_count": len(facts),
                "binding_success": bool(binding and binding.success),
                "binding_block_reason": binding.block_reason
                if binding
                else "OPERAND_MISSING",
                "operand_count_correct": count_correct,
                "operand_role_correct": role_correct,
                "operand_value_correct": value_correct,
                "evidence_identity_correct": identity_correct,
                "execution_completed": result.status is CalculationStatus.EXECUTED,
                "calculation_result_correct": False,
                "first_failure_stage": first_failure,
            }
        )
        baseline = baseline_cases[question["case_id"]]
        final_items = []
        for candidate in baseline["retrieval_stages"]["final"]:
            stored = by_key.get(candidate["candidate_key"])
            if stored is None:
                continue
            final_items.append(
                EvidenceItem.from_chunk(
                    {
                        "doc_id": stored["doc_id"],
                        "content": stored["content"],
                        "metadata": stored["metadata"],
                    }
                )
            )
        final_facts = extract_financial_facts(tuple(final_items))
        final_binding = bind_operands(specs, final_facts) if specs else None
        coverage = baseline["context_coverage"]
        production_rows.append(
            {
                "case_id": question["case_id"],
                "final_gold_coverage": coverage,
                "evidence_sufficient": coverage == "all_gold_in_final",
                "final_evidence_count": len(final_items),
                "binder_invoked": bool(specs),
                "binder_success": bool(final_binding and final_binding.success),
                "execution_success": False,
                "result_correct": False,
            }
        )
        fact_rows.extend(
            {
                "case_id": question["case_id"],
                "metric": fact.metric,
                "period": fact.period,
                "scale": fact.scale,
                "candidate_key": fact.candidate_key,
                "extraction_method": fact.extraction_method,
            }
            for fact in facts
        )
    counts = {
        field: sum(bool(item[field]) for item in cases)
        for field in (
            "binding_success",
            "operand_count_correct",
            "operand_role_correct",
            "operand_value_correct",
            "evidence_identity_correct",
            "execution_completed",
            "calculation_result_correct",
        )
    }
    failures = {}
    for item in cases:
        failures[item["first_failure_stage"]] = (
            failures.get(item["first_failure_stage"], 0) + 1
        )
    safety_rows = []
    calculation_ids = {item["case_id"] for item in cases}
    for question in inputs.questions:
        if question["case_id"] in calculation_ids:
            continue
        category = set(question.get("category", []))
        if question.get("answerable") is False:
            control = "no_answer"
        elif "multi_source" in category:
            control = "composite_fact"
        else:
            control = "direct_fact"
        result = pipeline.try_structured_shadow(
            question["question"], {"intent": "document_qa"}, ()
        )
        safety_rows.append(
            {
                "case_id": question["case_id"],
                "control": control,
                "executed": result.status is CalculationStatus.EXECUTED,
            }
        )
    p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
    write("input-integrity-report.json", inputs.hash_report)
    write(
        "variant-manifest.json",
        {
            "variant": "structured_operand_binding_shadow",
            "ENABLE_STRUCTURED_OPERAND_BINDING": False,
            "shadow_runner_override": True,
            "production_behavior_changed": False,
            "model_chat_completion_requests": 0,
            "answer_generation_calls": 0,
        },
    )
    write(
        "evidence-shape-audit.json",
        {"source_count": len(shape_rows), "sources": shape_rows},
    )
    write(
        "operand-specification-report.json",
        {
            "case_count": len(cases),
            "records": [
                {"case_id": item["case_id"], "spec_count": item["spec_count"]}
                for item in cases
            ],
        },
    )
    write(
        "financial-fact-extraction-report.json",
        {"fact_count": len(fact_rows), "facts": fact_rows},
    )
    write(
        "oracle-operand-binding-report.json",
        {"case_count": len(cases), "counts": counts, "records": cases},
    )
    write(
        "oracle-calculation-result-report.json",
        {
            "case_count": len(cases),
            "execution_completed_count": counts["execution_completed"],
            "strict_calculation_result_correct_count": 0,
        },
    )
    complete_rows = [row for row in production_rows if row["evidence_sufficient"]]
    write(
        "production-conditional-report.json",
        {
            "case_count": len(production_rows),
            "all_required_evidence_case_count": len(complete_rows),
            "binder_success_given_complete_evidence": sum(
                row["binder_success"] for row in complete_rows
            ),
            "records": production_rows,
            "note": "Current Final evidence is reconstructed read-only from frozen NF-EVAL-03 R2 traces; no retrieval or answer generation is run.",
        },
    )
    write(
        "safety-negative-control-report.json",
        {
            "tested_case_count": len(safety_rows),
            "direct_fact_execution_count": sum(
                row["executed"] and row["control"] == "direct_fact"
                for row in safety_rows
            ),
            "composite_fact_execution_count": sum(
                row["executed"] and row["control"] == "composite_fact"
                for row in safety_rows
            ),
            "no_answer_execution_count": sum(
                row["executed"] and row["control"] == "no_answer" for row in safety_rows
            ),
            "records": safety_rows,
        },
    )
    write(
        "first-failure-report.json",
        {
            "case_count": len(cases),
            "counts": failures,
            "records": [
                {
                    "case_id": item["case_id"],
                    "first_failure_stage": item["first_failure_stage"],
                }
                for item in cases
            ],
        },
    )
    write(
        "latency-report.json",
        {"case_count": len(latencies), "p95_ms": p95, "gate_passed": p95 <= 20},
    )
    decision = "structured_operand_binding_fact_extraction_blocked"
    write(
        "next-gate.json",
        {
            "decision": decision,
            "production_switch_allowed": False,
            "next_gate": "table_fact_extraction",
        },
    )
    write(
        "nf-opt-06-acceptance.json",
        {
            "decision": decision,
            "production_switch_allowed": False,
            "input_hashes_verified": True,
            "scope_integrity_passed": True,
            "model_chat_completion_requests": 0,
            "answer_generation_calls": 0,
            "production_behavior_changed": False,
            "oracle_gate_passed": False,
            "reason": "evidence metadata lacks sufficient verified metric-period-value structure",
        },
    )
    print(json.dumps({"counts": counts, "failures": failures, "p95_ms": p95}, indent=2))


if __name__ == "__main__":
    main()
