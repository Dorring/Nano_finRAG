#!/usr/bin/env python3
"""NF-E2E-08 R0: offline deterministic-fact selection feasibility audit.

This gate audits the already frozen SADA Top5 and Statement-Aware fields.  It
does not execute a model, retrieval, reranker, answer builder, or selector.
The audit is intentionally conservative: a value embedded in the frozen
Statement-Aware serialization is not promoted to ``parsed_numeric_value``
because the sealed candidate contract contains no typed numeric fact object.
That distinction is the pre-registered feasibility gate for DFS-V1.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/nf-e2e-08-r0-deterministic-fact-selection-recovery"
NF06 = ROOT / "artifacts/evaluation/nf-e2e-06-r0-citation-binding-recovery"
NF07 = ROOT / "artifacts/evaluation/nf-e2e-07-r0-claim-grounding-recovery"
NF01 = ROOT / "artifacts/evaluation/nf-e2e-01-r0-frozen-retrieval-integration-review"
NF24 = ROOT / "artifacts/evaluation/nf-opt-24-r0-deep-supply-top100-admission"
NF26 = ROOT / "artifacts/evaluation/nf-opt-26-r0-internal-retrieval-freeze"

GATE = "NF-E2E-08-R0"
BASE_COMMIT = "3a36a2cd6eec4d300f5b2e91b75c296447f4e761"
NF26_SHA = "70048502ec918ae6ee56246a788da42129df3b073c2be8682e14f97e409e7c80"
QUESTION_TOTAL = 72
FACT_TOTAL = 46
ANSWERABLE_TOTAL = 64
NO_ANSWER_TOTAL = 8
CALC_TOTAL = 11
CALC_READY = 5
CONTEXT_TOP_K = 5
CONTEXT_TOKENS = 1100
DFS_THRESHOLD = 15


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write(
                    (
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pct(value: int, total: int) -> float:
    return round(100.0 * value / total, 4) if total else 0.0


def import_e2e01():
    from scripts.evaluation import run_nf_e2e_01_r0_frozen_retrieval_integration_review as module

    return module


def load_questions() -> dict[str, dict[str, Any]]:
    """Read only original question text and document scope.

    The benchmark file also contains answer and review annotations.  They are
    deliberately not accessed here; only the question surface and the already
    existing document scope are allowed query signals.
    """
    path = ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"
    questions: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        case_id = str(raw["case_id"])
        questions[case_id] = {
            "case_id": case_id,
            "question": str(raw["question"]),
            "document_scope": tuple(str(item) for item in (raw.get("document_scope") or ())),
        }
    if len(questions) != QUESTION_TOTAL:
        raise RuntimeError(f"expected {QUESTION_TOTAL} original questions, got {len(questions)}")
    return questions


def load_state() -> dict[str, Any]:
    """Load frozen upstream state without model/retrieval/Gold execution."""
    nf26_manifest = NF26 / "final-evidence-manifest.json"
    if sha256_file(nf26_manifest) != NF26_SHA:
        raise RuntimeError("NF-OPT-26 manifest SHA mismatch")
    if (NF26 / "final-evidence-manifest.sha256").read_text(encoding="utf-8").strip() != NF26_SHA:
        raise RuntimeError("NF-OPT-26 manifest seal mismatch")
    method = read_json(NF26 / "internal-retrieval-method-freeze.json")
    metrics = read_json(NF26 / "final-internal-retrieval-metrics.json")
    if method.get("selected_internal_shadow_method") != "sada_statement_aware_v1":
        raise RuntimeError("selected internal method changed")
    if metrics.get("sada_top100", {}).get("hits") != 78:
        raise RuntimeError("SADA Top100 supply changed")
    if method.get("production_switch_allowed") is not False:
        raise RuntimeError("production guardrail missing")

    nf07_decision = read_json(NF07 / "decision.json")
    if nf07_decision.get("next_gate") != "deterministic_fact_selection_recovery":
        raise RuntimeError("NF-E2E-07 did not hand off to deterministic fact selection")
    if nf07_decision.get("model_execution") is not False:
        raise RuntimeError("NF-E2E-07 model execution was not false")

    nf01_context = read_json(NF01 / "context-budget-contract.json")
    if nf01_context.get("candidates_entering_context") != CONTEXT_TOP_K or nf01_context.get("token_budget") != CONTEXT_TOKENS:
        raise RuntimeError("Top5/context budget contract changed")

    response_seal = read_json(NF06 / "response-seal.json")
    response_path = NF06 / "reconstructed-responses.jsonl.gz"
    if not response_seal.get("complete") or response_seal.get("case_count") != QUESTION_TOTAL:
        raise RuntimeError("NF-E2E-06 response seal incomplete")
    if response_seal.get("model_execution") is not False:
        raise RuntimeError("NF-E2E-06 response artifact records model execution")
    if response_seal.get("gold_reads_during_reconstruction") != 0:
        raise RuntimeError("NF-E2E-06 response reconstruction read Gold")
    if sha256_file(response_path) != response_seal.get("response_sha256"):
        raise RuntimeError("NF-E2E-06 response SHA mismatch")

    routing = read_json(NF06 / "deterministic-routing-audit.json")
    routing_rows = {str(row["question_id"]): row for row in routing["rows"]}
    if len(routing_rows) != QUESTION_TOTAL:
        raise RuntimeError("routing audit is incomplete")
    if sum(int(row.get("route") == "deterministic_fact") for row in routing_rows.values()) != FACT_TOTAL:
        raise RuntimeError("deterministic fact denominator changed")
    responses = {
        str(row["question_id"]): row for row in read_jsonl_gz(response_path)
    }
    if len(responses) != QUESTION_TOTAL:
        raise RuntimeError("sealed responses are incomplete")

    e2e01 = import_e2e01()
    cases, inventory = e2e01.load_sada_inputs(ROOT)
    if len(cases) != QUESTION_TOTAL or sum(len(items) for items in cases.values()) != QUESTION_TOTAL * 100:
        raise RuntimeError("frozen SADA candidate universe is not 72 x 100")
    if any(len(items[:CONTEXT_TOP_K]) != CONTEXT_TOP_K for items in cases.values()):
        raise RuntimeError("frozen Top5 is incomplete")
    return {
        "method": method,
        "metrics": metrics,
        "routing": routing,
        "routing_by_id": routing_rows,
        "responses": responses,
        "response_seal": response_seal,
        "cases": cases,
        "inventory": inventory,
        "questions": load_questions(),
        "nf26_manifest_sha256": NF26_SHA,
        "nf01_context": nf01_context,
    }


def normalize_text(value: Any) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text or None


def normalize_period(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    match = re.search(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", text)
    return f"FY{match.group(1)}" if match else None


def fact_inventory_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id in sorted(state["cases"]):
        for item in state["cases"][case_id][:CONTEXT_TOP_K]:
            parsed = item["parsed"]
            metric = normalize_text(parsed.get("metric_path") or parsed.get("row_label"))
            headers = [str(header) for header in (parsed.get("column_headers") or [])]
            periods = [period for period in (normalize_period(header) for header in headers) if period]
            physical_complete = bool(
                parsed.get("document_id")
                and parsed.get("table_id")
                and parsed.get("row_id")
                and parsed.get("page") is not None
            )
            # The frozen contract exposes Period / Value as serialization text.
            # It does not expose a typed raw_value or parsed_numeric_value.
            rows.append(
                {
                    "case_id": case_id,
                    "candidate_id": item["candidate_key"],
                    "candidate_rank": item["rank"],
                    "physical_source_id": parsed.get("physical_source_id"),
                    "document_id": parsed.get("document_id"),
                    "pdf_page": parsed.get("page"),
                    "statement_id": parsed.get("statement"),
                    "table_id": parsed.get("table_id"),
                    "table_title": parsed.get("table_title"),
                    "metric": parsed.get("metric_path") or parsed.get("row_label"),
                    "normalized_metric": metric,
                    "row_label": parsed.get("row_label"),
                    "row_id": parsed.get("row_id"),
                    "column_header": headers,
                    "normalized_periods": periods,
                    "period_value_bindings": list(parsed.get("period_value_bindings") or []),
                    "raw_value": None,
                    "parsed_numeric_value": None,
                    "value_field_status": "not_available_as_typed_runtime_field",
                    "currency": parsed.get("currency"),
                    "scale": parsed.get("scale"),
                    "unit": None,
                    "cell_id": None,
                    "physical_source_identity_complete": physical_complete,
                    "machine_readable_metric": metric is not None,
                    "machine_readable_period": bool(periods),
                    "machine_readable_value": False,
                    "full_machine_readable_provenance": False,
                    "source_text": item["serialization"],
                    "statement_serialization_sha256": item["serialization_sha256"],
                }
            )
    write_jsonl_gz(OUT / "deterministic-fact-candidate-inventory.jsonl.gz", rows)
    return rows


def query_signal(profile: Any) -> dict[str, Any]:
    return {
        "task_type": profile.task_type,
        "issuer": profile.issuer,
        "metric_phrases": [
            {
                "raw_text": item.raw_text,
                "normalized_text": item.normalized_text,
                "role": item.role,
            }
            for item in profile.metric_phrases
        ],
        "periods": [
            {
                "raw_text": item.raw_text,
                "normalized_period": item.normalized_period,
                "role": item.role,
            }
            for item in profile.periods
        ],
        "operation": profile.operation,
        "expected_operand_count": profile.expected_operand_count,
        "requires_multiple_sources": profile.requires_multiple_sources,
        "statement_hint": profile.statement_hint,
        "unresolved_reasons": list(profile.unresolved_reasons),
    }


def write_frozen_contract(state: dict[str, Any]) -> None:
    write_json(
        OUT / "frozen-e2e-contract.json",
        {
            "gate": GATE,
            "base_commit": BASE_COMMIT,
            "evaluation_role": "development_shadow_deterministic_fact_selection_recovery",
            "fresh_blind_evaluation": False,
            "model_calls": 0,
            "retrieval_calls": 0,
            "reranker_calls": 0,
            "training": False,
            "production_switch_allowed": False,
            "selected_internal_shadow_method": "sada_statement_aware_v1",
            "sada_top100": {"hits": 78, "total": 80, "recall": pct(78, 80)},
            "context": {"top_k": CONTEXT_TOP_K, "token_budget": CONTEXT_TOKENS},
            "original_query": True,
            "statement_aware_unchanged": True,
            "bica_unchanged": True,
            "binder_unchanged": True,
            "calculator_unchanged": True,
            "gcca_unchanged": True,
            "nf_opt_26_manifest_sha256": state["nf26_manifest_sha256"],
            "calculation_preservation": {"binder_ready": 5, "calculator_strict_correct": 5, "final_numeric": 5, "period": 5, "unit": 5},
            "no_answer_preservation": {"correct": 5, "false_answer_release": 3},
            "gold_reads_during_stage_a": 0,
        },
    )


def current_fact_contract(state: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "gate": GATE,
        "route": "deterministic_fact",
        "router_entrypoint": "src.retrieval_v3.query_router.route_question",
        "deterministic_fact_entrypoint": "src.generation.deterministic_answers.DeterministicAnswerExtractor.answer_numeric_query_from_chunks",
        "fact_selector_entrypoint": "src.generation.deterministic_answers.DeterministicAnswerExtractor._select_distinct_evidence/_select_answer_values",
        "answer_builder_entrypoint": "src.application.rag_orchestrator.RAGOrchestrator",
        "response_builder_entrypoint": "src.application.rag_orchestrator.RAGOrchestrator._validate_and_repair_once",
        "candidate_consumer": "Top5 child chunks from frozen SADA context",
        "current_value_production": "numeric windows over chunk content",
        "sealed_selection_relation": {
            "selected_candidate_id": False,
            "selected_fact_id": False,
            "selected_source_id": False,
            "row_cell_span": False,
            "exact_derivation_relation": False,
        },
        "observer_selection_state_serialized_in_sealed_artifact": False,
        "answer_value_structured_in_sealed_artifact": False,
        "gold_reads": 0,
    }
    write_json(OUT / "current-deterministic-fact-contract.json", payload)
    return payload


def runtime_audit(state: dict[str, Any], inventory: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from src.retrieval_v3.query_router import route_question

    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in inventory:
        by_case.setdefault(row["case_id"], []).append(row)
    rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    for case_id in sorted(state["routing_by_id"]):
        if state["routing_by_id"][case_id].get("route") != "deterministic_fact":
            continue
        question = state["questions"][case_id]
        profile = route_question(question["question"], document_scope=question["document_scope"])
        signals = query_signal(profile)
        candidates = by_case[case_id]
        metric = sum(int(row["machine_readable_metric"]) for row in candidates)
        period = sum(int(row["machine_readable_period"]) for row in candidates)
        value = sum(int(row["machine_readable_value"]) for row in candidates)
        full = sum(int(row["full_machine_readable_provenance"]) for row in candidates)
        structured = bool(metric or period or value)
        metric_period = bool(metric and period)
        if not structured:
            primary = "FS6_no_machine_readable_fact_candidate"
        elif not value or not metric_period:
            primary = "FS4_structured_fields_incomplete"
        elif full == 0:
            primary = "FS4_structured_fields_incomplete"
        elif full > 1:
            primary = "FS7_multiple_fact_tuples_remain_ambiguous"
        else:
            primary = "FS0_unique_exact_fact_already_exists"
        answer = state["responses"].get(case_id, {})
        rows.append(
            {
                "question_id": case_id,
                "query": question["question"],
                "route": "deterministic_fact",
                "router_entrypoint": "src.retrieval_v3.query_router.route_question",
                "fact_selector_entrypoint": "src.generation.deterministic_answers.DeterministicAnswerExtractor.answer_numeric_query_from_chunks",
                "answer_builder_entrypoint": "src.application.rag_orchestrator.RAGOrchestrator",
                "top5_candidate_ids": [row["candidate_id"] for row in candidates],
                "answer_text_observed": str(answer.get("released_answer") or ""),
                "answer_value": None,
                "answer_period": None,
                "answer_unit": None,
                "how_answer_value_was_produced": "sealed deterministic numeric-window path; selected fact relation was not serialized",
                "selected_candidate_id_if_any": None,
                "selected_fact_id_if_any": None,
                "selected_source_id_if_any": None,
                "value_known": bool(answer.get("released_answer")),
                "selected_candidate_known": False,
                "selected_fact_known": False,
                "exact_provenance_known": False,
                "structured_fact_available": structured,
                "metric_resolvable": bool(metric),
                "period_resolvable": bool(period),
                "value_resolvable": bool(value),
                "metric_period_fact_available": metric_period and bool(value),
                "provenance_resolvable": bool(full),
                "unique_fact_tuple_possible": full == 1,
                "candidate_field_counts": {"metric": metric, "period": period, "value": value, "full_provenance": full},
                "primary_failure_reason": primary,
                "query_signals": signals,
            }
        )
        signal_rows.append({"question_id": case_id, **signals})
    counts = Counter(row["primary_failure_reason"] for row in rows)
    audit = {
        "gate": GATE,
        "denominator": len(rows),
        "counts": dict(counts),
        "rows": rows,
        "model_calls": 0,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "response_modification": False,
        "gold_reads": 0,
    }
    write_json(OUT / "deterministic-fact-runtime-audit.json", audit)
    write_json(
        OUT / "existing-query-signal-contract.json",
        {
            "gate": GATE,
            "component": "src.retrieval_v3.query_router.route_question",
            "existing_before_gate": True,
            "input": "original question text + existing document_scope",
            "gold_access": False,
            "candidate_access": False,
            "reference_answer_access": False,
            "deterministic": True,
            "selected_for_dfs_v1": True,
            "fields": {
                "company_document_scope": "available",
                "metric": "available",
                "period": "available",
                "operation": "available_or_null",
                "fact_type": "task_type",
                "currency": "not_available",
                "unit": "not_available",
            },
            "field_coverage": {
                "metric": sum(bool(row["query_signals"]["metric_phrases"]) for row in rows),
                "period": sum(bool(row["query_signals"]["periods"]) for row in rows),
                "operation": sum(bool(row["query_signals"]["operation"]) for row in rows),
            },
            "rows": signal_rows,
            "gold_reads": 0,
        },
    )
    return rows, audit


def write_fact_taxonomy(audit: dict[str, Any]) -> None:
    counts = {name: int(audit["counts"].get(name, 0)) for name in (
        "FS0_unique_exact_fact_already_exists",
        "FS1_multiple_candidates_same_fact_value",
        "FS2_same_metric_multiple_periods",
        "FS3_same_period_multiple_rows_or_metrics",
        "FS4_structured_fields_incomplete",
        "FS5_current_answer_from_unstructured_text_heuristic",
        "FS6_no_machine_readable_fact_candidate",
        "FS7_multiple_fact_tuples_remain_ambiguous",
        "FS8_other",
    )}
    rows = [
        {
            "question_id": row["question_id"],
            "primary_reason": row["primary_failure_reason"],
            "candidate_field_counts": row["candidate_field_counts"],
            "structured_fact_available": row["structured_fact_available"],
            "full_provenance_available": row["provenance_resolvable"],
            "gold_reads": 0,
        }
        for row in audit["rows"]
    ]
    write_json(OUT / "fact-selection-failure-taxonomy.json", {"gate": GATE, "denominator": FACT_TOTAL, "counts": counts, "rows": rows, "gold_reads": 0})


def write_funnel(audit: dict[str, Any]) -> dict[str, Any]:
    rows = audit["rows"]
    funnel = {
        "gate": GATE,
        "deterministic_fact": FACT_TOTAL,
        "structured_fact_available": sum(int(row["structured_fact_available"]) for row in rows),
        "metric_resolvable": sum(int(row["metric_resolvable"]) for row in rows),
        "period_resolvable": sum(int(row["period_resolvable"]) for row in rows),
        "metric_period_fact_available": sum(int(row["metric_period_fact_available"]) for row in rows),
        "full_machine_readable_provenance_available": sum(int(row["provenance_resolvable"]) for row in rows),
        "unique_fact_tuple_possible": sum(int(row["unique_fact_tuple_possible"]) for row in rows),
        "model_calls": 0,
        "retrieval_calls": 0,
        "gold_reads": 0,
    }
    write_json(OUT / "pre-dfs-fact-provenance-funnel.json", funnel)
    write_json(OUT / "fact-provenance-funnel.json", {"baseline": funnel, "post_dfs": None, "stage_d_executed": False})
    return funnel


def write_feasibility(funnel: dict[str, Any]) -> dict[str, Any]:
    available = int(funnel["full_machine_readable_provenance_available"])
    allowed = available >= DFS_THRESHOLD
    decision = {
        "gate": GATE,
        "stage": "A.2_pre_frozen_gate",
        "evaluation_role": "development_shadow_deterministic_fact_selection_recovery",
        "fresh_blind_evaluation": False,
        "full_machine_readable_fact_provenance_available": available,
        "deterministic_fact_queries": FACT_TOTAL,
        "minimum_required_for_dfs_v1": DFS_THRESHOLD,
        "dfs_v1_allowed": allowed,
        "decision": "dfs_v1_authorized" if allowed else "structured_fact_representation_insufficient",
        "reason": (
            "The frozen Top5 provides at least the pre-registered minimum of typed fact tuples."
            if allowed
            else "The frozen candidates expose metric/period serialization text but no typed parsed_numeric_value or complete fact object; full machine-readable provenance is below the pre-registered threshold."
        ),
        "model_calls": 0,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "gold_reads": 0,
    }
    write_json(OUT / "dfs-v1-feasibility-decision.json", decision)
    return decision


def write_disabled_dfs_artifacts(funnel: dict[str, Any], feasibility: dict[str, Any]) -> None:
    policy = "DFS-V1 not activated: Stage A feasibility gate failed because typed full fact provenance was below 15/46.\n"
    policy_path = OUT / "dfs-v1-policy.txt"
    policy_path.write_text(policy, encoding="utf-8")
    policy_sha = sha256_file(policy_path)
    (OUT / "dfs-v1-policy.sha256").write_text(policy_sha + "\n", encoding="utf-8")
    write_json(
        OUT / "dfs-v1-contract.json",
        {
            "name": "DFS-V1",
            "allowed": False,
            "executed": False,
            "policy_sha256": policy_sha,
            "selection_rule": "not activated",
            "uses_only_existing_query_constraints": True,
            "uses_normalized_equality_only": True,
            "rank_tie_break": False,
            "can_use_gold": False,
            "can_use_reference_answer": False,
            "can_use_expected_value": False,
            "can_search_top5": False,
            "reason_not_executed": feasibility["reason"],
            "shadow_only": True,
        },
    )
    empty = OUT / "dfs-v1-predictions.jsonl.gz"
    write_jsonl_gz(empty, [])
    write_json(
        OUT / "dfs-v1-prediction-seal.json",
        {
            "gate": GATE,
            "complete": False,
            "executed": False,
            "case_count": 0,
            "prediction_sha256": sha256_file(empty),
            "model_calls": 0,
            "retrieval_calls": 0,
            "reranker_calls": 0,
            "gold_reads_before_prediction_seal": 0,
            "reason": "DFS-V1 was not authorized by the pre-frozen feasibility gate.",
        },
    )
    write_json(
        OUT / "dfs-v1-selection-metrics.json",
        {
            "stage_b_executed": False,
            "selector_ready": None,
            "selector_missing": None,
            "selector_ambiguous": None,
            "exact_selected_fact_provenance": None,
            "strict_answer_correct_among_ready": None,
            "strict_source_correct_among_ready": None,
            "false_source_binding": 0,
            "gold_reads": 0,
            "reason": feasibility["reason"],
        },
    )


def write_post_artifacts(state: dict[str, Any], funnel: dict[str, Any], feasibility: dict[str, Any]) -> None:
    baseline_calc = {
        "binder_ready": 5,
        "calculator_strict_correct": 5,
        "final_numeric_correct": 5,
        "period_correct": 5,
        "unit_correct": 5,
        "citation_valid": 3,
        "false_binding": 0,
        "false_execution": 0,
        "executed_incorrect": 0,
    }
    baseline_no_answer = {"correct_safe_response": 5, "false_answer_release": 3}
    write_json(
        OUT / "wrong-source-safety.json",
        {
            "baseline_wrong_source": 7,
            "post_wrong_source": None,
            "historical_wrong_source_not_evaluated": True,
            "false_source_binding": 0,
            "stage_d_executed": False,
            "formal_result_invalid": False,
        },
    )
    write_json(
        OUT / "grounding-metrics.json",
        {"baseline": {"grounded_pass": 3, "denominator": ANSWERABLE_TOTAL}, "post_dfs": None, "stage_d_executed": False},
    )
    write_json(
        OUT / "citation-metrics.json",
        {"baseline": {"citation_full_recall": 23, "denominator": ANSWERABLE_TOTAL}, "post_dfs": None, "stage_d_executed": False},
    )
    write_json(OUT / "calculation-preservation.json", {"baseline": baseline_calc, "post_dfs": None, "stage_d_executed": False, "route_invocations": {"deterministic_fact": 0, "deterministic_calculation": 0}})
    write_json(OUT / "no-answer-preservation.json", {"baseline": baseline_no_answer, "post_dfs": None, "path_preserved": True, "dfs_invocations": {"safe_response": 0}, "stage_d_executed": False})
    write_json(
        OUT / "baseline-vs-dfs.json",
        {
            "baseline": {"grounded": 3, "citation_full_recall": 23, "answerable_released": 55, "wrong_source": 7},
            "post_dfs": None,
            "stage_d_executed": False,
            "unsupported_or_untraceable_release_baseline": None,
        },
    )
    write_json(
        OUT / "full-shadow-replay.json",
        {
            "stage_d_executed": False,
            "questions": QUESTION_TOTAL,
            "dfs_invocations": {"deterministic_fact": 0, "deterministic_calculation": 0, "safe_response": 0},
            "model_calls": 0,
            "retrieval_calls": 0,
            "reranker_calls": 0,
            "gold_reads": 0,
            "reason": feasibility["reason"],
        },
    )
    write_json(
        OUT / "safety-analysis.json",
        {
            "stage_d_executed": False,
            "false_source_binding": 0,
            "false_binding": 0,
            "false_execution": 0,
            "executed_incorrect": 0,
            "no_answer_correct": 5,
            "no_answer_false_release": 3,
            "production_switch_allowed": False,
            "formal_result_invalid": False,
        },
    )
    write_json(
        OUT / "bottleneck-analysis.json",
        {
            "dominant_residual_bottleneck": "structured_fact_representation",
            "structured_fact_representation_insufficient": not feasibility["dfs_v1_allowed"],
            "full_machine_readable_provenance_available": funnel["full_machine_readable_provenance_available"],
            "next_gate": "structured_fact_representation_review",
        },
    )
    decision = {
        "gate": GATE,
        "evaluation_role": "development_shadow_deterministic_fact_selection_recovery",
        "fresh_blind_evaluation": False,
        "model_execution": False,
        "retrieval_execution": False,
        "reranker_execution": False,
        "training": False,
        "production_switch_allowed": False,
        "deterministic_fact_queries": FACT_TOTAL,
        "structured_fact_available": funnel["structured_fact_available"],
        "metric_resolvable": funnel["metric_resolvable"],
        "period_resolvable": funnel["period_resolvable"],
        "full_provenance_available": funnel["full_machine_readable_provenance_available"],
        "unique_fact_tuple_possible": funnel["unique_fact_tuple_possible"],
        "dfs_v1_allowed": feasibility["dfs_v1_allowed"],
        "dfs_v1_executed": False,
        "selector_ready": None,
        "selector_missing": None,
        "selector_ambiguous": None,
        "exact_selected_fact_provenance_baseline": 0,
        "exact_selected_fact_provenance_post": None,
        "provenance_safe_fact_coverage": None,
        "false_source_binding": 0,
        "baseline_grounded_pass": 3,
        "post_grounded_pass": None,
        "baseline_citation_full_recall": 23,
        "post_citation_full_recall": None,
        "baseline_answerable_released": 55,
        "post_answerable_released": None,
        "baseline_wrong_source": 7,
        "post_wrong_source": None,
        "calculation_preserved": True,
        "no_answer_path_preserved": True,
        "deterministic_fact_selection_recovery_effective": False,
        "structured_fact_representation_insufficient": not feasibility["dfs_v1_allowed"],
        "dominant_residual_bottleneck": "structured_fact_representation",
        "next_gate": "structured_fact_representation_review",
        "gold_reads_before_prediction_seal": 0,
        "gold_reads_during_stage_a": 0,
    }
    write_json(OUT / "decision.json", decision)
    (OUT / "README.md").write_text(
        "# NF-E2E-08 R0 — Provenance-Aware Deterministic Fact Selection Recovery\n\n"
        "This development-shadow gate audited the frozen SADA Top5 without model, "
        "retrieval, reranker, or response execution. The Statement-Aware contract "
        "contains metric/period/value serialization text and physical provenance, "
        "but the frozen runtime candidate schema has no typed parsed_numeric_value "
        "or complete fact object. Consequently full machine-readable fact provenance "
        "is below the pre-frozen DFS-V1 threshold and DFS-V1 was not activated. No "
        "prediction or 72-question replay was performed. The next gate is "
        "structured_fact_representation_review. Production switch allowed: false.\n",
        encoding="utf-8",
    )


def run() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    state = load_state()
    write_frozen_contract(state)
    current_fact_contract(state)
    inventory = fact_inventory_rows(state)
    _rows, audit = runtime_audit(state, inventory)
    write_fact_taxonomy(audit)
    funnel = write_funnel(audit)
    feasibility = write_feasibility(funnel)
    write_disabled_dfs_artifacts(funnel, feasibility)
    write_post_artifacts(state, funnel, feasibility)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
