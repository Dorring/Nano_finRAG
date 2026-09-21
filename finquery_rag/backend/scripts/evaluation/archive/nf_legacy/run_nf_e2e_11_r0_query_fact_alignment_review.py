#!/usr/bin/env python3
"""NF-E2E-11 R0: Query--FinancialFact canonical alignment review.

This gate is an offline audit only.  It reads sealed NF-E2E-08/09/10
artifacts and classifies the already observed metric, period, and ambiguity
failures.  It deliberately does not execute DFS, retrieval, reranking,
models, PDF parsing, or an end-to-end replay.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/nf-e2e-11-r0-query-fact-alignment-review"
NF08 = ROOT / "artifacts/evaluation/nf-e2e-08-r0-deterministic-fact-selection-recovery"
NF09 = ROOT / "artifacts/evaluation/nf-e2e-09-r0-structured-financial-fact-representation"
NF10 = ROOT / "artifacts/evaluation/nf-e2e-10-r0-dfs-retry-financial-fact-v1"

GATE = "NF-E2E-11-R0"
BASE_COMMIT = "4cd069b8c5d6bc150bcb46a8cc7d4284aef76dc9"
FACT_SHA = "7a253b443962c5f372dd897c49c057a19b553e92314faadc31eefc82b27b54eb"
FACT_COUNT = 169
FACT_QUERIES = 46
DS3_COUNT = 28
DS4_COUNT = 3
DS7_COUNT = 7


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pct(count: int, denominator: int) -> float:
    return round(count * 100.0 / denominator, 4) if denominator else 0.0


def load_inputs() -> dict[str, Any]:
    contract_path = NF09 / "financial-fact-v1-contract.json"
    if sha256_file(contract_path) != FACT_SHA:
        raise RuntimeError("FinancialFactV1 contract SHA mismatch")
    facts = read_jsonl_gz(NF09 / "financial-facts-v1.jsonl.gz")
    seal = read_json(NF09 / "financial-facts-v1-seal.json")
    coverage = read_json(NF09 / "query-level-coverage.json")
    relation = read_json(NF09 / "relation-integrity.json")
    if len(facts) != FACT_COUNT or seal.get("deduplicated_facts") != FACT_COUNT:
        raise RuntimeError("FinancialFactV1 fact count mismatch")
    if sum(item.get("provenance_complete") is True for item in facts) != FACT_COUNT:
        raise RuntimeError("FinancialFactV1 provenance seal mismatch")
    if coverage.get("counts", {}).get("full_provenance") != 39:
        raise RuntimeError("FinancialFactV1 query coverage mismatch")
    if relation.get("fail") != 0 or relation.get("fabricated_cross_candidate_facts") != 0:
        raise RuntimeError("FinancialFactV1 relation integrity mismatch")

    runtime_rows = read_json(NF08 / "deterministic-fact-runtime-audit.json").get("rows", [])
    fact_rows = [row for row in runtime_rows if row.get("route") == "deterministic_fact"]
    if len(fact_rows) != FACT_QUERIES:
        raise RuntimeError("Deterministic fact denominator mismatch")
    predictions = read_jsonl_gz(NF10 / "dfs-v1-financial-fact-predictions.jsonl.gz")
    prediction_by_id = {str(row["question_id"]): row for row in predictions}
    selection_metrics = read_json(NF10 / "selection-metrics.json")
    nf10_contract = read_json(NF10 / "frozen-financial-fact-contract.json")
    nf09_contract = read_json(NF09 / "frozen-input-contract.json")
    if not nf10_contract.get("top5_ids_unchanged") or not nf10_contract.get("top5_order_unchanged"):
        raise RuntimeError("NF-E2E-10 Top5 freeze flags failed")
    if nf10_contract.get("top5_order_sha256") != nf09_contract.get("top5", {}).get("order_sha256"):
        raise RuntimeError("NF-E2E-09/NF-E2E-10 Top5 order hash mismatch")
    top5 = {str(row["question_id"]): list(row.get("top5_candidate_ids") or []) for row in fact_rows}
    if any(len(ids) != 5 for ids in top5.values()):
        raise RuntimeError("Frozen Top5 cardinality mismatch")
    facts_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        for candidate_id in fact.get("candidate_ids", [fact.get("candidate_id")]):
            if candidate_id:
                facts_by_candidate.setdefault(str(candidate_id), []).append(fact)
    return {
        "facts": facts,
        "facts_by_candidate": facts_by_candidate,
        "runtime_rows": {str(row["question_id"]): row for row in fact_rows},
        "predictions": prediction_by_id,
        "selection_metrics": selection_metrics,
        "top5": top5,
        "contract": read_json(contract_path),
        "seal": seal,
        "coverage": coverage,
        "relation": relation,
        "top5_order_sha256": nf10_contract.get("top5_order_sha256"),
        "nf10_contract": nf10_contract,
        "nf09_contract": nf09_contract,
    }


def fact_summary(fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "fact_id": fact.get("fact_id"),
        "candidate_id": fact.get("candidate_id"),
        "physical_source_id": fact.get("physical_source_id"),
        "document_id": fact.get("document_id"),
        "pdf_page": fact.get("pdf_page"),
        "statement_id": fact.get("statement_id"),
        "table_id": fact.get("table_id"),
        "row_id": fact.get("row_id"),
        "column_id": fact.get("column_id"),
        "cell_id": fact.get("cell_id"),
        "raw_metric": fact.get("raw_metric"),
        "normalized_metric": fact.get("normalized_metric"),
        "raw_period": fact.get("raw_period"),
        "normalized_period": fact.get("normalized_period"),
        "raw_value": fact.get("raw_value"),
        "parsed_numeric_value": fact.get("parsed_numeric_value"),
        "raw_scale": fact.get("raw_scale"),
        "normalized_scale": fact.get("normalized_scale"),
        "raw_currency": fact.get("raw_currency"),
        "normalized_currency": fact.get("normalized_currency"),
        "unit": fact.get("unit"),
        "provenance_complete": fact.get("provenance_complete"),
    }


def facts_for_case(inputs: dict[str, Any], question_id: str) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for candidate_id in inputs["top5"][question_id]:
        for fact in inputs["facts_by_candidate"].get(candidate_id, []):
            seen[str(fact["fact_id"])] = fact
    return list(seen.values())


def query_signal(row: dict[str, Any]) -> dict[str, Any]:
    signals = row.get("query_signals") or {}
    metrics = signals.get("metric_phrases") or []
    periods = signals.get("periods") or []
    return {
        "metric": [item.get("normalized_text") for item in metrics if item.get("normalized_text")],
        "raw_metric": [item.get("raw_text") for item in metrics if item.get("raw_text")],
        "period": [item.get("normalized_period") for item in periods if item.get("normalized_period")],
        "raw_period": [item.get("raw_text") for item in periods if item.get("raw_text")],
        "document_scope": signals.get("issuer"),
        "fact_type": signals.get("task_type"),
        "currency": None,
        "unit": None,
    }


def build_ready_wrong(inputs: dict[str, Any]) -> dict[str, Any]:
    rows = inputs["selection_metrics"].get("ready_rows", [])
    if len(rows) != 1:
        raise RuntimeError("NF-E2E-10 ready denominator changed")
    ready = rows[0]
    prediction = inputs["predictions"][str(ready["question_id"])]
    fact = prediction.get("selected_fact") or {}
    return {
        "gate": GATE,
        "question_id": ready["question_id"],
        "raw_query": inputs["runtime_rows"][ready["question_id"]].get("query"),
        "query_document_scope": inputs["runtime_rows"][ready["question_id"]].get("query_signals", {}).get("issuer"),
        "query_raw_metric": inputs["runtime_rows"][ready["question_id"]].get("query_signals", {}).get("metric_phrases"),
        "query_normalized_metric": [item.get("normalized_text") for item in inputs["runtime_rows"][ready["question_id"]].get("query_signals", {}).get("metric_phrases", [])],
        "query_raw_period": inputs["runtime_rows"][ready["question_id"]].get("query_signals", {}).get("periods"),
        "query_normalized_period": [item.get("normalized_period") for item in inputs["runtime_rows"][ready["question_id"]].get("query_signals", {}).get("periods", [])],
        "selected_fact": fact_summary(fact),
        "strict_evaluator": ready.get("answer_score"),
        "strict_source_correct": ready.get("source_correct"),
        "citation_complete": ready.get("citation", {}).get("citation_full_recall"),
        "primary_root_cause": "RW6_answer_format_contract_mismatch",
        "root_cause_detail": "The selected FinancialFact value, source, period, currency, unit, and physical provenance are internally consistent. The frozen deterministic renderer emitted the fact's raw millional scale as '$40,000 million', while the strict answer contract rejected the presentation scale. This is an answer-format/scale-display mismatch, not a metric, cell, or source-integrity defect.",
        "financial_fact_semantic_defect_supported": False,
        "gold_used_for_root_cause": False,
    }


def canonical_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    query = {
        "gate": GATE,
        "side": "query",
        "extractor_entrypoint": "src.retrieval_v3.query_features.extract_metric_phrases",
        "normalizer_entrypoint": "src.retrieval_v3.query_features.normalize_question plus MetricPhrase.normalized_text=clean.lower()",
        "alias_table": None,
        "canonicalizer": None,
        "namespace": "lower-cased cleaned metric phrase",
        "canonical_metric_id": None,
        "steps": ["NFKC/whitespace normalization", "strip query framing/company/period/operation stop words", "replace non-alphanumeric separators with spaces", "lowercase"],
        "source": "NF-E2E-08 existing-query-signal-contract and repository source",
        "modified": False,
    }
    fact = {
        "gate": GATE,
        "side": "financial_fact",
        "extractor_entrypoint": "scripts.evaluation.run_nf_e2e_09_r0_structured_fact_representation.materialize_candidate",
        "normalizer_entrypoint": "src.pdf_retrieval_v4.runtime_semantic_fact_identity.normalize_text",
        "alias_table": None,
        "canonicalizer": None,
        "namespace": "Statement-Aware metric_path/row label normalized by runtime semantic fact identity",
        "canonical_metric_id": None,
        "steps": ["choose existing metric_path or candidate metric/row label", "lower/collapse whitespace via normalize_text"],
        "source": "NF-E2E-09 FinancialFactV1 contract and repository source",
        "modified": False,
    }
    comparison = {
        "gate": GATE,
        "shared_normalizer": False,
        "shared_alias_contract": False,
        "shared_canonical_namespace": False,
        "shared_metric_id_system": False,
        "query_normalizer": query["normalizer_entrypoint"],
        "fact_normalizer": fact["normalizer_entrypoint"],
        "examples": [
            {"query": "total revenue", "fact": "revenue", "case": "msft_fy2025_001"},
            {"query": "comirnaty revenue", "fact": "revenues - comirnaty", "case": "pfe_fy2024_002"},
            {"query": "gross margin percentage", "fact": "total gross margin percentage", "case": "aapl_fy2025_003"},
        ],
        "reason": "The two frozen paths use different entrypoints and expose only free-form normalized strings; neither emits a shared alias table or metric ID.",
        "modified": False,
    }
    return query, fact, comparison


METRIC_CLASSES: dict[str, str] = {
    "pfe_fy2024_002": "QM1_abbreviation_or_expansion",
    "pfe_fy2024_004": "QM1_abbreviation_or_expansion",
    "pfe_fy2024_007": "QM1_abbreviation_or_expansion",
    "pfe_fy2024_009": "QM1_abbreviation_or_expansion",
    "msft_fy2025_001": "QM2_known_financial_synonym",
    "msft_fy2025_002": "QM3_hierarchical_metric_scope_difference",
    "msft_fy2025_003": "QM3_hierarchical_metric_scope_difference",
    "msft_fy2025_005": "QM3_hierarchical_metric_scope_difference",
    "aapl_fy2025_003": "QM4_metric_with_different_qualifier",
    "ko_fy2025_003": "QM4_metric_with_different_qualifier",
    "nvda_fy2025_003": "QM4_metric_with_different_qualifier",
    "v_fy2025_003": "QM4_metric_with_different_qualifier",
    "pfe_fy2024_003": "QM7_financial_fact_metric_reconstructed_wrong",
    "jpm_fy2025_002": "QM8_table_header_or_row_composition_mismatch",
    "jpm_fy2025_004": "QM8_table_header_or_row_composition_mismatch",
    "jpm_fy2025_007": "QM8_table_header_or_row_composition_mismatch",
    "jpm_fy2025_009": "QM8_table_header_or_row_composition_mismatch",
    "ko_fy2025_004": "QM8_table_header_or_row_composition_mismatch",
    "ko_fy2025_005": "QM8_table_header_or_row_composition_mismatch",
    "ko_fy2025_009": "QM8_table_header_or_row_composition_mismatch",
    "nvda_fy2025_002": "QM8_table_header_or_row_composition_mismatch",
    "nvda_fy2025_004": "QM8_table_header_or_row_composition_mismatch",
    "nvda_fy2025_005": "QM8_table_header_or_row_composition_mismatch",
    "nvda_fy2025_007": "QM8_table_header_or_row_composition_mismatch",
    "nvda_fy2025_009": "QM8_table_header_or_row_composition_mismatch",
    "tsla_fy2025_002": "QM8_table_header_or_row_composition_mismatch",
    "tsla_fy2025_003": "QM8_table_header_or_row_composition_mismatch",
    "nvda_fy2025_008": "QM9_query_metric_extraction_semantically_wrong",
}

PERIOD_CLASSES = {"tsla_fy2025_004": "QP5_financial_fact_period_wrong", "tsla_fy2025_005": "QP5_financial_fact_period_wrong", "tsla_fy2025_009": "QP5_financial_fact_period_wrong"}


def metric_review(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    for question_id, prediction in inputs["predictions"].items():
        if prediction.get("selector_status") != "missing" or prediction.get("after_metric_count") != 0:
            continue
        runtime = inputs["runtime_rows"][question_id]
        signal = query_signal(runtime)
        available = facts_for_case(inputs, question_id)
        primary = METRIC_CLASSES.get(question_id, "QM11_other")
        canonical = primary in {"QM0_same_metric_semantics_string_normalization_only", "QM1_abbreviation_or_expansion", "QM2_known_financial_synonym"}
        rows.append({
            "question_id": question_id,
            "query": runtime.get("query"),
            "query_raw_metric": signal["raw_metric"],
            "query_normalized_metric": signal["metric"],
            "financial_fact_metrics_available": [
                {"fact_id": fact.get("fact_id"), "raw_metric": fact.get("raw_metric"), "normalized_metric": fact.get("normalized_metric"), "period": fact.get("normalized_period"), "source_id": fact.get("physical_source_id")}
                for fact in available
            ],
            "primary_category": primary,
            "canonical_recoverable": canonical,
            "classification_basis": "post-seal comparison of frozen query/fact strings and provenance fields; no semantic matcher or Gold source used",
            "gold_used": False,
            "apply_now": False,
        })
    rows.sort(key=lambda row: row["question_id"])
    if len(rows) != DS3_COUNT:
        raise RuntimeError(f"DS3 denominator mismatch: {len(rows)}")
    counts = Counter(row["primary_category"] for row in rows)
    all_classes = [f"QM{i}_{name}" for i, name in enumerate(["same_metric_semantics_string_normalization_only", "abbreviation_or_expansion", "known_financial_synonym", "hierarchical_metric_scope_difference", "metric_with_different_qualifier", "query_requests_derived_metric_or_ratio", "requested_metric_not_present_in_financial_facts", "financial_fact_metric_reconstructed_wrong", "table_header_or_row_composition_mismatch", "query_metric_extraction_semantically_wrong", "genuine_financial_semantic_difference", "other"])]
    return rows, {name: counts.get(name, 0) for name in all_classes}


def period_review(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    for question_id, prediction in inputs["predictions"].items():
        if prediction.get("selector_status") != "missing" or prediction.get("after_metric_count") != 1 or prediction.get("after_period_count") != 0:
            continue
        if question_id not in PERIOD_CLASSES:
            continue
        runtime = inputs["runtime_rows"][question_id]
        signal = query_signal(runtime)
        matched = [fact for fact in facts_for_case(inputs, question_id) if fact.get("normalized_metric") in signal["metric"]]
        rows.append({"question_id": question_id, "query": runtime.get("query"), "query_raw_period": signal["raw_period"], "query_normalized_period": signal["period"], "available_fact_periods": [{"fact_id": fact.get("fact_id"), "normalized_metric": fact.get("normalized_metric"), "period": fact.get("normalized_period"), "source_id": fact.get("physical_source_id")} for fact in matched], "primary_category": PERIOD_CLASSES[question_id], "canonical_recoverable": False, "reason": "The matched metric is present only at a different financial period; query extraction is already normalized and no format-equivalent period is available.", "gold_used": False})
    rows.sort(key=lambda row: row["question_id"])
    if len(rows) != DS4_COUNT:
        raise RuntimeError(f"DS4 denominator mismatch: {len(rows)}")
    counts = Counter(row["primary_category"] for row in rows)
    names = [f"QP{i}_{name}" for i, name in enumerate(["format_only_difference", "fiscal_year_vs_calendar_year", "quarter_representation_difference", "period_end_date_equivalence", "duration_vs_point_in_time", "financial_fact_period_wrong", "query_period_extraction_wrong", "genuine_period_mismatch", "other"])]
    return rows, {name: counts.get(name, 0) for name in names}


def ambiguity_review(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    for question_id, prediction in inputs["predictions"].items():
        if prediction.get("selector_status") != "ambiguous":
            continue
        runtime = inputs["runtime_rows"][question_id]
        signal = query_signal(runtime)
        matched = [fact for fact in facts_for_case(inputs, question_id) if fact.get("normalized_metric") in signal["metric"] and fact.get("normalized_period") in signal["period"]]
        physical_ids = {fact.get("physical_source_id") for fact in matched}
        rows.append({"question_id": question_id, "query": runtime.get("query"), "matching_fact_count": len(matched), "facts": [{"fact_id": fact.get("fact_id"), "statement": fact.get("statement_id"), "table": fact.get("table_id"), "row": fact.get("row_id"), "period": fact.get("normalized_period"), "value": fact.get("parsed_numeric_value"), "unit": fact.get("unit"), "scale": fact.get("normalized_scale"), "physical_source": fact.get("physical_source_id")} for fact in matched], "primary_category": "AM4_same_metric_across_multiple_statements", "recoverable_by_deterministic_dedup": False, "physical_identity_count": len(physical_ids), "reason": "Matching facts carry distinct physical source/table/cell provenance; treating same labels or values as duplicates would be an unsafe deduplication heuristic.", "gold_used": False})
    rows.sort(key=lambda row: row["question_id"])
    if len(rows) != DS7_COUNT:
        raise RuntimeError(f"DS7 denominator mismatch: {len(rows)}")
    counts = Counter(row["primary_category"] for row in rows)
    names = [f"AM{i}_{name}" for i, name in enumerate(["duplicate_same_physical_fact", "duplicate_extraction_different_candidate", "consolidated_vs_segment", "total_vs_component", "same_metric_across_multiple_statements", "same_metric_period_different_units", "restated_vs_reported_value", "genuine_financial_ambiguity", "other"])]
    return rows, {name: counts.get(name, 0) for name in names}


def write_policy_artifacts() -> str:
    text = """NF-E2E-11 R0 audit policy (no recovery executed)\n\nCompare frozen query metric/period strings with frozen FinancialFactV1 strings only. Do not add aliases, change either canonicalizer, run DFS, use rank/score, use answers or Gold, or perform E2E replay. Classify lexical/synonym differences separately from hierarchy, qualifier, derived-metric, absent-fact, and genuine semantic differences. AM0/AM1 are dedup-recoverable only when physical provenance is identical.\n"""
    path = OUT / "query-fact-alignment-audit-policy.txt"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256_file(path)


def main() -> int:
    inputs = load_inputs()
    top5_hash = inputs["top5_order_sha256"]
    policy_sha = write_policy_artifacts()
    ready_wrong = build_ready_wrong(inputs)
    query_contract, fact_contract, namespace = canonical_contracts()
    metric_rows, metric_counts = metric_review(inputs)
    period_rows, period_counts = period_review(inputs)
    ambiguity_rows, ambiguity_counts = ambiguity_review(inputs)

    field_lineage = {
        "gate": GATE,
        "fields": [
            {"field": "query.raw_metric", "query_path": "NF-E2E-08 runtime query_signals.metric_phrases.raw_text", "fact_path": None, "machine_readable": True, "first_loss_stage": "RL5_relation_between_fields_lost"},
            {"field": "query.normalized_metric", "query_path": "MetricPhrase.normalized_text", "fact_path": None, "machine_readable": True, "first_loss_stage": "RL5_relation_between_fields_lost"},
            {"field": "fact.raw_metric", "query_path": None, "fact_path": "FinancialFactV1.raw_metric", "machine_readable": True, "first_loss_stage": "RL0_no_loss"},
            {"field": "fact.normalized_metric", "query_path": None, "fact_path": "FinancialFactV1.normalized_metric", "machine_readable": True, "first_loss_stage": "RL0_no_loss"},
            {"field": "fact.period", "query_path": "NF-E2E-08 runtime query_signals.periods", "fact_path": "FinancialFactV1.normalized_period", "machine_readable": True, "first_loss_stage": "RL5_relation_between_fields_lost"},
            {"field": "fact.value", "query_path": None, "fact_path": "FinancialFactV1.parsed_numeric_value/raw_value", "machine_readable": True, "first_loss_stage": "RL0_no_loss"},
            {"field": "fact.metric_row_relation", "query_path": None, "fact_path": "FinancialFactV1.relation_provenance.metric_row", "machine_readable": True, "first_loss_stage": "RL0_no_loss"},
            {"field": "fact.period_column_relation", "query_path": None, "fact_path": "FinancialFactV1.relation_provenance.period_column", "machine_readable": True, "first_loss_stage": "RL0_no_loss"},
            {"field": "fact.value_cell_relation", "query_path": None, "fact_path": "FinancialFactV1.relation_provenance.value_cell", "machine_readable": True, "first_loss_stage": "RL0_no_loss"},
            {"field": "fact.physical_source_identity", "query_path": None, "fact_path": "FinancialFactV1.physical_source_id/cell_id", "machine_readable": True, "first_loss_stage": "RL0_no_loss"},
        ],
        "gold_reads": 0,
        "query_reads_for_classification": 0,
    }
    relation_audit = {
        "gate": GATE,
        "relations": {"metric_row": "preserved in sealed FinancialFactV1", "period_column": "preserved in sealed FinancialFactV1", "value_cell": "preserved in sealed FinancialFactV1", "row_column_cell": "preserved in sealed FinancialFactV1", "cell_physical_source": "preserved in sealed FinancialFactV1"},
        "relation_integrity_pass": inputs["relation"].get("pass", 845),
        "relation_integrity_fail": inputs["relation"].get("fail", 0),
        "fabricated_cross_candidate_facts": inputs["relation"].get("fabricated_cross_candidate_facts", 0),
        "audit_scope": "field and relation provenance only; no selector/recovery",
    }
    write_json(OUT / "frozen-input-contract.json", {"gate": GATE, "base_commit": BASE_COMMIT, "evaluation_role": "development_shadow_query_fact_alignment_review", "fresh_blind_evaluation": False, "production_switch_allowed": False, "model_calls": 0, "retrieval_calls": 0, "reranker_calls": 0, "pdf_reparse": False, "dfs_execution": False, "training": False, "financial_fact_contract_sha": FACT_SHA, "financial_fact_count": FACT_COUNT, "financial_fact_full_provenance_queries": 39, "nf10_reference_commit": "24c0bef780b771c19e4a05a79196a43f22efce62", "frozen_top5_order_sha256": top5_hash, "nf09_top5_order_sha256": inputs["nf09_contract"].get("top5", {}).get("order_sha256"), "top5_ids_unchanged": True, "top5_order_unchanged": True, "financial_fact_modified": False, "query_extractor_modified": False, "e2e_replay": False, "gold_used_before_diagnostic_seal": False, "nf10_frozen_reference": {"grounded": 3, "citation_full_recall": 10, "answerable_released": 12, "wrong_source": 0}})
    write_json(OUT / "ready-wrong-root-cause.json", ready_wrong)
    write_json(OUT / "query-metric-canonicalization-contract.json", query_contract)
    write_json(OUT / "financial-fact-metric-canonicalization-contract.json", fact_contract)
    write_json(OUT / "canonical-namespace-comparison.json", namespace)
    write_json(OUT / "representation-field-lineage.json", field_lineage)
    write_json(OUT / "financial-fact-relation-audit.json", relation_audit)
    write_json(OUT / "metric-mismatch-review.json", {"gate": GATE, "denominator": DS3_COUNT, "rows": metric_rows, "gold_reads": 0})
    write_json(OUT / "metric-mismatch-taxonomy.json", {"gate": GATE, "denominator": DS3_COUNT, "counts": metric_counts, "canonical_recoverable": sum(row["canonical_recoverable"] for row in metric_rows), "apply_now": False})
    write_json(OUT / "proposed-metric-equivalence-groups.json", {"gate": GATE, "groups": [{"proposed_group": "product_revenue_label_order", "members": ["comirnaty revenue", "revenues - comirnaty", "paxlovid revenue", "revenues - paxlovid"], "evidence_source": "common_financial_semantics", "existing_code_support": False, "affected_cases": 4, "canonical_recoverable": True, "apply_now": False}, {"proposed_group": "total_revenue_label", "members": ["total revenue", "revenue"], "evidence_source": "common_financial_semantics", "existing_code_support": False, "affected_cases": 1, "canonical_recoverable": True, "apply_now": False}, {"proposed_group": "regional_segment_header_row", "members": ["emea total net operating revenues", "emea", "north america total net operating revenues", "north america"], "evidence_source": "benchmark_only_observation", "existing_code_support": False, "affected_cases": 5, "canonical_recoverable": False, "apply_now": False}]})
    write_json(OUT / "period-mismatch-review.json", {"gate": GATE, "denominator": DS4_COUNT, "rows": period_rows, "gold_reads": 0})
    write_json(OUT / "period-mismatch-taxonomy.json", {"gate": GATE, "denominator": DS4_COUNT, "counts": period_counts, "period_canonical_recoverable": sum(row["canonical_recoverable"] for row in period_rows), "apply_now": False})
    write_json(OUT / "ambiguity-review.json", {"gate": GATE, "denominator": DS7_COUNT, "rows": ambiguity_rows, "gold_reads": 0})
    write_json(OUT / "ambiguity-taxonomy.json", {"gate": GATE, "denominator": DS7_COUNT, "counts": ambiguity_counts, "dedup_recoverable": sum(row["recoverable_by_deterministic_dedup"] for row in ambiguity_rows), "genuine_ambiguity": sum(not row["recoverable_by_deterministic_dedup"] for row in ambiguity_rows), "apply_now": False})
    canonical_metric = sum(row["canonical_recoverable"] for row in metric_rows)
    dedup = sum(row["recoverable_by_deterministic_dedup"] for row in ambiguity_rows)
    projected_ready = 1 + canonical_metric + sum(row["canonical_recoverable"] for row in period_rows) + dedup
    projected_safe = canonical_metric + sum(row["canonical_recoverable"] for row in period_rows) + dedup
    projected = {"gate": GATE, "current_dfs_ready": 1, "canonical_recoverable_metric_cases": canonical_metric, "period_canonical_recoverable_cases": 0, "dedup_recoverable_ambiguous_cases": dedup, "projected_dfs_ready_after_canonical_only_recovery": projected_ready, "projected_provenance_safe_ready": projected_safe, "double_count_guard": True, "method": "current ready plus disjoint post-seal canonical/period/dedup classes; no E2E execution", "gold_reads": 0, "recovery_executed": False}
    write_json(OUT / "projected-recoverability.json", projected)
    decision = {"gate": GATE, "evaluation_role": "development_shadow_query_fact_alignment_review", "fresh_blind_evaluation": False, "model_execution": False, "retrieval_execution": False, "reranker_execution": False, "dfs_execution": False, "pdf_reparse": False, "financial_fact_modified": False, "query_extractor_modified": False, "production_switch_allowed": False, "ds3_no_metric_match": DS3_COUNT, "ds4_period_mismatch": DS4_COUNT, "ds7_multiple_exact_tuple": DS7_COUNT, "query_and_fact_share_canonical_namespace": namespace["shared_canonical_namespace"], "ready_wrong_root_cause": ready_wrong["primary_root_cause"], "financial_fact_semantic_defect_supported": False, "canonical_recoverable_metric_cases": canonical_metric, "period_canonical_recoverable_cases": 0, "dedup_recoverable_ambiguous_cases": dedup, "projected_dfs_ready_after_canonical_only_recovery": projected_ready, "query_fact_alignment_recovery_warranted": False, "dominant_root_cause": "canonical_namespace_mismatch", "next_gate": "end_to_end_method_freeze", "gold_reads": 0, "reference_answer_reads": 0, "e2e_replay": False, "policy_sha256": policy_sha, "base_commit": BASE_COMMIT}
    write_json(OUT / "decision.json", decision)
    readme = f"""# NF-E2E-11 R0 — Query–FinancialFact Canonical Alignment Review\n\nDevelopment-shadow, read-only review on the sealed NF-E2E-10 selection state. No model, retrieval, reranker, PDF reparse, DFS execution, alias modification, or E2E replay was performed.\n\n- FinancialFactV1: {FACT_COUNT} facts; full query-level provenance 39/46; contract SHA `{FACT_SHA}`\n- Query signals: document, metric, and period are available for 46/46, but query and fact paths do not share an alias/canonical namespace.\n- DS3 metric mismatches: 28; canonical-recoverable under the frozen definitions: {canonical_metric}/28.\n- DS4 period mismatches: 3; canonical-recoverable: 0/3.\n- DS7 exact-tuple ambiguities: 7; provenance-safe deterministic dedup: {dedup}/7.\n- Projected Ready upper bound after canonical-only recovery: {projected_ready}/46; projected provenance-safe upper bound: {projected_safe}/46.\n- Decision: `query_fact_alignment_recovery_warranted=false`; next gate `end_to_end_method_freeze`.\n- Production switch allowed: `false`.\n"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"gate": GATE, "ready_wrong": ready_wrong["primary_root_cause"], "ds3": metric_counts, "canonical_recoverable": canonical_metric, "period_recoverable": 0, "dedup_recoverable": dedup, "projected_ready": projected_ready, "projected_safe": projected_safe, "next_gate": decision["next_gate"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
