#!/usr/bin/env python3
"""NF-V2-02: query-independent FinancialFactV1 materialization over Top20.

This gate reuses the sealed SADA Top100 ordering and the existing NF-E2E-09
SFFM-V1 materializer.  It deliberately does not retrieve, rerank, call a
model, read query text, or run any downstream component while facts are being
materialized.  Gold/label data is opened only after the materialization seal
has been written, for cohort attribution and supply reporting.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts/evaluation/nf-v2-02-top20-financial-fact-expansion"
NF01 = ROOT / "artifacts/evaluation/nf-e2e-01-r0-frozen-retrieval-integration-review"
NF08 = ROOT / "artifacts/evaluation/nf-e2e-08-r0-deterministic-fact-selection-recovery"
NF09 = ROOT / "artifacts/evaluation/nf-e2e-09-r0-structured-financial-fact-representation"
NF24 = ROOT / "artifacts/evaluation/nf-opt-24-r0-deep-supply-top100-admission"
NF26 = ROOT / "artifacts/evaluation/nf-opt-26-r0-internal-retrieval-freeze"
GATE03 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-03-r2"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
PLANS = ROOT / "artifacts/evaluation/nf-v2-01-r1-bailian-formal-72-attempt-2/supervisor-plans.jsonl.gz"

GATE = "NF-V2-02"
BASE_COMMIT = "f9278578518a764845993907d45223914c9f8194"
NF26_MANIFEST_SHA = "70048502ec918ae6ee56246a788da42129df3b073c2be8682e14f97e409e7c80"
FACT_CONTRACT_SHA = "7a253b443962c5f372dd897c49c057a19b553e92314faadc31eefc82b27b54eb"
TOP100_PREDICTIONS = NF24 / "sada-v1-top100-predictions.jsonl.gz"
TOP100_SERIALIZATION = NF24 / "serialization-manifest.jsonl.gz"
TOP100_CONTRACT = NF24 / "deep-supply-contract.json"
TOP100_STATEMENT_CONTRACT = NF24 / "frozen-statement-aware-contract.json"
TOP100_ORDER_CONTRACT = NF24 / "current-top100-contract.json"
TOP_K = 20
TOP5 = 5
QUESTION_TOTAL = 72
HISTORICAL_FACT_TOTAL = 46
DIRECT_TOTAL = 56
MULTI_TOTAL = 5
HISTORICAL_MULTI_TOTAL = 16
CALC_TOTAL = 11


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pct(count: int, total: int) -> float:
    return round(count * 100.0 / total, 4) if total else 0.0


def p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))]


def import_nf01():
    from scripts.evaluation import run_nf_e2e_01_r0_frozen_retrieval_integration_review as module

    return module


def import_nf09():
    from scripts.evaluation import run_nf_e2e_09_r0_structured_fact_representation as module

    return module


def verify_frozen_top100() -> dict[str, Any]:
    """Load only sealed retrieval artifacts and verify their hashes/order."""

    manifest = NF26 / "final-evidence-manifest.json"
    if sha256_file(manifest) != NF26_MANIFEST_SHA:
        raise RuntimeError("NF-OPT-26 final evidence manifest hash mismatch")
    if (NF26 / "final-evidence-manifest.sha256").read_text(encoding="utf-8").strip() != NF26_MANIFEST_SHA:
        raise RuntimeError("NF-OPT-26 sidecar hash mismatch")
    method = read_json(NF26 / "internal-retrieval-method-freeze.json")
    if method.get("selected_internal_shadow_method") != "sada_statement_aware_v1":
        raise RuntimeError("unexpected frozen retrieval method")
    if method.get("production_switch_allowed") is not False:
        raise RuntimeError("production guardrail missing")
    nf01 = import_nf01()
    cases, counts = nf01.load_sada_inputs(ROOT)
    if len(cases) != QUESTION_TOTAL or any(len(items) != 100 for items in cases.values()):
        raise RuntimeError("frozen SADA Top100 is not 72 x 100")
    top100_order = {case_id: [item["candidate_key"] for item in items] for case_id, items in sorted(cases.items())}
    top20_order = {case_id: ids[:TOP_K] for case_id, ids in top100_order.items()}
    top5_order = {case_id: ids[:TOP5] for case_id, ids in top100_order.items()}
    candidate_occurrences = sum(len(ids) for ids in top20_order.values())
    if candidate_occurrences != QUESTION_TOTAL * TOP_K:
        raise RuntimeError("Top20 occurrence count mismatch")
    return {
        "method": method,
        "cases": cases,
        "top100_order": top100_order,
        "top20_order": top20_order,
        "top5_order": top5_order,
        "counts": counts,
        "top100_predictions_sha256": sha256_file(TOP100_PREDICTIONS),
        "serialization_manifest_sha256": sha256_file(TOP100_SERIALIZATION),
        "deep_supply_contract_sha256": sha256_file(TOP100_CONTRACT),
        "statement_aware_contract_sha256": sha256_file(TOP100_STATEMENT_CONTRACT),
        "top100_contract_sha256": sha256_file(TOP100_ORDER_CONTRACT),
        "top100_order_sha256": stable_sha(top100_order),
        "top20_order_sha256": stable_sha(top20_order),
        "top5_order_sha256": stable_sha(top5_order),
        "candidate_occurrences": candidate_occurrences,
        "unique_top20_candidates": len({candidate for values in top20_order.values() for candidate in values}),
        "top20_candidate_ids_sha256": stable_sha(sorted({candidate for values in top20_order.values() for candidate in values})),
    }


def candidate_rows(state: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Convert existing Statement-Aware serialization to materializer input."""

    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    unique: dict[str, dict[str, Any]] = {}
    for case_id, items in sorted(state["cases"].items()):
        rows: list[dict[str, Any]] = []
        for item in items[:TOP_K]:
            parsed = item["parsed"]
            row = {
                "case_id": case_id,
                "candidate_id": str(item["candidate_key"]),
                "candidate_rank": int(item["rank"]),
                "physical_source_id": parsed.get("physical_source_id"),
                "document_id": parsed.get("document_id"),
                "pdf_page": parsed.get("page"),
                "statement_id": parsed.get("statement"),
                "table_id": parsed.get("table_id"),
                "table_title": parsed.get("table_title"),
                "metric": parsed.get("metric_path") or parsed.get("row_label"),
                "normalized_metric": parsed.get("metric_path") or parsed.get("row_label"),
                "row_label": parsed.get("row_label"),
                "row_id": parsed.get("row_id"),
                "column_header": list(parsed.get("column_headers") or []),
                "normalized_periods": [],
                "period_value_bindings": list(parsed.get("period_value_bindings") or []),
                "raw_value": None,
                "parsed_numeric_value": None,
                "currency": parsed.get("currency"),
                "scale": parsed.get("scale"),
                "unit": None,
                "cell_id": None,
                "physical_source_identity_complete": bool(parsed.get("document_id") and parsed.get("table_id") and parsed.get("row_id") and parsed.get("page") is not None),
                "source_text": item["serialization"],
                "statement_serialization_sha256": item["serialization_sha256"],
            }
            rows.append(row)
            unique.setdefault(str(row["candidate_id"]), {key: value for key, value in row.items() if key != "case_id"})
        rows_by_case[case_id] = rows
    return rows_by_case, unique


def candidate_materialization(state: dict[str, Any]) -> dict[str, Any]:
    nf09 = import_nf09()
    atomic, atomic_index = nf09.load_atomic_facts()
    _, unique = candidate_rows(state)
    raw_facts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for candidate_id in sorted(unique):
        facts, candidate_failures = nf09.materialize_candidate(unique[candidate_id], atomic_index)
        raw_facts.extend(facts)
        failures.extend(candidate_failures)
    wall_ms = round((time.perf_counter() - started) * 1000.0, 3)
    facts, duplicate_count = nf09.dedup_facts(raw_facts)
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for fact in raw_facts:
        duplicate_groups[str(fact["fact_id"])].append(str(fact["candidate_id"]))
    duplicate_groups = {fact_id: sorted(set(candidate_ids)) for fact_id, candidate_ids in duplicate_groups.items() if len(set(candidate_ids)) > 1}
    return {
        "atomic_count": len(atomic),
        "atomic_index": atomic_index,
        "unique": unique,
        "raw_facts": raw_facts,
        "facts": facts,
        "failures": failures,
        "duplicate_count": duplicate_count,
        "duplicate_groups": duplicate_groups,
        "wall_ms": wall_ms,
    }


def verify_materialized_relations(facts: list[dict[str, Any]]) -> dict[str, Any]:
    required = ("metric_row", "period_column", "value_cell", "row_column_cell", "cell_physical_source")
    failures: list[dict[str, Any]] = []
    passed = 0
    for fact in facts:
        provenance = fact.get("relation_provenance") or {}
        for relation in required:
            if provenance.get(relation) is True:
                passed += 1
            else:
                failures.append({"fact_id": fact.get("fact_id"), "relation": relation})
        if not fact.get("candidate_id") or not fact.get("physical_source_id") or not fact.get("cell_id") or fact.get("provenance_complete") is not True:
            failures.append({"fact_id": fact.get("fact_id"), "relation": "complete_provenance_identity"})
    return {"relation_integrity_pass": passed, "relation_integrity_fail": len(failures), "facts_checked": len(facts), "relations_per_fact": len(required), "failures": failures, "fabricated_cross_candidate_facts": 0}


def verify_no_fabrication(facts: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for fact in facts:
        candidate_ids = fact.get("candidate_ids") or [fact.get("candidate_id")]
        provenance = fact.get("relation_provenance") or {}
        if not candidate_ids or not fact.get("candidate_id") or not provenance.get("candidate_atomic_identity_bridge"):
            failures.append({"fact_id": fact.get("fact_id"), "reason": "missing_single_candidate_provenance_chain"})
    return {"fabricated_cross_candidate_facts": 0 if not failures else len(failures), "facts_checked": len(facts), "failures": failures, "cross_candidate_composition": False, "single_candidate_chain_required": True}


def load_top5_facts() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    facts = read_jsonl_gz(NF09 / "financial-facts-v1.jsonl.gz")
    seal = read_json(NF09 / "financial-facts-v1-seal.json")
    return facts, seal


def fact_map(facts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        for candidate_id in fact.get("candidate_ids", [fact.get("candidate_id")]):
            by_candidate[str(candidate_id)].append(fact)
    return by_candidate


def candidate_fact_counts(top20: dict[str, list[dict[str, Any]]], facts: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate = fact_map(facts)
    all_candidates = {candidate for values in top20.values() for candidate in values}
    counts = [len({fact["fact_id"] for fact in by_candidate.get(candidate, [])}) for candidate in sorted(all_candidates)]
    table = sum(1 for fact in facts if fact.get("table_id") and fact.get("row_id") and fact.get("cell_id"))
    narrative = sum(1 for fact in facts if not (fact.get("table_id") and fact.get("row_id") and fact.get("cell_id")))
    return {
        "unique_top20_candidates": len(all_candidates),
        "candidates_with_facts": sum(int(count > 0) for count in counts),
        "candidates_with_provenance_complete_fact": sum(int(count > 0) for count in counts),
        "facts_per_candidate": {"mean": round(statistics.mean(counts), 4) if counts else 0.0, "median": statistics.median(counts) if counts else 0.0, "p95": p95(counts), "max": max(counts) if counts else 0},
        "facts_per_candidate_distribution": counts,
        "facts_materialized": len(facts),
        "provenance_complete_facts": sum(int(fact.get("provenance_complete") is True) for fact in facts),
        "numeric_parse_success": sum(int(fact.get("parsed_numeric_value") is not None) for fact in facts),
        "table_backed_facts": table,
        "narrative_facts": narrative,
        "other_facts": len(facts) - table - narrative,
    }


def load_postseal_annotations() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    labels = {str(row["case_id"]): row for row in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()) if row}
    plans = {str(row["question_id"]): row for row in read_jsonl_gz(PLANS)}
    return labels, plans


def expected_source_keys(label: dict[str, Any]) -> list[str]:
    return [str(item["candidate_key"]) for item in label.get("expected_sources", []) if item.get("candidate_key")]


def top20_query_rows(top20: dict[str, list[str]], facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate = fact_map(facts)
    rows: list[dict[str, Any]] = []
    for case_id in sorted(top20):
        available = [fact for candidate in top20[case_id] for fact in by_candidate.get(candidate, [])]
        fact_ids = sorted({fact["fact_id"] for fact in available})
        rows.append({"question_id": case_id, "top20_candidate_count": len(top20[case_id]), "financial_fact_count": len(fact_ids), "financial_fact_available": bool(fact_ids), "provenance_complete_fact_available": bool(fact_ids), "provenance_complete_fact_count": len(fact_ids)})
    return rows


def postseal_cohort_supply(top20: dict[str, list[str]], facts: list[dict[str, Any]], labels: dict[str, dict[str, Any]], plans: dict[str, dict[str, Any]], historical_ids: set[str]) -> dict[str, Any]:
    by_candidate = fact_map(facts)
    all_rows = top20_query_rows(top20, facts)
    direct_ids = sorted(qid for qid, plan in plans.items() if plan.get("intent") == "DIRECT_FACT")
    multi_ids = sorted(qid for qid, plan in plans.items() if plan.get("intent") == "MULTI_EVIDENCE")
    calc_ids = sorted(qid for qid, label in labels.items() if label.get("calculation") is not None)

    def source_supply(case_id: str) -> dict[str, Any]:
        required = expected_source_keys(labels.get(case_id, {}))
        candidates = set(top20.get(case_id, []))
        admitted = [key in candidates for key in required]
        covered = [bool(by_candidate.get(key)) for key in required]
        return {"question_id": case_id, "required_physical_source_count": len(required), "required_candidate_keys": required, "any_required_source_admitted": any(admitted), "all_required_physical_sources_admitted": bool(required) and all(admitted), "any_required_source_fact_supply": any(covered), "all_required_sources_with_provenance_complete_fact_supply": bool(required) and all(covered)}

    def cohort(ids: list[str]) -> dict[str, Any]:
        rows = [source_supply(case_id) for case_id in ids]
        return {"denominator": len(ids), "any_required_evidence_represented": sum(int(row["any_required_source_fact_supply"]) for row in rows), "all_required_physical_sources_admitted": sum(int(row["all_required_physical_sources_admitted"]) for row in rows), "all_required_sources_with_provenance_complete_fact_supply": sum(int(row["all_required_sources_with_provenance_complete_fact_supply"]) for row in rows), "rows": rows}

    return {
        "all_72_query_rows": all_rows,
        "v2_direct_fact_56": {"expected_denominator": DIRECT_TOTAL, **cohort(direct_ids)},
        "v2_multi_evidence_5": {"expected_denominator": MULTI_TOTAL, **cohort(multi_ids)},
        "calculation_11": {"expected_denominator": CALC_TOTAL, **cohort(calc_ids)},
        "historical_multi_evidence_16": {"expected_denominator": HISTORICAL_MULTI_TOTAL, **cohort(sorted(set(labels) & set(qid for qid, label in labels.items() if len(label.get("expected_sources", [])) > 1)))},
        "cohort_counts_observed": {"direct_fact": len(direct_ids), "multi_evidence": len(multi_ids), "calculation": len(calc_ids), "historical_fact": len(historical_ids)},
    }


def top5_top20_attribution(state: dict[str, Any], top20: dict[str, list[str]], facts: list[dict[str, Any]], labels: dict[str, dict[str, Any]], historical_ids: set[str]) -> dict[str, Any]:
    old_coverage = {str(row["question_id"]): row for row in read_json(NF09 / "query-level-coverage.json").get("rows", [])}
    by_candidate = fact_map(facts)
    counts = Counter()
    rows: list[dict[str, Any]] = []
    for case_id in sorted(historical_ids):
        expected = expected_source_keys(labels.get(case_id, {}))
        ranks = {candidate: rank for rank, candidate in enumerate(top20.get(case_id, []), 1)}
        # Keep the compatible cohort coverage definition aligned with NF-E2E-09
        # (any complete fact in the query's pool), while separately requiring
        # the expected physical source for T20-1/T20-2 recovery attribution.
        top20_full = any(bool(by_candidate.get(candidate)) for candidate in top20.get(case_id, []))
        expected_source_fact_supply = any(bool(by_candidate.get(key)) for key in expected if key in ranks)
        top5_full = bool(old_coverage.get(case_id, {}).get("full_provenance"))
        if top5_full:
            category = "T20-0_already_covered_in_top5"
        elif expected_source_fact_supply:
            new_ranks = [ranks[key] for key in expected if key in ranks and ranks[key] > TOP5]
            if new_ranks and min(new_ranks) <= 10:
                category = "T20-1_correct_candidate_rank_6_10"
            elif new_ranks and min(new_ranks) <= TOP_K:
                category = "T20-2_correct_candidate_rank_11_20"
            else:
                category = "T20-5_other"
        else:
            admitted = [key for key in expected if key in ranks and ranks[key] > TOP5]
            if admitted and not any(by_candidate.get(key) for key in admitted):
                category = "T20-3_candidate_admitted_fact_unavailable"
            elif admitted:
                category = "T20-4_fact_materialized_incomplete_provenance"
            else:
                category = "T20-5_other"
        counts[category] += 1
        rows.append({"question_id": case_id, "top5_full_provenance": top5_full, "top20_full_provenance": top20_full, "expected_source_fact_supply": expected_source_fact_supply, "expected_candidate_keys": expected, "expected_candidate_ranks": {key: ranks.get(key) for key in expected}, "primary_attribution": category})
    names = ["T20-0_already_covered_in_top5", "T20-1_correct_candidate_rank_6_10", "T20-2_correct_candidate_rank_11_20", "T20-3_candidate_admitted_fact_unavailable", "T20-4_fact_materialized_incomplete_provenance", "T20-5_other"]
    return {"denominator": len(historical_ids), "counts": {name: counts.get(name, 0) for name in names}, "newly_recovered": sum(counts.get(name, 0) for name in (names[1], names[2])), "rows": rows}


def calculation_supply(top20: dict[str, list[str]], facts: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_candidate = fact_map(facts)
    ids = sorted(case_id for case_id, label in labels.items() if label.get("calculation") is not None)
    rows: list[dict[str, Any]] = []
    for case_id in ids:
        label = labels[case_id]
        required = expected_source_keys(label)
        admitted = [key in set(top20.get(case_id, [])) for key in required]
        covered = [bool(by_candidate.get(key)) for key in required]
        rows.append({"question_id": case_id, "required_operand_source_count": len(required), "any_operand_fact_supply": any(covered), "all_required_operand_physical_sources_admitted": bool(required) and all(admitted), "all_required_operand_sources_with_provenance_complete_fact_supply": bool(required) and all(covered), "required_candidate_keys": required, "candidate_admitted": admitted, "fact_supply": covered})
    return {"denominator": len(ids), "any_operand_fact_supply": sum(int(row["any_operand_fact_supply"]) for row in rows), "all_required_operand_physical_sources_admitted": sum(int(row["all_required_operand_physical_sources_admitted"]) for row in rows), "calculation_fact_supply_complete": sum(int(row["all_required_operand_sources_with_provenance_complete_fact_supply"]) for row in rows), "rows": rows}


def multi_supply(top20: dict[str, list[str]], facts: list[dict[str, Any]], labels: dict[str, dict[str, Any]], plans: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_candidate = fact_map(facts)
    v2_ids = sorted(qid for qid, plan in plans.items() if plan.get("intent") == "MULTI_EVIDENCE")
    historical_ids = sorted(qid for qid, label in labels.items() if len(label.get("expected_sources", [])) > 1)

    def measure(ids: list[str]) -> dict[str, Any]:
        rows = []
        for case_id in ids:
            keys = expected_source_keys(labels[case_id])
            candidates = set(top20.get(case_id, []))
            admitted = [key in candidates for key in keys]
            covered = [bool(by_candidate.get(key)) for key in keys]
            rows.append({"question_id": case_id, "required_source_count": len(keys), "any_required_evidence_represented": any(covered), "all_required_physical_sources_admitted": bool(keys) and all(admitted), "all_required_sources_with_provenance_complete_fact_supply": bool(keys) and all(covered), "required_candidate_keys": keys})
        return {"denominator": len(ids), "any_required_evidence_represented": sum(int(row["any_required_evidence_represented"]) for row in rows), "all_required_physical_sources_admitted": sum(int(row["all_required_physical_sources_admitted"]) for row in rows), "all_required_sources_with_provenance_complete_fact_supply": sum(int(row["all_required_sources_with_provenance_complete_fact_supply"]) for row in rows), "rows": rows}

    return {"v2_supervisor_multi_evidence": measure(v2_ids), "historical_multi_evidence": measure(historical_ids), "v2_expected_denominator": MULTI_TOTAL, "historical_expected_denominator": HISTORICAL_MULTI_TOTAL}


def binder_projection(top20: dict[str, list[str]], facts: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate = fact_map(facts)
    rows = []
    for case_id in sorted(top20):
        fact_ids = {fact["fact_id"] for candidate in top20[case_id] for fact in by_candidate.get(candidate, [])}
        rows.append({"question_id": case_id, "number_of_top20_candidates": len(top20[case_id]), "number_of_materialized_facts": len(fact_ids), "number_of_provenance_complete_facts": len(fact_ids)})
    values = [row["number_of_materialized_facts"] for row in rows]
    return {"rows": rows, "denominator": len(rows), "median_facts_per_query": statistics.median(values) if values else 0.0, "p95_facts_per_query": p95(values), "maximum_facts_per_query": max(values) if values else 0, "projection_only": True, "binder_called": False}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    state = verify_frozen_top100()
    _, unique_rows = candidate_rows(state)
    top20_order = state["top20_order"]
    top5_order = state["top5_order"]
    write_json(OUT / "frozen-input-contract.json", {
        "gate": GATE,
        "base_commit": BASE_COMMIT,
        "evaluation_role": "development_shadow_v2_top20_financial_fact_expansion",
        "production_default": "V1",
        "production_switch_allowed": False,
        "model_calls": 0,
        "retrieval_execution": False,
        "reranker_execution": False,
        "frozen_retrieval_artifacts_reused": True,
        "top100_predictions_sha256": state["top100_predictions_sha256"],
        "serialization_manifest_sha256": state["serialization_manifest_sha256"],
        "sada_configuration_sha256": state["deep_supply_contract_sha256"],
        "statement_aware_configuration_sha256": state["statement_aware_contract_sha256"],
        "top100_order_sha256": state["top100_order_sha256"],
        "top20_order_sha256": state["top20_order_sha256"],
        "top5_order_sha256": state["top5_order_sha256"],
        "questions": QUESTION_TOTAL,
        "top20_candidate_occurrences": state["candidate_occurrences"],
        "unique_top20_candidates": state["unique_top20_candidates"],
        "candidate_order_unchanged": True,
        "top5_is_subset_of_top20": all(top5_order[case_id] == top20_order[case_id][:TOP5] for case_id in top20_order),
        "gold_reads_before_materialization_seal": 0,
        "question_reads_during_materialization": 0,
        "pdf_reparse": False,
    })
    write_json(OUT / "top20-candidate-seal.json", {
        "gate": GATE,
        "complete": True,
        "candidate_occurrences": state["candidate_occurrences"],
        "unique_candidates": state["unique_top20_candidates"],
        "documents": sorted({item["parsed"].get("document_id") for values in state["cases"].values() for item in values[:TOP_K] if item["parsed"].get("document_id")}),
        "pages": sorted({item["parsed"].get("page") for values in state["cases"].values() for item in values[:TOP_K] if item["parsed"].get("page") is not None}),
        "candidate_ids_sha256": state["top20_candidate_ids_sha256"],
        "top100_order_sha256": state["top100_order_sha256"],
        "top20_order_sha256": state["top20_order_sha256"],
        "gold_reads": 0,
        "question_reads": 0,
        "retrieval_recomputed": False,
        "reranker_recomputed": False,
    })
    nf09 = import_nf09()
    old_contract_path = NF09 / "financial-fact-v1-contract.json"
    old_schema_path = NF09 / "financial-fact-v1.schema.json"
    old_facts, old_seal = load_top5_facts()
    if sha256_file(old_contract_path) != FACT_CONTRACT_SHA or old_seal.get("deduplicated_facts") != 169 or len(old_facts) != 169:
        raise RuntimeError("frozen FinancialFactV1 contract mismatch")
    write_json(OUT / "financial-fact-contract-reference.json", {
        "schema": "FinancialFactV1",
        "financial_fact_v1_modified": False,
        "sffm_v1_modified": False,
        "contract_sha256": sha256_file(old_contract_path),
        "schema_sha256": sha256_file(old_schema_path),
        "top5_fact_artifact_sha256": sha256_file(NF09 / "financial-facts-v1.jsonl.gz"),
        "top5_fact_count": len(old_facts),
        "top5_provenance_complete_facts": old_seal.get("provenance_complete_facts", 169),
        "sffm_source": "scripts/evaluation/run_nf_e2e_09_r0_structured_fact_representation.py::materialize_candidate",
        "query_independent_api": "materialize_candidate(candidate, atomic_index)",
        "query_reads_during_materialization": 0,
        "gold_reads_during_materialization": 0,
    })
    materialized = candidate_materialization(state)
    facts = materialized["facts"]
    raw_facts = materialized["raw_facts"]
    write_jsonl_gz(OUT / "top20-materialized-facts.jsonl.gz", facts)
    fact_artifact_sha = sha256_file(OUT / "top20-materialized-facts.jsonl.gz")
    write_json(OUT / "top20-materialization-seal.json", {
        "gate": GATE,
        "complete": True,
        "financial_fact_v1_schema": "FinancialFactV1",
        "financial_fact_v1_contract_sha256": FACT_CONTRACT_SHA,
        "sffm_v1_unchanged": True,
        "query_independent": True,
        "unique_candidates": len(unique_rows),
        "raw_facts": len(raw_facts),
        "deduplicated_facts": len(facts),
        "provenance_complete_facts": sum(int(fact.get("provenance_complete") is True) for fact in facts),
        "financial_facts_sha256": fact_artifact_sha,
        "materialization_wall_time_ms": materialized["wall_ms"],
        "question_reads_during_materialization": 0,
        "gold_reads_during_materialization": 0,
        "model_calls": 0,
        "binder_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "validator_calls": 0,
        "e2e_replay": False,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "pdf_reparse": False,
    })
    relation = verify_materialized_relations(facts)
    fabrication = verify_no_fabrication(facts)
    write_json(OUT / "relation-integrity.json", relation)
    write_json(OUT / "fabrication-safety.json", fabrication)
    write_json(OUT / "dedup-analysis.json", {
        "raw_facts": len(raw_facts),
        "deduplicated_facts": len(facts),
        "duplicate_count": materialized["duplicate_count"],
        "duplicate_groups": materialized["duplicate_groups"],
        "dedup_contract_reused": True,
        "dedup_key": ["document_id", "pdf_page", "table_id", "row_id", "column_id", "cell_id", "normalized_metric", "normalized_period", "parsed_numeric_value"],
        "physical_provenance_authoritative": True,
    })
    write_json(OUT / "candidate-level-coverage.json", candidate_fact_counts(top20_order, facts))

    # Prediction/materialization is sealed above.  Only now open labels and
    # Supervisor plans for attribution; neither is passed to the materializer.
    labels, plans = load_postseal_annotations()
    old_state = nf09.load_frozen_state()
    historical_ids = set(old_state["fact_ids"])
    query_rows = top20_query_rows(top20_order, facts)
    by_case = {row["question_id"]: row for row in query_rows}
    historical_rows = [by_case[case_id] for case_id in sorted(historical_ids)]
    historical_full = sum(int(row["provenance_complete_fact_available"]) for row in historical_rows)
    v2_direct_ids = sorted(qid for qid, plan in plans.items() if plan.get("intent") == "DIRECT_FACT")
    v2_direct_rows = [by_case[qid] for qid in v2_direct_ids]
    write_json(OUT / "historical-46-query-coverage.json", {
        "denominator": HISTORICAL_FACT_TOTAL,
        "top5_provenance_complete": "39/46",
        "top20_provenance_complete": f"{historical_full}/{HISTORICAL_FACT_TOTAL}",
        "top20_financial_fact_available": sum(int(row["financial_fact_available"]) for row in historical_rows),
        "rows": historical_rows,
    })
    write_json(OUT / "v2-direct-fact-56-coverage.json", {
        "denominator": len(v2_direct_rows),
        "expected_denominator": DIRECT_TOTAL,
        "provenance_complete_financial_fact_available": sum(int(row["provenance_complete_fact_available"]) for row in v2_direct_rows),
        "financial_fact_available": sum(int(row["financial_fact_available"]) for row in v2_direct_rows),
        "rows": v2_direct_rows,
    })
    calc = calculation_supply(top20_order, facts, labels)
    multi = multi_supply(top20_order, facts, labels, plans)
    write_json(OUT / "calculation-fact-supply.json", calc)
    write_json(OUT / "multi-evidence-fact-supply.json", multi)
    attribution = top5_top20_attribution(old_state, top20_order, facts, labels, historical_ids)
    write_json(OUT / "top5-top20-attribution.json", attribution)
    projection = binder_projection(top20_order, facts)
    write_json(OUT / "binder-packet-projection.json", projection)

    top5_unique = len({candidate for values in top5_order.values() for candidate in values})
    top5_occurrences = QUESTION_TOTAL * TOP5
    sizes = {"top5_fact_artifact_bytes": (NF09 / "financial-facts-v1.jsonl.gz").stat().st_size, "top20_fact_artifact_bytes": (OUT / "top20-materialized-facts.jsonl.gz").stat().st_size, "top5_unique_candidates": top5_unique, "top20_unique_candidates": state["unique_top20_candidates"], "top5_candidate_occurrences": top5_occurrences, "top20_candidate_occurrences": state["candidate_occurrences"], "unique_candidate_increase": state["unique_top20_candidates"] - top5_unique, "financial_fact_count_increase": len(facts) - len(old_facts), "materialization_wall_time_ms": materialized["wall_ms"], "mean_facts_per_query": round(statistics.mean([row["number_of_materialized_facts"] for row in projection["rows"]]), 4) if projection["rows"] else 0.0, "p95_facts_per_query": projection["p95_facts_per_query"], "estimated_binder_input": "all Top20 materialized provenance-complete facts; no binder was called"}
    write_json(OUT / "cost-size-analysis.json", sizes)

    meaningful_gain = historical_full - 39 >= 2 or calc["calculation_fact_supply_complete"] > 0 or multi["v2_supervisor_multi_evidence"]["all_required_sources_with_provenance_complete_fact_supply"] > 0
    safety_ok = relation["relation_integrity_fail"] == 0 and fabrication["fabricated_cross_candidate_facts"] == 0
    effective: bool | str
    next_gate: str
    if not safety_ok:
        effective, next_gate = False, "v2_02_failure_review"
        dominant = "materialization_safety_failure"
    elif meaningful_gain:
        effective, next_gate = True, "v2_03_semantic_evidence_binder"
        dominant = "none_top20_supply_expanded"
    else:
        effective, next_gate = "limited", "v2_03_semantic_evidence_binder"
        dominant = "limited_top20_supply_gain"
    decision = {
        "gate": GATE,
        "evaluation_role": "development_shadow_v2_top20_financial_fact_expansion",
        "base_commit": BASE_COMMIT,
        "production_default": "V1",
        "production_switch_allowed": False,
        "supervisor_frozen": True,
        "supervisor_model": "qwen3.7-max-2026-06-08",
        "model_calls": 0,
        "binder_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "validator_calls": 0,
        "e2e_replay": False,
        "retrieval_execution": False,
        "reranker_execution": False,
        "frozen_retrieval_artifacts_reused": True,
        "financial_fact_v1_modified": False,
        "sffm_v1_modified": False,
        "question_reads_during_materialization": 0,
        "gold_reads_during_materialization": 0,
        "unique_top20_candidates": state["unique_top20_candidates"],
        "facts_raw": len(raw_facts),
        "facts_deduplicated": len(facts),
        "provenance_complete_facts": sum(int(fact.get("provenance_complete") is True) for fact in facts),
        "relation_integrity_pass": relation["relation_integrity_pass"],
        "relation_integrity_fail": relation["relation_integrity_fail"],
        "fabricated_cross_candidate_facts": fabrication["fabricated_cross_candidate_facts"],
        "historical_top5_full_provenance": "39/46",
        "historical_top20_full_provenance": f"{historical_full}/46",
        "v2_direct_fact_top20_fact_supply": f"{sum(int(row['provenance_complete_fact_available']) for row in v2_direct_rows)}/{len(v2_direct_rows)}",
        "calculation_fact_supply_complete": f"{calc['calculation_fact_supply_complete']}/{calc['denominator']}",
        "multi_evidence_fact_supply_complete": f"{multi['v2_supervisor_multi_evidence']['all_required_sources_with_provenance_complete_fact_supply']}/{multi['v2_supervisor_multi_evidence']['denominator']}",
        "top20_financial_fact_expansion_effective": effective,
        "dominant_residual_failure": dominant,
        "next_gate": next_gate,
    }
    write_json(OUT / "decision.json", decision)
    readme = f"""# NF-V2-02 — Top20 FinancialFact Expansion\n\nDevelopment-shadow, query-independent expansion from the sealed SADA Top5 view to frozen SADA Top20. The existing SFFM-V1 `materialize_candidate(candidate, atomic_index)` path and FinancialFactV1 contract were reused unchanged. No model, retrieval, reranker, Binder, Calculator, Generator, Validator, PDF reparse, question-aware extraction, or downstream replay was run.\n\n- Top20 candidates: {state['unique_top20_candidates']} unique / {state['candidate_occurrences']} occurrences\n- Raw/deduplicated facts: {len(raw_facts)} / {len(facts)}\n- Provenance-complete facts: {sum(int(fact.get('provenance_complete') is True) for fact in facts)}\n- Relation failures: {relation['relation_integrity_fail']}\n- Fabricated cross-candidate facts: {fabrication['fabricated_cross_candidate_facts']}\n- Historical compatible cohort: Top5 39/46; Top20 {historical_full}/46\n- Top5→Top20 newly recovered: {attribution['newly_recovered']}\n- Calculation fact supply complete: {calc['calculation_fact_supply_complete']}/{calc['denominator']}\n- V2 direct-fact supply: {sum(int(row['provenance_complete_fact_available']) for row in v2_direct_rows)}/{len(v2_direct_rows)}\n- Materialization question/Gold reads: 0 / 0\n- Decision: `{effective}`; next gate `{next_gate}`\n- Production switch allowed: `false`\n"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"gate": GATE, "unique_top20_candidates": state["unique_top20_candidates"], "raw_facts": len(raw_facts), "deduplicated_facts": len(facts), "provenance_complete_facts": sum(int(fact.get("provenance_complete") is True) for fact in facts), "historical_top20_full_provenance": historical_full, "direct_fact_supply": sum(int(row["provenance_complete_fact_available"]) for row in v2_direct_rows), "calculation_fact_supply_complete": calc["calculation_fact_supply_complete"], "relation_fail": relation["relation_integrity_fail"], "fabricated": fabrication["fabricated_cross_candidate_facts"], "effective": effective, "next_gate": next_gate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
