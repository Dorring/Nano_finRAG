"""Gate 08B: run hierarchical Shadow Retrieval and seal candidates.

This script intentionally has no imports from benchmark governance or labels.
It only reads the sealed Query Plans, Gate 06 Shadow indexes, the frozen Raw
stage replay and the production candidate store used for identity traceback.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.query_plan_models import QueryPlan  # noqa: E402
from src.pdf_retrieval_v4.serialization import query_plan_from_dict  # noqa: E402
from src.pdf_retrieval_v4.shadow_index_reader import ShadowIndexReader  # noqa: E402
from src.pdf_retrieval_v4.v4_gate08_pool import (  # noqa: E402
    ProductionCandidateMapper,
    build_query,
    hit_record,
    merge_raw_protected,
)


DEFAULT_PLANS = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-07/query-plan-predictions.json"
DEFAULT_RAW = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08/raw-parity.json"
DEFAULT_RUNTIME = ROOT / "artifacts/runtime/pdf-retrieval-v4-gate-06-r2"
DEFAULT_DB = ROOT.parents[3] / "backend/rag_bm25.db"
DEFAULT_CORPUS = ROOT / "benchmarks/financial_rag_v1/corpus.json"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08"


BUDGETS = {
    "section_bm25_k": 20,
    "section_dense_k": 20,
    "section_fused_k": 10,
    "table_bm25_k": 20,
    "table_dense_k": 20,
    "table_fused_k": 10,
    "local_rows_bm25_k": 5,
    "local_rows_dense_k": 5,
    "local_row_pool_cap": 40,
    "atomic_slot_k": 20,
    "comparison_slot_k": 10,
    "bucket_slot_k": 10,
    "structured_strict_pool_k": 40,
    "rrf_k": 60,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _scope_filter(reader: ShadowIndexReader, unit_type: str, scope: Iterable[str]) -> set[str]:
    return reader.view_ids_for_scope(unit_type, scope)


def _logical_filter(reader: ShadowIndexReader, unit_type: str, ids: set[str], table_ids: set[str]) -> set[str]:
    if not table_ids:
        return ids
    return {
        view_id
        for view_id in ids
        if str((reader.view(view_id) or {}).get("metadata", {}).get("logical_table_id", "")) in table_ids
    }


def _row_filter(reader: ShadowIndexReader, ids: set[str], row_ids: set[str]) -> set[str]:
    if not row_ids:
        return ids
    return {
        view_id
        for view_id in ids
        if str((reader.view(view_id) or {}).get("metadata", {}).get("row_id", "")) in row_ids
    }


def _period_match(reader: ShadowIndexReader, view_id: str, period: str | None) -> bool:
    if not period:
        return True
    metadata = (reader.view(view_id) or {}).get("metadata", {})
    periods = {str(value) for value in metadata.get("periods", [])}
    binding = metadata.get("temporal_binding") or {}
    if isinstance(binding, dict):
        periods.update(str(value) for value in (binding.get("period"), binding.get("base_period"), binding.get("current_period"), binding.get("reporting_period")) if value)
    return str(period) in periods


def _search_facts(
    reader: ShadowIndexReader,
    plan: QueryPlan,
    slot: dict[str, Any],
    table_ids: set[str],
    row_ids: set[str],
    unit_type: str,
    *,
    k: int,
) -> dict[str, list[dict[str, Any]]]:
    scoped = _scope_filter(reader, unit_type, plan.document_scope)
    table_scoped = _logical_filter(reader, unit_type, scoped, table_ids)
    row_scoped = _row_filter(reader, table_scoped, row_ids)
    exact = {view_id for view_id in row_scoped if _period_match(reader, view_id, slot.get("period"))}
    query = build_query(plan, slot=slot, stage="fact")
    exact_hits = reader.search(unit_type, query, allowed_ids=exact, bm25_k=k, dense_k=k, fused_k=k)
    fallback_hits = reader.search(unit_type, query, allowed_ids=row_scoped, bm25_k=k, dense_k=k, fused_k=k)
    return {
        "exact_period": [hit_record(reader, hit, route=f"{unit_type}_exact_period", slot_id=str(slot.get("slot_id"))) | {"period_match": "exact"} for hit in exact_hits],
        "unfiltered_temporal_fallback": [
            hit_record(reader, hit, route=f"{unit_type}_fallback", slot_id=str(slot.get("slot_id")))
            | {"period_match": "exact" if _period_match(reader, hit.retrieval_view_id, slot.get("period")) else "missing_or_compatible"}
            for hit in fallback_hits
            if not ((reader.view(hit.retrieval_view_id) or {}).get("metadata", {}).get("temporal_binding", {}) or {}).get("kind") == "comparison" or _period_match(reader, hit.retrieval_view_id, slot.get("period"))
        ],
    }


def _slot_dict(slot: Any) -> dict[str, Any]:
    if hasattr(slot, "__dict__"):
        return dict(slot.__dict__)
    return dict(slot)


def _structured_records(
    reader: ShadowIndexReader,
    mapper: ProductionCandidateMapper,
    plan: QueryPlan,
    route_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    priority = {"atomic_fact": 0, "comparison_fact": 1, "bucket_fact": 2, "row": 3, "cell": 4, "section": 5}
    mapped: list[dict[str, Any]] = []
    ambiguous = 0
    for item in route_records:
        view = reader.view(str(item.get("retrieval_view_id"))) or {}
        mapping = mapper.map_view(view)
        if mapping.get("strict_candidate_status") != "unique":
            if mapping.get("strict_candidate_status") == "ambiguous":
                ambiguous += 1
            continue
        record = dict(item)
        record.update({
            "original_candidate_identity": mapping["candidate_key"],
            "strict_candidate_status": "unique",
            "mapping_score": mapping.get("mapping_score"),
            "route_priority": priority.get(str(item.get("unit_type")), 99),
        })
        mapped.append(record)
    # Slot round-robin is applied to all required multi-source slots before
    # final identity de-duplication.
    slot_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in mapped:
        slot_groups[str(item.get("slot_id") or "__none__")].append(item)
    for values in slot_groups.values():
        values.sort(key=lambda item: (item.get("route_priority", 99), int(item.get("fused_rank") or 10**6), str(item.get("retrieval_view_id"))))
    ordered: list[dict[str, Any]] = []
    if len(slot_groups) > 1 and plan.requires_multiple_sources:
        keys = sorted(slot_groups)
        index = 0
        while True:
            appended = False
            for key in keys:
                if index < len(slot_groups[key]):
                    ordered.append(slot_groups[key][index])
                    appended = True
            if not appended:
                break
            index += 1
    else:
        ordered = sorted(mapped, key=lambda item: (item.get("route_priority", 99), str(item.get("slot_id") or ""), int(item.get("fused_rank") or 10**6), str(item.get("retrieval_view_id"))))
    dedup: dict[str, dict[str, Any]] = {}
    supporting: dict[str, list[str]] = defaultdict(list)
    for item in ordered:
        key = str(item["original_candidate_identity"])
        supporting[key].append(str(item["retrieval_view_id"]))
        dedup.setdefault(key, item)
    result: list[dict[str, Any]] = []
    for rank, (key, item) in enumerate(dedup.items(), 1):
        record = dict(item)
        record["structured_rank"] = rank
        record["supporting_view_ids"] = supporting[key]
        result.append(record)
        if len(result) >= BUDGETS["structured_strict_pool_k"]:
            break
    return result, ambiguous


def predict_case(
    reader: ShadowIndexReader,
    mapper: ProductionCandidateMapper,
    plan: QueryPlan,
    raw_case: dict[str, Any],
) -> dict[str, Any]:
    scope = tuple(plan.document_scope)
    route_types = {str(route.index_type) for route in plan.retrieval_routes}
    is_narrative = plan.task_type == "narrative_or_note"
    section_candidates: list[dict[str, Any]] = []
    if "section" in route_types or is_narrative:
        section_hits = reader.search(
            "section", build_query(plan, stage="section"),
            allowed_ids=_scope_filter(reader, "section", scope),
            bm25_k=BUDGETS["section_bm25_k"], dense_k=BUDGETS["section_dense_k"], fused_k=BUDGETS["section_fused_k"],
        )
        section_candidates = [hit_record(reader, hit, route="section") for hit in section_hits]
    table_candidates: list[dict[str, Any]] = []
    if not is_narrative and plan.task_type != "unsupported":
        table_hits = reader.search(
            "table", build_query(plan),
            allowed_ids=_scope_filter(reader, "table", scope),
            bm25_k=BUDGETS["table_bm25_k"], dense_k=BUDGETS["table_dense_k"], fused_k=BUDGETS["table_fused_k"],
        )
        table_candidates = [hit_record(reader, hit, route="table") for hit in table_hits]
    table_ids = {
        str(item.get("metadata", {}).get("logical_table_id"))
        for item in table_candidates
        if item.get("metadata", {}).get("logical_table_id")
    }
    table_order = [str(item["metadata"]["logical_table_id"]) for item in table_candidates if item.get("metadata", {}).get("logical_table_id")]
    slots = [_slot_dict(slot) for slot in plan.operand_slots]
    if not slots and plan.metric_phrases:
        slots = [{"slot_id": "fact", "role": "value", "raw_metric_phrase": plan.metric_phrases[0], "period": plan.periods[0] if plan.periods else None, "concept_candidates": []}]
    local_rows_by_slot: dict[str, list[dict[str, Any]]] = {}
    local_row_records: list[dict[str, Any]] = []
    if table_ids and not is_narrative and plan.task_type != "unsupported":
        for slot in slots or [{"slot_id": "__query__"}]:
            query = build_query(plan, slot=slot, stage="row")
            rows = reader.local_rows(table_order, query, bm25_k=BUDGETS["local_rows_bm25_k"], dense_k=BUDGETS["local_rows_dense_k"], total_cap=BUDGETS["local_row_pool_cap"])
            records = [hit_record(reader, hit, route="local_row", slot_id=str(slot.get("slot_id"))) for hit in rows]
            local_rows_by_slot[str(slot.get("slot_id"))] = records
            local_row_records.extend(records)
    row_ids = {
        str(item.get("metadata", {}).get("row_id"))
        for item in local_row_records
        if item.get("metadata", {}).get("row_id")
    }
    atomic_by_slot: dict[str, dict[str, list[dict[str, Any]]]] = {}
    comparison_by_slot: dict[str, dict[str, list[dict[str, Any]]]] = {}
    bucket_by_slot: dict[str, dict[str, list[dict[str, Any]]]] = {}
    route_records: list[dict[str, Any]] = list(local_row_records)
    for slot in slots:
        slot_id = str(slot.get("slot_id"))
        if "atomic_fact" in route_types or plan.task_type in {"table_single_fact", "single_metric_multi_period", "multi_metric_comparison", "calculation_multi_operand"}:
            result = _search_facts(reader, plan, slot, table_ids, row_ids, "atomic_fact", k=BUDGETS["atomic_slot_k"])
            atomic_by_slot[slot_id] = result
            route_records.extend(result["exact_period"])
            route_records.extend(result["unfiltered_temporal_fallback"])
        if "comparison_fact" in route_types:
            result = _search_facts(reader, plan, slot, table_ids, row_ids, "comparison_fact", k=BUDGETS["comparison_slot_k"])
            comparison_by_slot[slot_id] = result
            route_records.extend(result["exact_period"])
            route_records.extend(result["unfiltered_temporal_fallback"])
        if "bucket_fact" in route_types:
            result = _search_facts(reader, plan, slot, table_ids, row_ids, "bucket_fact", k=BUDGETS["bucket_slot_k"])
            bucket_by_slot[slot_id] = result
            route_records.extend(result["exact_period"])
            route_records.extend(result["unfiltered_temporal_fallback"])
    cell_candidates: list[dict[str, Any]] = []
    if "cell" in route_types:
        allowed = _scope_filter(reader, "cell", scope)
        allowed = _logical_filter(reader, "cell", allowed, table_ids)
        cell_hits = reader.search("cell", build_query(plan, stage="fact"), allowed_ids=allowed, bm25_k=20, dense_k=20, fused_k=20)
        cell_candidates = [hit_record(reader, hit, route="cell_auxiliary") for hit in cell_hits]
        route_records.extend(cell_candidates)
    # Section views are strict candidates only when they uniquely trace to a
    # production source; table views remain navigation-only by contract.
    for record in section_candidates:
        route_records.append(record)
    def annotate(items: list[dict[str, Any]]) -> None:
        for item in items:
            view = reader.view(str(item.get("retrieval_view_id"))) or {}
            mapping = mapper.map_view(view)
            item["mapped_candidate_identity"] = mapping.get("candidate_key")
            item["strict_candidate_status"] = mapping.get("strict_candidate_status")
            item["mapping_score"] = mapping.get("mapping_score")
    annotate(section_candidates)
    annotate(table_candidates)
    annotate(local_row_records)
    for values in atomic_by_slot.values():
        for lane in values.values():
            annotate(lane)
    for values in comparison_by_slot.values():
        for lane in values.values():
            annotate(lane)
    for values in bucket_by_slot.values():
        for lane in values.values():
            annotate(lane)
    annotate(cell_candidates)
    structured, ambiguous = _structured_records(reader, mapper, plan, route_records)
    merged = merge_raw_protected(list(raw_case["raw_full_rrf_candidates"]), structured, structured_k=BUDGETS["structured_strict_pool_k"])
    combined = merged["combined_pool"]
    scope_violations = 0
    for record in structured:
        view = reader.view(str(record.get("retrieval_view_id"))) or {}
        if str((view.get("metadata") or {}).get("document_id", "")) not in scope:
            scope_violations += 1
    return {
        "case_id": plan.plan_id if False else raw_case["case_id"],
        "query_plan_id": plan.plan_id,
        "task_type": plan.task_type,
        "document_scope": list(scope),
        "raw_full_rrf_candidates": raw_case["raw_full_rrf_candidates"],
        "raw_rrf_at_40": raw_case["raw_rrf_at_40"],
        "section_candidates": section_candidates,
        "table_candidates": table_candidates,
        "local_rows_by_slot": local_rows_by_slot,
        "atomic_candidates_by_slot": atomic_by_slot,
        "comparison_candidates_by_slot": comparison_by_slot,
        "bucket_candidates_by_slot": bucket_by_slot,
        "cell_auxiliary_candidates": cell_candidates,
        "structured_strict_source_pool": structured,
        "combined_pool": combined,
        "raw_candidate_hash": raw_case["raw_candidate_hash"],
        "raw_candidate_loss": merged["raw_candidate_loss"],
        "raw_candidate_rank_mutation": merged["raw_candidate_hash_before"] != merged["raw_candidate_hash_after"],
        "raw_candidate_score_mutation": merged["raw_candidate_hash_before"] != merged["raw_candidate_hash_after"],
        "structured_ambiguous_mapping_count": ambiguous,
        "cross_document_candidate_count": scope_violations,
        "soft_continuation_expansion": False,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "answer_generation_calls": 0,
    }


def write_jsonl_gzip(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as handle:
            for record in records:
                handle.write((json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    return sha(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--raw-parity", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--raw-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--code-commit", default=None)
    args = parser.parse_args()
    plans_payload = json.loads(args.plans.read_text(encoding="utf-8"))
    raw_payload = json.loads(args.raw_parity.read_text(encoding="utf-8"))
    raw_by_case = {str(item["case_id"]): item for item in raw_payload["raw_cases"]}
    plans: list[tuple[str, QueryPlan]] = [(str(item["case_id"]), query_plan_from_dict(item["plan"])) for item in plans_payload["plans"]]
    code_commit = args.code_commit or commit()
    protocol = {
        "schema": "pdf-retrieval-v4/gate-08/prediction/v1",
        "gate": "pdf_retrieval_v4_gate_08b",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "implementation_commit": code_commit,
        "artifact_seal_commit": code_commit,
        "budgets": BUDGETS,
        "raw_protection": "raw_full_pool_prefix_unchanged_plus_structured_residual",
        "embedding_model": "all-MiniLM-L6-v2",
        "rrf": {"k": 60, "bm25_weight": 1, "dense_weight": 1},
        "input_hashes": {"plans": sha(args.plans), "raw_parity": sha(args.raw_parity)},
        "prediction_inputs": ["question_text_in_sealed_plan", "document_scope", "gate_07_query_plan", "gate_06_r2_shadow_index", "gate_0_raw_replay", "production_candidate_store_identity_only"],
        "forbidden_inputs": ["labels.golden.jsonl", "benchmark-governance.jsonl", "evidence-family-map.json", "expected_value", "reference_answer", "original_final_hit_identity"],
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "reference_answer_reads_before_seal": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
        "parameter_scan": False,
        "per_query_oracle": False,
        "soft_continuation_expansion": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write(args.out_dir / "gate-08-protocol.json", protocol)
    write(args.out_dir / "gate-08-input-integrity.json", {
        "plans_sha256": sha(args.plans),
        "raw_parity_sha256": sha(args.raw_parity),
        "runtime_dir": str(args.runtime),
        "runtime_metadata_present": (args.runtime / "metadata/metadata.sqlite").is_file(),
        "runtime_index_identity": "pdf-retrieval-v4-gate-06-r2",
        "prediction_count": len(plans),
    })
    with ShadowIndexReader(args.runtime, rrf_k=BUDGETS["rrf_k"]) as reader:
        mapper = ProductionCandidateMapper(args.raw_db, args.corpus)
        records: list[dict[str, Any]] = [{"stream": "header", "schema": "pdf-retrieval-v4/gate-08/predictions/v1", "record_count": len(plans)}]
        for case_id, plan in plans:
            if case_id not in raw_by_case:
                raise RuntimeError(f"missing_raw_case:{case_id}")
            records.append(predict_case(reader, mapper, plan, raw_by_case[case_id]))
    prediction_path = args.out_dir / "retrieval-predictions.jsonl.gz"
    prediction_hash = write_jsonl_gzip(prediction_path, records)
    write(args.out_dir / "retrieval-prediction-manifest.json", {
        "record_count": len(plans),
        "gzip_record_count_including_header": len(records),
        "prediction_sha256": prediction_hash,
        "structured_pool_budget": BUDGETS["structured_strict_pool_k"],
        "raw_candidate_loss_count": sum(bool(item.get("raw_candidate_loss")) for item in records[1:]),
        "raw_rank_mutation_count": sum(bool(item.get("raw_candidate_rank_mutation")) for item in records[1:]),
        "cross_document_candidate_count": sum(int(item.get("cross_document_candidate_count", 0)) for item in records[1:]),
    })
    write(args.out_dir / "retrieval-prediction-seal.json", {
        "prediction_count": len(plans),
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "reference_answer_reads_before_seal": 0,
        "index_reads_before_seal": len(plans),
        "retrieval_runs": len(plans),
        "reranker_calls": 0,
        "calculator_calls": 0,
        "answer_generation_calls": 0,
        "parameter_scan": False,
        "per_query_oracle": False,
        "prediction_hash": prediction_hash,
        "protocol_hash": sha(args.out_dir / "gate-08-protocol.json"),
        "sealed": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
