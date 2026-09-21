"""Evaluate canonical retrieval with raw-intent reranking (Gate E0/E1)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts.evaluation.run_pdf_retrieval_v2_lite import _bm25, _write
from scripts.evaluation.run_pdf_v2_lite_gate_b3 import _rank_dense, _run_variant
from src.evaluation.pdf_query_representation_v2 import concept_family, normalize_label
from src.retrieval.embedding_provider import ExistingMiniLMEmbeddingProvider
from src.services.reranker import HeuristicReranker
from src.services.retrieval_config import get_embedding_model_name

ROOT = Path(__file__).resolve().parents[2]
CONCEPT_OUT = ROOT / "artifacts/evaluation/pdf-query-representation-v2"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-query-representation-v2-gate-e"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cases(records: list[dict[str, object]], canonical: bool) -> list[dict[str, object]]:
    output = []
    for record in records:
        candidates = list(record["concept_candidates"])
        label = str(candidates[0]["canonical_label"]) if candidates else str(record["metric_phrase"])
        query = " ".join(part for part in (str(record["issuer"]), label, str(record.get("period") or "")) if part) if canonical else str(record["natural_question"])
        output.append({"case_id": record["query_id"], "document_id": record["document_id"], "query": query, "gold_candidate_key": record["development_target_candidate_key"]})
    return output


def _hit_ids(result: dict[str, object]) -> set[str]:
    return {str(trace["case_id"]) for trace in result["traces"] if trace["final_hit"]}


def _rank_payload(trace: dict[str, object]) -> dict[str, object]:
    return {key.removesuffix("_rank"): trace[key] for key in ("bm25_rank", "dense_rank", "rrf_rank", "reranker_rank")}


def _reason(record: dict[str, object], raw: dict[str, object], canonical: dict[str, object], views_by_key: dict[str, dict[str, object]]) -> str:
    allowed = set(record["allowed_concept_ids"])
    candidates = list(record["concept_candidates"])
    if not candidates or candidates[0]["concept_id"] not in allowed:
        return "wrong_top1_concept"
    raw_terms = set(normalize_label(str(record["metric_phrase"])).split())
    canonical_terms = set(normalize_label(str(candidates[0]["canonical_label"])).split())
    if raw_terms - canonical_terms:
        return "raw_qualifier_removed"
    competing = [views_by_key[key] for key in canonical["final_candidate_keys"] if key in views_by_key]
    if any(concept_family(str(item["metric"])) == str(candidates[0]["concept_family"]) for item in competing):
        return "same_family_competition"
    if canonical["rrf_rank"] is not None and not canonical["final_hit"]:
        return "reranker_lexical_displacement"
    return "unknown"


def _dual_lane(
    *,
    views: list[dict[str, object]],
    raw_cases: list[dict[str, object]],
    canonical_cases: list[dict[str, object]],
    raw_vectors: object,
    canonical_vectors: object,
    candidate_vectors: object,
) -> dict[str, object]:
    keys = [str(view["candidate_key"]) for view in views]
    key_to_index = {key: index for index, key in enumerate(keys)}
    documents = [str(view["enriched_retrieval_text"]) for view in views]
    reranker = HeuristicReranker()
    hits = 0
    traces = []
    for index, (raw_case, canonical_case) in enumerate(zip(raw_cases, canonical_cases, strict=True)):
        rankings = (
            _bm25(str(raw_case["query"]), documents)[:40],
            _rank_dense(raw_vectors[index], candidate_vectors)[:40],
            _bm25(str(canonical_case["query"]), documents)[:40],
            _rank_dense(canonical_vectors[index], candidate_vectors)[:40],
        )
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, candidate_index in enumerate(ranking, 1):
                scores[candidate_index] = scores.get(candidate_index, 0.0) + 1 / (60 + rank)
        fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        chunks = [{"doc_id": keys[candidate_index], "content": str(views[candidate_index]["raw_row_text"]), "score": score, "metadata": {"doc_name": views[candidate_index]["document_id"], "page": views[candidate_index]["pdf_page"], "row_label": views[candidate_index]["metric"], "section_path": views[candidate_index]["statement_or_section"]}} for candidate_index, score in fused]
        final = reranker.rerank(str(raw_case["query"]), chunks, top_k=5)
        final_keys = [str(item["doc_id"]) for item in final]
        gold = str(raw_case["gold_candidate_key"])
        hit = gold in final_keys
        hits += int(hit)
        traces.append({"case_id": raw_case["case_id"], "final_hit": hit, "final_candidate_keys": final_keys, "gold_in_four_lane_pool": key_to_index[gold] in scores})
    return {"variant": "raw_canonical_four_lane_raw_rerank", "lane_count": 4, "per_lane_top_k": 40, "rrf_k": 60, "equal_weights": True, "case_count": len(raw_cases), "final_5_hit_count": hits, "final_5_recall": hits / len(raw_cases), "traces": traces}


def run(args: argparse.Namespace) -> int:
    concept_acceptance_path = args.concept_dir / "acceptance.json"
    concept_acceptance = json.loads(concept_acceptance_path.read_text(encoding="utf-8"))
    if not concept_acceptance["concept_gate_passed"]:
        raise RuntimeError("Concept gate must pass")
    records = json.loads((args.concept_dir / "concept-resolution-results.json").read_text(encoding="utf-8"))["records"]
    views_path = args.runtime_dir / "pdf-v2-lite-retrieval-views.json"
    views = json.loads(views_path.read_text(encoding="utf-8"))["views"]
    views_by_key = {str(view["candidate_key"]): view for view in views}
    raw_cases, canonical_cases = _cases(records, False), _cases(records, True)
    raw_queries = [str(case["query"]) for case in raw_cases]
    provider = ExistingMiniLMEmbeddingProvider(model_name_or_path=get_embedding_model_name(), device=args.device)
    candidate_vectors = provider.encode_documents([str(view["enriched_retrieval_text"]) for view in views])
    raw_vectors = provider.encode_queries(raw_queries)
    canonical_vectors = provider.encode_queries([str(case["query"]) for case in canonical_cases])
    common = {"bm25_field": "enriched_retrieval_text", "dense_field": "enriched_retrieval_text", "views": views, "embeddings": {"enriched_retrieval_text": candidate_vectors}, "reranker_text_field": "raw_row_text"}
    baseline = _run_variant(name="raw_retrieval_raw_rerank", cases=raw_cases, query_vectors=raw_vectors, **common)
    canonical_full = _run_variant(name="canonical_retrieval_canonical_rerank", cases=canonical_cases, query_vectors=canonical_vectors, **common)
    e1 = _run_variant(name="canonical_retrieval_raw_rerank", cases=canonical_cases, query_vectors=canonical_vectors, reranker_queries=raw_queries, **common)
    raw_trace = {str(item["case_id"]): item for item in baseline["traces"]}
    canonical_trace = {str(item["case_id"]): item for item in canonical_full["traces"]}
    records_by_id = {str(item["query_id"]): item for item in records}
    raw_hits, canonical_hits, e1_hits = map(_hit_ids, (baseline, canonical_full, e1))
    attributed = []
    for query_id in sorted((raw_hits - canonical_hits) | (canonical_hits - raw_hits)):
        record = records_by_id[query_id]
        raw, canonical = raw_trace[query_id], canonical_trace[query_id]
        concepts = list(record["concept_candidates"])
        attributed.append({"query_id": query_id, "change_type": "regression" if query_id in raw_hits else "new_hit", "natural_question": record["natural_question"], "metric_phrase": record["metric_phrase"], "top_1_concept": concepts[0] if concepts else None, "top_2_concept": concepts[1] if len(concepts) > 1 else None, "top_1_top_2_score_margin": None, "score_margin_status": "resolver_scores_not_exported", "raw_ranks": _rank_payload(raw), "canonical_ranks": _rank_payload(canonical), "top_5_competing_metrics": [views_by_key[key]["metric"] for key in canonical["final_candidate_keys"] if key in views_by_key], "regression_reason": _reason(record, raw, canonical, views_by_key) if query_id in raw_hits else None})
    count = len(records)
    new_hits, regressions = len(e1_hits - raw_hits), len(raw_hits - e1_hits)
    final_gain = (len(e1_hits) - len(raw_hits)) / count
    reranker_gain = e1["stage_hit_counts"]["reranker_20"] - baseline["stage_hit_counts"]["reranker_20"]
    development_passed = final_gain >= 0.08 and new_hits >= 8 and regressions <= 1 and reranker_gain >= 0
    dual_lane = None
    dual_lane_passed = False
    if final_gain >= 0.08 and 2 <= regressions <= 3:
        dual_lane = _dual_lane(views=views, raw_cases=raw_cases, canonical_cases=canonical_cases, raw_vectors=raw_vectors, canonical_vectors=canonical_vectors, candidate_vectors=candidate_vectors)
        dual_hits = {str(trace["case_id"]) for trace in dual_lane["traces"] if trace["final_hit"]}
        dual_new = len(dual_hits - raw_hits)
        dual_regressions = len(raw_hits - dual_hits)
        dual_gain = (len(dual_hits) - len(raw_hits)) / count
        dual_lane.update({"new_strict_hit_count": dual_new, "regressed_strict_hit_count": dual_regressions, "final_recall_at_5_gain": dual_gain})
        dual_lane_passed = dual_gain >= 0.08 and dual_new >= 8 and dual_regressions <= 1
    # Inventory proves no additional non-benchmark issuer corpus is available.
    holdout_available = False
    selected_development_passed = development_passed or dual_lane_passed
    overall_passed = selected_development_passed and holdout_available
    for result in (baseline, canonical_full, e1):
        result.pop("final_hit_keys")
        result.pop("traces")
    _write(args.out_dir / "gate-e0-regression-attribution.json", {"record_count": len(attributed), "records": attributed})
    _write(args.out_dir / "gate-e1-hybrid-results.json", {"baseline": baseline, "canonical_full": canonical_full, "canonical_retrieval_raw_rerank": e1, "development_final_recall_at_5_gain": final_gain, "new_strict_hit_count": new_hits, "regressed_strict_hit_count": regressions, "reranker_20_hit_gain": reranker_gain})
    if dual_lane is not None:
        traces = dual_lane.pop("traces")
        _write(args.out_dir / "gate-e2-dual-lane-results.json", dual_lane)
        _write(args.out_dir / "gate-e2-dual-lane-traces.json", {"records": traces})
    _write(args.out_dir / "holdout-corpus-availability.json", {"eligible_additional_nonbenchmark_issuer_count": 0, "available_full_pdf_issuers": ["Adobe", "Salesforce", "Walmart"], "already_used_for_strategy_development": ["Adobe", "Salesforce", "Walmart"], "test_pdf_excluded_reason": "not_a_frozen_financial_issuer_corpus", "frozen_benchmark_issuers_excluded": 8, "holdout_validation_possible": False})
    _write(args.out_dir / "hard-no-answer-report.json", {"status": "not_run", "reason": "no_unused_issuer_corpus_and_no_abstention_contract_in_current_hybrid_pipeline", "synthetic_out_of_domain_controls_are_not_counted_as_hard_no_answer": True, "false_retrieval_at_5": None, "no_answer_precision": None, "no_answer_recall": None, "abstention_f1": None})
    decision = "query_role_decoupling_development_pass_holdout_blocked" if selected_development_passed else "canonical_query_retrieval_gain_not_regression_safe"
    acceptance = {"schema": "pdf-query-representation-v2/gate-e/acceptance/v1", "concept_acceptance_sha256": _sha(concept_acceptance_path), "runtime_views_sha256": _sha(views_path), "development_query_count": count, "e1_development_strategy_passed": development_passed, "dual_lane_development_strategy_passed": dual_lane_passed, "selected_development_strategy": "e1_canonical_retrieval_raw_rerank" if development_passed else "e2_raw_canonical_four_lane" if dual_lane_passed else None, "development_strategy_passed": selected_development_passed, "issuer_disjoint_holdout_available": holdout_available, "gate_e_passed": overall_passed, "final_recall_at_5_gain": final_gain, "new_strict_hit_count": new_hits, "regressed_strict_hit_count": regressions, "reranker_20_hit_gain": reranker_gain, "dual_lane_executed": dual_lane is not None, "parameter_scan": False, "per_query_oracle_selection": False, "hard_no_answer_evaluation_complete": False, "frozen_72_question_reads": 0, "frozen_gold_source_reads": 0, "expected_value_reads": 0, "model_training_calls": 0, "answer_generation_calls": 0, "production_index_writes": 0, "production_behavior_changed": False, "production_switch_allowed": False, "frozen_transfer_allowed": False, "decision": decision, "next_gate": "validate_fixed_strategy_on_unused_issuer" if selected_development_passed else "stop_query_representation_v2"}
    _write(args.out_dir / "next-gate.json", {"decision": decision, "next_gate": acceptance["next_gate"], "production_switch_allowed": False})
    _write(args.out_dir / "acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--concept-dir", type=Path, default=CONCEPT_OUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default=os.getenv("PDF_QUERY_V2_EMBEDDING_DEVICE", "cpu"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
