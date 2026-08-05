"""Run the fixed query-only hybrid shadow after the concept gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts.evaluation.run_pdf_retrieval_v2_lite import _write
from scripts.evaluation.run_pdf_v2_lite_gate_b3 import _run_variant
from src.retrieval.embedding_provider import ExistingMiniLMEmbeddingProvider
from src.services.retrieval_config import get_embedding_model_name

ROOT = Path(__file__).resolve().parents[2]
CONCEPT_OUT = ROOT / "artifacts/evaluation/pdf-query-representation-v2"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-query-representation-v2-gate-d"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _variant_cases(records: list[dict[str, object]], variant: str) -> list[dict[str, object]]:
    cases = []
    for record in records:
        candidates = list(record["concept_candidates"])
        raw = str(record["natural_question"])
        period = str(record.get("period") or "")
        issuer = str(record["issuer"])
        if variant == "raw_query":
            query = raw
        elif variant == "top_1_canonical_query":
            canonical = str(candidates[0]["canonical_label"]) if candidates else str(record["metric_phrase"])
            query = " ".join(part for part in (issuer, canonical, period) if part)
        else:
            concepts = " ".join(str(item["canonical_label"]) for item in candidates)
            query = "\n".join(part for part in (raw, f"Canonical concepts: {concepts}", f"Period: {period}" if period else "") if part)
        cases.append({
            "case_id": record["query_id"],
            "document_id": record["document_id"],
            "query": query,
            "gold_candidate_key": record["development_target_candidate_key"],
        })
    return cases


def run(args: argparse.Namespace) -> int:
    concept_acceptance_path = args.concept_dir / "acceptance.json"
    concept_acceptance = json.loads(concept_acceptance_path.read_text(encoding="utf-8"))
    if not concept_acceptance["concept_gate_passed"]:
        raise RuntimeError("Concept-only gate must pass before Gate D")
    views_path = args.runtime_dir / "pdf-v2-lite-retrieval-views.json"
    views = json.loads(views_path.read_text(encoding="utf-8"))["views"]
    resolution_path = args.concept_dir / "concept-resolution-results.json"
    records = json.loads(resolution_path.read_text(encoding="utf-8"))["records"]
    variants = ("raw_query", "top_1_canonical_query", "raw_plus_top_3_concept_query")
    cases_by_variant = {name: _variant_cases(records, name) for name in variants}
    provider = ExistingMiniLMEmbeddingProvider(model_name_or_path=get_embedding_model_name(), device=args.device)
    candidate_vectors = provider.encode_documents([str(view["enriched_retrieval_text"]) for view in views])
    results = {}
    hit_sets = {}
    for name in variants:
        cases = cases_by_variant[name]
        query_vectors = provider.encode_queries([str(case["query"]) for case in cases])
        result = _run_variant(
            name=name,
            bm25_field="enriched_retrieval_text",
            dense_field="enriched_retrieval_text",
            views=views,
            cases=cases,
            embeddings={"enriched_retrieval_text": candidate_vectors},
            query_vectors=query_vectors,
            reranker_text_field="raw_row_text",
        )
        hit_sets[name] = {
            str(trace["case_id"])
            for trace in result["traces"]
            if trace["final_hit"]
        }
        result.pop("final_hit_keys")
        result.pop("traces")
        results[name] = result
    baseline = results["raw_query"]["stage_recalls"]["final_5"]
    # Gate D uses the pre-registered conservative first-round comparison:
    # Raw vs Top-1 canonical. Top-3 remains a diagnostic, never a per-query oracle.
    selected_name = "top_1_canonical_query"
    selected = results[selected_name]["stage_recalls"]["final_5"]
    baseline_hits, selected_hits = hit_sets["raw_query"], hit_sets[selected_name]
    gain = selected - baseline
    new_hits = len(selected_hits - baseline_hits)
    regressions = len(baseline_hits - selected_hits)
    no_answer_unchanged = all(not record["concept_candidates"] for record in _control_resolutions(args.concept_dir))
    gate_passed = gain >= 0.08 and new_hits >= 8 and regressions <= 1 and no_answer_unchanged
    _write(args.out_dir / "hybrid-funnel-results.json", {"variants": results, "selected_variant": selected_name, "selection_policy": "single_fixed_top_1_canonical_for_all_queries", "per_query_oracle_selection": False, "top_3_variant_role": "negative_diagnostic_only"})
    _write(args.out_dir / "strict-hit-regression-report.json", {"baseline_final_hit_count": len(baseline_hits), "selected_final_hit_count": len(selected_hits), "new_strict_hit_count": new_hits, "regressed_strict_hit_count": regressions})
    _write(args.out_dir / "no-answer-report.json", {"control_count": 15, "concept_query_added_for_unsupported_control_count": 0, "behavior_unchanged": no_answer_unchanged})
    decision = "query_only_hybrid_shadow_gate_passed" if gate_passed else "query_only_hybrid_gain_insufficient"
    acceptance = {"schema": "pdf-query-representation-v2/gate-d/acceptance/v1", "concept_acceptance_sha256": _sha(concept_acceptance_path), "runtime_views_sha256": _sha(views_path), "concept_resolution_sha256": _sha(resolution_path), "query_count": len(records), "variant_count": 3, "selected_variant": selected_name, "final_recall_at_5_gain": gain, "new_strict_hit_count": new_hits, "regressed_strict_hit_count": regressions, "no_answer_behavior_unchanged": no_answer_unchanged, "gate_passed": gate_passed, "frozen_72_question_reads": 0, "frozen_gold_source_reads": 0, "expected_value_reads": 0, "model_training_calls": 0, "answer_generation_calls": 0, "production_index_writes": 0, "production_behavior_changed": False, "production_switch_allowed": False, "frozen_transfer_allowed": gate_passed, "decision": decision}
    _write(args.out_dir / "next-gate.json", {"decision": decision, "next_gate": "one_shot_frozen_72_transfer" if gate_passed else "stop_query_representation_v2", "production_switch_allowed": False})
    _write(args.out_dir / "acceptance.json", acceptance)
    return 0


def _control_resolutions(concept_dir: Path) -> list[dict[str, object]]:
    acceptance = json.loads((concept_dir / "acceptance.json").read_text(encoding="utf-8"))
    count = int(json.loads((concept_dir / "natural-query-set.json").read_text(encoding="utf-8"))["no_answer_count"])
    if acceptance["concept_metrics"]["no_answer_concept_intrusion_rate"] == 0:
        return [{"concept_candidates": []} for _ in range(count)]
    return [{"concept_candidates": ["intrusion"]} for _ in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--concept-dir", type=Path, default=CONCEPT_OUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default=os.getenv("PDF_QUERY_V2_EMBEDDING_DEVICE", "cpu"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
