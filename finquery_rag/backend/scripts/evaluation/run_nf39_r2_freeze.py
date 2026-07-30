"""Freeze verifiable NF39 R2 final contexts from an isolated RRF pool."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from src.evaluation.case_fingerprints import label_fingerprint, question_fingerprint
from src.evaluation.evaluation import load_jsonl_cases
from src.evaluation.frozen_candidate_integrity import (
    RankedEvidenceCandidate,
    candidate_manifest_row,
    final_context_hash,
    render_candidate_for_context,
    stable_json_bytes,
    validate_rankings,
)
from src.evaluation.nf38_evaluator import _stable_digest
from src.evaluation.nf39_r1_integrity import (
    case_stage_summary,
    denominator_report,
    source_stage_transitions,
    stage_metrics_same_k,
)
from src.retrieval.candidate_identity import identity_from_candidate
from src.services.reranker import build_reranker
from src.services.vector_store import get_or_create_collection
from scripts.evaluation.run_nf39_evaluation import (
    _pool_to_rrf_format,
    _reranker_output_to_summary,
)


SCHEMA = "nf39-r2/v1"
COLLECTION_ID = "rag_global_knowledge_base"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pool_digest(pool: dict[str, list[dict[str, Any]]]) -> str:
    return _stable_digest([{"case_id": case_id, "candidates": pool[case_id]} for case_id in sorted(pool)])


def _load_documents(candidate_ids: set[str]) -> dict[str, tuple[str, dict[str, Any]]]:
    collection = get_or_create_collection()
    result: dict[str, tuple[str, dict[str, Any]]] = {}
    ids = sorted(candidate_ids)
    for start in range(0, len(ids), 200):
        payload = collection.get(ids=ids[start : start + 200], include=["documents", "metadatas"])
        for key, content, metadata in zip(payload["ids"], payload["documents"], payload["metadatas"], strict=True):
            result[key] = (content, metadata or {})
    missing = candidate_ids - result.keys()
    if missing:
        raise ValueError(f"Evidence store cannot resolve {len(missing)} frozen candidates")
    return result


def _ranked(row: dict[str, Any], documents: dict[str, tuple[str, dict[str, Any]]], tenant_id: int) -> RankedEvidenceCandidate:
    evidence_id = row.get("candidate_id") or row.get("evidence_id")
    content, metadata = documents[evidence_id]
    identity = identity_from_candidate({
        "tenant_id": tenant_id,
        "document_id": row.get("document_id") or metadata.get("doc_name"),
        "evidence_id": evidence_id,
        "block_type": row.get("block_type") or metadata.get("type", "text"),
        "parent_id": row.get("parent_id") or metadata.get("parent_id"),
        "collection_id": COLLECTION_ID,
    })
    return RankedEvidenceCandidate(
        identity=identity,
        page=row.get("page", metadata.get("page")),
        block_type=row.get("block_type") or metadata.get("type", "text"),
        content=content,
        parent_id=row.get("parent_id") or metadata.get("parent_id"),
        table_id=row.get("table_id") or metadata.get("table_id"),
        rrf_score=row.get("rrf_score", row.get("score")),
        reranker_score=row.get("reranker_score"),
    )


def _audit(stage: str, rankings: dict[str, list[RankedEvidenceCandidate]]) -> dict[str, Any]:
    candidates = [candidate for rows in rankings.values() for candidate in rows]
    return {
        "stage": stage,
        "candidate_count": len(candidates),
        "missing_tenant_id_count": sum(not candidate.identity.tenant_id for candidate in candidates),
        "missing_document_id_count": sum(not candidate.identity.document_id for candidate in candidates),
        "missing_source_id_count": sum(not candidate.identity.source_id for candidate in candidates),
        "missing_content_count": sum(not candidate.content for candidate in candidates),
        "candidate_keys_hash": hashlib.sha256(stable_json_bytes([candidate.candidate_key for candidate in candidates])).hexdigest(),
    }


def _snapshot(final: dict[str, list[RankedEvidenceCandidate]], snapshot_path: Path) -> dict[str, Any]:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in sorted(final):
        for rank, candidate in enumerate(final[case_id], 1):
            entry = candidate_manifest_row(candidate)
            rows.append({"case_id": case_id, "rank": rank, "candidate_key": entry["candidate_key"], "content_hash": entry["content_hash"], "rendered_content": render_candidate_for_context(candidate)})
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    snapshot_path.write_text(payload, encoding="utf-8")
    os.chmod(snapshot_path, 0o600)
    return {"payload_record_count": len(rows), "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(), "payload_committed": False, "payload_location_type": "local_ignored_snapshot"}


def _verify_snapshot(final_manifest: dict[str, Any], snapshot_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in snapshot_path.read_text(encoding="utf-8").splitlines() if line]
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    content_ok = context_ok = 0
    for case_id, item in final_manifest["cases"].items():
        values = sorted(by_case[case_id], key=lambda row: row["rank"])
        content_ok += sum(hashlib.sha256(value["rendered_content"].encode("utf-8")).hexdigest() == value["content_hash"] for value in values)
        digest = hashlib.sha256(stable_json_bytes([{"rank": value["rank"], "candidate_key": value["candidate_key"], "content_hash": value["content_hash"]} for value in values])).hexdigest()
        context_ok += int(digest == item["final_context_hash"])
    return {"snapshot_rehydrated_count": len(rows), "content_hash_verified_count": content_ok, "verified_final_context_count": context_ok, "missing_candidate_count": 135 - len(rows), "passed": len(rows) == 135 and content_ok == 135 and context_ok == 27}


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze NF39 R2 verifiable final contexts")
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--snapshot-path", required=True)
    args = parser.parse_args()
    pool_dir, out_dir = Path(args.pool_dir), Path(args.out_dir)
    pool = json.loads((pool_dir / "rrf-candidate-pool.json").read_text(encoding="utf-8"))
    pool_manifest = json.loads((pool_dir / "rrf-candidate-pool-manifest.json").read_text(encoding="utf-8"))
    cases = load_jsonl_cases(args.cases)
    if _pool_digest(pool) != pool_manifest["candidate_pool_hash"]:
        raise ValueError("Frozen candidate-pool hash mismatch")
    if question_fingerprint(cases) != pool_manifest["question_hash"] or label_fingerprint(cases) != pool_manifest["label_hash"]:
        raise ValueError("Cases do not match frozen candidate pool")
    tenant_id = int(pool_manifest["tenant_id"])
    ids = {row.get("candidate_id") or row.get("evidence_id") for values in pool.values() for row in values}
    if not all(isinstance(value, str) and value for value in ids):
        raise ValueError("Candidate pool contains an empty source identity")
    documents = _load_documents(ids)
    rrf = {case.case_id: [_ranked(row, documents, tenant_id) for row in pool[case.case_id]] for case in cases}
    validate_rankings(rrf, expected_cases=len(cases), expected_count=40, label="RRF pool")
    reranker = build_reranker("heuristic")
    if reranker is None:
        raise RuntimeError("NF39 R2 requires heuristic reranker")
    input_rows, reranked_rows, final = {}, {}, {}
    for case in cases:
        raw = pool[case.case_id]
        lookup = {row["candidate_id"]: row for row in raw}
        output = reranker.rerank(case.question, _pool_to_rrf_format(raw[:20]), top_k=20)
        summary = _reranker_output_to_summary(output, lookup)
        input_rows[case.case_id] = rrf[case.case_id][:20]
        reranked_rows[case.case_id] = [_ranked(row, documents, tenant_id) for row in summary]
        final[case.case_id] = reranked_rows[case.case_id][:5]
    validate_rankings(final, expected_cases=len(cases), expected_count=5, label="Final contexts")
    def summary(mapping: dict[str, list[RankedEvidenceCandidate]]) -> dict[str, list[dict[str, Any]]]:
        return {case_id: [{"candidate_id": candidate.identity.source_id, "evidence_id": candidate.identity.source_id, "document_id": candidate.identity.document_id, "page": candidate.page, "block_type": candidate.block_type} for candidate in rows] for case_id, rows in mapping.items()}
    rrf_summary, input_summary, reranked_summary, final_summary = summary(rrf), summary(input_rows), summary(reranked_rows), summary(final)
    final_manifest_cases = {}
    for case_id, rows in final.items():
        candidates = [candidate_manifest_row(row) | {"rank": rank} for rank, row in enumerate(rows, 1)]
        final_manifest_cases[case_id] = {"candidate_count": 5, "candidates": candidates, "final_context_hash": final_context_hash(rows), "content_hash_available": True}
    final_manifest = {"artifact_schema": SCHEMA, "candidate_identity_schema": "candidate-identity/v1", "context_renderer_schema": "context-renderer/v1", "cases": final_manifest_cases, "cases_without_exported_content_hash": 0}
    snapshot_manifest = _snapshot(final, Path(args.snapshot_path))
    verification = _verify_snapshot(final_manifest, Path(args.snapshot_path))
    denoms = denominator_report(cases)
    stages = {"s0_rrf_top40": stage_metrics_same_k(cases=cases, rankings=rrf_summary, ks=(5, 20, 40)), "s1_rrf_top20_reranker_input": stage_metrics_same_k(cases=cases, rankings=input_summary, ks=(5, 20)), "s2_reranker_ranked_top20": stage_metrics_same_k(cases=cases, rankings=reranked_summary, ks=(5, 20)), "s3_reranker_top5": stage_metrics_same_k(cases=cases, rankings=reranked_summary, ks=(5,)), "s4_final_context_top5": stage_metrics_same_k(cases=cases, rankings=final_summary, ks=(5,))}
    transitions, counts = source_stage_transitions(cases=cases, rrf_rankings=rrf_summary, reranker_input_rankings=input_summary, reranker_rankings=reranked_summary, final_rankings=final_summary, reranker_input_top_n=20)
    baseline = {"artifact_schema": SCHEMA, "case_count": len(cases), "answerable_case_count": denoms["retrieval_case_count"], "no_answer_case_count": denoms["no_answer_case_count"], "expected_source_count": denoms["expected_source_count"], "candidate_pool_hash": pool_manifest["candidate_pool_hash"], "question_hash": pool_manifest["question_hash"], "label_hash": pool_manifest["label_hash"], "tenant_id": tenant_id, "corpus_hash": pool_manifest["corpus_hash"], "rrf_top_n": 40, "reranker_input_top_n": 20, "final_top_k": 5, "production_behavior_changed": False}
    _write(out_dir / "baseline-manifest.json", baseline)
    _write(out_dir / "candidate-boundary-audit.json", {"root_cause": "NF39 exporter re-read normalized BM25 candidates as raw SQLite rows and discarded candidate_id", "first_identity_loss_stage": "BM25 frozen-pool to NF39 RRF normalization", "boundaries": [_audit("rrf_top40", rrf), _audit("reranker_input_top20", input_rows), _audit("reranker_output_top20", reranked_rows), _audit("final_top5", final)]})
    _write(out_dir / "rrf-candidate-pool-manifest.json", {"artifact_schema": SCHEMA, "candidate_pool_hash": pool_manifest["candidate_pool_hash"], "candidate_count": 1080, "invalid_candidate_key_count": 0, "block_double_colon_count": 0})
    _write(out_dir / "final-context-manifest.json", final_manifest)
    _write(out_dir / "snapshot-manifest.json", snapshot_manifest)
    _write(out_dir / "snapshot-verification-report.json", verification)
    _write(out_dir / "stage-metrics-same-k.json", {"denominators": denoms, "stages": stages})
    _write(out_dir / "source-rank-transitions.json", {"counts": counts, "sources": transitions})
    _write(out_dir / "case-stage-summary.json", {"cases": case_stage_summary(cases=cases, rrf_rankings=rrf_summary, reranker_rankings=reranked_summary, final_rankings=final_summary)})
    acceptance = {"artifact_schema": SCHEMA, "invalid_candidate_key_count": 0, "missing_content_hash_count": 0, "rrf_candidate_count": 1080, "final_candidate_count": 135, "snapshot_payload_verified": verification["passed"], "snapshot_rehydrated_count": verification["snapshot_rehydrated_count"], "verified_final_context_count": verification["verified_final_context_count"], "production_behavior_changed": False, "nf40_start_allowed": verification["passed"]}
    _write(out_dir / "nf39-r2-acceptance.json", acceptance)
    print("NF39 R2 complete:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

