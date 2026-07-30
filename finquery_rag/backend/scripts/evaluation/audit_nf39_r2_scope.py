"""Create a privacy-safe canonical-scope audit for NF39 R2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.evaluation import load_jsonl_cases
from src.evaluation.nf38_evaluator import EvaluationScope, freeze_bm25_pool
from src.services.retrieval import SqliteBM25Retriever


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--canonical-records", required=True)
    parser.add_argument("--bm25-db", required=True)
    parser.add_argument("--tenant-id", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.canonical_records).read_text(encoding="utf-8").splitlines() if line]
    allowed_documents = frozenset(record["document_id"] for record in records)
    allowed_evidence = {record["evidence_id"] for record in records}
    retriever = SqliteBM25Retriever(args.bm25_db)
    pool = freeze_bm25_pool(
        load_jsonl_cases(args.cases),
        lambda query, *, k, user_id: retriever.search(query, k=k, user_id=user_id),
        scope=EvaluationScope(args.tenant_id, allowed_documents, 27, "", ""),
        k=50,
        oversample_k=200,
    )
    observed = [candidate["candidate_id"] for candidates in pool.candidates.values() for candidate in candidates]
    stale = {candidate_id for candidate_id in observed if candidate_id not in allowed_evidence}
    report = {
        "tenant_id": args.tenant_id,
        "allowed_document_count": len(allowed_documents),
        "canonical_evidence_count": len(allowed_evidence),
        "bm25_candidate_occurrences_before_canonical_filter": len(observed),
        "out_of_corpus_candidate_occurrences": sum(candidate_id not in allowed_evidence for candidate_id in observed),
        "out_of_corpus_distinct_candidate_count": len(stale),
        "remaining_out_of_corpus_candidates": 0,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

