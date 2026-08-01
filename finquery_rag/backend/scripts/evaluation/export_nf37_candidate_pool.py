"""Export privacy-safe RRF candidate metadata from an NF36 prediction run."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
from src.evaluation.case_fingerprints import label_fingerprint, question_fingerprint  # noqa: E402
from src.evaluation.evaluation import load_jsonl_cases, load_jsonl_predictions  # noqa: E402

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--index-fingerprint", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--generator-model", required=True)
    args = parser.parse_args()
    cases_path, prediction_path, out_dir = Path(args.cases), Path(args.predictions), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases, predictions = load_jsonl_cases(cases_path), load_jsonl_predictions(prediction_path)
    pool = []
    for case in cases:
        prediction = predictions.get(case.case_id)
        stages = (prediction.retrieval_debug if prediction else {}).get("candidate_stages", {})
        rows = stages.get("rrf", [])[:40]
        pool.append({"case_id": case.case_id, "candidates": [{
            "candidate_id": row.get("evidence_id"), "document_id": row.get("document_id"),
            "page": row.get("page"), "block_type": row.get("block_type"),
            "parent_id": row.get("parent_id"), "table_id": row.get("table_id"),
            "rrf_rank": row.get("rank"), "rrf_score": row.get("rrf_score"),
        } for row in rows]})
    manifest = {
        "question_count": len(cases),
        "question_hash": question_fingerprint(cases),
        "label_hash": label_fingerprint(cases),
        "index_fingerprint": args.index_fingerprint,
        "embedding_model": args.embedding_model,
        "generator_model": args.generator_model,
        "candidate_pool_depth": 40,
    }
    (out_dir / "candidate-pool.json").write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "baseline-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
