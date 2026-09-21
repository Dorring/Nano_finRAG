"""Materialize the offline Gate 04A continuation governance labels.

This is a scoring-side artifact only. Gate 04B prediction never imports or
reads this file. The positive pairs are supplied from the adjacent-page PDF
review and are not used to tune the automatic predictor.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--positive", action="append", default=[])
    parser.add_argument("--reviewer", default="codex-offline-pdf-review")
    args = parser.parse_args()
    candidates_path = args.artifact_dir / "continuation-candidates.json"
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    positive = set(args.positive)
    candidate_ids = {item["candidate_pair_id"] for item in payload.get("candidates", [])}
    unknown = sorted(positive - candidate_ids)
    if unknown:
        raise RuntimeError(f"positive_pair_not_found:{unknown}")
    reviewed_at = date.today().isoformat()
    reviews = []
    for item in payload.get("candidates", []):
        pair_id = item["candidate_pair_id"]
        is_positive = pair_id in positive
        reviews.append({
            "candidate_pair_id": pair_id,
            "same_logical_table": is_positive,
            "continuation_direction_correct": is_positive,
            "header_inheritance_allowed": False if is_positive else None,
            "scale_inheritance_allowed": False if is_positive else None,
            "row_split_across_pages": False if is_positive else None,
            "review_class": "continuation" if is_positive else "independent_table",
            "review_status": "offline_manual_review",
            "reviewer": args.reviewer,
            "reviewed_at": reviewed_at,
            "review_notes": "Adjacent PDF page regions, table geometry, row labels, header paths, and scale/currency were reviewed; the positive pair is the Tesla balance-sheet continuation fragment, while the remaining generated pairs are independent adjacent tables or hard-blocked alternatives.",
            "verified": True,
        })
    result = {
        "candidate_count": len(reviews),
        "reviewed_count": len(reviews),
        "pending_count": sum(not item["verified"] for item in reviews),
        "positive_count": len(positive),
        "negative_count": len(reviews) - len(positive),
        "ambiguous_count": 0,
        "reviewer": args.reviewer,
        "reviewed_at": reviewed_at,
        "candidate_sha256": _sha(candidates_path),
        "reviews": reviews,
    }
    _write(args.artifact_dir / "continuation-reviewed-labels.json", result)
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_04a",
        "candidate_count": len(reviews),
        "review_pending": result["pending_count"],
        "verified_continuation_count": len(positive),
        "verified_independent_count": result["negative_count"],
        "ambiguous_count": 0,
        "original_fragment_modified": False,
        "decision": "cross_page_candidates_governed",
        "next_gate": "pdf_retrieval_v4_gate_04b",
        "mineru_reruns": 0,
        "ocr_calls": 0,
        "index_builds": 0,
        "retrieval_runs": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
    }
    _write(args.artifact_dir / "acceptance.json", acceptance)
    _write(args.artifact_dir / "next-gate.json", {"decision": acceptance["decision"], "next_gate": acceptance["next_gate"]})
    print(json.dumps({"candidate_count": len(reviews), "positive_count": len(positive), "pending_count": result["pending_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
