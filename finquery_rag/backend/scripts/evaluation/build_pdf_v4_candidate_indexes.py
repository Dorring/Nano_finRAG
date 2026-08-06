"""Gate 08 R2: Build 4 shadow lane indexes from candidate views.

Loads view-pairs.jsonl, reconstructs CandidateAlignedView objects, and
builds 4 isolated shadow lanes (candidate_raw_bm25, candidate_raw_dense,
candidate_structured_bm25, candidate_structured_dense) plus a shared
candidate-metadata.sqlite mapping view_id -> candidate_key.

The dense lanes encode all view texts with sentence-transformers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.candidate_aligned_view import (  # noqa: E402
    CandidateAlignedView,
    CandidateViewPair,
)
from src.pdf_retrieval_v4.candidate_view_index import (  # noqa: E402
    CandidateViewIndexBuilder,
)

DEFAULT_VIEWS = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2/candidate-views"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2/candidate-indexes"


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _view_from_dict(payload: dict[str, Any]) -> CandidateAlignedView:
    return CandidateAlignedView(
        candidate_key=str(payload["candidate_key"]),
        view_type=str(payload["view_type"]),
        view_id=str(payload["view_id"]),
        retrieval_text=str(payload["retrieval_text"]),
        document_id=str(payload.get("document_id") or ""),
        pdf_page=payload.get("pdf_page"),
        logical_table_ids=tuple(payload.get("logical_table_ids") or []),
        row_ids=tuple(payload.get("row_ids") or []),
        fact_ids=tuple(payload.get("fact_ids") or []),
        metric_paths=tuple(payload.get("metric_paths") or []),
        periods=tuple(payload.get("periods") or []),
        temporal_types=tuple(payload.get("temporal_types") or []),
        bridge_grade=str(payload.get("bridge_grade") or "raw_only"),
    )


def load_view_pairs(path: Path) -> list[CandidateViewPair]:
    pairs: list[CandidateViewPair] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        raw_view = _view_from_dict(record["raw_view"])
        structured_view: CandidateAlignedView | None = None
        sv = record.get("structured_view")
        if sv:
            structured_view = _view_from_dict(sv)
        pairs.append(
            CandidateViewPair(
                candidate_key=str(record["candidate_key"]),
                raw_view=raw_view,
                structured_view=structured_view,
            )
        )
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views-dir", type=Path, default=DEFAULT_VIEWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--encoder-model", default="all-MiniLM-L6-v2",
        help="sentence-transformers model name for dense lanes",
    )
    args = parser.parse_args()

    views_path = args.views_dir / "view-pairs.jsonl"
    if not views_path.is_file():
        raise RuntimeError(f"view_pairs_not_found:{views_path}")

    pairs = load_view_pairs(views_path)
    builder = CandidateViewIndexBuilder(args.out_dir, args.encoder_model)
    report = builder.build(pairs)

    write(args.out_dir / "index-build-report.json", report)

    print("Gate 08 R2 candidate index build complete.")
    print(f"  Total pairs:           {report['total_pairs']}")
    print(f"  Unique candidate keys: {report['unique_candidate_keys']}")
    print(f"  Integrity OK:          {report['integrity']['ok']}")
    for lane, lane_stat in report["lanes"].items():
        print(f"  Lane {lane}: {lane_stat['view_count']} views")
    print(f"  Output:                {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
