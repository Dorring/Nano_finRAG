"""Gate 08 R2: Build candidate-aligned dual views and save as JSONL.

For each Production Candidate, build a Raw View (original candidate text)
and, when a Grade-A structural mapping exists, a Structured View aggregating
V4 structural metadata (metric_paths, periods, facts, temporal types).

Runs over the FULL Production Candidate Universe -- no Gold/Question/
Expected Value is read.  Each JSONL line carries the CandidateViewPair
summary fields plus the full raw_view / structured_view dicts so the
index builder can reconstruct CandidateAlignedView objects.
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

from src.pdf_retrieval_v4.candidate_view_builder import CandidateViewBuilder  # noqa: E402
from src.pdf_retrieval_v4.structural_gold_mapper import StructuralGoldMapper  # noqa: E402
from src.pdf_retrieval_v4.v4_gate08_pool import ProductionCandidateMapper  # noqa: E402

DEFAULT_DB = ROOT / "rag_bm25.db"
DEFAULT_CORPUS = ROOT / "benchmarks/financial_rag_v1/corpus.json"
DEFAULT_RUNTIME = ROOT / "artifacts/runtime/pdf-retrieval-v4-gate-06-r2"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2/candidate-views"


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    metadata_db = args.runtime_dir / "metadata" / "metadata.sqlite"
    if not metadata_db.is_file():
        raise RuntimeError(f"metadata_db_not_found:{metadata_db}")

    mapper = ProductionCandidateMapper(args.db_path, args.corpus, tenant_id=1)
    gold_mapper = StructuralGoldMapper(metadata_db, mapper)
    builder = CandidateViewBuilder(mapper, gold_mapper)

    pairs = builder.build_all()
    stats = builder.build_stats(pairs)

    records: list[dict[str, Any]] = []
    for pair in pairs:
        record = dict(pair.to_dict())
        record["raw_view"] = pair.raw_view.to_dict()
        record["structured_view"] = (
            pair.structured_view.to_dict() if pair.structured_view else None
        )
        records.append(record)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    views_path = out_dir / "view-pairs.jsonl"
    write_jsonl(views_path, records)
    write(out_dir / "view-builder-stats.json", stats)

    gold_mapper.close()

    print("Gate 08 R2 candidate view build complete.")
    print(f"  Total candidates:     {stats['total_candidates']}")
    print(f"  With structured view: {stats['with_structured_view']}")
    print(f"  Raw only:             {stats['raw_only']}")
    print(f"  Bridge grades:        {stats['bridge_grade_counts']}")
    print(f"  Output:               {views_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
