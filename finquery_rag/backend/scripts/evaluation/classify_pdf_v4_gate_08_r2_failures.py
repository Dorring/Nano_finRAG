"""Gate 08 R2: Classify unrecovered B-class candidates into failure stages.

Uses classify_b_class_failure from candidate_retrieval_attribution to
attribute each B-class candidate (from the R1.1 gold-coverage-
classification) to a structured failure stage based on its presence in
the R2 candidate direct retrieval pool, lane hits, and RRF ranking.

No predictions are re-run.  Gold/labels are read only for offline
attribution after the R2 seal has been written.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.candidate_retrieval_attribution import (  # noqa: E402
    classify_b_class_failure,
)
from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit  # noqa: E402
from src.pdf_retrieval_v4.candidate_view_index import CandidateSearchHit  # noqa: E402

DEFAULT_R2_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2"
DEFAULT_R11_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r1-1"
DEFAULT_LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2/failure-classification"

B_CLASS = "strict_mapped_not_retrieved"


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_predictions(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if value.get("stream") != "header":
                    records.append(value)
    return records


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reconstruct_search_hit(
    lane: str, item: dict[str, Any]
) -> CandidateSearchHit:
    rank = item.get("rank")
    score = item.get("score")
    if "bm25" in lane:
        return CandidateSearchHit(
            candidate_key=str(item.get("candidate_key") or ""),
            view_id=str(item.get("view_id") or ""),
            lane=lane,
            bm25_rank=rank,
            dense_rank=None,
            bm25_score=score,
            dense_score=None,
        )
    return CandidateSearchHit(
        candidate_key=str(item.get("candidate_key") or ""),
        view_id=str(item.get("view_id") or ""),
        lane=lane,
        bm25_rank=None,
        dense_rank=rank,
        bm25_score=None,
        dense_score=score,
    )


def _reconstruct_rrf_hit(item: dict[str, Any]) -> CandidateRRFHit:
    return CandidateRRFHit(
        candidate_key=str(item.get("candidate_key") or ""),
        rrf_score=float(item.get("rrf_score") or 0.0),
        lane_ranks={
            str(k): int(v)
            for k, v in (item.get("lane_ranks") or {}).items()
        },
        supporting_view_ids={
            str(k): str(v)
            for k, v in (item.get("supporting_view_ids") or {}).items()
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-out", type=Path, default=DEFAULT_R2_OUT)
    parser.add_argument("--r11-out", type=Path, default=DEFAULT_R11_OUT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    # Load R2 predictions.
    r2_predictions_path = args.r2_out / "predictions.jsonl.gz"
    r2_predictions_list = load_predictions(r2_predictions_path)
    r2_predictions = {
        str(item["case_id"]): item for item in r2_predictions_list
    }

    # Load R1.1 gold-coverage-classification.
    r11_classification_path = args.r11_out / "gold-coverage-classification.json"
    r11_classification = json.loads(
        r11_classification_path.read_text(encoding="utf-8")
    )
    r11_rows = r11_classification.get("rows") or []
    r11_by_identity = {
        str(row["gold_source_identity"]): row for row in r11_rows
    }

    # Load labels (for gold candidate keys).
    labels_list = load_jsonl(args.labels)
    labels = {str(item["case_id"]): item for item in labels_list}

    # ------------------------------------------------------------------
    # Classify B-class candidates
    # ------------------------------------------------------------------
    attribution_rows: list[dict[str, Any]] = []
    for case_id in sorted(labels):
        if case_id not in r2_predictions:
            continue
        label = labels[case_id]
        sources = [
            item
            for item in label.get("expected_sources") or []
            if item.get("candidate_key")
        ]
        r2_pred = r2_predictions[case_id]

        candidate_direct_pool = r2_pred.get("candidate_direct_pool") or []
        is_multi_slot = bool(r2_pred.get("is_multi_slot"))

        # Reconstruct lane_hits and rrf_hits as dataclass objects.
        lane_hits: dict[str, list[CandidateSearchHit]] = {}
        for lane, hits in (r2_pred.get("lane_hits") or {}).items():
            lane_hits[str(lane)] = [
                _reconstruct_search_hit(str(lane), item) for item in hits
            ]
        rrf_hits: list[CandidateRRFHit] = [
            _reconstruct_rrf_hit(item)
            for item in (r2_pred.get("rrf_hits") or [])
        ]

        for idx, source in enumerate(sources):
            identity = f"{case_id}#{idx}"
            r11_row = r11_by_identity.get(identity, {})
            if str(r11_row.get("coverage_class") or "") != B_CLASS:
                continue
            gold_key = str(source.get("candidate_key"))

            result = classify_b_class_failure(
                candidate_key=gold_key,
                case_id=case_id,
                candidate_direct_pool=candidate_direct_pool,
                lane_hits=lane_hits,
                rrf_hits=rrf_hits,
                is_multi_slot=is_multi_slot,
            )
            attribution_rows.append(
                {
                    "gold_source_identity": identity,
                    "case_id": case_id,
                    "gold_candidate_key": gold_key,
                    "first_failure_stage": result.first_failure_stage.value,
                    "best_rank": result.best_rank,
                    "in_top50": result.in_top50,
                    "in_top40": result.in_top40,
                    "detail": result.detail,
                }
            )

    failure_counts: Counter[str] = Counter(
        row["first_failure_stage"] for row in attribution_rows
    )
    recovered_count = failure_counts.get("recovered", 0)
    unrecovered_count = len(attribution_rows) - recovered_count

    failure_attribution = {
        "gate": "pdf_retrieval_v4_gate_08_r2",
        "b_class_total": len(attribution_rows),
        "recovered": recovered_count,
        "unrecovered": unrecovered_count,
        "failure_stage_counts": dict(failure_counts),
        "rows": attribution_rows,
    }

    failure_summary = {
        "gate": "pdf_retrieval_v4_gate_08_r2",
        "b_class_total": len(attribution_rows),
        "recovered": recovered_count,
        "unrecovered": unrecovered_count,
        "failure_stage_counts": dict(failure_counts),
        "primary_failure_stages": sorted(
            [
                {"stage": stage, "count": count}
                for stage, count in failure_counts.items()
                if stage != "recovered"
            ],
            key=lambda x: -x["count"],
        ),
    }

    write(args.out_dir / "failure-attribution.json", failure_attribution)
    write(args.out_dir / "failure-summary.json", failure_summary)

    print("Gate 08 R2 failure classification complete.")
    print(f"  B-class total:    {len(attribution_rows)}")
    print(f"  Recovered:        {recovered_count}")
    print(f"  Unrecovered:      {unrecovered_count}")
    print(f"  Failure stages:   {dict(failure_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
