#!/usr/bin/env python3
"""Gate 08 R3-D: First Failure Attribution and Candidate Competition Audit."""

from __future__ import annotations
import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any
from collections import Counter

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"
R3_DIR = ARTIFACTS / "evaluation" / "pdf-retrieval-v4-gate-08-r3"
R5_DIR = ARTIFACTS / "evaluation" / "pdf-retrieval-v4-gate-05-r5"

PREDICTIONS_GZ = R3_DIR / "predictions.jsonl.gz"
SEAL_JSON = R3_DIR / "prediction-seal.json"
GOLD_LABELS = ROOT / "benchmarks" / "financial_rag_v1" / "data" / "labels.golden.jsonl"
UNIVERSE_SCORING = R5_DIR / "universe-scoring.json"
STRUCTURED_VIEWS = R5_DIR / "structured-views.jsonl"

OUTPUT_DIR = R3_DIR


def load_gzip_jsonl(path: Path) -> list[dict]:
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_universe_by_case(universe_scoring: dict) -> dict[str, list[dict]]:
    """Build case_id -> ordered list of universe details (index = position in expected_sources)."""
    details = universe_scoring.get("details", [])
    by_case: dict[str, list[dict]] = {}
    for d in details:
        cid = d.get("case_id", "")
        by_case.setdefault(cid, []).append(d)
    return by_case


def build_universe_by_key(universe_scoring: dict) -> dict[tuple[str, str], dict]:
    """Build (case_id, gold_candidate_key) -> detail mapping as fallback."""
    details = universe_scoring.get("details", [])
    by_key: dict[tuple[str, str], dict] = {}
    for d in details:
        cid = d.get("case_id", "")
        ck = d.get("gold_candidate_key", "")
        if cid and ck:
            by_key[(cid, ck)] = d
    return by_key


def build_view_index(views: list[dict]) -> dict[str, dict]:
    """Build candidate_key -> {metric_paths, pdf_page, document_id}."""
    index: dict[str, dict] = {}
    for v in views:
        ck = v.get("candidate_key")
        if ck is None:
            continue
        index[ck] = {
            "metric_paths": v.get("metric_paths", []),
            "pdf_page": v.get("pdf_page"),
            "document_id": v.get("document_id"),
        }
    return index


def get_pool_keys(pred: dict) -> set[str]:
    return {
        p["candidate_key"]
        for p in pred.get("e3_expanded_pool", [])
        if "candidate_key" in p
    }


def get_sorted_pool(pred: dict) -> list[dict]:
    pool = list(pred.get("e3_expanded_pool", []))
    return sorted(pool, key=lambda x: x.get("rank", 9999))


def get_lane_top_keys(
    pred: dict, lane: str, structured_field: str, top_n: int | None = None
) -> set[str]:
    structured = pred.get(structured_field, {})
    items = structured.get(lane, [])
    if top_n is not None:
        items = items[:top_n]
    return {item["candidate_key"] for item in items if "candidate_key" in item}


def get_fused_rank(
    pred: dict, ck: str, structured_field: str = "structured_expanded"
) -> int | None:
    fused = pred.get(structured_field, {}).get("fused", [])
    for item in fused:
        if item.get("candidate_key") == ck:
            return item.get("rank")
    return None


def classify_first_failure(
    gold_ck: str, pred: dict, universe_detail: dict | None
) -> str:
    """Classify the first failure stage for a missed gold (not in e3_expanded_pool)."""
    # Stage 1: outside_grade_a_universe
    if universe_detail is not None and universe_detail.get("new_status") != "mapped":
        return "outside_grade_a_universe"

    # Stage 2: structured_bm25_and_dense_top50_miss
    bm25_top50 = get_lane_top_keys(pred, "bm25", "structured_expanded", top_n=50)
    dense_top50 = get_lane_top_keys(pred, "dense", "structured_expanded", top_n=50)
    if gold_ck not in bm25_top50 and gold_ck not in dense_top50:
        return "structured_bm25_and_dense_top50_miss"

    # Stage 3: structured_fused_rank_41_to_50
    fused_rank = get_fused_rank(pred, gold_ck)
    if fused_rank is not None and 40 < fused_rank <= 50:
        return "structured_fused_rank_41_to_50"

    # Stage 4: structured_fused_top40_but_post_filter_excluded
    if fused_rank is not None and fused_rank <= 40:
        return "structured_fused_top40_but_post_filter_excluded"

    # Stage 5: slot_budget_truncated
    if pred.get("is_multi_slot", False):
        return "slot_budget_truncated"

    # Stage 6: structured_pool_budget_truncated (fused rank > 40, beyond stage 3 range)
    if fused_rank is not None and fused_rank > 40:
        return "structured_pool_budget_truncated"

    # Stage 7: recovered_by_structured (in structured_expanded but not in final pool)
    exp_bm25 = get_lane_top_keys(pred, "bm25", "structured_expanded")
    exp_dense = get_lane_top_keys(pred, "dense", "structured_expanded")
    exp_fused = get_lane_top_keys(pred, "fused", "structured_expanded")
    if gold_ck in exp_bm25 or gold_ck in exp_dense or gold_ck in exp_fused:
        return "recovered_by_structured"

    # Stage 8: recovered_by_raw_only (in structured_legacy but not in structured_expanded)
    leg_bm25 = get_lane_top_keys(pred, "bm25", "structured_legacy")
    leg_dense = get_lane_top_keys(pred, "dense", "structured_legacy")
    leg_fused = get_lane_top_keys(pred, "fused", "structured_legacy")
    if gold_ck in leg_bm25 or gold_ck in leg_dense or gold_ck in leg_fused:
        return "recovered_by_raw_only"

    return "unclassified"


def competition_audit(gold_ck: str, pred: dict, view_index: dict[str, dict]) -> dict:
    """Analyze competition in E3-Expanded pool top20 for a missed gold."""
    sorted_pool = get_sorted_pool(pred)
    top20 = sorted_pool[:20]

    gold_view = view_index.get(gold_ck, {})
    gold_metric_paths = set(gold_view.get("metric_paths", []))
    gold_pdf_page = gold_view.get("pdf_page")
    gold_document_id = gold_view.get("document_id")

    same_metric = 0
    same_page = 0
    same_doc = 0

    for item in top20:
        ck = item.get("candidate_key")
        if ck is None or ck == gold_ck:
            continue
        v = view_index.get(ck, {})
        if gold_metric_paths and set(v.get("metric_paths", [])) & gold_metric_paths:
            same_metric += 1
        if gold_pdf_page is not None and v.get("pdf_page") == gold_pdf_page:
            same_page += 1
        if gold_document_id is not None and v.get("document_id") == gold_document_id:
            same_doc += 1

    return {
        "candidate_key": gold_ck,
        "same_metric_competition": same_metric,
        "same_page_competition": same_page,
        "same_document_competition": same_doc,
        "pool_size": len(sorted_pool),
        "top20_size": len(top20),
    }


def main():
    parser = argparse.ArgumentParser(description="Gate 08 R3 First Failure Attribution")
    parser.add_argument("--verbose", action="store_true")
    parser.parse_args()

    print(f"[1/5] Loading predictions from {PREDICTIONS_GZ}")
    predictions_raw = load_gzip_jsonl(PREDICTIONS_GZ)
    predictions = [p for p in predictions_raw if p.get("stream") != "header"]
    pred_by_case = {p["case_id"]: p for p in predictions if "case_id" in p}
    print(
        f"  Loaded {len(predictions_raw)} raw records, {len(predictions)} after header filter"
    )

    print(f"[2/5] Loading gold labels from {GOLD_LABELS}")
    gold_records = load_jsonl(GOLD_LABELS)
    print(f"  Loaded {len(gold_records)} gold cases")

    print(f"[3/5] Loading universe scoring from {UNIVERSE_SCORING}")
    universe_scoring = load_json(UNIVERSE_SCORING)
    universe_by_case = build_universe_by_case(universe_scoring)
    universe_by_key = build_universe_by_key(universe_scoring)
    total_universe = len(universe_scoring.get("details", []))
    mapped_count = sum(
        1
        for d in universe_scoring.get("details", [])
        if d.get("new_status") == "mapped"
    )
    outside_count = total_universe - mapped_count
    print(
        f"  Loaded {total_universe} universe details ({mapped_count} mapped, {outside_count} outside)"
    )

    print(f"[4/5] Loading structured views from {STRUCTURED_VIEWS}")
    views = load_jsonl(STRUCTURED_VIEWS)
    view_index = build_view_index(views)
    print(f"  Loaded {len(views)} views, indexed {len(view_index)} candidate keys")

    print("[5/5] Classifying failures and auditing competition")
    print()

    failure_counts: Counter = Counter()
    failure_details: list[dict] = []
    competition_results: list[dict] = []

    total_gold = 0
    in_pool = 0
    missed = 0
    outside_universe_missed = 0
    in_universe_missed = 0

    for gold_rec in gold_records:
        case_id = gold_rec.get("case_id")
        expected_sources = gold_rec.get("expected_sources", [])
        pred = pred_by_case.get(case_id)
        if pred is None:
            print(f"  WARNING: No prediction for case {case_id}")
            continue

        pool_keys = get_pool_keys(pred)
        universe_details = universe_by_case.get(case_id, [])

        for idx, src in enumerate(expected_sources):
            total_gold += 1
            ck = src.get("candidate_key")
            if ck is None:
                continue

            # Match universe detail by (case_id, index), fallback to (case_id, candidate_key)
            universe_detail = None
            if idx < len(universe_details):
                universe_detail = universe_details[idx]
                # Verify candidate_key match; fallback if mismatch
                if (
                    universe_detail.get("gold_candidate_key")
                    and universe_detail.get("gold_candidate_key") != ck
                ):
                    alt = universe_by_key.get((case_id, ck))
                    if alt is not None:
                        universe_detail = alt
            else:
                universe_detail = universe_by_key.get((case_id, ck))

            is_in_universe = (
                universe_detail is not None
                and universe_detail.get("new_status") == "mapped"
            )

            if ck in pool_keys:
                in_pool += 1
                continue

            missed += 1
            if not is_in_universe:
                outside_universe_missed += 1
            else:
                in_universe_missed += 1

            stage = classify_first_failure(ck, pred, universe_detail)
            failure_counts[stage] += 1

            failure_details.append(
                {
                    "case_id": case_id,
                    "source_index": idx,
                    "candidate_key": ck,
                    "document_id": src.get("document_id"),
                    "page": src.get("page"),
                    "metric": src.get("metric"),
                    "period": src.get("period"),
                    "in_universe": is_in_universe,
                    "universe_status": universe_detail.get("new_status")
                    if universe_detail
                    else "no_detail",
                    "first_failure_stage": stage,
                }
            )

            # Competition audit only for in-universe missed golds
            if is_in_universe:
                audit = competition_audit(ck, pred, view_index)
                audit["case_id"] = case_id
                audit["source_index"] = idx
                competition_results.append(audit)

    print("=" * 60)
    print("First Failure Attribution Summary")
    print("=" * 60)
    print(f"Total gold sources:    {total_gold}")
    print(f"In E3-Expanded pool:   {in_pool}")
    print(f"Missed:                {missed}")
    print(f"  Outside universe:    {outside_universe_missed}")
    print(f"  In universe (missed):{in_universe_missed}")
    print()
    print("Failure Stage Breakdown:")
    for stage, count in failure_counts.most_common():
        print(f"  {stage:55s} {count:4d}")
    print()

    print("=" * 60)
    print("Candidate Competition Audit")
    print("=" * 60)
    print(f"Audited {len(competition_results)} in-universe missed golds")
    if competition_results:
        avg_metric = sum(
            c["same_metric_competition"] for c in competition_results
        ) / len(competition_results)
        avg_page = sum(c["same_page_competition"] for c in competition_results) / len(
            competition_results
        )
        avg_doc = sum(
            c["same_document_competition"] for c in competition_results
        ) / len(competition_results)
        print(f"  Avg same_metric_competition:     {avg_metric:.2f}")
        print(f"  Avg same_page_competition:       {avg_page:.2f}")
        print(f"  Avg same_document_competition:   {avg_doc:.2f}")
        print()
        print("Top 10 highest competition entries:")
        sorted_comp = sorted(
            competition_results,
            key=lambda x: x["same_metric_competition"],
            reverse=True,
        )
        for c in sorted_comp[:10]:
            print(
                f"  case={c['case_id']:40s} metric={c['same_metric_competition']:3d} "
                f"page={c['same_page_competition']:3d} doc={c['same_document_competition']:3d}"
            )

    # Write outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    first_failure_output = {
        "gate": "pdf_retrieval_v4_gate_08_r3",
        "total_gold_sources": total_gold,
        "in_pool": in_pool,
        "missed": missed,
        "outside_universe": outside_universe_missed,
        "in_universe_missed": in_universe_missed,
        "failure_stage_counts": dict(failure_counts),
        "failure_details": failure_details,
    }
    ff_path = OUTPUT_DIR / "first-failure-attribution.json"
    with open(ff_path, "w", encoding="utf-8") as f:
        json.dump(first_failure_output, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {ff_path}")

    comp_output = {
        "gate": "pdf_retrieval_v4_gate_08_r3",
        "audited_count": len(competition_results),
        "audit_details": competition_results,
    }
    comp_path = OUTPUT_DIR / "candidate-competition-audit.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comp_output, f, indent=2, ensure_ascii=False)
    print(f"Wrote {comp_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
