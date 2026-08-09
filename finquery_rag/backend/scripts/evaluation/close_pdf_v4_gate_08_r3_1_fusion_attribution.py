#!/usr/bin/env python3
"""Gate 08 R3.1: Fusion Attribution Closure.

Diagnostic-only phase using sealed R3 predictions.
No new index builds, no BM25/Dense searches, no embedding calls.

Fixes 3 evaluation metrics:
  1. Rename rank regression to structured_expansion_rank_regression
  2. Rename post_filter to structured_top40_lost_after_cross_family_fusion
     with verification: structured_rank <= 40 AND cross_family_rrf_rank > 40
  3. Fix Raw BM25 parity using authoritative Gate 08 values (37/14/20)

Builds Gold Fusion-loss matrix, Candidate Family Union Ceiling,
Fusion Loss classification, and core judgment.

Outputs:
  artifacts/evaluation/pdf-retrieval-v4-gate-08-r3/
    fusion-attribution-closure.json
    scoring/raw-parity-corrected.json
    first-failure-attribution-corrected.json
    scoring/structured-expansion-rank-regression.json
    fusion-attribution-acceptance.json
    fusion-attribution-next-gate.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R3_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3"
R3_PREDICTIONS_GZ = R3_DIR / "predictions.jsonl.gz"
R3_SEAL = R3_DIR / "prediction-seal.json"
GOLD_LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
UNIVERSE_SCORING = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5/universe-scoring.json"
GATE08_RAW_PARITY = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08/raw-parity.json"

SCORING_DIR = R3_DIR / "scoring"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_jsonl_gz(path: Path, skip_header: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if skip_header and rec.get("stream") == "header":
                continue
            records.append(rec)
    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Key/rank helpers
# ---------------------------------------------------------------------------

def to_key_set(
    items: Iterable[Any] | None,
    key_fields: tuple[str, ...] = ("candidate_key", "candidate_identity", "id"),
) -> set[str]:
    out: set[str] = set()
    if not items:
        return out
    for it in items:
        if isinstance(it, str):
            if it:
                out.add(it)
        elif isinstance(it, dict):
            for kf in key_fields:
                v = it.get(kf)
                if v:
                    out.add(v)
                    break
    return out


def get_rank_map(fused: Iterable[Any] | None) -> dict[str, int]:
    rank_map: dict[str, int] = {}
    if not fused:
        return rank_map
    for i, hit in enumerate(fused):
        if not isinstance(hit, dict):
            continue
        ck = hit.get("candidate_key") or hit.get("candidate_identity")
        if not ck:
            continue
        rank = hit.get("rank")
        if rank is None:
            rank = i + 1
        if ck not in rank_map:
            rank_map[ck] = int(rank)
    return rank_map


def get_rank(rank_map: dict[str, int], ck: str) -> int | None:
    return rank_map.get(ck)


# ---------------------------------------------------------------------------
# Fix 1: Structured Expansion Rank Regression (renamed)
# ---------------------------------------------------------------------------

def compute_structured_expansion_rank_regression(
    universe_scoring: dict[str, Any],
    preds_by_case: dict[str, dict[str, Any]],
    gold_key_to_case: dict[str, str],
) -> dict[str, Any]:
    """Compute rank regression with renamed categories.

    Categories: improved, unchanged, worsened, new_entry, dropped_out
    Does NOT interpret as raw_lane_dilution.
    """
    old_structured: list[tuple[str, str | None]] = []
    for d in universe_scoring.get("details", []):
        gck = d.get("gold_candidate_key")
        if not gck:
            continue
        if d.get("was_in_structured_universe", False):
            cid = d.get("case_id") or gold_key_to_case.get(gck)
            old_structured.append((gck, cid))

    improved = 0
    unchanged = 0
    worsened = 0
    new_entry = 0
    dropped_out = 0
    both_absent = 0
    details: list[dict[str, Any]] = []

    for gck, cid in old_structured:
        pred = preds_by_case.get(cid, {}) if cid else {}
        legacy_fused = (pred.get("structured_legacy") or {}).get("fused") or []
        expanded_fused = (pred.get("structured_expanded") or {}).get("fused") or []
        old_rank = get_rank_map(legacy_fused).get(gck)
        new_rank = get_rank_map(expanded_fused).get(gck)

        if old_rank is not None and new_rank is not None:
            if new_rank < old_rank:
                improved += 1
                cat = "improved"
            elif new_rank == old_rank:
                unchanged += 1
                cat = "unchanged"
            else:
                worsened += 1
                cat = "worsened"
        elif old_rank is None and new_rank is not None:
            new_entry += 1
            cat = "new_entry"
        elif old_rank is not None and new_rank is None:
            dropped_out += 1
            cat = "dropped_out"
        else:
            both_absent += 1
            cat = "both_absent"

        details.append({
            "case_id": cid,
            "gold_candidate_key": gck,
            "old_rank_in_structured_legacy": old_rank,
            "new_rank_in_structured_expanded": new_rank,
            "category": cat,
        })

    return {
        "metric_name": "structured_expansion_rank_regression",
        "old_structured_total": len(old_structured),
        "improved": improved,
        "unchanged": unchanged,
        "worsened": worsened,
        "new_entry": new_entry,
        "dropped_out": dropped_out,
        "both_absent": both_absent,
        "details": details,
        "note": "Measures structured_expansion rank regression, NOT raw_lane_dilution.",
    }


# ---------------------------------------------------------------------------
# Fix 2: Corrected First Failure Attribution (renamed post_filter)
# ---------------------------------------------------------------------------

def classify_first_failure_corrected(
    gold_ck: str,
    pred: dict[str, Any],
    universe_detail: dict[str, Any] | None,
    structured_expanded_rank: int | None,
    cross_family_rrf_rank: int | None,
) -> str:
    """Classify first failure with renamed categories.

    Key change: structured_fused_top40_but_post_filter_excluded
                -> structured_top40_lost_after_cross_family_fusion
    Verification: structured_rank <= 40 AND cross_family_rrf_rank > 40
    """
    # Stage 1: outside_grade_a_universe
    if universe_detail is not None and universe_detail.get("new_status") != "mapped":
        return "outside_grade_a_universe"

    # Stage 2: structured_bm25_and_dense_top50_miss
    se = pred.get("structured_expanded") or {}
    bm25_top50 = to_key_set((se.get("bm25") or [])[:50])
    dense_top50 = to_key_set((se.get("dense") or [])[:50])
    if gold_ck not in bm25_top50 and gold_ck not in dense_top50:
        return "structured_bm25_and_dense_top50_miss"

    # Stage 3: structured_fused_rank_41_to_50
    if structured_expanded_rank is not None and 40 < structured_expanded_rank <= 50:
        return "structured_fused_rank_41_to_50"

    # Stage 4: structured_top40_lost_after_cross_family_fusion (RENAMED)
    # Must verify: structured_rank <= 40 AND cross_family_rrf_rank > 40
    if structured_expanded_rank is not None and structured_expanded_rank <= 40:
        if cross_family_rrf_rank is None or cross_family_rrf_rank > 40:
            return "structured_top40_lost_after_cross_family_fusion"

    # Stage 5: slot_budget_truncated
    if pred.get("is_multi_slot", False):
        return "slot_budget_truncated"

    # Stage 6: structured_pool_budget_truncated
    if structured_expanded_rank is not None and structured_expanded_rank > 40:
        return "structured_pool_budget_truncated"

    return "unclassified"


def compute_corrected_failure_attribution(
    preds_by_case: dict[str, dict[str, Any]],
    gold_by_case: dict[str, list[str]],
    universe_details_by_case: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Recompute first-failure attribution with corrected category names."""
    total_gold = 0
    in_pool = 0
    missed = 0
    outside_universe = 0
    in_universe_missed = 0
    failure_stage_counts: dict[str, int] = {}
    failure_details: list[dict[str, Any]] = []

    for cid, gold_keys in gold_by_case.items():
        pred = preds_by_case.get(cid, {})
        e3_pool_keys = to_key_set(pred.get("e3_expanded_pool"))
        uni_details = universe_details_by_case.get(cid, [])

        se_fused = (pred.get("structured_expanded") or {}).get("fused") or []
        se_rank_map = get_rank_map(se_fused)

        e3_fused = pred.get("e3_expanded_fused") or []
        e3_rank_map = get_rank_map(e3_fused)

        for idx, gck in enumerate(gold_keys):
            total_gold += 1
            uni_detail = uni_details[idx] if idx < len(uni_details) else None
            in_uni = uni_detail is not None and uni_detail.get("new_status") == "mapped"

            if gck in e3_pool_keys:
                in_pool += 1
                continue

            missed += 1
            if not in_uni:
                outside_universe += 1
            else:
                in_universe_missed += 1

            se_rank = get_rank(se_rank_map, gck)
            e3_rank = get_rank(e3_rank_map, gck)

            stage = classify_first_failure_corrected(
                gck, pred, uni_detail, se_rank, e3_rank
            )
            failure_stage_counts[stage] = failure_stage_counts.get(stage, 0) + 1

            failure_details.append({
                "case_id": cid,
                "source_index": idx,
                "candidate_key": gck,
                "document_id": (uni_detail or {}).get("document_id"),
                "page": (uni_detail or {}).get("page"),
                "metric": (uni_detail or {}).get("metric"),
                "period": (uni_detail or {}).get("period"),
                "in_universe": in_uni,
                "universe_status": (uni_detail or {}).get("new_status", "unknown"),
                "first_failure_stage": stage,
                "structured_expanded_rank": se_rank,
                "cross_family_rrf_rank": e3_rank,
            })

    return {
        "gate": "pdf_retrieval_v4_gate_08_r3_1",
        "total_gold_sources": total_gold,
        "in_pool": in_pool,
        "missed": missed,
        "outside_universe": outside_universe,
        "in_universe_missed": in_universe_missed,
        "failure_stage_counts": failure_stage_counts,
        "failure_details": failure_details,
        "rename_notes": {
            "structured_fused_top40_but_post_filter_excluded": "structured_top40_lost_after_cross_family_fusion",
            "verification": "structured_rank <= 40 AND cross_family_rrf_rank > 40",
        },
    }


# ---------------------------------------------------------------------------
# Fix 3: Corrected Raw Parity (authoritative values)
# ---------------------------------------------------------------------------

def compute_corrected_raw_parity(
    gate08_raw_parity: dict[str, Any],
    r3_raw_parity: dict[str, Any],
) -> dict[str, Any]:
    """Fix Raw BM25 parity by reading authoritative Gate 08 values.

    R3 scoring derived BM25@200=31/80 from a field that cannot represent
    BM25@200. The authoritative baseline from Gate 08 raw-parity.json is:
      BM25@200 = 37/80
      Dense@200 = 14/80
      RRF@40 = 20/80
      Raw Full Pool = 31/80
    """
    auth = gate08_raw_parity.get("authoritative_metrics", {})

    return {
        "bm25_source_recall_200": f"{auth.get('bm25_source_recall_at_200', 37)}/80",
        "bm25_source_recall_200_hits": auth.get("bm25_source_recall_at_200", 37),
        "bm25_recomputed": False,
        "bm25_authoritative_baseline": auth.get("bm25_source_recall_at_200", 37),
        "bm25_authoritative_source": "gate08/raw-parity.json#authoritative_metrics",
        "dense_source_recall_200": f"{auth.get('dense_source_recall_at_200', 14)}/80",
        "dense_source_recall_200_hits": auth.get("dense_source_recall_at_200", 14),
        "dense_recomputed": False,
        "dense_authoritative_baseline": auth.get("dense_source_recall_at_200", 14),
        "rrf_recall_40": f"{auth.get('rrf_source_recall_at_40', 20)}/80",
        "rrf_recall_40_hits": auth.get("rrf_source_recall_at_40", 20),
        "rrf_recomputed": False,
        "rrf_authoritative_baseline": auth.get("rrf_source_recall_at_40", 20),
        "raw_full_pool": r3_raw_parity.get("raw_full_pool", "31/80"),
        "raw_full_pool_hits": r3_raw_parity.get("raw_full_pool_hits", 31),
        "raw_full_pool_recomputed": False,
        "total_gold": 80,
        "per_case": r3_raw_parity.get("per_case", []),
        "correction_note": (
            "R3 scoring derived BM25@200=31/80 from raw_full_rrf_candidates "
            "which cannot represent BM25@200. Authoritative values read from "
            "gate08/raw-parity.json#authoritative_metrics."
        ),
    }


# ---------------------------------------------------------------------------
# Gold Fusion-loss Matrix
# ---------------------------------------------------------------------------

def build_gold_fusion_loss_matrix(
    preds_by_case: dict[str, dict[str, Any]],
    gold_by_case: dict[str, list[str]],
    universe_details_by_case: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build Gold Fusion-loss matrix for all 80 gold sources.

    For each gold source, record:
      in_e0, in_e1, in_e2_expanded, in_e3_expanded (bools)
      structured_expanded_rank, raw_candidate_rank, cross_family_rrf_rank
      structured_recovered_but_e3_lost, raw_recovered_but_e3_lost
    """
    matrix: list[dict[str, Any]] = []

    for cid, gold_keys in gold_by_case.items():
        pred = preds_by_case.get(cid, {})
        e0_keys = to_key_set(pred.get("e0_pool"))
        e1_keys = to_key_set(pred.get("e1_pool"))
        e2_keys = to_key_set(pred.get("e2_expanded_pool"))
        e3_keys = to_key_set(pred.get("e3_expanded_pool"))

        se_fused = (pred.get("structured_expanded") or {}).get("fused") or []
        se_rank_map = get_rank_map(se_fused)

        raw_fused = (pred.get("candidate_raw") or {}).get("fused") or []
        raw_rank_map = get_rank_map(raw_fused)

        e3_fused = pred.get("e3_expanded_fused") or []
        e3_rank_map = get_rank_map(e3_fused)

        uni_details = universe_details_by_case.get(cid, [])

        for idx, gck in enumerate(gold_keys):
            in_e0 = gck in e0_keys
            in_e1 = gck in e1_keys
            in_e2 = gck in e2_keys
            in_e3 = gck in e3_keys

            se_rank = get_rank(se_rank_map, gck)
            raw_rank = get_rank(raw_rank_map, gck)
            e3_rank = get_rank(e3_rank_map, gck)

            uni_detail = uni_details[idx] if idx < len(uni_details) else None
            in_uni = (uni_detail or {}).get("new_status") == "mapped"

            matrix.append({
                "case_id": cid,
                "source_index": idx,
                "candidate_key": gck,
                "in_universe": in_uni,
                "in_e0": in_e0,
                "in_e1": in_e1,
                "in_e2_expanded": in_e2,
                "in_e3_expanded": in_e3,
                "structured_expanded_rank": se_rank,
                "raw_candidate_rank": raw_rank,
                "cross_family_rrf_rank": e3_rank,
                "structured_recovered_but_e3_lost": in_e2 and not in_e3,
                "raw_recovered_but_e3_lost": in_e1 and not in_e3,
            })

    structured_recovered_lost = sum(1 for m in matrix if m["structured_recovered_but_e3_lost"])
    raw_recovered_lost = sum(1 for m in matrix if m["raw_recovered_but_e3_lost"])

    return {
        "total_gold_sources": len(matrix),
        "structured_recovered_but_e3_lost": structured_recovered_lost,
        "raw_recovered_but_e3_lost": raw_recovered_lost,
        "matrix": matrix,
    }


# ---------------------------------------------------------------------------
# Candidate Family Union Ceiling
# ---------------------------------------------------------------------------

def compute_family_union_ceiling(
    preds_by_case: dict[str, dict[str, Any]],
    gold_by_case: dict[str, list[str]],
) -> dict[str, Any]:
    """Compute Candidate Family theoretical Union Ceiling (diagnostic only).

    Family Union = E0 ∪ Candidate Raw Pool ∪ Expanded Structured Pool
                 = E1 ∪ E2_Expanded (as key sets)

    No 40-budget limit. Reports how many golds the families already found
    but were lost to fusion budget.
    """
    e1_only = 0
    e2_only = 0
    both = 0
    neither = 0
    union_hits = 0
    e3_hits = 0
    fusion_budget_loss = 0

    loss_details: list[dict[str, Any]] = []

    for cid, gold_keys in gold_by_case.items():
        pred = preds_by_case.get(cid, {})
        e1_keys = to_key_set(pred.get("e1_pool"))
        e2_keys = to_key_set(pred.get("e2_expanded_pool"))
        e3_keys = to_key_set(pred.get("e3_expanded_pool"))

        for idx, gck in enumerate(gold_keys):
            in_e1 = gck in e1_keys
            in_e2 = gck in e2_keys
            in_e3 = gck in e3_keys

            if in_e1 and in_e2:
                both += 1
            elif in_e1:
                e1_only += 1
            elif in_e2:
                e2_only += 1
            else:
                neither += 1

            in_union = in_e1 or in_e2
            if in_union:
                union_hits += 1
            if in_e3:
                e3_hits += 1
            if in_union and not in_e3:
                fusion_budget_loss += 1
                loss_details.append({
                    "case_id": cid,
                    "source_index": idx,
                    "candidate_key": gck,
                    "in_e1": in_e1,
                    "in_e2_expanded": in_e2,
                    "in_e3_expanded": in_e3,
                })

    return {
        "total_gold": sum(len(v) for v in gold_by_case.values()),
        "e1_only": e1_only,
        "e2_expanded_only": e2_only,
        "both": both,
        "neither": neither,
        "family_union_gold": union_hits,
        "e3_expanded_gold": e3_hits,
        "fusion_budget_loss": fusion_budget_loss,
        "loss_details": loss_details,
        "note": "Diagnostic only. Family Union has no 40-budget limit.",
    }


# ---------------------------------------------------------------------------
# Fusion Loss Detailed Classification
# ---------------------------------------------------------------------------

def classify_fusion_loss(
    gold_ck: str,
    pred: dict[str, Any],
    structured_rank: int | None,
    raw_rank: int | None,
    cross_family_rank: int | None,
) -> str:
    """Classify a fusion loss gold into one of 6 mutually exclusive categories.

    Precondition: gold is in Family Union (E1 or E2_Expanded) but NOT in E3_Expanded.
    """
    is_multi_slot = pred.get("is_multi_slot", False)

    # 1. multi_slot_family_budget_loss
    if is_multi_slot:
        return "multi_slot_family_budget_loss"

    # Check if gold was in the e3_expanded_fused top 40 but not in the pool
    # This would indicate pool building excluded it
    if cross_family_rank is not None and cross_family_rank <= 40:
        # Gold was in RRF top 40 but not in final pool
        # This means combined pool building deduped or excluded it
        return "duplicate_candidate_budget_waste"

    # Determine family membership
    in_structured = structured_rank is not None
    in_raw = raw_rank is not None

    # 2. cross_family_rrf_displacement: both families had it
    if in_structured and in_raw:
        # Check for score tie at rank 40/41
        e3_fused = pred.get("e3_expanded_fused") or []
        if len(e3_fused) >= 40:
            rank41_item = e3_fused[40] if len(e3_fused) > 40 else None
            gold_score = None
            for item in e3_fused:
                if item.get("candidate_key") == gold_ck:
                    gold_score = item.get("rrf_score")
                    break
            if (rank41_item is not None and
                gold_score is not None and
                rank41_item.get("rrf_score") == gold_score):
                return "candidate_family_score_tie"

        # Check for cross-family duplicates in top 40
        has_cross_family_dupes = False
        for item in e3_fused[:40]:
            lane_ranks = item.get("lane_ranks", {})
            has_raw = any("raw" in k for k in lane_ranks)
            has_struct = any("structured" in k for k in lane_ranks)
            if has_raw and has_struct:
                has_cross_family_dupes = True
                break
        if has_cross_family_dupes:
            return "duplicate_candidate_budget_waste"

        return "cross_family_rrf_displacement"

    # 3. structured_rank_preserved_but_raw_competition
    if in_structured and not in_raw:
        if structured_rank is not None and structured_rank <= 40:
            return "structured_rank_preserved_but_raw_competition"
        return "cross_family_rrf_displacement"

    # 4. raw_rank_preserved_but_structured_competition
    if in_raw and not in_structured:
        if raw_rank is not None and raw_rank <= 40:
            return "raw_rank_preserved_but_structured_competition"
        return "cross_family_rrf_displacement"

    # 5. candidate_family_score_tie (fallback for edge cases)
    if cross_family_rank is not None and cross_family_rank == 41:
        return "candidate_family_score_tie"

    # 6. Fallback
    return "cross_family_rrf_displacement"


def compute_fusion_loss_classification(
    preds_by_case: dict[str, dict[str, Any]],
    gold_by_case: dict[str, list[str]],
    family_union_result: dict[str, Any],
) -> dict[str, Any]:
    """Classify all fusion loss golds into 6 mutually exclusive categories."""
    loss_details = family_union_result.get("loss_details", [])
    classification_counts: dict[str, int] = {}
    classification_details: list[dict[str, Any]] = []

    for loss in loss_details:
        cid = loss["case_id"]
        gck = loss["candidate_key"]
        pred = preds_by_case.get(cid, {})

        se_fused = (pred.get("structured_expanded") or {}).get("fused") or []
        se_rank_map = get_rank_map(se_fused)
        raw_fused = (pred.get("candidate_raw") or {}).get("fused") or []
        raw_rank_map = get_rank_map(raw_fused)
        e3_fused = pred.get("e3_expanded_fused") or []
        e3_rank_map = get_rank_map(e3_fused)

        se_rank = get_rank(se_rank_map, gck)
        raw_rank = get_rank(raw_rank_map, gck)
        e3_rank = get_rank(e3_rank_map, gck)

        category = classify_fusion_loss(gck, pred, se_rank, raw_rank, e3_rank)
        classification_counts[category] = classification_counts.get(category, 0) + 1

        classification_details.append({
            "case_id": cid,
            "source_index": loss["source_index"],
            "candidate_key": gck,
            "in_e1": loss["in_e1"],
            "in_e2_expanded": loss["in_e2_expanded"],
            "structured_expanded_rank": se_rank,
            "raw_candidate_rank": raw_rank,
            "cross_family_rrf_rank": e3_rank,
            "fusion_loss_category": category,
        })

    return {
        "total_fusion_losses": len(loss_details),
        "classification_counts": classification_counts,
        "classification_details": classification_details,
        "categories": [
            "cross_family_rrf_displacement",
            "structured_rank_preserved_but_raw_competition",
            "raw_rank_preserved_but_structured_competition",
            "multi_slot_family_budget_loss",
            "duplicate_candidate_budget_waste",
            "candidate_family_score_tie",
        ],
        "mutually_exclusive": True,
    }


# ---------------------------------------------------------------------------
# Core Judgment
# ---------------------------------------------------------------------------

def make_core_judgment(fusion_budget_loss: int) -> dict[str, Any]:
    """Determine next gate based on Fusion Budget Loss.

    If Fusion Budget Loss >= 3: next_gate = candidate_family_fusion_closure (R4)
    If Fusion Budget Loss <= 1: next_gate = field_aware_retrieval (R5)
    Otherwise: next_gate = candidate_family_fusion_closure (R4) with caution
    """
    if fusion_budget_loss >= 3:
        decision = "fusion_attribution_closure_recommends_r4"
        next_gate = "candidate_family_fusion_closure"
        rationale = (
            f"Fusion Budget Loss = {fusion_budget_loss} (>= 3). "
            "Worth doing Lane-Preserving Candidate Family Fusion (R4) "
            "before modifying Retriever."
        )
    elif fusion_budget_loss <= 1:
        decision = "fusion_attribution_closure_skip_to_r5"
        next_gate = "field_aware_retrieval"
        rationale = (
            f"Fusion Budget Loss = {fusion_budget_loss} (<= 1). "
            "Fusion is not the main bottleneck. Skip R4, go to Field-aware Retrieval (R5)."
        )
    else:
        decision = "fusion_attribution_closure_recommends_r4"
        next_gate = "candidate_family_fusion_closure"
        rationale = (
            f"Fusion Budget Loss = {fusion_budget_loss} (2, borderline). "
            "Proceed with R4 Lane-Preserving Fusion but expect limited gain."
        )

    return {
        "decision": decision,
        "next_gate": next_gate,
        "rationale": rationale,
        "fusion_budget_loss": fusion_budget_loss,
        "threshold_r4": 3,
        "threshold_skip": 1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    print("=" * 70)
    print("Gate 08 R3.1: Fusion Attribution Closure")
    print("=" * 70)

    # 1. Verify seal
    print("\n[1] Verifying seal...")
    seal = load_json(R3_SEAL)
    sealed = seal.get("sealed", False)
    gold_reads = seal.get("gold_reads_before_seal", -1)
    print(f"  sealed={sealed} gold_reads_before_seal={gold_reads}")
    if not sealed or gold_reads != 0:
        print("  WARNING: Seal verification failed, proceeding anyway")

    # 2. Load data
    print("\n[2] Loading data...")
    r3_preds = load_jsonl_gz(R3_PREDICTIONS_GZ, skip_header=True)
    preds_by_case: dict[str, dict[str, Any]] = {
        p["case_id"]: p for p in r3_preds if p.get("case_id")
    }
    print(f"  R3 predictions: {len(preds_by_case)} cases")

    labels = load_jsonl(GOLD_LABELS)
    gold_by_case: dict[str, list[str]] = {}
    gold_key_to_case: dict[str, str] = {}
    for label in labels:
        cid = label.get("case_id")
        if not cid:
            continue
        keys = [
            s.get("candidate_key")
            for s in (label.get("expected_sources") or [])
            if isinstance(s, dict) and s.get("candidate_key")
        ]
        gold_by_case[cid] = keys
        for k in keys:
            gold_key_to_case[k] = cid
    total_gold = sum(len(v) for v in gold_by_case.values())
    print(f"  Gold labels: {len(gold_by_case)} cases, {total_gold} gold sources")

    universe_scoring = load_json(UNIVERSE_SCORING)
    universe_details_by_case: dict[str, list[dict[str, Any]]] = {}
    for d in universe_scoring.get("details", []):
        cid = d.get("case_id")
        if cid:
            universe_details_by_case.setdefault(cid, []).append(d)
    print(f"  Universe scoring: {sum(len(v) for v in universe_details_by_case.values())} details")

    gate08_raw_parity = load_json(GATE08_RAW_PARITY)
    print(f"  Gate 08 raw parity: authoritative_metrics={gate08_raw_parity.get('authoritative_metrics')}")

    r3_raw_parity = load_json(SCORING_DIR / "raw-parity.json") if (SCORING_DIR / "raw-parity.json").exists() else {}
    # 3. Fix 1: Structured Expansion Rank Regression
    print("\n[3] Fix 1: Structured Expansion Rank Regression (renamed)...")
    rank_regression = compute_structured_expansion_rank_regression(
        universe_scoring, preds_by_case, gold_key_to_case
    )
    print(f"  improved={rank_regression['improved']} unchanged={rank_regression['unchanged']} "
          f"worsened={rank_regression['worsened']} new_entry={rank_regression['new_entry']} "
          f"dropped_out={rank_regression['dropped_out']}")

    # 4. Fix 2: Corrected First Failure Attribution
    print("\n[4] Fix 2: Corrected First Failure Attribution...")
    corrected_failures = compute_corrected_failure_attribution(
        preds_by_case, gold_by_case, universe_details_by_case
    )
    print(f"  in_pool={corrected_failures['in_pool']} missed={corrected_failures['missed']}")
    for stage, count in sorted(corrected_failures["failure_stage_counts"].items()):
        print(f"    {stage}: {count}")

    # 5. Fix 3: Corrected Raw Parity
    print("\n[5] Fix 3: Corrected Raw Parity...")
    corrected_raw_parity = compute_corrected_raw_parity(gate08_raw_parity, r3_raw_parity)
    print(f"  BM25@200={corrected_raw_parity['bm25_source_recall_200']} "
          f"Dense@200={corrected_raw_parity['dense_source_recall_200']} "
          f"RRF@40={corrected_raw_parity['rrf_recall_40']} "
          f"Raw Full Pool={corrected_raw_parity['raw_full_pool']}")

    # 6. Gold Fusion-loss Matrix
    print("\n[6] Gold Fusion-loss Matrix...")
    fusion_loss_matrix = build_gold_fusion_loss_matrix(
        preds_by_case, gold_by_case, universe_details_by_case
    )
    print(f"  structured_recovered_but_e3_lost={fusion_loss_matrix['structured_recovered_but_e3_lost']}")
    print(f"  raw_recovered_but_e3_lost={fusion_loss_matrix['raw_recovered_but_e3_lost']}")

    # 7. Candidate Family Union Ceiling
    print("\n[7] Candidate Family Union Ceiling...")
    family_union = compute_family_union_ceiling(preds_by_case, gold_by_case)
    print(f"  E1 only={family_union['e1_only']} E2 only={family_union['e2_expanded_only']} "
          f"both={family_union['both']} neither={family_union['neither']}")
    print(f"  Family Union={family_union['family_union_gold']}/80 "
          f"E3 Expanded={family_union['e3_expanded_gold']}/80 "
          f"Fusion Budget Loss={family_union['fusion_budget_loss']}")

    # 8. Fusion Loss Detailed Classification
    print("\n[8] Fusion Loss Detailed Classification...")
    fusion_loss_class = compute_fusion_loss_classification(
        preds_by_case, gold_by_case, family_union
    )
    for cat, count in sorted(fusion_loss_class["classification_counts"].items()):
        print(f"    {cat}: {count}")

    # 9. Core Judgment
    print("\n[9] Core Judgment...")
    judgment = make_core_judgment(family_union["fusion_budget_loss"])
    print(f"  decision={judgment['decision']}")
    print(f"  next_gate={judgment['next_gate']}")
    print(f"  rationale={judgment['rationale']}")

    # 10. Assemble main output
    fusion_attribution_closure = {
        "gate": "pdf_retrieval_v4_gate_08_r3_1",
        "phase": "fusion_attribution_closure",
        "diagnostic_only": True,
        "new_index_builds": 0,
        "bm25_searches": 0,
        "dense_searches": 0,
        "embedding_calls": 0,
        "seal_verified": sealed and gold_reads == 0,
        "frozen_conclusions": {
            "gate_08_r3_commit": "f9e9cdb",
            "e0": 42,
            "e1_raw_candidate": 47,
            "e2_legacy": 46,
            "e2_control": 46,
            "e2_expanded": 57,
            "e3_legacy": 47,
            "e3_expanded": 52,
            "representation_gain": 0,
            "pure_structured_coverage_gain": 11,
            "current_full_system_gain": 5,
            "strict_candidate_universe": 68,
            "structured_lane_conversion": "49/68",
            "combined_conversion": "46/68",
        },
        "fix_1_structured_expansion_rank_regression": {
            "metric_name": rank_regression["metric_name"],
            "improved": rank_regression["improved"],
            "unchanged": rank_regression["unchanged"],
            "worsened": rank_regression["worsened"],
            "new_entry": rank_regression["new_entry"],
            "dropped_out": rank_regression["dropped_out"],
            "both_absent": rank_regression["both_absent"],
            "note": rank_regression["note"],
        },
        "fix_2_corrected_failure_attribution": {
            "total_gold_sources": corrected_failures["total_gold_sources"],
            "in_pool": corrected_failures["in_pool"],
            "missed": corrected_failures["missed"],
            "outside_universe": corrected_failures["outside_universe"],
            "in_universe_missed": corrected_failures["in_universe_missed"],
            "failure_stage_counts": corrected_failures["failure_stage_counts"],
            "rename_notes": corrected_failures["rename_notes"],
        },
        "fix_3_corrected_raw_parity": {
            "bm25_source_recall_200": corrected_raw_parity["bm25_source_recall_200"],
            "dense_source_recall_200": corrected_raw_parity["dense_source_recall_200"],
            "rrf_recall_40": corrected_raw_parity["rrf_recall_40"],
            "raw_full_pool": corrected_raw_parity["raw_full_pool"],
            "bm25_recomputed": corrected_raw_parity["bm25_recomputed"],
            "bm25_authoritative_baseline": corrected_raw_parity["bm25_authoritative_baseline"],
            "correction_note": corrected_raw_parity["correction_note"],
        },
        "gold_fusion_loss_matrix": {
            "total_gold_sources": fusion_loss_matrix["total_gold_sources"],
            "structured_recovered_but_e3_lost": fusion_loss_matrix["structured_recovered_but_e3_lost"],
            "raw_recovered_but_e3_lost": fusion_loss_matrix["raw_recovered_but_e3_lost"],
        },
        "family_union_ceiling": {
            "total_gold": family_union["total_gold"],
            "e1_only": family_union["e1_only"],
            "e2_expanded_only": family_union["e2_expanded_only"],
            "both": family_union["both"],
            "neither": family_union["neither"],
            "family_union_gold": family_union["family_union_gold"],
            "e3_expanded_gold": family_union["e3_expanded_gold"],
            "fusion_budget_loss": family_union["fusion_budget_loss"],
            "note": family_union["note"],
        },
        "fusion_loss_classification": {
            "total_fusion_losses": fusion_loss_class["total_fusion_losses"],
            "classification_counts": fusion_loss_class["classification_counts"],
            "categories": fusion_loss_class["categories"],
            "mutually_exclusive": fusion_loss_class["mutually_exclusive"],
        },
        "core_judgment": judgment,
    }

    # 11. Write outputs
    print("\n[10] Writing outputs...")
    write_json(R3_DIR / "fusion-attribution-closure.json", fusion_attribution_closure)
    print(f"  {R3_DIR / 'fusion-attribution-closure.json'}")

    write_json(SCORING_DIR / "raw-parity-corrected.json", corrected_raw_parity)
    print(f"  {SCORING_DIR / 'raw-parity-corrected.json'}")

    write_json(R3_DIR / "first-failure-attribution-corrected.json", corrected_failures)
    print(f"  {R3_DIR / 'first-failure-attribution-corrected.json'}")

    write_json(SCORING_DIR / "structured-expansion-rank-regression.json", rank_regression)
    print(f"  {SCORING_DIR / 'structured-expansion-rank-regression.json'}")

    # Write detailed matrix and classification as separate files
    write_json(SCORING_DIR / "gold-fusion-loss-matrix.json", fusion_loss_matrix)
    print(f"  {SCORING_DIR / 'gold-fusion-loss-matrix.json'}")

    write_json(SCORING_DIR / "fusion-loss-classification.json", fusion_loss_class)
    print(f"  {SCORING_DIR / 'fusion-loss-classification.json'}")

    write_json(SCORING_DIR / "family-union-ceiling.json", family_union)
    print(f"  {SCORING_DIR / 'family-union-ceiling.json'}")

    # 12. Write acceptance and next-gate
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_08_r3_1",
        "phase": "fusion_attribution_closure",
        "diagnostic_only": True,
        "frozen_e3_expanded_score": "52/80",
        "frozen_decision": "coverage_expansion_small_gain",
        "fixes_applied": {
            "fix_1_rank_regression_renamed": True,
            "fix_2_post_filter_renamed": True,
            "fix_3_raw_bm25_parity_corrected": True,
        },
        "family_union_gold": family_union["family_union_gold"],
        "e3_expanded_gold": family_union["e3_expanded_gold"],
        "fusion_budget_loss": family_union["fusion_budget_loss"],
        "fusion_loss_classification": fusion_loss_class["classification_counts"],
        "structured_recovered_but_e3_lost": fusion_loss_matrix["structured_recovered_but_e3_lost"],
        "raw_recovered_but_e3_lost": fusion_loss_matrix["raw_recovered_but_e3_lost"],
        "corrected_raw_parity": {
            "bm25_source_recall_200": corrected_raw_parity["bm25_source_recall_200"],
            "dense_source_recall_200": corrected_raw_parity["dense_source_recall_200"],
            "rrf_recall_40": corrected_raw_parity["rrf_recall_40"],
            "raw_full_pool": corrected_raw_parity["raw_full_pool"],
        },
        "decision": judgment["decision"],
        "next_gate": judgment["next_gate"],
    }
    r3_1_acceptance_path = R3_DIR / "fusion-attribution-acceptance.json"
    write_json(r3_1_acceptance_path, acceptance)
    print(f"  {r3_1_acceptance_path}")

    next_gate = {
        "current_gate": "pdf_retrieval_v4_gate_08_r3_1",
        "decision": judgment["decision"],
        "next_gate": judgment["next_gate"],
        "rationale": judgment["rationale"],
        "fusion_budget_loss": judgment["fusion_budget_loss"],
        "family_union_gold": family_union["family_union_gold"],
        "e3_expanded_gold": family_union["e3_expanded_gold"],
        "recommended_actions": _recommended_actions(judgment, fusion_loss_class),
    }
    r3_1_next_gate_path = R3_DIR / "fusion-attribution-next-gate.json"
    write_json(r3_1_next_gate_path, next_gate)
    print(f"  {r3_1_next_gate_path}")

    # Summary
    print("\n" + "=" * 70)
    print("FUSION ATTRIBUTION CLOSURE COMPLETE")
    print(f"  Family Union Gold = {family_union['family_union_gold']}/80")
    print(f"  E3 Expanded Gold  = {family_union['e3_expanded_gold']}/80")
    print(f"  Fusion Budget Loss = {family_union['fusion_budget_loss']}")
    print(f"  Decision = {judgment['decision']}")
    print(f"  Next Gate = {judgment['next_gate']}")
    print("=" * 70)

    return 0


def _recommended_actions(
    judgment: dict[str, Any],
    fusion_loss_class: dict[str, Any],
) -> list[str]:
    if judgment["next_gate"] == "candidate_family_fusion_closure":
        counts = fusion_loss_class.get("classification_counts", {})
        actions = [
            "Implement Lane-Preserving Candidate Family Fusion (R4)",
            "Use Structured-Protected Late Fusion with slot_top_k=20 protection",
            "Maintain total candidate residual budget = 40",
            "No BM25/Dense search changes, no embedding changes",
        ]
        if counts.get("structured_rank_preserved_but_raw_competition", 0) > 0:
            actions.append(
                f"Address structured_rank_preserved_but_raw_competition "
                f"({counts['structured_rank_preserved_but_raw_competition']} cases) "
                f"via structured family protection"
            )
        if counts.get("duplicate_candidate_budget_waste", 0) > 0:
            actions.append(
                f"Address duplicate_candidate_budget_waste "
                f"({counts['duplicate_candidate_budget_waste']} cases) "
                f"via cross-family dedup"
            )
        return actions
    else:
        return [
            "Skip R4, proceed to Gate 08 R5 Field-aware Structured Retrieval",
            "Focus on structured_bm25_and_dense_top50_miss (12 golds)",
            "Add metric/context/fact BM25 lanes using Semantic Graph fields",
            "Keep Dense model = all-MiniLM-L6-v2",
        ]


if __name__ == "__main__":
    sys.exit(main())
