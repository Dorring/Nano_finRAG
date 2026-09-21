#!/usr/bin/env python3
"""Gate 08 R3-C: Strict Gold Scoring for all experiment groups."""

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

# ---------------------------------------------------------------------------
# Paths (all relative to backend ROOT)
# ---------------------------------------------------------------------------
R3_PREDICTIONS_GZ = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3/predictions.jsonl.gz"
)
R3_SEAL = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3/prediction-seal.json"
GOLD_LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
UNIVERSE_SCORING = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5/universe-scoring.json"
)
GATE08_RAW_PARITY = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08/raw-parity.json"
)
GATE08_PREDICTIONS_GZ = (
    ROOT
    / "artifacts/evaluation/pdf-retrieval-v4-gate-08/retrieval-predictions.jsonl.gz"
)
R2_SCORING = (
    ROOT
    / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2/scoring/scoring-report.json"
)
OUTPUT_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3/scoring"

# Experiment group -> pool field name in R3 predictions
EXPERIMENT_GROUPS: list[tuple[str, str]] = [
    ("e0", "e0_pool"),
    ("e1", "e1_pool"),
    ("e2_legacy", "e2_legacy_pool"),
    ("e2_control", "e2_control_pool"),
    ("e2_expanded", "e2_expanded_pool"),
    ("e3_legacy", "e3_legacy_pool"),
    ("e3_control", "e3_control_pool"),
    ("e3_expanded", "e3_expanded_pool"),
]

# Baselines for delta computation
FULL_SYSTEM_BASELINE = 47  # from R2 combined_strict_recall
RAW_PROTECTED_BASELINE = 42  # raw-protected lane baseline


# ---------------------------------------------------------------------------
# IO helpers
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


def to_key_set(
    items: Iterable[Any] | None,
    key_fields: tuple[str, ...] = ("candidate_key", "candidate_identity", "id"),
) -> set[str]:
    """Normalize an iterable of strings or dicts into a set of keys."""
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
    """Build candidate_key -> 1-based rank from a fused RRF hit list."""
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


def parse_fraction_num(value: Any) -> int | None:
    """Parse '47/80' -> 47. Accepts int directly."""
    if isinstance(value, int):
        return value
    if not value or not isinstance(value, str):
        return None
    head = value.split("/")[0].strip()
    try:
        return int(head)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Seal verification (MUST happen before reading gold labels)
# ---------------------------------------------------------------------------
def verify_seal(seal: dict[str, Any]) -> tuple[bool, str]:
    if not seal.get("sealed"):
        return False, "seal.sealed is not true"
    if seal.get("gold_reads_before_seal", 0) != 0:
        return (
            False,
            f"gold_reads_before_seal={seal.get('gold_reads_before_seal')} (expected 0)",
        )
    if seal.get("governance_reads_before_seal", 0) != 0:
        return (
            False,
            f"governance_reads_before_seal={seal.get('governance_reads_before_seal')} (expected 0)",
        )
    return True, "ok"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_group(
    pool_field: str,
    preds_by_case: dict[str, dict[str, Any]],
    gold_by_case: dict[str, list[str]],
    universe_set: set[str],
    outside_set: set[str],
) -> dict[str, Any]:
    total_hits = 0
    universe_hits = 0
    outside_hits = 0
    total_gold = 0
    universe_total = 0
    outside_total = 0
    hit_keys: list[str] = []
    missed_keys: list[str] = []

    for cid, gold_keys in gold_by_case.items():
        pred = preds_by_case.get(cid)
        pool_keys: set[str] = set()
        if pred is not None:
            pool_keys = to_key_set(pred.get(pool_field))

        for idx, k in enumerate(gold_keys):
            total_gold += 1
            in_uni = (cid, idx) in universe_set
            in_out = (cid, idx) in outside_set
            if in_uni:
                universe_total += 1
            if in_out:
                outside_total += 1
            if k in pool_keys:
                total_hits += 1
                hit_keys.append(k)
                if in_uni:
                    universe_hits += 1
                if in_out:
                    outside_hits += 1
            else:
                missed_keys.append(k)

    return {
        "total_gold": total_gold,
        "total_hits": total_hits,
        "total_recall": f"{total_hits}/{total_gold}",
        "universe_total": universe_total,
        "universe_hits": universe_hits,
        "universe_recall": f"{universe_hits}/{universe_total}",
        "outside_total": outside_total,
        "outside_hits": outside_hits,
        "outside_recall": f"{outside_hits}/{outside_total}",
        "hit_keys": sorted(hit_keys),
        "missed_keys": sorted(missed_keys),
    }


def compute_raw_parity(
    raw_cases: list[dict[str, Any]],
    gold_by_case: dict[str, list[str]],
) -> dict[str, Any]:
    bm25_hits = 0
    rrf40_hits = 0
    raw_full_hits = 0
    total_gold = 0
    dense_hits = None  # may not be directly available

    per_case: list[dict[str, Any]] = []
    for rc in raw_cases:
        cid = rc.get("case_id")
        gold_keys = gold_by_case.get(cid, [])
        if not gold_keys:
            continue

        rrf_cands = rc.get("raw_full_rrf_candidates") or []
        rrf_set = to_key_set(rrf_cands)
        rrf_top40_set = to_key_set(rrf_cands[:40])

        bm25_cands = (
            rc.get("raw_candidate_ids")
            or rc.get("bm25_candidate_ids")
            or rc.get("raw_bm25_candidates")
            or []
        )
        bm25_set = to_key_set(bm25_cands)

        dense_cands = (
            rc.get("raw_dense_candidate_ids")
            or rc.get("dense_candidate_ids")
            or rc.get("raw_dense_candidates")
            or []
        )
        dense_available = bool(dense_cands)
        dense_set = to_key_set(dense_cands)

        case_bm25 = 0
        case_rrf40 = 0
        case_raw_full = 0
        case_dense = 0
        for k in gold_keys:
            total_gold += 1
            if k in bm25_set:
                bm25_hits += 1
                case_bm25 += 1
            if k in rrf_top40_set:
                rrf40_hits += 1
                case_rrf40 += 1
            if k in rrf_set:
                raw_full_hits += 1
                case_raw_full += 1
            if dense_available and k in dense_set:
                case_dense += 1

        per_case.append(
            {
                "case_id": cid,
                "gold_count": len(gold_keys),
                "bm25_hits": case_bm25,
                "rrf_top40_hits": case_rrf40,
                "raw_full_hits": case_raw_full,
                "dense_hits": case_dense if dense_available else None,
            }
        )

    result: dict[str, Any] = {
        "total_gold": total_gold,
        "bm25_source_recall_200": f"{bm25_hits}/{total_gold}",
        "bm25_source_recall_200_hits": bm25_hits,
        "rrf_recall_40": f"{rrf40_hits}/{total_gold}",
        "rrf_recall_40_hits": rrf40_hits,
        "raw_full_pool": f"{raw_full_hits}/{total_gold}",
        "raw_full_pool_hits": raw_full_hits,
        "dense_source_recall_200": None,
        "per_case": per_case,
    }
    if dense_hits is not None:
        result["dense_source_recall_200"] = f"{dense_hits}/{total_gold}"
        result["dense_source_recall_200_hits"] = dense_hits
    return result


def compute_structured_lane_recall(
    preds_by_case: dict[str, dict[str, Any]],
    gold_by_case: dict[str, list[str]],
    universe_set: set[str],
    gate08_structured_identities: set[str],
) -> dict[str, Any]:
    fused_hits = 0
    combined_hits = 0
    gate08_hits = 0
    universe_total = 0

    fused_hit_keys: list[str] = []
    combined_hit_keys: list[str] = []
    gate08_hit_keys: list[str] = []

    for cid, gold_keys in gold_by_case.items():
        pred = preds_by_case.get(cid, {})

        structured_expanded = pred.get("structured_expanded") or {}
        fused = structured_expanded.get("fused") or []
        fused_keys = to_key_set(fused)

        e3_pool_keys = to_key_set(pred.get("e3_expanded_pool"))

        for idx, k in enumerate(gold_keys):
            if (cid, idx) not in universe_set:
                continue
            universe_total += 1
            if k in fused_keys:
                fused_hits += 1
                fused_hit_keys.append(k)
            if k in e3_pool_keys:
                combined_hits += 1
                combined_hit_keys.append(k)
            if k in gate08_structured_identities:
                gate08_hits += 1
                gate08_hit_keys.append(k)

    return {
        "universe_total": universe_total,
        "grade_a_gold_in_structured_expanded_fused": fused_hits,
        "structured_lane_recall": f"{fused_hits}/{universe_total}",
        "grade_a_gold_in_e3_expanded_pool": combined_hits,
        "combined_conversion": f"{combined_hits}/{universe_total}",
        "grade_a_gold_in_gate08_structured_pool": gate08_hits,
        "gate08_structured_lane_recall": f"{gate08_hits}/{universe_total}",
        "fused_hit_keys": sorted(fused_hit_keys),
        "combined_hit_keys": sorted(combined_hit_keys),
        "gate08_hit_keys": sorted(gate08_hit_keys),
    }


def analyze_newly_bridged(
    universe_scoring: dict[str, Any],
    preds_by_case: dict[str, dict[str, Any]],
    gold_key_to_case: dict[str, str],
) -> dict[str, Any]:
    newly_bridged: list[tuple[str, str | None]] = []
    for d in universe_scoring.get("details", []):
        gck = d.get("gold_candidate_key")
        if not gck:
            continue
        if d.get("new_status") == "mapped" and not d.get(
            "was_in_structured_universe", False
        ):
            _cid = d.get("case_id") or gold_key_to_case.get(gck)
            newly_bridged.append((gck, _cid))

    in_top10 = 0
    in_top20 = 0
    in_top40 = 0
    in_top50 = 0
    in_e3_expanded = 0
    details: list[dict[str, Any]] = []

    for gck, cid in newly_bridged:
        pred = preds_by_case.get(cid, {}) if cid else {}
        fused = (pred.get("structured_expanded") or {}).get("fused") or []
        rank_map = get_rank_map(fused)
        rank = rank_map.get(gck)
        pool_keys = to_key_set(pred.get("e3_expanded_pool"))
        in_pool = gck in pool_keys

        b10 = rank is not None and rank <= 10
        b20 = rank is not None and rank <= 20
        b40 = rank is not None and rank <= 40
        b50 = rank is not None and rank <= 50

        if b10:
            in_top10 += 1
        if b20:
            in_top20 += 1
        if b40:
            in_top40 += 1
        if b50:
            in_top50 += 1
        if in_pool:
            in_e3_expanded += 1

        details.append(
            {
                "gold_candidate_key": gck,
                "case_id": cid,
                "rank_in_structured_expanded": rank,
                "in_top10": b10,
                "in_top20": b20,
                "in_top40": b40,
                "in_top50": b50,
                "in_e3_expanded_pool": in_pool,
            }
        )

    return {
        "newly_bridged_total": len(newly_bridged),
        "in_structured_expanded_top10": in_top10,
        "in_structured_expanded_top20": in_top20,
        "in_structured_expanded_top40": in_top40,
        "in_structured_expanded_top50": in_top50,
        "in_e3_expanded_pool": in_e3_expanded,
        "details": details,
    }


def analyze_rank_regression(
    universe_scoring: dict[str, Any],
    preds_by_case: dict[str, dict[str, Any]],
    gold_key_to_case: dict[str, str],
) -> dict[str, Any]:
    old_structured: list[tuple[str, str | None]] = []
    for d in universe_scoring.get("details", []):
        gck = d.get("gold_candidate_key")
        if not gck:
            continue
        if d.get("was_in_structured_universe", False):
            _cid = d.get("case_id") or gold_key_to_case.get(gck)
            old_structured.append((gck, _cid))

    improved = 0
    unchanged = 0
    worsened = 0
    new_entry = 0
    dropped_out = 0
    both_absent = 0
    dropped_out_top40 = 0
    dropped_out_top50 = 0
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

        if (
            old_rank is not None
            and old_rank <= 40
            and (new_rank is None or new_rank > 40)
        ):
            dropped_out_top40 += 1
        if (
            old_rank is not None
            and old_rank <= 50
            and (new_rank is None or new_rank > 50)
        ):
            dropped_out_top50 += 1

        details.append(
            {
                "gold_candidate_key": gck,
                "case_id": cid,
                "old_rank_in_structured_legacy": old_rank,
                "new_rank_in_structured_expanded": new_rank,
                "category": cat,
            }
        )

    return {
        "old_structured_total": len(old_structured),
        "improved": improved,
        "unchanged": unchanged,
        "worsened": worsened,
        "new_entry": new_entry,
        "dropped_out": dropped_out,
        "both_absent": both_absent,
        "dropped_out_top40": dropped_out_top40,
        "dropped_out_top50": dropped_out_top50,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate 08 R3-C strict gold scoring (ablation)."
    )
    parser.add_argument(
        "--strict-seal", action="store_true", help="Exit non-zero on seal failure."
    )
    args = parser.parse_args()

    # 1. Verify seal BEFORE reading gold labels.
    print(f"[seal] loading {R3_SEAL}")
    seal = load_json(R3_SEAL)
    ok, msg = verify_seal(seal)
    print(f"[seal] verified={ok} ({msg})")
    if not ok:
        if args.strict_seal:
            return 2
        print(
            "[seal] WARNING: proceeding despite seal issue (use --strict-seal to enforce)"
        )

    # 2. Load data sources.
    print(f"[load] R3 predictions: {R3_PREDICTIONS_GZ}")
    r3_preds = load_jsonl_gz(R3_PREDICTIONS_GZ, skip_header=True)
    preds_by_case: dict[str, dict[str, Any]] = {
        p["case_id"]: p for p in r3_preds if p.get("case_id")
    }
    print(f"[load] R3 predictions: {len(preds_by_case)} cases")

    print(f"[load] gold labels: {GOLD_LABELS}")
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
    print(f"[load] gold labels: {len(gold_by_case)} cases, {total_gold} gold sources")

    print(f"[load] universe scoring: {UNIVERSE_SCORING}")
    universe_scoring = load_json(UNIVERSE_SCORING)

    print(f"[load] gate08 raw parity: {GATE08_RAW_PARITY}")
    raw_parity_data = load_json(GATE08_RAW_PARITY)
    raw_cases = (
        raw_parity_data.get("raw_cases", [])
        if isinstance(raw_parity_data, dict)
        else []
    )

    print(f"[load] gate08 predictions: {GATE08_PREDICTIONS_GZ}")
    gate08_preds = load_jsonl_gz(GATE08_PREDICTIONS_GZ, skip_header=True)

    print(f"[load] R2 scoring: {R2_SCORING}")
    r2_scoring = load_json(R2_SCORING) if R2_SCORING.exists() else {}

    # 3. Build universe / outside sets keyed by (case_id, index).
    # The universe-scoring details align positionally with each case's
    # expected_sources, so classify each gold SOURCE by (case_id, index).
    # This avoids collapsing duplicate candidate_keys (some gold sources share
    # the same candidate_key across cases/details) so reported counts reflect
    # gold-source counts (68 mapped / 12 outside) instead of unique-key counts.
    universe_set: set[tuple[str, int]] = set()
    outside_set: set[tuple[str, int]] = set()
    was_in_structured_map: dict[tuple[str, int], bool] = {}
    _uni_details_by_case: dict[str, list[dict[str, Any]]] = {}
    for d in universe_scoring.get("details", []):
        _cid = d.get("case_id")
        if _cid is None:
            continue
        _uni_details_by_case.setdefault(_cid, []).append(d)
    for _cid, _dlist in _uni_details_by_case.items():
        for _idx, d in enumerate(_dlist):
            _ci = (_cid, _idx)
            was_in_structured_map[_ci] = bool(
                d.get("was_in_structured_universe", False)
            )
            if d.get("new_status") == "mapped":
                universe_set.add(_ci)
            else:
                outside_set.add(_ci)
    print(f"[universe] mapped={len(universe_set)} outside={len(outside_set)}")

    # 4. Score each experiment group.
    groups: dict[str, dict[str, Any]] = {}
    for name, pool_field in EXPERIMENT_GROUPS:
        groups[name] = score_group(
            pool_field, preds_by_case, gold_by_case, universe_set, outside_set
        )
        g = groups[name]
        print(
            f"[group] {name:12s} total={g['total_recall']} universe={g['universe_recall']} outside={g['outside_recall']}"
        )

    # 5. Compute deltas.
    e2_legacy_hits = groups["e2_legacy"]["total_hits"]
    e2_control_hits = groups["e2_control"]["total_hits"]
    e2_expanded_hits = groups["e2_expanded"]["total_hits"]
    e3_expanded_hits = groups["e3_expanded"]["total_hits"]
    e3_legacy_hits = groups["e3_legacy"]["total_hits"]

    full_system_baseline = (
        parse_fraction_num(r2_scoring.get("combined_strict_recall"))
        or FULL_SYSTEM_BASELINE
    )
    raw_protected_baseline = (
        parse_fraction_num(
            r2_scoring.get("raw_protected_recall")
            or r2_scoring.get("raw_strict_recall")
            or r2_scoring.get("e3_legacy_recall")
        )
        or RAW_PROTECTED_BASELINE
    )

    deltas = {
        "representation_gain": e2_control_hits - e2_legacy_hits,
        "pure_coverage_gain": e2_expanded_hits - e2_control_hits,
        "full_system_gain": e3_expanded_hits - full_system_baseline,
        "raw_protected_gain": e3_expanded_hits - raw_protected_baseline,
        "baselines": {
            "full_system_baseline": full_system_baseline,
            "full_system_baseline_source": "r2.combined_strict_recall",
            "raw_protected_baseline": raw_protected_baseline,
            "raw_protected_baseline_source": "constant-or-r2",
            "e3_legacy_hits_for_reference": e3_legacy_hits,
            "r2_combined_strict_recall_raw": r2_scoring.get("combined_strict_recall"),
        },
    }
    print(
        f"[deltas] rep={deltas['representation_gain']} "
        f"pure_cov={deltas['pure_coverage_gain']} "
        f"full_sys={deltas['full_system_gain']} "
        f"raw_prot={deltas['raw_protected_gain']}"
    )

    # 6. Raw production parity.
    raw_parity_report = compute_raw_parity(raw_cases, gold_by_case)
    print(
        f"[raw-parity] bm25={raw_parity_report['bm25_source_recall_200']} "
        f"rrf40={raw_parity_report['rrf_recall_40']} "
        f"raw_full={raw_parity_report['raw_full_pool']}"
    )

    # 7. Build gate08 structured identity set & structured-lane conversion.
    gate08_structured_identities: set[str] = set()
    for rec in gate08_preds:
        pool = rec.get("structured_strict_source_pool") or []
        for item in pool:
            if isinstance(item, dict):
                oid = item.get("original_candidate_identity") or item.get(
                    "candidate_key"
                )
                if oid:
                    gate08_structured_identities.add(oid)
            elif isinstance(item, str) and item:
                gate08_structured_identities.add(item)
    print(f"[gate08] structured identities: {len(gate08_structured_identities)}")

    structured_lane_report = compute_structured_lane_recall(
        preds_by_case, gold_by_case, universe_set, gate08_structured_identities
    )
    print(
        f"[struct-lane] fused={structured_lane_report['structured_lane_recall']} "
        f"combined={structured_lane_report['combined_conversion']} "
        f"gate08={structured_lane_report['gate08_structured_lane_recall']}"
    )

    # 8. Newly bridged 13 gold analysis.
    newly_bridged_report = analyze_newly_bridged(
        universe_scoring, preds_by_case, gold_key_to_case
    )
    print(
        f"[new-bridge] total={newly_bridged_report['newly_bridged_total']} "
        f"top40={newly_bridged_report['in_structured_expanded_top40']} "
        f"e3_pool={newly_bridged_report['in_e3_expanded_pool']}"
    )

    # 9. Old structured rank regression.
    rank_regression_report = analyze_rank_regression(
        universe_scoring, preds_by_case, gold_key_to_case
    )
    print(
        f"[rank-reg] improved={rank_regression_report['improved']} "
        f"unchanged={rank_regression_report['unchanged']} "
        f"worsened={rank_regression_report['worsened']} "
        f"dropped_top40={rank_regression_report['dropped_out_top40']}"
    )

    # 10. Assemble main ablation report.
    ablation_report = {
        "run_id": "pdf-retrieval-v4-gate-08-r3",
        "seal": {
            "verified": ok,
            "message": msg,
            "sealed": seal.get("sealed"),
            "gold_reads_before_seal": seal.get("gold_reads_before_seal"),
            "governance_reads_before_seal": seal.get("governance_reads_before_seal"),
        },
        "counts": {
            "total_gold": total_gold,
            "universe_mapped": len(universe_set),
            "outside_universe": len(outside_set),
            "cases": len(gold_by_case),
        },
        "experiment_groups": {
            name: {k: v for k, v in g.items()} for name, g in groups.items()
        },
        "deltas": deltas,
    }

    # 11. Write output files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "ablation-metrics.json", ablation_report)
    write_json(OUTPUT_DIR / "raw-parity.json", raw_parity_report)
    write_json(OUTPUT_DIR / "structured-lane-recall.json", structured_lane_report)
    write_json(
        OUTPUT_DIR / "structured-universe-conversion.json",
        {
            "universe_mapped": len(universe_set),
            "universe_total_expected": universe_scoring.get("mapped"),
            "outside_universe": len(outside_set),
            "outside_total_expected": universe_scoring.get("total_gold", 0)
            - universe_scoring.get("mapped", 0)
            if isinstance(universe_scoring.get("total_gold"), int)
            else len(outside_set),
            "structured_lane_recall": structured_lane_report["structured_lane_recall"],
            "combined_conversion": structured_lane_report["combined_conversion"],
            "gate08_structured_lane_recall": structured_lane_report[
                "gate08_structured_lane_recall"
            ],
            "newly_bridged_total": newly_bridged_report["newly_bridged_total"],
            "old_structured_total": rank_regression_report["old_structured_total"],
        },
    )
    write_json(OUTPUT_DIR / "newly-bridged-13-gold.json", newly_bridged_report)
    write_json(
        OUTPUT_DIR / "old-structured-rank-regression.json", rank_regression_report
    )

    # 12. Summary.
    print("\n" + "=" * 72)
    print("Gate 08 R3-C Ablation Scoring Summary")
    print("=" * 72)
    for name, _ in EXPERIMENT_GROUPS:
        g = groups[name]
        print(
            f"  {name:12s} total={g['total_recall']:>8}  universe={g['universe_recall']:>8}  outside={g['outside_recall']:>6}"
        )
    print("-" * 72)
    print(
        f"  Representation Gain (E2-Control - E2-Legacy)   : {deltas['representation_gain']}"
    )
    print(
        f"  Pure Coverage Gain   (E2-Expanded - E2-Control): {deltas['pure_coverage_gain']}"
    )
    print(
        f"  Full System Gain     (E3-Expanded - {full_system_baseline})          : {deltas['full_system_gain']}"
    )
    print(
        f"  Raw-protected Gain   (E3-Expanded - {raw_protected_baseline})          : {deltas['raw_protected_gain']}"
    )
    print("-" * 72)
    print(f"  BM25 Source Recall@200 : {raw_parity_report['bm25_source_recall_200']}")
    print(f"  RRF Recall@40          : {raw_parity_report['rrf_recall_40']}")
    print(f"  Raw Full Pool          : {raw_parity_report['raw_full_pool']}")
    print("-" * 72)
    print(
        f"  Structured-lane recall (fused)  : {structured_lane_report['structured_lane_recall']}"
    )
    print(
        f"  Combined conversion (E3-Expanded): {structured_lane_report['combined_conversion']}"
    )
    print(
        f"  Newly bridged gold (top40/e3pool): {newly_bridged_report['in_structured_expanded_top40']}/{newly_bridged_report['in_e3_expanded_pool']} of {newly_bridged_report['newly_bridged_total']}"
    )
    print(
        f"  Rank regression (imp/unch/worse): {rank_regression_report['improved']}/{rank_regression_report['unchanged']}/{rank_regression_report['worsened']}"
    )
    print("=" * 72)
    print(f"[done] outputs written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
