"""NF-OPT-22 R0: confidence-gated Qwen/BM25 late-fusion shadow audit.

The prediction path is deliberately Gold-blind.  It reads the sealed Qwen
Top100, the sealed BM25 ranks, and the executable near-boundary contract from
NF-OPT-19.  Only after the CGLRRF prediction is hashed and sealed are the
strict/semantic diagnostic artifacts loaded.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


BASE_COMMIT = "2b34d2b27df734824cb8c2ac124d14ff37c5ee1f"
OUT_NAME = "nf-opt-22-r0-confidence-gated-late-fusion"
RRF_K = 60
K_VALUES = (1, 3, 5, 10, 20, 50, 100)
EXPECTED_TOP100_SHA = "ced014c357a3c8862a5ae3723a4c618556156542fd63aa2a8fd4bf766b60f01a"
EXPECTED_QWEN_SHA = "26b901607e99a27cf6afd86e278ab2b2216ab740f4537ba7a81974dc108cac22"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return sha256_file(path)


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def top_keys(items: list[dict[str, Any]], k: int) -> list[str]:
    return [item["candidate_key"] for item in items[:k]]


def rank_map(items: list[dict[str, Any]]) -> dict[str, int]:
    return {item["candidate_key"]: index + 1 for index, item in enumerate(items)}


def load_r0_helpers() -> Any:
    # The established NF-OPT-21 module contains only pure metric helpers at
    # import time; its main() is not invoked here.
    import run_nf_opt_21_r0_qwen_bm25_late_fusion as r0

    return r0


def load_frozen_rankings(evaluation_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    qwen_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3" / "main_rerank_predictions.jsonl.gz"
    qwen_seal_path = qwen_path.parent / "prediction-seal.json"
    bounded_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r2a-2" / "bounded-top100-predictions.jsonl.gz"
    bounded_seal_path = bounded_path.parent / "prediction-seal.json"
    qwen_rows = read_gzip_jsonl(qwen_path)
    bounded_rows = read_gzip_jsonl(bounded_path)
    bounded_by_case = {row["case_id"]: row["candidates"] for row in bounded_rows}
    qwen_by_case: dict[str, list[dict[str, Any]]] = {}
    identity_mismatch = 0
    rank_mismatch = 0
    bm25_missing = 0
    score_nonfinite = 0
    for row in qwen_rows:
        items = sorted(row["ranked_candidates"], key=lambda item: int(item["post_rerank_rank"]))
        if len(items) != 100 or [int(item["post_rerank_rank"]) for item in items] != list(range(1, 101)):
            raise RuntimeError(f"invalid frozen Qwen ranks: {row['case_id']}")
        bounded = bounded_by_case.get(row["case_id"], [])
        if set(item["candidate_key"] for item in items) != set(item["candidate_key"] for item in bounded):
            identity_mismatch += 1
        bounded_by_key = {item["candidate_key"]: item for item in bounded}
        prepared: list[dict[str, Any]] = []
        for item in items:
            key = item["candidate_key"]
            bitem = bounded_by_key.get(key)
            qrank = int(item["post_rerank_rank"])
            brank = item.get("pre_rerank_rank")
            brank = int(brank) if brank is not None else None
            if brank is None or bitem is None:
                bm25_missing += 1
            if bitem is not None:
                bounded_rank = bitem.get("rank", bitem.get("final_candidate_rank"))
                if bounded_rank is not None and brank != int(bounded_rank):
                    rank_mismatch += 1
            score = float(item["reranker_score"])
            if not math.isfinite(score):
                score_nonfinite += 1
            clone = dict(item)
            clone["original_qwen_rank"] = qrank
            clone["original_bm25_rank"] = brank
            prepared.append(clone)
        qwen_by_case[row["case_id"]] = prepared
    qwen_sha = sha256_file(qwen_path)
    bounded_sha = sha256_file(bounded_path)
    qwen_seal = read_json(qwen_seal_path)
    bounded_seal = read_json(bounded_seal_path)
    if qwen_sha != qwen_seal.get("main_prediction_sha256") or qwen_sha != EXPECTED_QWEN_SHA:
        raise RuntimeError("Qwen frozen prediction SHA mismatch")
    if bounded_sha != bounded_seal.get("prediction_sha256") or bounded_sha != EXPECTED_TOP100_SHA:
        raise RuntimeError("bounded Top100 frozen prediction SHA mismatch")
    if identity_mismatch or rank_mismatch or score_nonfinite:
        raise RuntimeError(f"frozen input mismatch: identity={identity_mismatch}, rank={rank_mismatch}, nonfinite={score_nonfinite}")
    contract = {
        "gate": "NF-OPT-22-R0",
        "base_commit": BASE_COMMIT,
        "evaluation_role": "development_shadow_confidence_gated_late_fusion",
        "fresh_blind_evaluation": False,
        "model_execution": False,
        "retrieval_rerun": False,
        "training": False,
        "candidate_depth": 100,
        "fusion_candidate_depth": 10,
        "qwen_prediction_sha256": qwen_sha,
        "qwen_prediction_sha_expected": EXPECTED_QWEN_SHA,
        "bounded_top100_sha256": bounded_sha,
        "bounded_top100_sha_expected": EXPECTED_TOP100_SHA,
        "candidate_identity_mismatch": identity_mismatch,
        "bm25_rank_mismatch": rank_mismatch,
        "bm25_rank_missing": bm25_missing,
        "qwen_score_unchanged": True,
        "qwen_rank_unchanged": True,
        "top10_candidate_set_invariant": True,
        "top20_candidate_set_invariant": True,
        "top50_candidate_set_invariant": True,
        "top100_candidate_set_invariant": True,
        "gold_reads_during_prediction": 0,
    }
    return qwen_by_case, bounded_by_case, contract


def load_near_boundary(evaluation_root: Path) -> float:
    path = evaluation_root / "nf-opt-19-r0-setwise-ranking-audit" / "score-margin-analysis.json"
    payload = read_json(path)
    rule = payload.get("rule")
    if rule != "near_boundary iff C1 margin <= median(C1 margins)":
        raise RuntimeError(f"unexpected near-boundary rule: {rule!r}")
    threshold = payload.get("threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise RuntimeError("near-boundary threshold is not executable")
    counts = payload.get("counts", {})
    if counts.get("near_boundary") != 13 or counts.get("clear_loss") != 12:
        raise RuntimeError(f"near-boundary cohort count mismatch: {counts}")
    # Only the executable rule and threshold are read before prediction.  The
    # per-case cohort records are Gold-dependent diagnostics and remain locked
    # until after the prediction seal.
    return float(threshold)


def partition_candidate(item: dict[str, Any], rank5_score: float, threshold: float) -> tuple[str, float]:
    margin = rank5_score - float(item["reranker_score"])
    # Signed margin is fixed relative to the rank-5 boundary.  A candidate
    # more than the sealed threshold above rank five is PROTECTED; the sealed
    # near-boundary band is BORDER; candidates below it are LOW.
    if margin < -threshold:
        partition = "PROTECTED"
    elif margin <= threshold:
        partition = "BORDER"
    else:
        partition = "LOW"
    return partition, margin


def fusion_score(item: dict[str, Any]) -> float:
    qrank = int(item["original_qwen_rank"])
    brank = item.get("original_bm25_rank")
    score = 1.0 / (RRF_K + qrank)
    if brank is not None:
        score += 1.0 / (RRF_K + int(brank))
    return score


def materialize_cglrrf(items: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = [dict(item) for item in items]
    top10 = [dict(item) for item in source[:10]]
    rank5_score = float(top10[4]["reranker_score"])
    for item in top10:
        partition, margin = partition_candidate(item, rank5_score, threshold)
        item["confidence_partition"] = partition
        item["rank5_score_margin"] = margin
        item["late_rrf_score"] = fusion_score(item) if partition == "BORDER" else None
    protected = [item for item in top10[:5] if item["confidence_partition"] == "PROTECTED"]
    border = [item for item in top10 if item["confidence_partition"] == "BORDER"]
    low = [item for item in top10 if item["confidence_partition"] == "LOW"]
    border_sorted = sorted(
        border,
        key=lambda item: (-float(item["late_rrf_score"]), int(item["original_qwen_rank"]),
                          int(item["original_bm25_rank"]) if item.get("original_bm25_rank") is not None else math.inf,
                          item["candidate_key"]),
    )
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for item in protected:
        if item["candidate_key"] not in selected_keys and len(selected) < 5:
            selected.append(item)
            selected_keys.add(item["candidate_key"])
    remaining = 5 - len(selected)
    for item in border_sorted:
        if remaining <= 0:
            break
        if item["candidate_key"] not in selected_keys:
            selected.append(item)
            selected_keys.add(item["candidate_key"])
            remaining -= 1
    if remaining:
        # LOW candidates never receive a BM25 promotion.  This fallback is
        # only the original Qwen order, for the case where the protected and
        # border pools do not fill five slots.
        for item in top10:
            if remaining <= 0:
                break
            if item["candidate_key"] not in selected_keys:
                selected.append(item)
                selected_keys.add(item["candidate_key"])
                remaining -= 1
    if len(selected) != 5 or len(selected_keys) != 5:
        raise RuntimeError("CGLRRF failed to produce five unique candidates")
    selected_keys = {item["candidate_key"] for item in selected}
    tail = [item for item in top10 if item["candidate_key"] not in selected_keys]
    # Keep all unselected Top10 candidates in their original Qwen relative
    # order, followed by the untouched rank 11..100 tail.
    reordered = selected + tail + source[10:]
    for index, item in enumerate(reordered, start=1):
        item["cglrrf_rank"] = index
    meta = {
        "rank5_score": rank5_score,
        "protected_count": len(protected),
        "border_count": len(border),
        "low_count": len(low),
        "protected_ids": [item["candidate_key"] for item in protected],
        "border_ids": [item["candidate_key"] for item in border],
        "low_ids": [item["candidate_key"] for item in low],
        "selected_top5_ids": [item["candidate_key"] for item in selected],
        "original_top5_ids": [item["candidate_key"] for item in top10[:5]],
        "changed": [item["candidate_key"] for item in selected] != [item["candidate_key"] for item in top10[:5]],
    }
    return reordered, meta


def metrics_for(bindings: list[dict[str, Any]], ranked: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in K_VALUES:
        hits = sum(rank_map(ranked[row["case_id"]]).get(row["candidate_key"], 10**9) <= k for row in bindings)
        result[f"@{k}"] = {"hits": hits, "total": len(bindings), "rate": rate(hits, len(bindings))}
    return result


def movement(bindings: list[dict[str, Any]], before: dict[str, list[dict[str, Any]]], after: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rescued = damaged = 0
    rows: list[dict[str, Any]] = []
    for row in bindings:
        case_id = row["case_id"]
        key = row["candidate_key"]
        old = rank_map(before[case_id]).get(key, 10**9)
        new = rank_map(after[case_id]).get(key, 10**9)
        old_hit = old <= 5
        new_hit = new <= 5
        outcome = "unchanged"
        if not old_hit and new_hit:
            rescued += 1
            outcome = "rescued"
        elif old_hit and not new_hit:
            damaged += 1
            outcome = "damaged"
        rows.append({"case_id": case_id, "source_index": row.get("source_index", 0), "candidate_key": key, "before_rank": old, "after_rank": new, "outcome": outcome})
    return {"rescued": rescued, "damaged": damaged, "net": rescued - damaged, "rows": rows}


def parse_plan_rows(payload: Any) -> dict[str, dict[str, Any]]:
    rows = payload.get("plans", []) if isinstance(payload, dict) else payload
    return {row["case_id"]: row.get("plan", {}) for row in rows or []}


def load_fact_registry(registry_rows: list[dict[str, Any]]) -> dict[str, dict[str, set[str]]]:
    facts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for record in registry_rows:
        ids = set(record.get("semantic_fact_ids", []))
        ids.update(fact.get("semantic_fact_id") for fact in record.get("semantic_facts", []) or [] if fact.get("semantic_fact_id"))
        for occurrence in record.get("occurrences", []) or []:
            case_id = occurrence.get("case_id")
            if case_id:
                facts[case_id][record["candidate_key"]].update(ids)
    return facts


def main() -> int:
    backend_root = Path(__file__).resolve().parents[2]
    evaluation_root = backend_root / "artifacts" / "evaluation"
    out_dir = evaluation_root / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    qwen_by_case, bounded_by_case, frozen_contract = load_frozen_rankings(evaluation_root)
    threshold = load_near_boundary(evaluation_root)
    frozen_contract.update({"near_boundary_threshold": threshold})
    write_json(out_dir / "frozen-contract.json", frozen_contract)
    write_json(out_dir / "confidence-gate-contract.json", {
        "gate": "NF-OPT-22-R0",
        "method": "Confidence-Gated Qwen/BM25 Late Fusion V1",
        "short_name": "CGLRRF-V1",
        "near_boundary_rule": "near_boundary iff C1 margin <= median(C1 margins)",
        "near_boundary_threshold": threshold,
        "margin_definition": "rank5_score - candidate_score",
        "protected_rule": "margin < -near_boundary_threshold",
        "border_rule": "-near_boundary_threshold <= margin <= near_boundary_threshold",
        "low_rule": "margin > near_boundary_threshold",
        "protected_top5_preserved": True,
        "border_pool": "original_qwen_top10_only",
        "low_candidate_rule": "original_qwen_order_fallback_only",
        "rrf_k": RRF_K,
        "weights": {"qwen": 1.0, "bm25": 1.0},
        "tie_break": ["fusion_score_desc", "original_qwen_rank_asc", "original_bm25_rank_asc", "candidate_key_asc"],
        "candidate_generation": False,
        "model_execution": False,
        "retrieval_rerun": False,
        "training": False,
        "gold_reads_during_prediction": 0,
    })

    # Gold-blind candidate partition and prediction generation.
    partition_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    ranked_cglrrf: dict[str, list[dict[str, Any]]] = {}
    meta_by_case: dict[str, dict[str, Any]] = {}
    aggregate = Counter()
    changed_count = 0
    for case_id in sorted(qwen_by_case):
        ranked, meta = materialize_cglrrf(qwen_by_case[case_id], threshold)
        ranked_cglrrf[case_id] = ranked
        meta_by_case[case_id] = meta
        aggregate.update({"protected": meta["protected_count"], "border": meta["border_count"], "low": meta["low_count"]})
        changed_count += int(meta["changed"])
        partition_rows.append({"case_id": case_id, **meta})
        original_top10 = top_keys(qwen_by_case[case_id], 10)
        if set(original_top10) != set(top_keys(ranked, 10)):
            raise RuntimeError(f"Top10 candidate set changed for {case_id}")
        prediction_rows.append({
            "case_id": case_id,
            "candidate_budget": 100,
            "fusion_candidate_depth": 10,
            "input_top10_candidate_ids": original_top10,
            "selected_top5_candidate_ids": top_keys(ranked, 5),
            "ranked_candidates": ranked,
        })
    write_json(out_dir / "candidate-partition.json", {
        "queries": len(prediction_rows),
        "aggregate_candidate_counts": dict(aggregate),
        "average_candidate_counts_per_query": {key: aggregate[key] / len(prediction_rows) for key in ("protected", "border", "low")},
        "queries_top5_changed": changed_count,
        "queries_top5_unchanged": len(prediction_rows) - changed_count,
        "near_boundary_threshold": threshold,
        "records": partition_rows,
    })
    prediction_path = out_dir / "predictions.jsonl.gz"
    prediction_sha = write_gzip_jsonl(prediction_path, prediction_rows)
    prediction_seal = {
        "gate": "NF-OPT-22-R0",
        "sealed": True,
        "prediction_sha256": prediction_sha,
        "queries": len(prediction_rows),
        "candidate_depth": 100,
        "fusion_candidate_depth": 10,
        "gold_reads_during_prediction": 0,
        "model_execution": False,
        "retrieval_rerun": False,
        "training": False,
    }
    write_json(out_dir / "prediction-seal.json", prediction_seal)
    # Verify the just-written artifact before any Gold-dependent read.
    if sha256_file(prediction_path) != prediction_sha or len(prediction_rows) != 72:
        raise RuntimeError("CGLRRF prediction seal validation failed")

    # Post-seal only: all data below is diagnostic/Gold-dependent.
    r0 = load_r0_helpers()
    strict_path = evaluation_root / "pdf-retrieval-v4-strict-source-contract" / "strict-gold-source-bindings.jsonl"
    targets_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1" / "gold-semantic-targets.jsonl"
    registry_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1-p0" / "candidate-semantic-fact-registry.jsonl.gz"
    plan_path = evaluation_root / "pdf-retrieval-v4-gate-07" / "query-plan-predictions.json"
    calc_path = evaluation_root / "nf-opt-19-r0-setwise-ranking-audit" / "calculation-slot-coverage.json"
    r0_root = evaluation_root / "nf-opt-21-r0-qwen-bm25-late-fusion"
    strict_rows = read_jsonl(strict_path)
    targets = r0.load_targets(targets_path)
    facts = r0.build_case_facts(read_gzip_jsonl(registry_path))
    query_plans = parse_plan_rows(read_json(plan_path))
    bindings_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strict_rows:
        bindings_by_case[row["case_id"]].append(row)
    case_ids = sorted(qwen_by_case)
    qwen_ranked = qwen_by_case
    lrrf_rows = read_gzip_jsonl(r0_root / "lrrf-predictions.jsonl.gz")
    lrrf_ranked = {row["case_id"]: sorted(row["ranked_candidates"], key=lambda item: int(item.get("fusion_rank", item.get("post_rerank_rank", 0)))) for row in lrrf_rows}
    if set(lrrf_ranked) != set(case_ids):
        raise RuntimeError("LRRF historical artifact query mismatch")
    strict = {"qwen": metrics_for(strict_rows, qwen_ranked), "lrrf_v1": metrics_for(strict_rows, lrrf_ranked), "cglrrf_v1": metrics_for(strict_rows, ranked_cglrrf)}
    write_json(out_dir / "strict-metrics.json", {"strict_sources": len(strict_rows), "metrics": strict})
    semantic = {
        "qwen": r0.semantic_metrics(strict_rows, qwen_ranked, facts, targets),
        "lrrf_v1": r0.semantic_metrics(strict_rows, lrrf_ranked, facts, targets),
        "cglrrf_v1": r0.semantic_metrics(strict_rows, ranked_cglrrf, facts, targets),
    }
    write_json(out_dir / "semantic-metrics.json", semantic)

    move_qwen = movement(strict_rows, qwen_ranked, ranked_cglrrf)
    move_lrrf = movement(strict_rows, lrrf_ranked, ranked_cglrrf)
    move_qwen["queries_top5_changed"] = changed_count
    move_qwen["queries_top5_unchanged"] = len(case_ids) - changed_count
    move_lrrf["queries_top5_changed"] = sum(top_keys(lrrf_ranked[c], 5) != top_keys(ranked_cglrrf[c], 5) for c in case_ids)
    write_json(out_dir / "movement-vs-qwen.json", move_qwen)
    write_json(out_dir / "movement-vs-lrrf.json", move_lrrf)

    # Historical LRRF rescue/damage accounting is only read now, after seal.
    lrrf_rescues = []
    lrrf_damages = []
    for row in strict_rows:
        case_id, key = row["case_id"], row["candidate_key"]
        qhit = rank_map(qwen_ranked[case_id]).get(key, 10**9) <= 5
        lhit = rank_map(lrrf_ranked[case_id]).get(key, 10**9) <= 5
        ch = rank_map(ranked_cglrrf[case_id]).get(key, 10**9) <= 5
        if not qhit and lhit:
            lrrf_rescues.append({"case_id": case_id, "candidate_key": key, "retained": ch})
        if qhit and not lhit:
            lrrf_damages.append({"case_id": case_id, "candidate_key": key, "prevented": ch})
    write_json(out_dir / "lrrf-rescue-preservation.json", {"historical_rescues": len(lrrf_rescues), "retained": sum(row["retained"] for row in lrrf_rescues), "lost": sum(not row["retained"] for row in lrrf_rescues), "records": lrrf_rescues})
    write_json(out_dir / "lrrf-damage-prevention.json", {"historical_damages": len(lrrf_damages), "prevented": sum(row["prevented"] for row in lrrf_damages), "still_damaged": sum(not row["prevented"] for row in lrrf_damages), "damage_prevention_rate": rate(sum(row["prevented"] for row in lrrf_damages), len(lrrf_damages)), "records": lrrf_damages})

    near_records = [row for row in read_json(evaluation_root / "nf-opt-19-r0-setwise-ranking-audit" / "score-margin-analysis.json")["records"] if row.get("cohort") == "near_boundary"]
    clear_records = [row for row in read_json(evaluation_root / "nf-opt-19-r0-setwise-ranking-audit" / "score-margin-analysis.json")["records"] if row.get("cohort") == "clear_loss"]
    def cohort_hits(records: list[dict[str, Any]], ranked: dict[str, list[dict[str, Any]]]) -> int:
        return sum(rank_map(ranked[row["case_id"]]).get(row["candidate_key"], 10**9) <= 5 for row in records)
    write_json(out_dir / "near-boundary-analysis.json", {
        "near_boundary_sources": len(near_records),
        "clear_loss_sources": len(clear_records),
        "near_boundary": {name: cohort_hits(near_records, ranked) for name, ranked in (("qwen", qwen_ranked), ("lrrf_v1", lrrf_ranked), ("cglrrf_v1", ranked_cglrrf))},
        "clear_loss": {name: cohort_hits(clear_records, ranked) for name, ranked in (("qwen", qwen_ranked), ("lrrf_v1", lrrf_ranked), ("cglrrf_v1", ranked_cglrrf))},
    })

    multi_cases = sorted(case_id for case_id in case_ids if query_plans.get(case_id, {}).get("requires_multiple_sources") and bindings_by_case.get(case_id))
    calc_rows = read_json(calc_path)["cases"]
    multi = {name: {"@5": r0.coverage(bindings_by_case, multi_cases, ranked, 5), "@10": r0.coverage(bindings_by_case, multi_cases, ranked, 10)} for name, ranked in (("qwen", qwen_ranked), ("lrrf_v1", lrrf_ranked), ("cglrrf_v1", ranked_cglrrf))}
    write_json(out_dir / "multi-evidence-analysis.json", {"denominator": len(multi_cases), "variants": multi})
    calc = {name: {"@5": r0.calc_coverage(calc_rows, ranked, facts, 5), "@10": r0.calc_coverage(calc_rows, ranked, facts, 10)} for name, ranked in (("qwen", qwen_ranked), ("lrrf_v1", lrrf_ranked), ("cglrrf_v1", ranked_cglrrf))}
    write_json(out_dir / "calculation-slot-analysis.json", {"denominator": len(calc_rows), "variants": calc})

    cgl_r5 = strict["cglrrf_v1"]["@5"]["hits"]
    qwen_r5 = strict["qwen"]["@5"]["hits"]
    lrrf_r5 = strict["lrrf_v1"]["@5"]["hits"]
    cgl_sem5 = semantic["cglrrf_v1"]["@5"]["hits"]
    cgl_multi_all = multi["cglrrf_v1"]["@5"]["all"]
    cgl_calc_all = calc["cglrrf_v1"]["@5"]["all_slots"]
    gain = cgl_r5 - qwen_r5
    damage = move_qwen["damaged"]
    safe = cgl_sem5 >= 49 and cgl_multi_all >= 4 and cgl_calc_all >= 5
    strong = cgl_r5 >= 49 and gain >= 6 and damage <= 3 and cgl_sem5 >= 51 and cgl_multi_all >= 7 and cgl_calc_all >= 6
    marginal = 47 <= cgl_r5 <= 48 and damage < 6 and cgl_sem5 >= 49 and cgl_multi_all >= 4 and cgl_calc_all >= 5
    if strong:
        effective: bool | str = True
        selected = "cglrrf_v1"
        next_gate = "internal_retrieval_shadow_freeze"
    elif marginal:
        effective = "marginal"
        selected = "cglrrf_v1" if cgl_r5 > lrrf_r5 or (cgl_r5 == lrrf_r5 and damage < 6) else "lrrf_v1"
        next_gate = "internal_retrieval_shadow_freeze" if selected == "cglrrf_v1" else "internal_retrieval_method_freeze"
    else:
        effective = False
        selected = "lrrf_v1"
        next_gate = "internal_retrieval_method_freeze"
    decision = {
        "gate": "NF-OPT-22-R0",
        "base_commit": BASE_COMMIT,
        "evaluation_role": "development_shadow_confidence_gated_late_fusion",
        "fresh_blind_evaluation": False,
        "model_execution": False,
        "retrieval_rerun": False,
        "training": False,
        "strict_sources": len(strict_rows),
        "qwen_r5_hits": qwen_r5,
        "lrrf_r5_hits": lrrf_r5,
        "cglrrf_r5_hits": cgl_r5,
        "qwen_top10_supply": strict["qwen"]["@10"]["hits"],
        "rescued_vs_qwen": move_qwen["rescued"],
        "damaged_vs_qwen": move_qwen["damaged"],
        "net_vs_qwen": move_qwen["net"],
        "rescued_vs_lrrf": move_lrrf["rescued"],
        "damaged_vs_lrrf": move_lrrf["damaged"],
        "net_vs_lrrf": move_lrrf["net"],
        "lrrf_historical_rescued": len(lrrf_rescues),
        "lrrf_historical_damaged": len(lrrf_damages),
        "lrrf_rescues_retained": sum(row["retained"] for row in lrrf_rescues),
        "lrrf_damages_prevented": sum(row["prevented"] for row in lrrf_damages),
        "semantic_r5_hits": cgl_sem5,
        "semantic_r10_hits": semantic["cglrrf_v1"]["@10"]["hits"],
        "multi_evidence_all_at_5": cgl_multi_all,
        "calculation_all_slots_at_5": cgl_calc_all,
        "protected_candidate_count": aggregate["protected"],
        "border_candidate_count": aggregate["border"],
        "low_candidate_count": aggregate["low"],
        "queries_top5_changed": changed_count,
        "near_boundary_threshold": threshold,
        "top10_supply_invariant": strict["cglrrf_v1"]["@10"]["hits"] == strict["qwen"]["@10"]["hits"],
        "confidence_gated_late_fusion_effective": effective,
        "selected_internal_shadow_method": selected,
        "production_switch_allowed": False,
        "next_gate": next_gate,
        "decision_reason": [
            "CGLRRF is a Gold-blind, fixed-RRF correction restricted to the sealed Qwen Top10.",
            f"R@5 gain vs Qwen = {gain} sources; damage = {damage}; safety={safe}.",
        ],
    }
    write_json(out_dir / "decision.json", decision)
    (out_dir / "README.md").write_text(
        "# NF-OPT-22 R0 — Confidence-Gated Qwen/BM25 Late Fusion\n\n"
        "This development-shadow audit consumes only the sealed Qwen Top100 "
        "and BM25 ranks. Prediction generation is Gold-blind; strict, semantic, "
        "multi-evidence, and calculation diagnostics are loaded after the "
        "prediction seal.\n\n"
        f"Near-boundary threshold: `{threshold}`\n\n"
        f"RRF_K: `{RRF_K}`; selected shadow method: `{selected}`; "
        f"effectiveness: `{effective}`.\n\n"
        "Production switch allowed: `false`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact_dir": str(out_dir), "decision": decision}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
