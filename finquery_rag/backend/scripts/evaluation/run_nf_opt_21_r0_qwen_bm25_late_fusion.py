"""NF-OPT-21 R0: frozen Qwen/BM25 Top10 late-fusion shadow audit.

The prediction portion of this module is intentionally Gold-blind.  It reads
only the sealed Qwen and bounded Top100 rankings, derives BM25 rank from the
already persisted pre-rerank rank, and emits two fixed Top10 late-fusion
ablations.  Gold, semantic targets, and historical failure cohorts are loaded
only after both prediction artifacts have been written and sealed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RRF_K = 60
K_VALUES = (1, 3, 5, 10, 20, 50, 100)
OUT_NAME = "nf-opt-21-r0-qwen-bm25-late-fusion"
BASE_COMMIT = "3cef6bbb25561c68e1fc12f288eaaaad2144fa83"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return sha256_file(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def rank_map(items: list[dict[str, Any]]) -> dict[str, int]:
    return {item["candidate_key"]: index + 1 for index, item in enumerate(items)}


def top_keys(items: list[dict[str, Any]], k: int) -> list[str]:
    return [item["candidate_key"] for item in items[:k]]


def percentile(values: list[int | float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_plan_rows(payload: Any) -> dict[str, dict[str, Any]]:
    rows = payload.get("plans", []) if isinstance(payload, dict) else payload
    return {row["case_id"]: row.get("plan", {}) for row in rows or []}


def build_case_facts(registry_rows: list[dict[str, Any]]) -> dict[str, dict[str, set[str]]]:
    facts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for record in registry_rows:
        ids = set(record.get("semantic_fact_ids", []))
        for fact in record.get("semantic_facts", []) or []:
            if fact.get("semantic_fact_id"):
                ids.add(fact["semantic_fact_id"])
        for occurrence in record.get("occurrences", []) or []:
            case_id = occurrence.get("case_id")
            if case_id:
                facts[case_id][record["candidate_key"]].update(ids)
    return facts


def load_targets(path: Path) -> dict[str, dict[str, Any]]:
    return {row["binding_id"]: row for row in read_jsonl(path)}


def semantic_ids_for(binding: dict[str, Any], targets: dict[str, dict[str, Any]]) -> set[str]:
    target = targets.get(binding.get("binding_id"), {})
    ids = set(target.get("matching_semantic_fact_ids") or [])
    if target.get("gold_semantic_fact_id"):
        ids.add(target["gold_semantic_fact_id"])
    return ids


def semantic_union(case_id: str, ranked: list[dict[str, Any]], k: int, facts: dict[str, dict[str, set[str]]]) -> set[str]:
    values: set[str] = set()
    for item in ranked[:k]:
        values.update(facts.get(case_id, {}).get(item["candidate_key"], set()))
    return values


def strict_metrics(bindings: list[dict[str, Any]], ranked: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for k in K_VALUES:
        hits = 0
        for binding in bindings:
            if rank_map(ranked[binding["case_id"]]).get(binding["candidate_key"], 10**9) <= k:
                hits += 1
        output[f"@{k}"] = {"hits": hits, "total": len(bindings), "rate": rate(hits, len(bindings))}
    return output


def semantic_metrics(
    bindings: list[dict[str, Any]],
    ranked: dict[str, list[dict[str, Any]]],
    facts: dict[str, dict[str, set[str]]],
    targets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for k in (1, 3, 5, 10):
        hits = 0
        for binding in bindings:
            case_id = binding["case_id"]
            target_ids = semantic_ids_for(binding, targets)
            physical = binding["candidate_key"] in set(top_keys(ranked[case_id], k))
            hits += int(physical or bool(target_ids & semantic_union(case_id, ranked[case_id], k, facts)))
        output[f"@{k}"] = {"hits": hits, "total": len(bindings), "rate": rate(hits, len(bindings))}
    return output


def coverage(bindings_by_case: dict[str, list[dict[str, Any]]], case_ids: list[str], ranked: dict[str, list[dict[str, Any]]], k: int) -> dict[str, Any]:
    any_count = 0
    all_count = 0
    ratios: list[float] = []
    for case_id in case_ids:
        bindings = bindings_by_case.get(case_id, [])
        keys = set(top_keys(ranked[case_id], k))
        represented = sum(binding["candidate_key"] in keys for binding in bindings)
        any_count += int(represented > 0)
        all_count += int(bool(bindings) and represented == len(bindings))
        ratios.append(rate(represented, len(bindings)) or 0.0)
    return {
        "queries": len(case_ids),
        "any": any_count,
        "all": all_count,
        "any_rate": rate(any_count, len(case_ids)),
        "all_rate": rate(all_count, len(case_ids)),
        "average_required_source_coverage": statistics.mean(ratios) if ratios else None,
    }


def calc_coverage(
    calc_rows: list[dict[str, Any]],
    ranked: dict[str, list[dict[str, Any]]],
    facts: dict[str, dict[str, set[str]]],
    k: int,
) -> dict[str, Any]:
    represented_total = 0
    required_total = 0
    all_count = 0
    records: list[dict[str, Any]] = []
    for row in calc_rows:
        required = list(row.get("required_semantic_fact_ids") or [])
        union = semantic_union(row["case_id"], ranked[row["case_id"]], k, facts)
        represented = sum(value in union for value in required)
        represented_total += represented
        required_total += len(required)
        all_count += int(bool(required) and represented == len(required))
        records.append({"case_id": row["case_id"], "required_slots": len(required), "represented_slots": represented, "all_slots": bool(required) and represented == len(required)})
    return {"queries": len(calc_rows), "all_slots": all_count, "all_slots_rate": rate(all_count, len(calc_rows)), "represented_slots": represented_total, "required_slots": required_total, "average_slot_coverage": rate(represented_total, required_total), "records": records}


def make_late_score(item: dict[str, Any]) -> tuple[float, int, int, str]:
    qrank = int(item["original_qwen_rank"])
    brank = int(item["original_bm25_rank"]) if item["original_bm25_rank"] is not None else math.inf
    score = 1.0 / (RRF_K + qrank)
    if math.isfinite(brank):
        score += 1.0 / (RRF_K + brank)
    return score, qrank, brank, item["candidate_key"]


def materialize_variant(original: list[dict[str, Any]], variant: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = [dict(item) for item in original]
    top10 = source[:10]
    scored = []
    for item in top10:
        score, qrank, brank, key = make_late_score(item)
        scored.append((score, qrank, brank, key, item))
    selected: str | None = None
    if variant == "lrrf_v1":
        ordered_top10 = [entry[4] for entry in sorted(scored, key=lambda row: (-row[0], row[1], row[2], row[3]))]
    elif variant == "plrf_v1":
        ordered_bottom = [entry[4] for entry in sorted(scored[4:], key=lambda row: (-row[0], row[1], row[2], row[3]))]
        selected = ordered_bottom[0]["candidate_key"] if ordered_bottom else None
        ordered_top10 = top10[:4] + ([ordered_bottom[0]] if ordered_bottom else top10[4:5])
        ordered_top10 += [item for item in top10[5:] if item["candidate_key"] != selected]
        if selected is not None and len(ordered_top10) < 10:
            ordered_top10 += [top10[4]]
    else:
        raise ValueError(f"unknown variant {variant}")
    output = ordered_top10 + source[10:]
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(output, start=1):
        row = dict(item)
        row["fusion_rank"] = index
        row["original_qwen_rank"] = int(item["original_qwen_rank"])
        row["original_bm25_rank"] = item["original_bm25_rank"]
        row["late_rrf_score"] = float(item.get("late_rrf_score", 0.0)) if index > 10 else float(next(entry[0] for entry in scored if entry[4]["candidate_key"] == item["candidate_key"]))
        rows.append(row)
    return rows, {"variant": variant, "selected_slot5_candidate": selected, "top10_before": top_keys(source, 10), "top10_after": top_keys(rows, 10)}


def movement(bindings: list[dict[str, Any]], before: dict[str, list[dict[str, Any]]], after: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rescued = damaged = unchanged_hit = unchanged_miss = 0
    rows: list[dict[str, Any]] = []
    for binding in bindings:
        case_id = binding["case_id"]
        key = binding["candidate_key"]
        old = rank_map(before[case_id]).get(key, 10**9)
        new = rank_map(after[case_id]).get(key, 10**9)
        old_hit = old <= 5
        new_hit = new <= 5
        if not old_hit and new_hit:
            rescued += 1
            outcome = "rescued"
        elif old_hit and not new_hit:
            damaged += 1
            outcome = "damaged"
        elif old_hit:
            unchanged_hit += 1
            outcome = "unchanged_hit"
        else:
            unchanged_miss += 1
            outcome = "unchanged_miss"
        rows.append({"case_id": case_id, "source_index": binding.get("source_index", 0), "candidate_key": key, "before_rank": old, "after_rank": new, "outcome": outcome})
    return {"rescued": rescued, "damaged": damaged, "net": rescued - damaged, "unchanged_hit": unchanged_hit, "unchanged_miss": unchanged_miss, "rows": rows}


def source_category(bindings_by_case: dict[str, list[dict[str, Any]]], query_plans: dict[str, dict[str, Any]], calc_ids: set[str], case_id: str) -> str:
    if case_id in calc_ids:
        return "calculation"
    if len(bindings_by_case.get(case_id, [])) > 1 or query_plans.get(case_id, {}).get("requires_multiple_sources"):
        return "multi_evidence"
    return "single_evidence"


def rank_distribution(rows: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    values = [int(row["bm25_rank"]) for row in rows if row.get("bm25_rank") is not None and math.isfinite(float(row["bm25_rank"]))]
    return {"count": len(values), "mean": statistics.mean(values) if values else None, "p50": percentile(values, 0.5), "p90": percentile(values, 0.9), "direction": direction}


def main() -> int:
    backend_root = Path(__file__).resolve().parents[2]
    evaluation_root = backend_root / "artifacts" / "evaluation"
    out_dir = evaluation_root / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    qwen_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3" / "main_rerank_predictions.jsonl.gz"
    qwen_seal_path = qwen_path.parent / "prediction-seal.json"
    bounded_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r2a-2" / "bounded-top100-predictions.jsonl.gz"
    bounded_seal_path = bounded_path.parent / "prediction-seal.json"
    registry_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1-p0" / "candidate-semantic-fact-registry.jsonl.gz"
    plan_path = evaluation_root / "pdf-retrieval-v4-gate-07" / "query-plan-predictions.json"
    strict_path = evaluation_root / "pdf-retrieval-v4-strict-source-contract" / "strict-gold-source-bindings.jsonl"
    targets_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1" / "gold-semantic-targets.jsonl"
    calc_path = evaluation_root / "nf-opt-19-r0-setwise-ranking-audit" / "calculation-slot-coverage.json"
    nf20_root = evaluation_root / "nf-opt-20-r0-pointwise-discrimination-audit"

    qwen_rows = read_gzip_jsonl(qwen_path)
    bounded_rows = read_gzip_jsonl(bounded_path)
    qwen_by_case: dict[str, list[dict[str, Any]]] = {}
    bounded_by_case = {row["case_id"]: row["candidates"] for row in bounded_rows}
    rank_mismatch = 0
    identity_mismatch = 0
    bm25_missing = 0
    for row in qwen_rows:
        items = sorted(row["ranked_candidates"], key=lambda item: int(item["post_rerank_rank"]))
        if len(items) != 100 or [int(item["post_rerank_rank"]) for item in items] != list(range(1, 101)):
            raise RuntimeError(f"invalid Qwen frozen ranks: {row['case_id']}")
        bounded = bounded_by_case.get(row["case_id"], [])
        if set(item["candidate_key"] for item in items) != set(item["candidate_key"] for item in bounded):
            identity_mismatch += 1
        by_key = {item["candidate_key"]: item for item in bounded}
        prepared: list[dict[str, Any]] = []
        for item in items:
            bounded_item = by_key.get(item["candidate_key"])
            qrank = int(item["post_rerank_rank"])
            brank = int(item["pre_rerank_rank"]) if item.get("pre_rerank_rank") is not None else None
            if bounded_item is None or brank is None:
                bm25_missing += 1
            bounded_rank = None if bounded_item is None else bounded_item.get("rank", bounded_item.get("final_candidate_rank"))
            if bounded_rank is not None and brank != int(bounded_rank):
                rank_mismatch += 1
            clone = dict(item)
            clone["original_qwen_rank"] = qrank
            clone["original_bm25_rank"] = brank
            prepared.append(clone)
        qwen_by_case[row["case_id"]] = prepared
    qwen_sha = sha256_file(qwen_path)
    bounded_sha = sha256_file(bounded_path)
    qwen_seal = read_json(qwen_seal_path)
    bounded_seal = read_json(bounded_seal_path)
    if qwen_sha != qwen_seal.get("main_prediction_sha256"):
        raise RuntimeError("Qwen frozen prediction SHA mismatch")
    if bounded_sha != bounded_seal.get("prediction_sha256"):
        raise RuntimeError("bounded Top100 frozen prediction SHA mismatch")
    if identity_mismatch or rank_mismatch:
        raise RuntimeError(f"frozen identity/rank mismatch: identity={identity_mismatch}, rank={rank_mismatch}")
    write_json(out_dir / "frozen-input-contract.json", {
        "gate": "NF-OPT-21-R0", "base_commit": BASE_COMMIT,
        "model_execution": False, "retrieval_rerun": False, "candidate_generation": False, "training": False,
        "qwen_prediction_sha256": qwen_sha, "qwen_prediction_sha_expected": qwen_seal.get("main_prediction_sha256"),
        "bounded_top100_sha256": bounded_sha, "bounded_top100_sha_expected": bounded_seal.get("prediction_sha256"),
        "candidate_identity_mismatch": identity_mismatch, "bm25_rank_mismatch": rank_mismatch,
        "bm25_rank_source": "qwen.pre_rerank_rank equals bounded.rank", "bm25_rank_missing": bm25_missing,
        "qwen_rank_unchanged": True, "qwen_score_unchanged": True, "candidate_budget": 100, "fusion_candidate_depth": 10,
        "top10_candidate_set_invariant": True, "top20_candidate_set_invariant": True,
        "top50_candidate_set_invariant": True, "top100_candidate_set_invariant": True,
    })

    # Gold-blind generation.  The runtime has no strict bindings, semantic
    # targets, or historical failure cohort in scope.
    variant_rows: dict[str, list[dict[str, Any]]] = {}
    variant_meta: dict[str, dict[str, Any]] = {}
    for variant in ("lrrf_v1", "plrf_v1"):
        rows: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []
        for case_id in sorted(qwen_by_case):
            output, meta = materialize_variant(qwen_by_case[case_id], variant)
            rows.append({"case_id": case_id, "candidate_budget": 100, "fusion_candidate_depth": 10, "ranked_candidates": output})
            metas.append(meta)
        variant_rows[variant] = rows
        variant_meta[variant] = {"variant": variant, "queries": len(rows), "metas": metas}
        output_path = out_dir / ("lrrf-predictions.jsonl.gz" if variant == "lrrf_v1" else "plrf-predictions.jsonl.gz")
        prediction_sha = write_gzip_jsonl(output_path, rows)
        manifest = {"gate": "NF-OPT-21-R0", "variant": variant, "rows": len(rows), "candidate_count_per_query": 100, "fusion_candidate_depth": 10, "prediction_sha256": prediction_sha, "gold_reads_during_prediction": 0, "model_execution": False, "retrieval_rerun": False}
        write_json(out_dir / ("lrrf-prediction-seal.json" if variant == "lrrf_v1" else "plrf-prediction-seal.json"), {**manifest, "sealed": True})
        write_json(out_dir / ("lrrf-v1-contract.json" if variant == "lrrf_v1" else "plrf-v1-contract.json"), {
            "gate": "NF-OPT-21-R0", "method": "Qwen-BM25 Late RRF V1" if variant == "lrrf_v1" else "Protected Qwen-BM25 Boundary Fusion V1",
            "short_name": "LRRF-V1" if variant == "lrrf_v1" else "PLRF-V1", "rrf_k": RRF_K, "weights": {"qwen": 1.0, "bm25": 1.0},
            "candidate_source": "frozen_qwen_top10", "tie_break": ["late_rrf_score_desc", "original_qwen_rank_asc", "original_bm25_rank_asc", "candidate_key_asc"],
            "max_slot_mutations_per_query": 10 if variant == "lrrf_v1" else 1,
            "rank1_4_preserved": variant == "plrf_v1", "gold_reads_during_prediction": 0, "prediction_sha256": prediction_sha,
        })
        write_json(out_dir / ("lrrf-runtime-audit.json" if variant == "lrrf_v1" else "plrf-runtime-audit.json"), variant_meta[variant])

    # Post-seal only: load gold and diagnostic cohort metadata.
    strict_rows = read_jsonl(strict_path)
    targets = load_targets(targets_path)
    registry_rows = read_gzip_jsonl(registry_path)
    facts = build_case_facts(registry_rows)
    query_plans = parse_plan_rows(read_json(plan_path))
    bindings_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strict_rows:
        bindings_by_case[row["case_id"]].append(row)
    case_ids = sorted(qwen_by_case)
    qwen_ranked = qwen_by_case
    variant_ranked = {variant: {row["case_id"]: sorted(row["ranked_candidates"], key=lambda item: int(item["fusion_rank"])) for row in rows} for variant, rows in variant_rows.items()}
    calc_rows = read_json(calc_path)["cases"]

    strict_output = {"qwen": strict_metrics(strict_rows, qwen_ranked)}
    strict_output.update({variant: strict_metrics(strict_rows, ranked) for variant, ranked in variant_ranked.items()})
    write_json(out_dir / "strict-metrics.json", {"strict_sources": len(strict_rows), **strict_output})
    semantic_output = {"qwen": semantic_metrics(strict_rows, qwen_ranked, facts, targets)}
    semantic_output.update({variant: semantic_metrics(strict_rows, ranked, facts, targets) for variant, ranked in variant_ranked.items()})
    write_json(out_dir / "semantic-metrics.json", semantic_output)

    movements = {variant: movement(strict_rows, qwen_ranked, ranked) for variant, ranked in variant_ranked.items()}
    write_json(out_dir / "rank-movement.json", {"baseline": "qwen", "variants": movements, "queries": len(case_ids), "source_denominator": len(strict_rows)})

    # Post-seal cohort accounting.
    c_counts = {"C0": 0, "C1": 0, "C2": 0}
    for binding in strict_rows:
        rank = rank_map(qwen_ranked[binding["case_id"]]).get(binding["candidate_key"], 101)
        c_counts["C0" if rank <= 5 else "C1" if rank <= 100 else "C2"] += 1
    c1_pairs = read_gzip_jsonl(nf20_root / "c1-pairs.jsonl.gz")
    reversal_cases: list[dict[str, Any]] = []
    for pair in c1_pairs:
        competitors = [row for row in pair.get("competitors", []) if row.get("competitor_role") == "highest_ranked_wrong"]
        if not competitors:
            competitors = [row for row in pair.get("competitors", []) if row.get("competitor_role") == "rank5_cutoff"]
        if not competitors:
            continue
        row = competitors[0]
        if row.get("bm25_ordering") == "gold_above_competitor" and row.get("qwen_ordering") == "gold_below_competitor":
            gold_value = pair.get("gold_candidate")
            gold_key = gold_value.get("candidate_key") if isinstance(gold_value, dict) else gold_value
            reversal_cases.append({"case_id": pair["case_id"], "source_index": pair.get("source_index", 0), "gold_candidate_id": gold_key or row.get("gold_candidate_id"), "competitor_candidate_id": row.get("competitor_candidate_id"), "gold_qwen_rank": pair.get("gold_rank"), "competitor_qwen_rank": row.get("competitor_rank"), "gold_bm25_rank": row.get("gold_bm25_rank"), "competitor_bm25_rank": row.get("competitor_bm25_rank")})
    reversal_output: dict[str, Any] = {"frozen_cohort_total": len(reversal_cases)}
    for variant, ranked in variant_ranked.items():
        recovered = sum(rank_map(ranked[row["case_id"]]).get(row["gold_candidate_id"], 101) <= 5 for row in reversal_cases if row["gold_candidate_id"])
        damaged = sum(rank_map(qwen_ranked[row["case_id"]]).get(row["gold_candidate_id"], 101) <= 5 and rank_map(ranked[row["case_id"]]).get(row["gold_candidate_id"], 101) > 5 for row in reversal_cases if row["gold_candidate_id"])
        reversal_output[variant] = {"recovered": recovered, "damaged": damaged, "records": reversal_cases}
    write_json(out_dir / "bm25-reversal-recovery.json", reversal_output)

    rank_recovery: dict[str, Any] = {}
    for qrank in range(6, 11):
        source_rows = [binding for binding in strict_rows if rank_map(qwen_ranked[binding["case_id"]]).get(binding["candidate_key"], 101) == qrank]
        rank_recovery[str(qrank)] = {"qwen_gold_sources": len(source_rows), **{variant: sum(rank_map(ranked[b["case_id"]]).get(b["candidate_key"], 101) <= 5 for b in source_rows) for variant, ranked in variant_ranked.items()}}
    write_json(out_dir / "qwen-rank6-10-recovery.json", rank_recovery)

    bm25_analysis: dict[str, Any] = {}
    for variant, mov in movements.items():
        by_outcome = {"rescued": [], "damaged": []}
        for row in mov["rows"]:
            if row["outcome"] in by_outcome:
                by_outcome[row["outcome"]].append({**row, "bm25_rank": qwen_ranked[row["case_id"]][row["before_rank"] - 1].get("original_bm25_rank") if row["before_rank"] <= 100 else None})
        bm25_analysis[variant] = {outcome: rank_distribution(values, outcome) for outcome, values in by_outcome.items()}
    write_json(out_dir / "bm25-rank-analysis.json", bm25_analysis)

    # NF-OPT-19's frozen taxonomy counts all 72 benchmark questions, including
    # single-evidence questions without a strict physical binding.  Its
    # multi-evidence denominator is the 16 answerable queries whose frozen
    # QueryPlan requires multiple sources; the two published plans with no
    # strict binding are therefore not silently promoted into that denominator.
    multi_cases = sorted(
        case_id
        for case_id in case_ids
        if query_plans.get(case_id, {}).get("requires_multiple_sources") and bindings_by_case.get(case_id)
    )
    single_cases = sorted(set(case_ids) - set(multi_cases))
    single_output = {variant: {"query_count": len(single_cases), "strict_r5": strict_metrics([binding for case_id in single_cases for binding in bindings_by_case[case_id]], ranked)["@5"]} for variant, ranked in {"qwen": qwen_ranked, **variant_ranked}.items()}
    write_json(out_dir / "single-evidence-analysis.json", single_output)
    multi_output = {variant: {"@5": coverage(bindings_by_case, multi_cases, ranked, 5), "@10": coverage(bindings_by_case, multi_cases, ranked, 10)} for variant, ranked in {"qwen": qwen_ranked, **variant_ranked}.items()}
    write_json(out_dir / "multi-evidence-analysis.json", {"denominator": len(multi_cases), "variants": multi_output})
    calc_output = {variant: {"@5": calc_coverage(calc_rows, ranked, facts, 5), "@10": calc_coverage(calc_rows, ranked, facts, 10)} for variant, ranked in {"qwen": qwen_ranked, **variant_ranked}.items()}
    write_json(out_dir / "calculation-slot-analysis.json", {"denominator": len(calc_rows), "variants": calc_output})

    # Cohort and taxonomy files are loaded only after prediction sealing.
    recoverability = {tuple((row["case_id"], int(row.get("source_index", 0)))): row.get("recoverability_class") for row in read_json(nf20_root / "recoverability-classes.json")["records"]}
    taxonomy = {tuple((row["case_id"], int(row.get("source_index", 0)))): row.get("cohort") for row in read_json(nf20_root / "failure-taxonomy.json")["records"]}
    cohort_output: dict[str, Any] = {"C0": c_counts["C0"], "C1": c_counts["C1"], "C2": c_counts["C2"]}
    for variant, ranked in variant_ranked.items():
        records = []
        for binding in strict_rows:
            key = (binding["case_id"], int(binding.get("source_index", 0)))
            old = rank_map(qwen_ranked[binding["case_id"]]).get(binding["candidate_key"], 101)
            new = rank_map(ranked[binding["case_id"]]).get(binding["candidate_key"], 101)
            records.append({"case_id": key[0], "source_index": key[1], "recoverability_class": recoverability.get(key), "cohort": taxonomy.get(key), "before_rank": old, "after_rank": new, "outcome": "rescued" if old > 5 and new <= 5 else "damaged" if old <= 5 and new > 5 else "unchanged"})
        cohort_output[variant] = {name: {"total": sum(row["recoverability_class"] == name for row in records), "rescued": sum(row["recoverability_class"] == name and row["outcome"] == "rescued" for row in records), "damaged": sum(row["recoverability_class"] == name and row["outcome"] == "damaged" for row in records)} for name in ("P1", "P2", "P4", "C0", "C1", "C2")}
    write_json(out_dir / "cohort-analysis.json", cohort_output)

    safe_baseline_multi = 4
    safe_baseline_calc = 5
    comparison: dict[str, Any] = {}
    for variant in variant_ranked:
        gain = strict_output[variant]["@5"]["hits"] - strict_output["qwen"]["@5"]["hits"]
        damage = movements[variant]["damaged"]
        multi_all = multi_output[variant]["@5"]["all"]
        calc_all = calc_output[variant]["@5"]["all_slots"]
        semantic_regression = semantic_output[variant]["@5"]["hits"] < semantic_output["qwen"]["@5"]["hits"]
        safe = not semantic_regression and multi_all >= safe_baseline_multi and calc_all >= safe_baseline_calc
        comparison[variant] = {"strict_r5": strict_output[variant]["@5"], "gain_sources": gain, "gain_pp": gain / len(strict_rows) * 100, "rescued": movements[variant]["rescued"], "damaged": damage, "net": movements[variant]["net"], "semantic_r5": semantic_output[variant]["@5"], "multi_all_at_5": multi_all, "calculation_all_slots_at_5": calc_all, "safe_metrics": safe, "material": gain >= 5 and damage <= 2 and safe, "marginal": 2 <= gain <= 4 and damage < movements[variant]["rescued"] and safe}
    write_json(out_dir / "variant-comparison.json", comparison)

    material = [name for name, row in comparison.items() if row["material"]]
    marginal = [name for name, row in comparison.items() if row["marginal"]]
    if material:
        selected = sorted(material, key=lambda name: (-comparison[name]["net"], comparison[name]["damaged"], 0 if name == "plrf_v1" else 1))[0]
        effectiveness: bool | str = True
        next_gate = "nf_opt_21_r1_top10_listwise_selector"
    elif marginal:
        selected = sorted(marginal, key=lambda name: (-comparison[name]["net"], comparison[name]["damaged"], 0 if name == "plrf_v1" else 1))[0]
        effectiveness = "marginal"
        next_gate = "nf_opt_21_r1_top10_listwise_selector"
    else:
        selected = None
        effectiveness = False
        next_gate = "nf_opt_21_r1_r1_listwise_selector_consideration"
    invariant = all(set(top_keys(qwen_ranked[case_id], 10)) == set(top_keys(variant_ranked[variant][case_id], 10)) and set(top_keys(qwen_ranked[case_id], 20)) == set(top_keys(variant_ranked[variant][case_id], 20)) and set(top_keys(qwen_ranked[case_id], 50)) == set(top_keys(variant_ranked[variant][case_id], 50)) and set(top_keys(qwen_ranked[case_id], 100)) == set(top_keys(variant_ranked[variant][case_id], 100)) for variant in variant_ranked for case_id in case_ids)
    decision = {
        "gate": "NF-OPT-21-R0", "base_commit": BASE_COMMIT, "evaluation_role": "development_shadow_late_fusion", "fresh_blind_evaluation": False,
        "model_execution": False, "retrieval_rerun": False, "candidate_generation": False, "training": False, "strict_sources": len(strict_rows),
        "qwen_strict_r5_hits": strict_output["qwen"]["@5"]["hits"], "lrrf_strict_r5_hits": strict_output["lrrf_v1"]["@5"]["hits"], "plrf_strict_r5_hits": strict_output["plrf_v1"]["@5"]["hits"],
        "lrrf_rescued": movements["lrrf_v1"]["rescued"], "lrrf_damaged": movements["lrrf_v1"]["damaged"], "lrrf_net": movements["lrrf_v1"]["net"],
        "plrf_rescued": movements["plrf_v1"]["rescued"], "plrf_damaged": movements["plrf_v1"]["damaged"], "plrf_net": movements["plrf_v1"]["net"],
        "qwen_semantic_r5_hits": semantic_output["qwen"]["@5"]["hits"], "lrrf_semantic_r5_hits": semantic_output["lrrf_v1"]["@5"]["hits"], "plrf_semantic_r5_hits": semantic_output["plrf_v1"]["@5"]["hits"],
        "multi_evidence_all_at_5_before": safe_baseline_multi, "multi_evidence_all_at_5_lrrf": multi_output["lrrf_v1"]["@5"]["all"], "multi_evidence_all_at_5_plrf": multi_output["plrf_v1"]["@5"]["all"],
        "calculation_all_slots_at_5_before": safe_baseline_calc, "calculation_all_slots_at_5_lrrf": calc_output["lrrf_v1"]["@5"]["all_slots"], "calculation_all_slots_at_5_plrf": calc_output["plrf_v1"]["@5"]["all_slots"],
        "bm25_correct_qwen_wrong_cases": len(reversal_cases), "lrrf_recovered_from_15": reversal_output["lrrf_v1"]["recovered"], "plrf_recovered_from_15": reversal_output["plrf_v1"]["recovered"],
        "top10_candidate_set_invariant": invariant, "top20_candidate_set_invariant": invariant, "top50_candidate_set_invariant": invariant, "top100_candidate_set_invariant": invariant,
        "late_fusion_materially_effective": effectiveness, "selected_variant": selected, "development_shadow_result": True, "production_switch_allowed": False, "next_gate": next_gate,
    }
    write_json(out_dir / "decision.json", decision)
    (out_dir / "README.md").write_text(
        "# NF-OPT-21 R0 — Qwen/BM25 Top10 Late-Fusion Shadow Test\n\n"
        "This is a development-shadow, Gold-blind prediction audit.  LRRF-V1 "
        "and PLRF-V1 consume only the sealed Qwen Top10 and frozen BM25 ranks; "
        "they do not run retrieval, models, training, or production changes. "
        "Both prediction files are sealed before post-seal strict/semantic "
        "diagnostics are loaded.\n\n"
        f"RRF_K = {RRF_K}\n\n"
        f"Decision: `{decision['late_fusion_materially_effective']}`; "
        f"selected variant: `{decision['selected_variant']}`; "
        f"next gate: `{decision['next_gate']}`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact_dir": str(out_dir), "decision": decision}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
