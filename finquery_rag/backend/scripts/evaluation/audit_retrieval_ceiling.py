"""R0: how much retrieval headroom is there, and where is the loss?

Before changing any retrieval component, establish the ceiling.  The published
numbers are Hybrid RRF Recall@5/@10/@20 = 50.7 / 58.0 / 65.3 over 75 scorable
cases.  Those were produced with `lane_k=100` and a query that is the *raw
question only*.  Two very different explanations fit those numbers, and this
script separates them:

  1. **The cut.**  If gold sits at rank 60 in a lane, the fix is a deeper pool
     or a better fusion -- not a better retriever.  Measured by sweeping K far
     past 20 over the *same* one-query ranking.

  2. **The query.**  A single OR-of-all-tokens query cannot serve four different
     evidence needs in one multi-operand question.  The codebase already builds
     per-slot queries (`build_all_queries`); the benchmark path does not use
     them.  Measured by re-running identical lanes under three formulations:

       Q1 raw      the raw question                          (ships today)
       Q2 slot     the planner's per-slot queries, unioned    (exists, unwired)
       Q3 oracle   entity + metric + period, from gold        (analysis only)

     Q2 is gold-free -- it is `build_query_plan(question)`, the same call the
     benchmark makes -- so Q1->Q2 is headroom that needs *wiring*, not new
     capability.  Q3 is gold-derived and is **not a system configuration**: it
     bounds what any query-side fix could ever reach, and it is labelled ORACLE
     everywhere.  Quoting it as a result would be a fabricated number.

**The positive control runs first.**  Q1 at K=20 must reproduce the published
50.7 / 58.0 / 65.3.  If it does not, this audit is measuring something other
than the system, and nothing below it may be quoted.

Read-only: nothing here writes a store, an index, or a plan.

  .venv/bin/python scripts/evaluation/audit_retrieval_ceiling.py \\
      --out-dir /disk/qh/nano-finrag/artifacts/evaluation/r0-retrieval-ceiling
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_nf_v3_retrieval_benchmark as bench  # noqa: E402

BENCH_DIR = _BACKEND_DIR / "benchmarks" / "tv2_canonical_v1"

#: K values reported.  The first three are the published cut points; the rest
#: say whether a miss is a ranking problem or a depth problem.
KS: tuple[int, ...] = (5, 10, 20, 50, 100, 200, 500)

#: The lane_k the published numbers were produced with.
PUBLISHED_LANE_K = 100

#: Published Hybrid RRF Recall@K over 75 scorable cases.  The control.
PUBLISHED_HYBRID = {5: 0.507, 10: 0.580, 20: 0.653}

#: Tokens carrying no retrieval signal; they would inflate any overlap measure.
_STOP = frozenset(
    """a an the of for in on at to and or is was were be been what which how much did
    does do according filing reported report per its their this that these those with
    from by as if than then there here total net""".split()
)

FORMULATIONS: tuple[tuple[str, str], ...] = (
    ("q1_raw", "Q1 raw question           (ships today)"),
    ("q2_slot", "Q2 planner slot queries   (exists, unwired)"),
    ("q3_oracle", "Q3 ORACLE coordinate query (analysis only)"),
)


def _content_tokens(text: str) -> set[str]:
    from src.pdf_retrieval_v4.candidate_view_index import _tokenize

    return {t for t in _tokenize(str(text or "")) if t not in _STOP and len(t) > 1}


def _overlap(question: str, text: str) -> tuple[int, int, float]:
    """(shared content tokens, gold content tokens, fraction covered)."""
    q = _content_tokens(question)
    g = _content_tokens(text)
    if not g:
        return 0, 0, 0.0
    shared = len(q & g)
    return shared, len(g), shared / len(g)


# ---------------------------------------------------------------------------
# Rank capture
# ---------------------------------------------------------------------------


def _best_rank_of(
    reader: Any,
    lane: str,
    queries: Sequence[str],
    watch: set[str],
    depth: int,
) -> tuple[dict[str, int], bool]:
    """Best (lowest) rank each watched key reaches in this lane, across variants.

    Returns (ranks, saw_everything).  `saw_everything` is False when a search
    was truncated by the depth cap, which is what makes *absence* from a lane
    an observation rather than an artefact of the cap.
    """
    merged: dict[str, int] = {}
    truncated = False
    for query in queries:
        hits = reader.search(lane, query, allowed_candidate_keys=None, k=depth)
        if len(hits) >= depth:
            truncated = True
        for rank, hit in enumerate(hits, 1):
            key = hit.candidate_key
            if key in watch and (key not in merged or rank < merged[key]):
                merged[key] = rank
    return merged, truncated


def _fuse(
    lane_lists: Mapping[str, Sequence[str]],
    k: int,
) -> list[str]:
    """RRF over lane ranked lists, the way `fuse_candidate_hits` does it."""
    from src.pdf_retrieval_v4.candidate_rrf import fuse_candidate_hits
    from src.pdf_retrieval_v4.candidate_view_index import CandidateSearchHit

    lane_hits = {
        lane: [
            CandidateSearchHit(candidate_key=key, view_id=key, lane=lane,
                               bm25_rank=rank, dense_rank=None,
                               bm25_score=None, dense_score=None)
            for rank, key in enumerate(list(keys)[:PUBLISHED_LANE_K], 1)
        ]
        for lane, keys in lane_lists.items()
    }
    return [hit.candidate_key for hit in fuse_candidate_hits(lane_hits, rrf_k=60)][:k]


# ---------------------------------------------------------------------------
# Query formulations
# ---------------------------------------------------------------------------


def build_formulations(question: str) -> dict[str, list[str]]:
    """Q1 and Q2 for one question.  Takes no gold, so it cannot leak any.

    Q3 is built by the caller from gold and is appended there.
    """
    from src.pdf_retrieval_v4.candidate_query_builder import build_all_queries
    from src.pdf_retrieval_v4.planner import build_query_plan

    queries = build_all_queries(build_query_plan(question, ()))
    raw = queries["raw_question"][0] if queries.get("raw_question") else question
    slot_queries: list[str] = []
    for variants in (queries.get("slots") or {}).values():
        for variant in variants:
            if variant and variant not in slot_queries:
                slot_queries.append(variant)
    return {"q1_raw": [raw], "q2_slot": slot_queries or [raw]}


def oracle_query(gold: Mapping[str, Any], question_meta: Mapping[str, Any]) -> str | None:
    """entity + metric + period, from gold.  ORACLE -- analysis only."""
    entity = question_meta.get("entity") or question_meta.get("document_id")
    parts = [str(p) for p in (entity, gold.get("metric"), gold.get("period")) if p]
    return " ".join(parts) if len(parts) >= 2 else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _recall_stats(
    scorable: Sequence[tuple[dict[str, Any], list[str]]],
    ranks: Mapping[str, Any],
    formulation: str,
    k: int,
    stratum: str | None = None,
) -> dict[str, float]:
    """Union-across-lanes recall at K, in both aggregations.

    ``macro`` is the mean over cases of each case's own partial recall, and it
    is the benchmark's definition (`aggregate` in the runner does `_mean` over
    per-case partials).  ``micro`` pools gold ids instead, so a two-operand case
    counts twice.  They differ here and the difference is not cosmetic: the
    published control only matches on ``macro``, and reporting ``micro`` under a
    published column name would inflate every number by ~2 points.

    ``complete`` is the fraction of cases where *every* gold id is present --
    the same per-case view the benchmark reports beside the partial one.
    """
    macro_sum = micro_hits = total = complete = cases = 0
    for question, gold_keys in scorable:
        if stratum is not None and question.get("stratum") != stratum:
            continue
        per_lane = ranks[question["id"]].get(formulation) or {}
        best: dict[str, int] = {}
        for lane in bench.LANES:
            for key, rank in (per_lane.get(lane) or {}).items():
                if key not in best or rank < best[key]:
                    best[key] = rank
        present = sum(1 for key in gold_keys if best.get(key, 10**9) <= k)
        macro_sum += present / len(gold_keys)
        micro_hits += present
        total += len(gold_keys)
        cases += 1
        complete += 1 if present == len(gold_keys) else 0
    return {
        "macro": macro_sum / cases if cases else 0.0,
        "micro": micro_hits / total if total else 0.0,
        "complete": complete / cases if cases else 0.0,
        "gold_ids": total,
        "cases": cases,
    }


def _sweep(
    scorable: Sequence[tuple[dict[str, Any], list[str]]],
    ranks: Mapping[str, Any],
    formulation: str,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for stratum in ("ALL", "factual_lookup", "arithmetic_calculation"):
        key = None if stratum == "ALL" else stratum
        out[stratum] = {
            str(k): _recall_stats(scorable, ranks, formulation, k, key) for k in KS
        }
    return out


def _print_sweep(rows: Mapping[str, Any], label: str) -> None:
    print(f"  {label}")
    print("    " + " ".join(f"@{k:<7}" for k in KS))
    for stratum in ("ALL", "factual_lookup", "arithmetic_calculation"):
        entry = rows.get(stratum)
        if not entry:
            continue
        print(f"  {stratum:26}"
              + " ".join(f"{entry[str(k)]['macro']:>7.1%}" for k in KS))
        print(f"  {'  (case-complete)':26}"
              + " ".join(f"{entry[str(k)]['complete']:>7.1%}" for k in KS))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=BENCH_DIR / "canonical-eval-v1.jsonl")
    parser.add_argument("--gold-evidence", type=Path,
                        default=BENCH_DIR / "gold-evidence-v1.jsonl")
    parser.add_argument(
        "--v2-fact-store", type=Path,
        default=Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl"),
    )
    parser.add_argument(
        "--ixbrl-fact-store", type=Path,
        default=Path(
            "/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts-ixbrl-v1.jsonl"),
    )
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--depth-cap", type=int, default=0,
                        help="Cap the deep search. 0 = the largest lane, i.e. full depth.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    import run_p1_2_dual_track_benchmark as runner

    runner._load_deployment_env()

    from run_tv2_canonical_benchmark import FactStoreGroundingIndex  # noqa: E402

    from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader
    from src.runtime.trusted_v2_production import _path_env

    environ = dict(os.environ)
    index_dir = (
        args.index_dir if args.index_dir is not None
        else _path_env(environ, "TRUSTED_V2_R4_INDEX_DIR", directory=True)
    )

    questions = bench._load_jsonl(args.eval_set)
    gold_by_id = {row["id"]: row for row in bench._load_jsonl(args.gold_evidence)}
    if args.limit:
        questions = questions[: args.limit]

    reader = CandidateViewIndexReader(index_dir)
    index_candidate_keys = {key for (_lane, key) in reader._candidate_to_view}
    v2_index = FactStoreGroundingIndex(args.v2_fact_store)
    ixbrl_keys = bench.load_ixbrl_candidate_keys(args.ixbrl_fact_store)
    resolver = bench.GoldResolver(v2_index, ixbrl_keys, index_candidate_keys)

    lane_sizes = {lane: len(reader._lane_view_ids.get(lane, ())) for lane in bench.LANES}
    depth = args.depth_cap or max(lane_sizes.values())

    print("=" * 78)
    print("R0 RETRIEVAL CEILING AUDIT")
    print("=" * 78)
    print(f"  index dir         {index_dir}")
    print(f"  index candidates  {len(index_candidate_keys)}")
    for lane in bench.LANES:
        print(f"  lane {lane:28} {lane_sizes[lane]:>7} views")
    print(f"  search depth      {depth}"
          f"{'' if args.depth_cap else '   (full -- largest lane)'}")
    print()

    # -- the scorable slice --------------------------------------------------
    scorable: list[tuple[dict[str, Any], list[str]]] = []
    skipped: collections.Counter = collections.Counter()
    for question in questions:
        case_id = question["id"]
        gold = gold_by_id.get(case_id, {})
        raw_ids = gold.get("fact_ids") or []
        if not raw_ids:
            skipped["no gold (abstention)"] += 1
            continue
        keys: list[str] = []
        for raw in raw_ids:
            res = resolver.resolve(str(raw), document_id=question.get("document_id"),
                                   metric=gold.get("metric"), period=gold.get("period"))
            if res.indexed and res.candidate_key:
                keys.append(res.candidate_key)
        if len(keys) != len(raw_ids):
            skipped[f"gold not in index ({question.get('stratum')})"] += 1
            continue
        scorable.append((question, list(dict.fromkeys(keys))))

    print(f"  scorable cases    {len(scorable)}")
    for reason, count in skipped.most_common():
        print(f"  skipped           {count:>4}  {reason}")
    print()

    # -- retrieval -----------------------------------------------------------
    print(f"  retrieving {len(scorable)} cases x 3 formulations x 4 lanes at depth "
          f"{depth} ...", flush=True)
    started = time.perf_counter()
    ranks: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    lane_lists: dict[str, dict[str, list[str]]] = {}
    truncated_lanes: set[tuple[str, str]] = set()
    overlap_rows: list[dict[str, Any]] = []
    oracle_missing: list[str] = []

    for position, (question, gold_keys) in enumerate(scorable, 1):
        case_id = question["id"]
        watch = set(gold_keys)
        formulations = build_formulations(question["question"])
        oracle = oracle_query(gold_by_id.get(case_id, {}), question)
        if oracle:
            formulations["q3_oracle"] = [oracle]
        else:
            oracle_missing.append(case_id)

        per_case: dict[str, dict[str, dict[str, int]]] = {}
        for name, queries in formulations.items():
            lane_ranks: dict[str, dict[str, int]] = {}
            for lane in bench.LANES:
                merged, truncated = _best_rank_of(reader, lane, queries, watch, depth)
                lane_ranks[lane] = merged
                if truncated:
                    truncated_lanes.add((name, lane))
            per_case[name] = lane_ranks
        ranks[case_id] = per_case

        # Q1's lane orderings, truncated exactly as the shipped path truncates
        # them, so the control fuses what the benchmark fused.
        lane_lists[case_id] = {
            lane: [h.candidate_key for h in reader.search(
                lane, formulations["q1_raw"][0], allowed_candidate_keys=None,
                k=PUBLISHED_LANE_K)]
            for lane in bench.LANES
        }

        # Token overlap between the question and the gold's own indexed text.
        for key in gold_keys:
            view = reader.view_for_candidate("candidate_structured_bm25", key) or {}
            text = str(view.get("retrieval_text") or "")
            shared, total, frac = _overlap(question["question"], text)
            overlap_rows.append({
                "case_id": case_id, "stratum": question.get("stratum"), "key": key,
                "shared": shared, "gold_tokens": total, "coverage": round(frac, 4),
                "text": re.sub(r"\s+", " ", text)[:180],
            })

        if position % 15 == 0 or position == len(scorable):
            print(f"    [{position:>3}/{len(scorable)}] {time.perf_counter() - started:.0f}s",
                  flush=True)
    print()

    # -- positive control ----------------------------------------------------
    print("=" * 78)
    print("POSITIVE CONTROL -- Q1 hybrid RRF at the published cut points")
    print("=" * 78)
    control_ok = True
    control_rows = []
    for k in (5, 10, 20):
        # Macro, per case -- the benchmark's own `_mean(partials)`.  Computing
        # this over pooled gold ids instead gives 49.1/58.2/63.6, which reads
        # as a 2-point failure to reproduce a system that reproduces exactly.
        per_case: list[float] = []
        for question, gold_keys in scorable:
            head = set(_fuse(lane_lists[question["id"]], k))
            per_case.append(sum(1 for key in gold_keys if key in head) / len(gold_keys))
        observed = sum(per_case) / len(per_case) if per_case else 0.0
        want = PUBLISHED_HYBRID[k]
        ok = abs(observed - want) <= 0.007
        control_ok = control_ok and ok
        control_rows.append({"k": k, "observed": round(observed, 6), "published": want,
                             "match": ok})
        print(f"    hybrid RRF R@{k:<3} {observed:>8.3%}   published {want:>8.3%}   "
              f"{'MATCH' if ok else 'MISMATCH <<<'}")
    if not control_ok:
        print()
        print("    The control failed.  Everything below measures something other than")
        print("    the system that produced the published numbers, and must not be quoted.")
    print()

    # -- depth sweep ---------------------------------------------------------
    print("=" * 78)
    print("DEPTH -- the same Q1 ranking, cut at increasing K (union of 4 lanes)")
    print("=" * 78)
    depth_rows = _sweep(scorable, ranks, "q1_raw")
    _print_sweep(depth_rows, "Q1 raw question")
    print()

    # -- query formulation ---------------------------------------------------
    print("=" * 78)
    print("QUERY FORMULATION -- identical lanes, full depth, no cut")
    print("=" * 78)
    formulation_rows: dict[str, Any] = {}
    for name, label in FORMULATIONS:
        rows = _sweep(scorable, ranks, name)
        formulation_rows[name] = rows
        _print_sweep(rows, label)
        print()

    # -- lane ceiling --------------------------------------------------------
    print("=" * 78)
    print("LANE RECALL -- Q1, each lane alone, a real single-list score")
    print("=" * 78)
    lane_ceiling: dict[str, dict[str, float]] = {}
    for lane in bench.LANES:
        per_k = {}
        for k in KS:
            per_case = []
            for question, gold_keys in scorable:
                r = ranks[question["id"]]["q1_raw"].get(lane) or {}
                per_case.append(
                    sum(1 for key in gold_keys if r.get(key, 10**9) <= k) / len(gold_keys))
            per_k[str(k)] = round(sum(per_case) / len(per_case), 6) if per_case else 0.0
        lane_ceiling[lane] = per_k
        row = "  ".join(f"@{k} {per_k[str(k)]:>7.1%}" for k in (5, 20, 200, 500))
        print(f"    {lane:28} {row}")
    union_coverage_per_k = {
        str(k): round(_recall_stats(scorable, ranks, "q1_raw", k)["macro"], 6) for k in KS
    }
    print()
    print("    PER-LANE COVERAGE -- `gold is in some lane's top K`. This is NOT a")
    print("    single ranked list and NOT a score any one fusion could reach:")
    print("    four separate top-K lists cannot be concatenated into one top-K.")
    print("      " + "  ".join(f"@{k} {union_coverage_per_k[str(k)]:>7.1%}" for k in KS))
    print()

    # -- miss taxonomy -------------------------------------------------------
    print("=" * 78)
    print("MISS TAXONOMY -- Q1 gold ids absent from every lane's top-20")
    print("=" * 78)
    taxonomy = _taxonomy(scorable, ranks)
    for reason, count in taxonomy["counts"].most_common():
        print(f"    {count:>4}  {reason}")
    print()
    print(f"    of {taxonomy['missed']} missed gold ids:")
    print(f"      an ORACLE query reaches top-20   {taxonomy['oracle_reachable']}")
    print(f"      an ORACLE query does not         {taxonomy['oracle_unreachable']}")
    print()

    # -- token overlap -------------------------------------------------------
    print("=" * 78)
    print("TOKEN OVERLAP -- question content tokens present in gold's indexed text")
    print("=" * 78)
    buckets: collections.Counter = collections.Counter()
    for row in overlap_rows:
        frac = row["coverage"]
        buckets["0%     (no shared content token)" if frac == 0
                else "<25%" if frac < 0.25
                else "25-50%" if frac < 0.5
                else "50-75%" if frac < 0.75
                else ">=75%"] += 1
    for name in ("0%     (no shared content token)", "<25%", "25-50%", "50-75%", ">=75%"):
        print(f"    {buckets.get(name, 0):>4}  {name}")
    print()

    report = {
        "audit": "R0-retrieval-ceiling",
        "index_dir": str(index_dir),
        "index_candidates": len(index_candidate_keys),
        "lane_sizes": lane_sizes,
        "search_depth": depth,
        "truncated_search": sorted(f"{name}:{lane}" for name, lane in truncated_lanes),
        "scorable_cases": len(scorable),
        "skipped": dict(skipped),
        "control": {"rows": control_rows, "ok": control_ok,
                    "published": {str(k): v for k, v in PUBLISHED_HYBRID.items()}},
        "depth_sweep_q1": depth_rows,
        "formulations": formulation_rows,
        "lane_recall_q1": lane_ceiling,
        "union_coverage_q1": union_coverage_per_k,
        "taxonomy": taxonomy,
        "overlap_buckets": dict(buckets),
        "oracle_query_missing": oracle_missing,
        "overlap_rows": overlap_rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "retrieval-ceiling.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(f"  written to {args.out_dir / 'retrieval-ceiling.json'}")
    return 0 if control_ok else 1


def _taxonomy(
    scorable: Sequence[tuple[dict[str, Any], list[str]]],
    ranks: Mapping[str, Any],
) -> dict[str, Any]:
    """Why each gold id is missing from every lane's top-20 at Q1.

    Classified per lane rather than on a fused list: a key absent from every
    lane's top-20 is absent from any fusion of them, so this is the permissive
    reading and it makes the reason about ranking, not about RRF's ordering.
    """
    counts: collections.Counter = collections.Counter()
    examples: list[dict[str, Any]] = []
    missed = oracle_reachable = oracle_unreachable = 0

    for question, gold_keys in scorable:
        case_id = question["id"]
        q1 = ranks[case_id]["q1_raw"]
        oracle = ranks[case_id].get("q3_oracle") or {}
        for key in gold_keys:
            lane_rank = {lane: (q1.get(lane) or {}).get(key) for lane in bench.LANES}
            if any(rank and rank <= 20 for rank in lane_rank.values()):
                continue
            missed += 1
            ranked = {lane: rank for lane, rank in lane_rank.items() if rank}
            if ranked:
                best = min(ranked.values())
                reason = "BELOW_CUT -- ranked, but past 20 in every lane"
            else:
                best = None
                reason = "ABSENT_FROM_ALL_LANE_RANKINGS -- indexed, returned by no lane"
            reaches = any(
                ((oracle.get(lane) or {}).get(key) or 10**9) <= 20 for lane in bench.LANES)
            if reaches:
                oracle_reachable += 1
                reason += "  [oracle-closable]"
            else:
                oracle_unreachable += 1
                reason += "  [oracle cannot close it]"
            counts[reason] += 1
            examples.append({"case_id": case_id, "stratum": question.get("stratum"),
                             "key": key, "best_rank": best,
                             "oracle_reachable": reaches})

    return {"missed": missed, "oracle_reachable": oracle_reachable,
            "oracle_unreachable": oracle_unreachable, "counts": counts,
            "examples": examples}


if __name__ == "__main__":
    raise SystemExit(main())
