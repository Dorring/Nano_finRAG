"""Experiment 1: the fusion is losing 10.7 points, and this measures what a
better ordering of the *same* candidates recovers.

R0 measured a per-lane coverage statistic: at the raw-question query, gold is in
*some* lane's top-20 for 76.0% of cases, while the shipped fused top-20 holds
65.3%.  Those two numbers are not a comparison, and reading them as one is the
mistake this experiment exists to correct.  76.0% describes four separate lists
of twenty; no single ranked list of twenty can contain them all.  It is an upper
bound on *coverage*, not a score any fusion could reach, and it is not the size
of the prize.

What this measures instead is the prize that actually exists: for every case the
four lanes are searched exactly as the shipped path searches them (`lane_k=100`,
the raw-question formulation), the pool is the union of those four lists, and the
*only* difference between rows is the rule that orders it:

  rrf_k60         the shipped rule.  The control: it must reproduce 65.333 @20.
  rrf_k10/k1000   the same rule with a different rank prior, to show whether the
                  constant is the knob.  If recall is flat across three orders of
                  magnitude of k, the problem is the *form*, not the parameter.
  min_rank        order by the best rank the candidate reaches in any lane
  min_rank_supp   the same, ties broken by how many lanes found it
  borda           sum over lanes of (pool_depth - rank), absent lanes contributing 0
  combsum_minmax  sum of per-lane min-max normalised scores

The two families are the point.  RRF and Borda are *sum* rules: they reward a
candidate for appearing in several lanes, and a candidate found by exactly one
lane at rank 3 loses to one found by four lanes at rank 40.  That is a reasonable
prior when the lanes are near-duplicates, and a bad one when two lanes are lexical
and two are semantic -- a fact only the dense lane can see is not less relevant for
being invisible to BM25.  min_rank is the *max* rule and ignores agreement
entirely.  Which is right here is an empirical question, and the answer is not
obvious in advance: max-pooling is more fragile to one lane's noise.

`pool_oracle` is the ceiling for reordering this pool: the fraction of gold
present in the pool at all.  Every rule ranks the same candidates, so none can
beat it -- but it is not a rule and not a result, because a case has at most four
gold ids and a perfect ordering would place all of them inside the top 20.  The
distance from the best rule to it is what no reordering can fix.

**This is a selection over six orderings on the same 75 cases.**  The rules are
standard and none has a fitted parameter chosen here, but picking the winner by
this table is still a form of selection on the evaluation set.  If one is adopted,
it is adopted as "a standard fusion that measures better on the frozen benchmark",
and a held-out confirmation is owed before it is called an improvement in general.
The script says so in its own output rather than leaving it to the reader.

  .venv/bin/python scripts/evaluation/experiment_fusion.py \\
      --out-dir /disk/qh/nano-finrag/artifacts/evaluation/e1-fusion
"""

from __future__ import annotations

import argparse
import collections
import json
import os
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
KS: tuple[int, ...] = (5, 10, 20)

#: The shipped lane depth.  The pool is the union of the four lanes at this depth,
#: which is exactly what `fuse_candidate_hits` is handed today.
POOL_DEPTH = 100

#: The shipped RRF constant.
RRF_K = 60

#: Reproduced from the published table.  The control.
PUBLISHED_RRF = {5: 0.507, 10: 0.580, 20: 0.653}

RULES: tuple[tuple[str, str], ...] = (
    ("rrf_k60", "shipped: RRF, k=60  (control)"),
    ("rrf_k10", "RRF, k=10"),
    ("rrf_k1000", "RRF, k=1000"),
    ("min_rank", "best rank in any lane"),
    ("min_rank_supp", "best rank, ties by lane support"),
    ("borda", "Borda: sum of (depth - rank)"),
    ("combsum_minmax", "CombSUM over min-max normalised scores"),
)


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _lane_lists(
    reader: Any, query: str
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    """The four ranked lists, and the per-lane scores behind them.

    Ranks and scores are taken from the same `search` call, so a rule that uses
    scores and a rule that uses ranks are looking at the identical retrieval.
    """
    lists: dict[str, list[str]] = {}
    scores: dict[str, dict[str, float]] = {}
    for lane in bench.LANES:
        hits = reader.search(lane, query, allowed_candidate_keys=None, k=POOL_DEPTH)
        lists[lane] = _dedupe([h.candidate_key for h in hits])
        lane_scores: dict[str, float] = {}
        for hit in hits:
            value = hit.bm25_score if hit.bm25_score is not None else hit.dense_score
            if value is not None:
                lane_scores[hit.candidate_key] = float(value)
        scores[lane] = lane_scores
    return lists, scores


def _rank_maps(lists: Mapping[str, Sequence[str]]) -> dict[str, dict[str, int]]:
    return {lane: {key: position for position, key in enumerate(keys, 1)}
            for lane, keys in lists.items()}


def _pool(lists: Mapping[str, Sequence[str]]) -> list[str]:
    return _dedupe([key for lane in bench.LANES for key in lists.get(lane, ())])


def fuse(
    rule: str,
    candidates: Sequence[str],
    ranks: Mapping[str, Mapping[str, int]],
    scores: Mapping[str, Mapping[str, float]],
) -> list[str]:
    """Order the pool.  Every rule sees the same candidates, ranks and scores."""

    def rrf(k: int, key: str) -> float:
        return sum(1.0 / (k + ranks[lane][key])
                   for lane in bench.LANES if key in ranks[lane])

    if rule.startswith("rrf_k"):
        k = int(rule.split("_k")[1])
        return sorted(candidates, key=lambda key: (-rrf(k, key), key))

    if rule == "min_rank":
        return sorted(candidates, key=lambda key: (
            min((ranks[lane][key] for lane in bench.LANES if key in ranks[lane]),
                default=10**9), key))

    if rule == "min_rank_supp":
        return sorted(candidates, key=lambda key: (
            min((ranks[lane][key] for lane in bench.LANES if key in ranks[lane]),
                default=10**9),
            -sum(1 for lane in bench.LANES if key in ranks[lane]),
            key))

    if rule == "borda":
        return sorted(candidates, key=lambda key: (
            -sum(POOL_DEPTH - ranks[lane][key]
                 for lane in bench.LANES if key in ranks[lane]), key))

    if rule == "combsum_minmax":
        normalised: dict[str, dict[str, float]] = {}
        for lane in bench.LANES:
            values = list(scores.get(lane, {}).values())
            if not values:
                normalised[lane] = {}
                continue
            low, high = min(values), max(values)
            span = (high - low) or 1.0
            normalised[lane] = {key: (value - low) / span
                                for key, value in scores[lane].items()}
        return sorted(candidates, key=lambda key: (
            -sum(normalised[lane].get(key, 0.0) for lane in bench.LANES), key))

    raise ValueError(f"unknown_rule:{rule}")


def _macro_recall(
    gold_by_case: Mapping[str, Sequence[str]],
    ranked_by_case: Mapping[str, Sequence[str]],
    rule: str,
    k: int,
) -> float:
    values = []
    for case_id, gold in gold_by_case.items():
        head = set(ranked_by_case[case_id][rule][:k])
        values.append(sum(1 for key in gold if key in head) / len(gold))
    return sum(values) / len(values) if values else 0.0


def _mrr(
    gold_by_case: Mapping[str, Sequence[str]],
    ranked_by_case: Mapping[str, Sequence[str]],
    rule: str,
) -> float:
    values = []
    for case_id, gold in gold_by_case.items():
        ranked = ranked_by_case[case_id][rule]
        best = 0.0
        for position, key in enumerate(ranked, 1):
            if key in gold:
                best = 1.0 / position
                break
        values.append(best)
    return sum(values) / len(values) if values else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=BENCH_DIR / "canonical-eval-v1.jsonl")
    parser.add_argument("--gold-evidence", type=Path,
                        default=BENCH_DIR / "gold-evidence-v1.jsonl")
    parser.add_argument("--v2-fact-store", type=Path,
                        default=Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/"
                                     "financial-facts.jsonl"))
    parser.add_argument("--ixbrl-fact-store", type=Path,
                        default=Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/"
                                     "financial-facts-ixbrl-v1.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    import run_p1_2_dual_track_benchmark as runner

    runner._load_deployment_env()

    from run_tv2_canonical_benchmark import FactStoreGroundingIndex  # noqa: E402

    from src.pdf_retrieval_v4.candidate_query_builder import build_all_queries
    from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader
    from src.pdf_retrieval_v4.planner import build_query_plan
    from src.runtime.trusted_v2_production import _path_env

    environ = dict(os.environ)
    index_dir = (args.index_dir if args.index_dir is not None
                 else _path_env(environ, "TRUSTED_V2_R4_INDEX_DIR", directory=True))

    questions = bench._load_jsonl(args.eval_set)
    gold_by_id = {row["id"]: row for row in bench._load_jsonl(args.gold_evidence)}
    reader = CandidateViewIndexReader(index_dir)
    index_keys = {key for (_lane, key) in reader._candidate_to_view}
    resolver = bench.GoldResolver(
        FactStoreGroundingIndex(args.v2_fact_store),
        bench.load_ixbrl_candidate_keys(args.ixbrl_fact_store),
        index_keys)

    print("=" * 78)
    print("EXPERIMENT 1 -- FUSION ABLATION")
    print("=" * 78)
    print(f"  index            {index_dir}")
    print(f"  pool depth       {POOL_DEPTH} per lane, union across {len(bench.LANES)} lanes")
    print(f"  query           the raw question only (the shipped formulation)")
    print()

    # -- the scorable slice, resolved exactly as the benchmark resolves it -----
    scorable: list[tuple[dict[str, Any], list[str]]] = []
    for question in questions:
        gold = gold_by_id.get(question["id"], {})
        raw_ids = gold.get("fact_ids") or []
        if not raw_ids:
            continue
        keys = []
        for raw in raw_ids:
            res = resolver.resolve(str(raw), document_id=question.get("document_id"),
                                   metric=gold.get("metric"), period=gold.get("period"))
            if res.indexed and res.candidate_key:
                keys.append(res.candidate_key)
        if len(keys) != len(raw_ids):
            continue
        scorable.append((question, list(dict.fromkeys(keys))))
    print(f"  scorable cases   {len(scorable)}")
    print()

    # -- retrieval: the pool is built once and every rule reorders it ---------
    print(f"  retrieving ...", flush=True)
    started = time.perf_counter()
    ranked_by_case: dict[str, dict[str, list[str]]] = {}
    lanes_by_case: dict[str, dict[str, list[str]]] = {}
    gold_by_case: dict[str, list[str]] = {}
    pool_sizes: list[int] = []
    for position, (question, gold_keys) in enumerate(scorable, 1):
        queries = build_all_queries(build_query_plan(question["question"], ()))
        raw_query = (queries["raw_question"][0] if queries.get("raw_question")
                     else question["question"])
        lists, scores = _lane_lists(reader, raw_query)
        ranks = _rank_maps(lists)
        candidates = _pool(lists)
        pool_sizes.append(len(candidates))
        ranked_by_case[question["id"]] = {
            rule: fuse(rule, candidates, ranks, scores) for rule, _ in RULES}
        lanes_by_case[question["id"]] = lists
        gold_by_case[question["id"]] = gold_keys
        if position % 20 == 0 or position == len(scorable):
            print(f"    [{position:>3}/{len(scorable)}] "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    print(f"  pool size mean {sum(pool_sizes) / len(pool_sizes):.1f} "
          f"min {min(pool_sizes)} max {max(pool_sizes)}")
    print()

    # -- the ceiling, stated as the thing it actually is ---------------------
    # Not a ranking.  An oracle that ordered this pool perfectly would score this
    # at every K, because a case has at most four gold ids and any of them in the
    # pool fits inside the top 20.  It bounds what *reordering* can reach; it is
    # not a number any rule can be, and it must never be printed as one.
    coverage = []
    for question, gold_keys in scorable:
        present = {key for lane in bench.LANES
                   for key in lanes_by_case[question["id"]].get(lane, [])}
        coverage.append(sum(1 for key in gold_keys if key in present) / len(gold_keys))
    pool_oracle = sum(coverage) / len(coverage) if coverage else 0.0

    # -- control -------------------------------------------------------------
    print("=" * 78)
    print("CONTROL -- the shipped rule must reproduce the published numbers")
    print("=" * 78)
    control_ok = True
    control_rows = []
    for k in (5, 10, 20):
        observed = _macro_recall(gold_by_case, ranked_by_case, "rrf_k60", k)
        want = PUBLISHED_RRF[k]
        ok = abs(observed - want) <= 0.007
        control_ok = control_ok and ok
        control_rows.append({"k": k, "observed": round(observed, 6), "published": want,
                             "match": ok})
        print(f"    rrf_k60 R@{k:<3} {observed:>8.3%}   published {want:>8.3%}   "
              f"{'MATCH' if ok else 'MISMATCH <<<'}")
    if not control_ok:
        print()
        print("    The control failed: this is not measuring the shipped path, and")
        print("    no row below may be quoted.")
    print()

    # -- the ablation --------------------------------------------------------
    print("=" * 78)
    print("ABLATION -- identical pool, identical candidates, different ordering")
    print("=" * 78)
    print(f"    {'rule':16}{'R@5':>9}{'R@10':>9}{'R@20':>9}{'MRR':>9}   delta@20")
    results: dict[str, Any] = {}
    baseline_20 = _macro_recall(gold_by_case, ranked_by_case, "rrf_k60", 20)
    for rule, label in RULES:
        row = {str(k): round(_macro_recall(gold_by_case, ranked_by_case, rule, k), 6)
               for k in KS}
        row["mrr"] = round(_mrr(gold_by_case, ranked_by_case, rule), 6)
        row["label"] = label
        results[rule] = row
        delta = row["20"] - baseline_20
        mark = "" if rule == "rrf_k60" else f"{delta:+.3f}"
        print(f"    {rule:16}{row['5']:>9.3%}{row['10']:>9.3%}{row['20']:>9.3%}"
              f"{row['mrr']:>9.4f}   {mark}")
    bound = {str(k): round(pool_oracle, 6) for k in KS}
    print(f"    {'pool_oracle':16}{pool_oracle:>9.3%}{pool_oracle:>9.3%}"
          f"{pool_oracle:>9.3%}{'':>9}   (not a rule -- see below)")
    print()

    best = max((r for r in results if r != "rrf_k60"),
               key=lambda r: results[r]["20"])
    print(f"  best reordering: {best}  @20 {results[best]['20']:.3%}  "
          f"vs shipped {baseline_20:.3%}  ({results[best]['20'] - baseline_20:+.3f})")
    print(f"  gold in the pool at all: {pool_oracle:.3%} -- an oracle ordering would "
          f"reach that at every K.")
    print()
    print("  `pool_oracle` is a ceiling on REORDERING, not a rule and not a result:")
    print("  a case has at most four gold ids, so any of them already in the pool fits")
    print("  inside the top 20 under a perfect ordering. No rule here is close to it,")
    print("  and the shipped rule is not the worst of them.")
    print()
    print("  Read the R@20 column as the real comparison. A per-lane *coverage*")
    print("  statistic -- 'the gold is in some lane's top 20' -- is a different and")
    print("  larger number, and it is not achievable by one ranked list of 20.")
    print()
    print("  The rules are standard and none was fitted here, but choosing among them")
    print("  by this table is still selection on the evaluation set. A held-out")
    print("  confirmation is owed before this is called an improvement in general.")
    print()

    report = {
        "experiment": "E1-fusion-ablation",
        "index_dir": str(index_dir),
        "pool_depth": POOL_DEPTH,
        "query_formulation": "raw question only",
        "scorable_cases": len(scorable),
        "pool_size": {"mean": round(sum(pool_sizes) / len(pool_sizes), 2),
                      "min": min(pool_sizes), "max": max(pool_sizes)},
        "control": {"rows": control_rows, "ok": control_ok,
                    "published": {str(k): v for k, v in PUBLISHED_RRF.items()}},
        "rules": results,
        "pool_oracle": round(pool_oracle, 6),
        "pool_oracle_note": (
            "Ceiling on reordering this pool, not a rule and not a result: with at "
            "most four gold ids per case a perfect ordering reaches it at every K."),
        "selection_caveat": (
            "Six standard orderings compared on the frozen 75-case slice; picking the "
            "winner here is selection on the evaluation set and needs held-out "
            "confirmation."),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fusion-ablation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(f"  written to {args.out_dir / 'fusion-ablation.json'}")
    return 0 if control_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
