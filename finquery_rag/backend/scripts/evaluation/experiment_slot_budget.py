"""Experiment 2: does the planner's slot-query set beat the raw question when
the total number of retrieved candidates is held equal?

R0 measured 96.0% for slot queries against 76.0% for the raw question, but those
were per-lane coverage statistics over a union of *every* query variant.  More
queries is more draws, and more draws is more coverage almost by construction.
Quoting that as a retrieval improvement would not survive the first question
about query budget, so this experiment holds the budget fixed and re-asks.

**The budget is retrieved candidates, not returned ones.**  An earlier version of
this file capped the *output* list and swept that cap, which measures nothing:
with `K <= 20` and a cap of `B >= 20`, every arm is scored on `fused[:20]` and
the cap cannot bind.  What can be held equal, and what the question is actually
about, is how much of the corpus each arm is allowed to look at:

  raw         1 query   x 4 lanes x depth 100   -> shipped; the control
  slot_full   N queries x 4 lanes x depth 100   -> the same, with more compute
  slot_budget N queries x 4 lanes x depth 100/N -> the same total retrieved

`slot_budget` is the arm the question needs.  It issues N times as many searches
and takes 1/N the depth in each, so it looks at exactly as many candidates as the
raw arm does.  If slot queries still win there, the gain is from *asking better
queries* and not from looking at more.  `slot_full` is reported beside it because
the difference between the two is the price of the budget.

**How the arms stay comparable.**  Every arm fuses exactly four ranked lists, one
per lane, because the fusion unit is the lane; changing it would confound the
query question with a fusion question.  For a multi-query arm, a lane's list is
the candidates that lane returned for *any* variant, ordered by the best rank the
variant set gave them.  So an arm that searches more queries gets a better list
per lane, not more lists -- which is the mechanism under test.

  .venv/bin/python scripts/evaluation/experiment_slot_budget.py \\
      --out-dir /disk/qh/nano-finrag/artifacts/evaluation/e2-slot-budget
"""

from __future__ import annotations

import argparse
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

#: The shipped `lane_k`.  The raw arm uses it unchanged, so at this depth the raw
#: arm *is* the shipped hybrid -- which is what makes the control meaningful.
LANE_DEPTH = 100

RRF_K = 60

ARMS: tuple[tuple[str, str], ...] = (
    ("raw", "1 query, depth 100 (shipped)"),
    ("slot_full", "N queries, depth 100 (more compute)"),
    ("slot_budget", "N queries, depth 100/N (equal retrieved)"),
)

#: Reproduced from the published table.  The raw arm must match it exactly.
PUBLISHED_RRF = {5: 0.507, 10: 0.580, 20: 0.653}


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _lane_best_rank(
    reader: Any, lane: str, queries: Sequence[str], depth: int
) -> list[str]:
    """One lane's ranking for a query *set*: best rank across the variants.

    `order` is first-seen order and Python's sort is stable, so ordering by the
    best rank alone leaves ties in variant order -- deterministic, and without
    depending on set iteration.
    """
    best: dict[str, int] = {}
    order: list[str] = []
    for query in queries:
        for rank, hit in enumerate(
            reader.search(lane, query, allowed_candidate_keys=None, k=depth), 1
        ):
            key = hit.candidate_key
            if not key:
                continue
            if key not in best:
                best[key] = rank
                order.append(key)
            elif rank < best[key]:
                best[key] = rank
    return sorted(order, key=lambda key: best[key])


def _fuse(per_lane: Mapping[str, Sequence[str]]) -> list[str]:
    """RRF over the four lanes -- the shipped rule, unchanged for every arm."""
    ranks = {lane: {key: position for position, key in enumerate(keys, 1)}
             for lane, keys in per_lane.items()}
    candidates = _dedupe([key for lane in bench.LANES for key in per_lane.get(lane, ())])
    return sorted(
        candidates,
        key=lambda key: (-sum(1.0 / (RRF_K + ranks[lane][key])
                               for lane in bench.LANES if key in ranks[lane]), key))


def _macro_recall(gold: Mapping[str, Sequence[str]],
                  ranked: Mapping[str, Sequence[str]], arm: str, k: int) -> float:
    values = []
    for case_id, gold_keys in gold.items():
        head = set(ranked[case_id][arm][:k])
        values.append(sum(1 for key in gold_keys if key in head) / len(gold_keys))
    return sum(values) / len(values) if values else 0.0


def _mrr(gold: Mapping[str, Sequence[str]],
         ranked: Mapping[str, Sequence[str]], arm: str) -> float:
    values = []
    for case_id, gold_keys in gold.items():
        best = 0.0
        for position, key in enumerate(ranked[case_id][arm], 1):
            if key in gold_keys:
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
    print("EXPERIMENT 2 -- QUERY FORMULATION AT AN EQUAL RETRIEVAL BUDGET")
    print("=" * 78)
    print(f"  lane depth (raw, slot_full)  {LANE_DEPTH}")
    print(f"  fusion                       RRF k={RRF_K}, four lanes, every arm")
    print()

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

    print("  retrieving ...", flush=True)
    started = time.perf_counter()
    ranked: dict[str, dict[str, list[str]]] = {}
    gold_by_case: dict[str, list[str]] = {}
    searches = {arm: 0 for arm, _ in ARMS}
    retrieved = {arm: 0 for arm, _ in ARMS}
    variants_seen: list[int] = []

    for position, (question, gold_keys) in enumerate(scorable, 1):
        queries = build_all_queries(build_query_plan(question["question"], ()))
        raw_query = (queries["raw_question"][0] if queries.get("raw_question")
                     else question["question"])
        slot_queries: list[str] = []
        for variants in (queries.get("slots") or {}).values():
            for variant in variants:
                if variant and variant not in slot_queries:
                    slot_queries.append(variant)
        if not slot_queries:
            slot_queries = [raw_query]
        variants_seen.append(len(slot_queries))
        n = len(slot_queries)

        arm_specs = {
            "raw": ([raw_query], LANE_DEPTH),
            "slot_full": (slot_queries, LANE_DEPTH),
            "slot_budget": (slot_queries, max(1, LANE_DEPTH // n)),
        }
        per_case: dict[str, list[str]] = {}
        for arm, (query_set, depth) in arm_specs.items():
            searches[arm] += len(query_set) * len(bench.LANES)
            retrieved[arm] += len(query_set) * len(bench.LANES) * depth
            per_lane = {lane: _lane_best_rank(reader, lane, query_set, depth)
                        for lane in bench.LANES}
            per_case[arm] = _fuse(per_lane)
        ranked[question["id"]] = per_case
        gold_by_case[question["id"]] = gold_keys
        if position % 20 == 0 or position == len(scorable):
            print(f"    [{position:>3}/{len(scorable)}] "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    mean_variants = sum(variants_seen) / len(variants_seen)
    print(f"  slot variants per case: mean {mean_variants:.2f} "
          f"min {min(variants_seen)} max {max(variants_seen)}")
    print()

    # -- control -------------------------------------------------------------
    print("=" * 78)
    print("CONTROL -- the raw arm must be the shipped hybrid, exactly")
    print("=" * 78)
    control_ok = True
    control_rows = []
    for k in (5, 10, 20):
        observed = _macro_recall(gold_by_case, ranked, "raw", k)
        want = PUBLISHED_RRF[k]
        ok = abs(observed - want) <= 0.007
        control_ok = control_ok and ok
        control_rows.append({"k": k, "observed": round(observed, 6), "published": want,
                             "match": ok})
        print(f"    raw R@{k:<3} {observed:>8.3%}   published {want:>8.3%}   "
              f"{'MATCH' if ok else 'MISMATCH <<<'}")
    if not control_ok:
        print()
        print("    The control failed, so the raw arm is not the shipped configuration")
        print("    and no row below may be quoted.")
    print()

    # -- the comparison ------------------------------------------------------
    print("=" * 78)
    print("ARMS")
    print("=" * 78)
    print(f"    {'arm':14}{'R@5':>10}{'R@10':>10}{'R@20':>10}{'MRR':>10}")
    results: dict[str, Any] = {}
    for arm, label in ARMS:
        row = {str(k): round(_macro_recall(gold_by_case, ranked, arm, k), 6) for k in KS}
        row["mrr"] = round(_mrr(gold_by_case, ranked, arm), 6)
        row["label"] = label
        results[arm] = row
        print(f"    {arm:14}{row['5']:>10.3%}{row['10']:>10.3%}{row['20']:>10.3%}"
              f"{row['mrr']:>10.4f}")
    print()

    # -- cost ----------------------------------------------------------------
    print("  COST")
    print(f"    {'arm':14}{'lane searches':>15}{'candidates retrieved':>22}")
    for arm, _ in ARMS:
        print(f"    {arm:14}{searches[arm]:>15}{retrieved[arm]:>22}")
    same = retrieved["raw"] == retrieved["slot_budget"]
    print()
    print(f"    raw and slot_budget retrieved the same number of candidates: {same}")
    print()
    gain_full = results["slot_full"]["20"] - results["raw"]["20"]
    gain_budget = results["slot_budget"]["20"] - results["raw"]["20"]
    print(f"    slot_full   - raw  @20  {gain_full:+.3f}   (more compute allowed)")
    print(f"    slot_budget - raw  @20  {gain_budget:+.3f}   (equal retrieved)")
    print()
    print("  The number that answers the question is the second one: with the")
    print("  retrieval budget held equal, the gain is what better queries bought,")
    print("  and the first minus the second is what more looking bought.")
    print()

    report = {
        "experiment": "E2-slot-budget",
        "index_dir": str(index_dir),
        "lane_depth": LANE_DEPTH,
        "rrf_k": RRF_K,
        "scorable_cases": len(scorable),
        "control": {"rows": control_rows, "ok": control_ok,
                    "published": {str(k): v for k, v in PUBLISHED_RRF.items()}},
        "results": results,
        "cost": {"searches": searches, "retrieved": retrieved,
                 "equal_retrieved": same,
                 "mean_slot_variants": round(mean_variants, 3)},
        "gain_at_20": {"slot_full": round(gain_full, 6),
                       "slot_budget": round(gain_budget, 6)},
        "framing": ("slot_budget holds total retrieved candidates equal to raw; "
                    "slot_full is the same query set without that constraint, and "
                    "the difference between the two is what extra looking bought."),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "slot-budget.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(f"  written to {args.out_dir / 'slot-budget.json'}")
    return 0 if control_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
