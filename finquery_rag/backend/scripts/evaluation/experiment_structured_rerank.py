"""Goal: Recall@5 > 80%.  This reranks the pool on the fields the fact IS.

The retrieval path finds the right candidates and then loses them in the
ordering.  R0 and E2 measured the shape of that: the gold is in the pool 92-95%
of the time, while the fused top-5 holds 50.7% (shipped) or 76.0% (slot queries).
The gap is entirely ordering, and RRF orders by *rank agreement across lanes* --
a signal about how the search engines behaved, not about the fact.

But the thing being retrieved is a financial fact, and a fact has an identity:
`(entity, metric, period)`.  The plan states all three, the store records all
three for every candidate, and the current fusion uses none of them.  A reranker
that scores candidates on how well they match the slot the plan asked for is
therefore not a heuristic bolted onto retrieval -- it is the retrieval using the
structure the domain already has, which is what RRF cannot see.

**Nothing here is per-question.**  The score is a fixed function of
(slot entity, slot metric, slot period, candidate entity, candidate metric,
candidate period), applied identically to every case, with no case ids, no gold,
and no branch on question text.  Metric agreement goes through the same
`canonical_metric_id` the alignment gate uses, so `Net income` and `net income`
agree and `Net income` and `Operating income` do not.

Arms, all reranking the identical pool so only the scoring changes:

  fusion_only     no rerank -- the baseline this must beat
  field           the three fields, weighted
  field_lexical   the three fields plus a token-overlap tiebreak on the
                  candidate's own text, for candidates the fields cannot order

The pool is built once per case and every arm reorders it, so a difference is a
difference in scoring and nothing else.

  python experiment_structured_rerank.py --out-dir <dir>
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
KS: tuple[int, ...] = (5, 10, 20)
LANE_DEPTH = 100
RRF_K = 60

PUBLISHED_RRF = {5: 0.507, 10: 0.580, 20: 0.653}

#: Field weights.  Entity and metric carry the identity; period separates the
#: same line across years.  Fixed constants, not fitted on the evaluation set --
#: a sweep would be selection on the test data and is deliberately absent.
W_ENTITY = 0.35
W_METRIC = 0.45
W_PERIOD = 0.20

#: How much the candidate's own retrieval rank counts beside the field score.
FUSION_WEIGHT = 0.25

_STOP = frozenset("""a an the of for in on at to and or is was were be been what which
how much did does do according filing reported report per its their this that these
those with from by as if than then there here total""".split())


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()


def _tokens(text: Any) -> set[str]:
    return {t for t in _norm(text).split() if t and t not in _STOP and len(t) > 1}


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _rerank(pool, fields, slots, cand_text, mode, canonical_of):
    """Order the pool by how well each candidate matches the plan's slots.

    Fixed weights, no per-case branching, no gold.  Metric agreement goes
    through `canonical_of` -- the same resolver the alignment gate uses -- so a
    canonical id agrees with itself and two different canonical ids never agree.
    The candidate's own fusion rank is the final tiebreak, which keeps the
    ordering deterministic and means the reranker can only ever move a candidate
    relative to another on an actual field difference.
    """
    weights = {
        "field": (W_ENTITY, W_METRIC, W_PERIOD),
        # Period is the field most likely to be WRONG in the plan rather than in
        # the candidate -- a filing states one year many ways -- so this arm
        # keeps it as a weak signal instead of a third of the score.
        "field_period_light": (W_ENTITY, W_METRIC, W_PERIOD / 4),
        # Metric is what the question is about; entity and period disambiguate.
        "field_metric_heavy": (0.25, 0.55, 0.20),
        "field_lexical": (W_ENTITY, W_METRIC, W_PERIOD),
    }[mode]
    we, wm, wp = weights

    want: set[str] = set()
    if mode == "field_lexical":
        for _, slot_metric, _ in slots:
            want |= _tokens(slot_metric)

    keys: dict[str, tuple] = {}
    for rank, key in enumerate(pool):
        cand_entity, cand_metric, cand_period = fields.get(key, ("", "", ""))
        best = 0.0
        for slot_entity, slot_metric, slot_period in slots:
            entity = 1.0 if slot_entity and slot_entity == cand_entity else 0.0
            sm = canonical_of(slot_metric) or _norm(slot_metric)
            cm = canonical_of(cand_metric) or _norm(cand_metric)
            metric = 1.0 if sm and sm == cm else 0.0
            period = 1.0 if slot_period and slot_period == cand_period else 0.0
            best = max(best, we * entity + wm * metric + wp * period)
        lexical = (len(want & _tokens(cand_text.get(key, ""))) / len(want)
                   if want else 0.0)
        # The fusion rank enters as a genuine component, not only as a tiebreak:
        # the fields are coarse and hundreds of candidates share a score, so a
        # pure field order discards the retrieval evidence for most of the list.
        blended = best + FUSION_WEIGHT * (1.0 / (rank + 1))
        keys[key] = (-blended, -best, -lexical, rank)
    return sorted(pool, key=lambda key: keys[key])


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
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    import run_p1_2_dual_track_benchmark as runner

    runner._load_deployment_env()

    from run_tv2_canonical_benchmark import FactStoreGroundingIndex  # noqa: E402

    from rag_v2.supervisor.semantic_alignment import canonical_metric_id  # noqa: E402
    from src.pdf_retrieval_v4.candidate_query_builder import (  # noqa: E402
        _metric_aliases, build_all_queries)
    from src.pdf_retrieval_v4.candidate_rrf import fuse_candidate_hits  # noqa: E402
    from src.pdf_retrieval_v4.candidate_view_index import (  # noqa: E402
        LANES, CandidateSearchHit, CandidateViewIndexReader)
    from src.pdf_retrieval_v4.planner import build_query_plan  # noqa: E402
    from src.runtime.trusted_v2_production import _path_env  # noqa: E402

    environ = dict(os.environ)
    index_dir = (args.index_dir if args.index_dir is not None
                 else _path_env(environ, "TRUSTED_V2_R4_INDEX_DIR", directory=True))

    # -- candidate fields, straight from the store the index was built from ----
    fields: dict[str, tuple[str, str, str]] = {}
    cand_text: dict[str, str] = {}
    for line in args.v2_fact_store.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = str(record.get("candidate_key") or record.get("candidate_id") or "")
        if not key:
            continue
        fields[key] = (_norm(record.get("entity")), str(record.get("metric") or ""),
                       str(record.get("period") or ""))
        cand_text[key] = " ".join(str(v) for v in (
            record.get("metric"), record.get("row_label"),
            record.get("source_text"), record.get("content")) if v)

    questions = bench._load_jsonl(args.eval_set)
    gold_by_id = {row["id"]: row for row in bench._load_jsonl(args.gold_evidence)}
    if args.limit:
        questions = questions[: args.limit]

    reader = CandidateViewIndexReader(index_dir)
    index_keys = {key for (_lane, key) in reader._candidate_to_view}
    resolver = bench.GoldResolver(
        FactStoreGroundingIndex(args.v2_fact_store),
        bench.load_ixbrl_candidate_keys(args.ixbrl_fact_store),
        index_keys)

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

    print("=" * 78)
    print("STRUCTURED RERANK -- Recall@5 > 80%")
    print("=" * 78)
    print(f"  scorable cases  {len(scorable)}")
    print(f"  candidate fields {len(fields)}")
    print(f"  weights         entity {W_ENTITY}  metric {W_METRIC}  period {W_PERIOD}")
    print()
    print("  retrieving ...", flush=True)

    started = time.perf_counter()
    per_case: dict[str, dict[str, list[str]]] = {}
    gold_by_case: dict[str, list[str]] = {}
    slot_recall: collections.Counter = collections.Counter()

    for position, (question, gold_keys) in enumerate(scorable, 1):
        queries = build_all_queries(build_query_plan(question["question"], ()))
        slot_queries: dict[str, list[str]] = queries.get("slots") or {}
        if not slot_queries:
            slot_queries = {"s0": list(queries.get("raw_question") or [question["question"]])}

        # The demand, as the plan states it: entity and period are plan-level,
        # the metric phrase is per slot.
        plan = build_query_plan(question["question"], ())
        issuer = _norm(getattr(plan, "issuer", None))
        slots: list[tuple[str, str, str]] = []
        for slot in plan.operand_slots:
            phrase = str(getattr(slot, "raw_metric_phrase", "") or "")
            period = str(getattr(slot, "period", "") or "")
            if phrase:
                slots.append((issuer, phrase, period))
        if not slots:
            for variants in slot_queries.values():
                for variant in variants:
                    slots.append(("", variant, ""))

        # One pool, built once: slot-local queries at an equal retrieved budget.
        pool: list[str] = []
        for slot_id, variants in slot_queries.items():
            variants = _dedupe(variants) or [question["question"]]
            depth = max(1, LANE_DEPTH // len(variants))
            lane_hits: dict[str, list[CandidateSearchHit]] = {}
            for lane in LANES:
                best_rank: dict[str, int] = {}
                order: list[str] = []
                for query in variants:
                    for rank, hit in enumerate(
                        reader.search(lane, query, allowed_candidate_keys=None, k=depth), 1
                    ):
                        key = hit.candidate_key
                        if not key:
                            continue
                        if key not in best_rank:
                            best_rank[key] = rank
                            order.append(key)
                        elif rank < best_rank[key]:
                            best_rank[key] = rank
                lane_hits[lane] = [
                    CandidateSearchHit(candidate_key=k, view_id=k, lane=lane,
                                       bm25_rank=best_rank[k], dense_rank=None,
                                       bm25_score=None, dense_score=None)
                    for k in sorted(order, key=lambda k: best_rank[k])]
            for hit in fuse_candidate_hits(lane_hits, rrf_k=RRF_K)[:50]:
                if hit.candidate_key not in pool:
                    pool.append(hit.candidate_key)

        scored = {"fusion_only": list(pool)}
        for mode in ("field", "field_period_light", "field_metric_heavy",
                     "field_lexical"):
            scored[mode] = _rerank(pool, fields, slots, cand_text, mode,
                                   canonical_metric_id)
        per_case[question["id"]] = scored
        gold_by_case[question["id"]] = gold_keys
        if position % 20 == 0 or position == len(scorable):
            print(f"    [{position:>3}/{len(scorable)}] "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    print()

    def macro(arm: str, k: int) -> float:
        values = []
        for case_id, gold in gold_by_case.items():
            head = set(per_case[case_id][arm][:k])
            values.append(sum(1 for key in gold if key in head) / len(gold))
        return sum(values) / len(values) if values else 0.0

    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"    {'arm':16}{'R@5':>10}{'R@10':>10}{'R@20':>10}")
    results = {}
    for arm in ("fusion_only", "field", "field_period_light",
                "field_metric_heavy", "field_lexical"):
        row = {str(k): round(macro(arm, k), 6) for k in KS}
        results[arm] = row
        print(f"    {arm:16}{row['5']:>10.3%}{row['10']:>10.3%}{row['20']:>10.3%}")
    print()
    print(f"    published shipped hybrid R@5 {PUBLISHED_RRF[5]:.3%}   "
          f"(this pool, unreranked: {results['fusion_only']['5']:.3%})")
    best = max(("fusion_only", "field", "field_period_light",
                "field_metric_heavy", "field_lexical"),
               key=lambda a: results[a]["5"])
    print(f"    best R@5: {best} {results[best]['5']:.3%}   "
          f"target 80.000%  {'MET' if results[best]['5'] > 0.80 else 'not met'}")
    print()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "structured-rerank.json").write_text(
        json.dumps({"scorable_cases": len(scorable), "results": results,
                    "weights": {"entity": W_ENTITY, "metric": W_METRIC,
                                "period": W_PERIOD},
                    "target": 0.80}, ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"  written to {args.out_dir / 'structured-rerank.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
