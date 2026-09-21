"""P1.5-0c: would an entity-scoped query actually reach the gold?

P1.5-0b established that 0/21 gold facts enter the top-40 packet while all 21
are indexed.  The planner check that followed found why the packet is a single
shared pool: `build_query_plan` gives every cross-entity question
`task_type=general_single_fact` or `table_single_fact` and exactly ONE operand
slot, so `is_multi_slot` is False, `build_slot_pool` never runs, and a
four-company ranking shares one top-40.

That explains why per-slot retrieval does not engage.  It does not yet say
whether engaging it would help -- if the gold ranks 300th under a query that
names its own entity, metric and period, then a slot-scoped pool would be a
slot-scoped version of the same miss, and the fault is the scorer or the index
representation instead.  This measures that, and nothing else.

For each required slot it records the rank of the gold candidate under two
queries, at a lane depth far past production's 50:

    global  the production `raw_question` query, unchanged
    slot    "<entity> <metric> <period>", built from the SupervisorPlan

The per-slot query is diagnostic.  It is not wired into production and the
production path is not modified -- this only asks what the retriever would do
if it were asked a narrower question.

Rank is reported as an integer, or ``None`` when the fact is absent from the
candidate set at that depth: absent and rank-5000 are different findings.

  python probe_slot_retrieval_rank.py --lane-k 500 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_SCRIPTS = _BACKEND_DIR / "scripts/evaluation"
for _path in (str(_BACKEND_DIR), str(_SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_p1_2_dual_track_benchmark as runner  # noqa: E402

_ID_KEYS = ("candidate_key", "candidate_id", "fact_id", "evidence_id")


def _fused_order(retriever: Any, query: str, allowed: dict[str, Any]) -> list[str]:
    """The fused candidate order for one query, at the retriever's lane depth."""

    from src.pdf_retrieval_v4.candidate_rrf import fuse_candidate_hits

    lane_hits = retriever._search_lanes(query, allowed)
    fused = fuse_candidate_hits(lane_hits, rrf_k=retriever.rrf_k)
    order: list[str] = []
    for hit in fused:
        key = getattr(hit, "candidate_key", None)
        if key and key not in order:
            order.append(str(key))
    return order


def _rank(order: list[str], gold: str) -> int | None:
    return order.index(gold) + 1 if gold in order else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-k", type=int, default=500)
    parser.add_argument("--fixtures", type=Path, default=runner.DEFAULT_FIXTURES)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    base = _BACKEND_DIR / "benchmarks/tv2_canonical_v1"
    questions = {r["id"]: r for r in runner._load_jsonl(base / "canonical-eval-v1.jsonl")}
    fixtures = {r["id"]: r for r in runner._load_jsonl(args.fixtures)}
    gold = {r["id"]: r for r in runner._load_jsonl(base / "gold-evidence-v1.jsonl")}

    runner._load_deployment_env()
    import os

    from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever
    from src.pdf_retrieval_v4.candidate_query_builder import build_all_queries
    from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader
    from src.pdf_retrieval_v4.planner import build_query_plan
    from src.runtime.trusted_v2_production import _path_env

    # The reader alone, not `_load_resources`.  That helper also builds the
    # specialist checkpoint, which is CUDA-serialised and therefore pins this
    # measurement to a GPU it has no use for: ranking needs the index and the
    # BM25/dense lanes, and the dense encoder is `device="cpu"` already.  With
    # the reader built directly this runs beside the live backend instead of
    # requiring it stopped, and the thing being measured is unchanged -- the
    # same index directory, the same reader class, the same lanes.
    index_dir = _path_env(dict(os.environ), "TRUSTED_V2_R4_INDEX_DIR", directory=True)
    reader = CandidateViewIndexReader(index_dir)
    retriever = CandidateDirectRetriever(reader, lane_k=args.lane_k)
    retriever.final_pool_k = args.lane_k
    allowed = retriever._allowed_keys_for_scope(set())

    cases = [r for r in fixtures.values() if r["stratum"] == "cross_entity_comparison"]
    rows: list[dict[str, Any]] = []
    for fixture in cases:
        question = questions[fixture["id"]]["question"]
        golds = [str(x) for x in (gold.get(fixture["id"], {}).get("fact_ids") or [])]
        plan = build_query_plan(question, ())
        global_query = build_all_queries(plan)["raw_question"][0]
        global_order = _fused_order(retriever, global_query, allowed)
        slot_order_cache: dict[str, list[str]] = {}

        for index, slot in enumerate(fixture["plan"]["required_slots"]):
            gold_id = golds[index] if index < len(golds) else None
            parts = [slot.get("entity"), slot.get("metric"), slot.get("period")]
            slot_query = " ".join(str(p) for p in parts if p)
            if slot_query not in slot_order_cache:
                slot_order_cache[slot_query] = _fused_order(retriever, slot_query, allowed)
            rows.append(
                {
                    "case": fixture["id"],
                    "slot_id": slot["slot_id"],
                    "entity": slot.get("entity"),
                    "metric": slot.get("metric"),
                    "period": slot.get("period"),
                    "gold": gold_id,
                    "global_query": global_query,
                    "slot_query": slot_query,
                    "global_rank": None if gold_id is None else _rank(global_order, gold_id),
                    "slot_rank": None if gold_id is None else _rank(slot_order_cache[slot_query], gold_id),
                    "global_pool_size": len(global_order),
                    "slot_pool_size": len(slot_order_cache[slot_query]),
                }
            )
            print(
                f"  {fixture['id']:30} {slot['slot_id']:4} "
                f"global={rows[-1]['global_rank']} slot={rows[-1]['slot_rank']} "
                f"(pools {len(global_order)}/{len(slot_order_cache[slot_query])})",
                flush=True,
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "slot_retrieval_rank.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nwrote {args.out_dir / 'slot_retrieval_rank.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
