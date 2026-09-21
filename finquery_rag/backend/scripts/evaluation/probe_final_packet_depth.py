"""P1.5-1D2: does a deeper lane send more gold all the way to the Binder?

P1.5-1D established that the merge is innocent -- no budget eviction, no
cross-slot dedupe, no unfairness -- and that the loss is `build_slot_pool`'s
per-lane truncation, applied before the round-robin.  It also showed a depth
sweep rising 30 -> 35 and saturating, but that number counted the *merged pool*,
which is not what the Binder sees.

Four different populations have been living under the one name "in_lane", and
conflating them is what made the earlier numbers hard to reconcile:

    Raw Lane Recall       gold is anywhere in the slot's fused lane
    Lane Admission Recall gold survives slot_top_k
    Merged Recall         gold survives the round-robin merge
    Final Packet Recall   gold survives materialise -> entity/scope order -> cap

This measures all four at two lane depths and runs the real policy, so the last
column is the one that decides whether a fact is ever visible to the Binder.

Deterministic end to end: retrieval, materialisation, ordering and the cap are
all non-model steps, so this needs no GPU and changes nothing.

  python probe_final_packet_depth.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import collections
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

DEPTHS = (20, 40)


def _key_of(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("candidate_key") or "")
    return str(getattr(item, "candidate_key", "") or "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=runner.DEFAULT_FIXTURES)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    base = _BACKEND_DIR / "benchmarks/tv2_canonical_v1"
    fixtures = {r["id"]: r for r in runner._load_jsonl(args.fixtures)}
    gold = {r["id"]: r for r in runner._load_jsonl(base / "gold-evidence-v1.jsonl")}

    runner._load_deployment_env()
    import os

    from rag_v2.contracts.plan import SupervisorPlan
    from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever
    from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader
    from src.runtime.trusted_v2_production import (
        StructuredFactStore,
        _path_env,
    )
    from src.runtime.trusted_v2_r4 import (
        CandidateDirectR4Policy,
        R4RetrievalRequest,
        build_slot_retrieval_requests,
    )

    environ = dict(os.environ)
    reader = CandidateViewIndexReader(
        _path_env(environ, "TRUSTED_V2_R4_INDEX_DIR", directory=True)
    )
    store = StructuredFactStore(
        _path_env(environ, "TRUSTED_V2_FACT_STORE_PATH", directory=False)
    )

    cases = [
        row for row in fixtures.values() if row["stratum"] == "cross_entity_comparison"
    ]
    cases.sort(key=lambda row: row["id"])

    rows: list[dict[str, Any]] = []
    for depth in DEPTHS:
        retriever = CandidateDirectRetriever(reader)
        policy = CandidateDirectR4Policy(
            retriever,
            materializer=store.materialize,
            slot_top_k=depth,
        )
        for fixture in cases:
            plan = SupervisorPlan.from_dict(fixture["plan"])
            requests = build_slot_retrieval_requests(plan)
            golds = [str(x) for x in (gold.get(fixture["id"], {}).get("fact_ids") or [])]

            raw = retriever.retrieve_for_requests(
                requests, document_scope=set(), slot_top_k=depth
            )
            pools = raw["slot_pools"]
            merged = {_key_of(i) for i in raw["candidate_direct_pool"]}

            result = policy.retrieve(
                R4RetrievalRequest(
                    request_id=f"depth-{depth}-{fixture['id']}",
                    standalone_query=fixture["question"],
                    plan=plan,
                    reason_code="MISSING_SLOT",
                    target_slots=tuple(r.slot_id for r in requests),
                )
            )
            final = [str(c) for c in result.candidate_ids]
            gold_set = {g for g in golds}
            wrong = [c for c in final if c not in gold_set]

            for index, request in enumerate(requests):
                gold_id = golds[index] if index < len(golds) else None
                if gold_id is None:
                    continue
                lane_keys = [_key_of(h) for h in pools.get(request.slot_id, [])]
                rows.append(
                    {
                        "slot_top_k": depth,
                        "case": fixture["id"],
                        "slot_id": request.slot_id,
                        "metric": request.metric,
                        "gold": gold_id,
                        "raw_lane_recall": gold_id in lane_keys,
                        "raw_lane_rank": (
                            lane_keys.index(gold_id) + 1 if gold_id in lane_keys else None
                        ),
                        "lane_admission_recall": gold_id in lane_keys[:depth],
                        "merged_recall": gold_id in merged,
                        "final_packet_recall": gold_id in final,
                        "final_packet_size": len(final),
                        "wrong_candidates_in_final": len(wrong),
                        "candidate_ids": final,
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "final_packet_depth.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("=== the four recalls that were all being called 'in_lane' ===")
    print(f"{'slot_top_k':>10} {'raw':>7} {'admitted':>10} {'merged':>8} {'final':>7} {'pkt_size':>9} {'wrong':>7}")
    for depth in DEPTHS:
        sub = [r for r in rows if r["slot_top_k"] == depth]
        n = len(sub)
        print(
            f"{depth:>10} "
            f"{sum(1 for r in sub if r['raw_lane_recall']):>4}/{n:<3}"
            f"{sum(1 for r in sub if r['lane_admission_recall']):>7}/{n:<3}"
            f"{sum(1 for r in sub if r['merged_recall']):>5}/{n:<3}"
            f"{sum(1 for r in sub if r['final_packet_recall']):>4}/{n:<3}"
            f"{sum(r['final_packet_size'] for r in sub) / n:>9.1f} "
            f"{sum(r['wrong_candidates_in_final'] for r in sub) / n:>7.1f}"
        )
    print()
    gained = {
        (r["case"], r["slot_id"])
        for r in rows
        if r["slot_top_k"] == 40 and r["final_packet_recall"]
    } - {
        (r["case"], r["slot_id"])
        for r in rows
        if r["slot_top_k"] == 20 and r["final_packet_recall"]
    }
    lost = {
        (r["case"], r["slot_id"])
        for r in rows
        if r["slot_top_k"] == 20 and r["final_packet_recall"]
    } - {
        (r["case"], r["slot_id"])
        for r in rows
        if r["slot_top_k"] == 40 and r["final_packet_recall"]
    }
    print(f"  final-packet gold gained by depth 40: {len(gained)}")
    for case, slot in sorted(gained):
        print(f"    + {case} {slot}")
    print(f"  final-packet gold lost by depth 40:   {len(lost)}")
    for case, slot in sorted(lost):
        print(f"    - {case} {slot}")
    print(f"\nwrote {args.out_dir / 'final_packet_depth.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
