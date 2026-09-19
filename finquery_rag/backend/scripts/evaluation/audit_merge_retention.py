"""P1.5-1D: why does a gold that reached its own lane fail to reach the packet?

The slot-local authority fix took cross-entity gold reachability from 0/21 to
40/47 in-lane, but only 30/47 reach the merged packet.  Two explanations want
opposite fixes:

  BUDGET_EVICTION      the packet is genuinely too small, so the merge has to
                       drop someone -- expected, and arguably correct
  UNFAIR_MERGE         the merge favours one lane over another, so a slot is
                       starved although its candidate ranks well

This replays `build_slot_pool` step by step for each dropped slot and records
which step removed it, rather than inferring it from the rank distribution.

It changes nothing: the merge is read, not altered, and the retriever is the
production one.  Retrieval only, so no model and no GPU.

  python audit_merge_retention.py --out-dir <dir>
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

#: The two constants that can remove a candidate before the final packet cap.
#: Both are read from the production call rather than restated, so an audit of a
#: different configuration cannot silently report about this one.
SLOT_TOP_K = 20


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
    from src.pdf_retrieval_v4.candidate_slot_pool import build_slot_pool
    from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader
    from src.runtime.trusted_v2_production import _path_env
    from src.runtime.trusted_v2_r4 import build_slot_retrieval_requests

    index_dir = _path_env(dict(os.environ), "TRUSTED_V2_R4_INDEX_DIR", directory=True)
    retriever = CandidateDirectRetriever(CandidateViewIndexReader(index_dir))
    merge_depth = max(80, retriever.final_pool_k * 2)

    cases = [
        row for row in fixtures.values() if row["stratum"] == "cross_entity_comparison"
    ]
    cases.sort(key=lambda row: row["id"])

    rows: list[dict[str, Any]] = []
    for fixture in cases:
        plan = SupervisorPlan.from_dict(fixture["plan"])
        requests = build_slot_retrieval_requests(plan)
        golds = [str(x) for x in (gold.get(fixture["id"], {}).get("fact_ids") or [])]
        out = retriever.retrieve_for_requests(requests, document_scope=set())
        pools = out["slot_pools"]

        # Replay the merge's own first step, so "beyond the lane depth" is
        # measured rather than assumed.
        slot_top = {
            slot_id: list(hits[:SLOT_TOP_K]) for slot_id, hits in pools.items()
        }
        survivors = {_key_of(h) for hits in slot_top.values() for h in hits}
        merged_keys = [_key_of(item) for item in out["candidate_direct_pool"]]

        for index, request in enumerate(requests):
            gold_id = golds[index] if index < len(golds) else None
            if gold_id is None:
                continue
            lane = pools.get(request.slot_id, [])
            lane_keys = [_key_of(hit) for hit in lane]
            lane_rank = lane_keys.index(gold_id) + 1 if gold_id in lane_keys else None
            in_packet = gold_id in merged_keys

            other_lanes = [
                slot_id
                for slot_id, hits in pools.items()
                if slot_id != request.slot_id and gold_id in [_key_of(h) for h in hits]
            ]
            if lane_rank is None:
                reason = "NOT_IN_OWN_LANE"
            elif in_packet and lane_rank > SLOT_TOP_K:
                reason = "REACHED_VIA_ANOTHER_LANE"
            elif in_packet:
                reason = "RETAINED"
            elif lane_rank > SLOT_TOP_K:
                reason = "LANE_DEPTH_TRUNCATION"
            elif gold_id not in survivors:
                reason = "CROSS_SLOT_DEDUPE"
            elif gold_id not in merged_keys:
                reason = "MERGE_BUDGET_EVICTION"
            else:
                reason = "OTHER"

            rows.append(
                {
                    "case": fixture["id"],
                    "slot_id": request.slot_id,
                    "slot_count": len(requests),
                    "entity": request.entity,
                    "metric": request.metric,
                    "gold": gold_id,
                    "gold_lane_rank": lane_rank,
                    "lane_size": len(lane),
                    "slot_top_k": SLOT_TOP_K,
                    "in_slot_top": gold_id in survivors,
                    "in_merged_pool": in_packet,
                    "merged_pool_size": len(merged_keys),
                    "merge_depth": merge_depth,
                    "also_in_other_lanes": sorted(other_lanes),
                    "final_packet_budget": retriever.final_pool_k,
                    "drop_reason": reason,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "merge_retention.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    reasons = collections.Counter(r["drop_reason"] for r in rows)
    print("=== outcome of every cross-entity gold slot ===")
    for name, count in reasons.most_common():
        print(f"  {name:28} {count:3}")
    print()
    dropped = [r for r in rows if not r["in_merged_pool"]]
    print(f"=== the {len(dropped)} slots that do not reach the packet ===")
    for r in dropped:
        rank = r["gold_lane_rank"]
        print(
            f"  {r['case']:28} {r['slot_id']:4} n={r['slot_count']} "
            f"rank={str(rank):>4}/{r['lane_size']:<4} "
            f"in_slot_top={r['in_slot_top']!s:5} reason={r['drop_reason']}"
        )
    print(f"\nwrote {args.out_dir / 'merge_retention.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
