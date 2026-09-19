"""P1.5-1: does the alias expansion still earn its place, once lanes are slot-local?

Two changes are candidates, and they must not be measured as one:

    A  the authority fix      global query -> RequiredSlot-driven slot-local lane
    B  the alias capability   literal metric -> slot-local alias variants

Arm A is `SlotRetrievalRequestV1.query` alone.  Arm B is the same lane plus the
slot's own alias variants, each inheriting that slot's entity and period.  The
only variable between them is alias expansion on/off.

This matters because the deterministic query drops something measurably
load-bearing: four slots in the canonical set have a question surface that
differs from the filing label (``Revenues`` vs ``net sales``), and
`QuerySensitiveIndexReader` in the test suite exists precisely to show that
query wording changes what comes back.  One of those four -- `pctshare-005` --
currently releases.

Retrieval only, and therefore no model and no GPU: `gold_in_lane` and
`gold_in_final_packet` are decided entirely by the retriever.  Whether a
reachable gold then *binds* is a separate question that needs the Binder, and is
measured separately so a binding failure cannot be read as a retrieval one.

  python probe_slot_retrieval_ab.py --out-dir <dir>
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

#: The cases whose every required slot is measured: the stratum the authority
#: fix is for, plus the four slots that depend on alias expansion by name.
CROSS_ENTITY = "cross_entity_comparison"
ALIAS_DEPENDENT = (
    "tv2f01-s1-tsla-033",
    "tv2f01-s2-pctshare-005",
    "tv2f01-s4-ooc-001",
    "tv2f01-s4-oop-003",
)
#: The regression set: every percentage_share case, because the alias fix and
#: the P1.4 work both touch this stratum.
PCTSHARE_PREFIX = "tv2f01-s2-pctshare"


def _rank_of(order: list[Any], gold: str) -> int | None:
    for index, item in enumerate(order, 1):
        key = item.get("candidate_key") if isinstance(item, dict) else getattr(
            item, "candidate_key", None
        )
        if str(key) == gold:
            return index
    return None


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

    from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever
    from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader
    from src.runtime.trusted_v2_production import _path_env
    from src.runtime.trusted_v2_r4 import build_slot_retrieval_requests
    from rag_v2.contracts.plan import SupervisorPlan

    index_dir = _path_env(dict(os.environ), "TRUSTED_V2_R4_INDEX_DIR")
    retriever = CandidateDirectRetriever(CandidateViewIndexReader(index_dir))

    cases = [
        row
        for row in fixtures.values()
        if row["stratum"] == CROSS_ENTITY
        or row["id"] in ALIAS_DEPENDENT
        or row["id"].startswith(PCTSHARE_PREFIX)
    ]
    cases.sort(key=lambda row: row["id"])

    rows: list[dict[str, Any]] = []
    for fixture in cases:
        plan = SupervisorPlan.from_dict(fixture["plan"])
        requests = build_slot_retrieval_requests(plan)
        golds = [str(x) for x in (gold.get(fixture["id"], {}).get("fact_ids") or [])]
        for alias_expansion in (False, True):
            out = retriever.retrieve_for_requests(
                requests, document_scope=set(), alias_expansion=alias_expansion
            )
            pools = out["slot_pools"]
            packet = out["candidate_direct_pool"]
            for index, request in enumerate(requests):
                gold_id = golds[index] if index < len(golds) else None
                rows.append(
                    {
                        "case": fixture["id"],
                        "stratum": fixture["stratum"],
                        "arm": "B_aliases" if alias_expansion else "A_literal",
                        "slot_id": request.slot_id,
                        "entity": request.entity,
                        "metric": request.metric,
                        "query_variants": out["slot_query_variants"].get(request.slot_id),
                        "gold": gold_id,
                        "gold_in_lane": (
                            None if gold_id is None
                            else _rank_of(pools.get(request.slot_id, []), gold_id)
                        ),
                        "gold_in_packet": (
                            None if gold_id is None else _rank_of(packet, gold_id)
                        ),
                        "lane_size": len(pools.get(request.slot_id, [])),
                        "packet_size": len(packet),
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "slot_retrieval_ab.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    for arm in ("A_literal", "B_aliases"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        in_lane = sum(1 for r in arm_rows if r["gold_in_lane"] is not None)
        in_packet = sum(1 for r in arm_rows if r["gold_in_packet"] is not None)
        print(f"  {arm:10} slots={len(arm_rows):3}  gold_in_lane={in_lane:3}  gold_in_packet={in_packet:3}")

    print("\n=== the four alias-dependent slots ===")
    for case in ALIAS_DEPENDENT:
        for arm in ("A_literal", "B_aliases"):
            hit = [
                r for r in rows if r["case"] == case and r["arm"] == arm
            ]
            for r in hit:
                print(
                    f"  {case:30} {arm:10} lane={str(r['gold_in_lane']):>5} "
                    f"packet={str(r['gold_in_packet']):>5} variants={len(r['query_variants'] or [])}"
                )
    print(f"\nwrote {args.out_dir / 'slot_retrieval_ab.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
