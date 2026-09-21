#!/usr/bin/env python3
"""Coverage-C0 recon: what id space is the final pool, and does the bridge reach it?

Three things have to be measured rather than assumed before a pool-membership
trace is worth writing:

1. what ids `CandidateDirectR4Policy.retrieve(...).candidate_ids` actually hold;
2. whether the canonical identity bridge (`load_alias_map`, the one the runtime
   host carries) resolves the cross-entity gold's `ixbrl:` keys into that space;
3. which store the runtime is actually reading.

  .venv/bin/python scripts/evaluation/probe_p1_8_identity_bridge.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BACKEND = Path("/disk/qh/nano-finrag/finquery_rag/backend")
for _p in (str(BACKEND), str(BACKEND / "scripts/evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LEGACY = Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl")
IXBRL = Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts-ixbrl-v1.jsonl")
V2 = Path("/disk/qh/nano-finrag/artifacts/evaluation/p1-8-c-v2")
FIXTURE = Path("/disk/qh/nano-finrag/artifacts/evaluation/p1-8-d1-c3/plan-fixtures-v9.jsonl")

CASES = ["tv2f01-s3-crossdiff-003", "tv2f01-s3-compare-010", "tv2f01-s1-jpm-007"]


def aliases_over(paths) -> dict[str, str]:
    """`load_alias_map`, run over several stores instead of one.

    The cross-entity gold names iXBRL keys and the rest of the benchmark names
    legacy keys, so a single-store map is a map of half the vocabulary.
    """

    import score_nf_v3_final as scorer

    merged: dict[str, str] = {}
    for path in paths:
        merged.update(scorer.load_alias_map(path))
    return merged


def main() -> int:
    import run_p1_2_dual_track_benchmark as runner

    runner._load_deployment_env()
    from rag_v2.contracts.plan import SupervisorPlan
    from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever
    from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader
    from src.runtime.trusted_v2_production import StructuredFactStore, _path_env
    from src.runtime.trusted_v2_r4 import CandidateDirectR4Policy, R4RetrievalRequest, build_slot_retrieval_requests

    environ = dict(os.environ)
    print("=" * 78)
    print("0. which store and index is the runtime reading")
    print("=" * 78)
    for key in ("TRUSTED_V2_FACT_STORE_PATH", "TRUSTED_V2_R4_INDEX_DIR"):
        print(f"  {key} = {environ.get(key)}")
    print()

    reader = CandidateViewIndexReader(_path_env(environ, "TRUSTED_V2_R4_INDEX_DIR", directory=True))
    store = StructuredFactStore(_path_env(environ, "TRUSTED_V2_FACT_STORE_PATH", directory=False))
    retriever = CandidateDirectRetriever(reader)
    policy = CandidateDirectR4Policy(retriever, materializer=store.materialize)

    aliases = aliases_over([LEGACY, IXBRL])
    print(f"  alias map entries over both stores: {len(aliases)}")

    fixtures = {r["id"]: r for r in (json.loads(l) for l in FIXTURE.read_text(encoding="utf-8").splitlines() if l.strip())}
    gold = {r["id"]: r for r in (json.loads(l) for l in (V2 / "gold-evidence-v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}

    print("=" * 78)
    print("1. do the cross-entity gold ids resolve through the bridge at all")
    print("=" * 78)
    for case in CASES:
        golds = [str(x) for x in (gold[case].get("fact_ids") or [])]
        for g in golds:
            print(f"  {case:<28} {g[:34]:<36} -> {str(aliases.get(g))[:34]}")
    print()

    print("=" * 78)
    print("2. the final Binder-visible pool, per case")
    print("=" * 78)
    for case in CASES:
        fixture = fixtures[case]
        plan = SupervisorPlan.from_dict(fixture["plan"])
        requests = build_slot_retrieval_requests(plan)
        result = policy.retrieve(
            R4RetrievalRequest(
                request_id=f"recon-{case}",
                standalone_query=fixture["question"],
                plan=plan,
                reason_code="MISSING_SLOT",
                target_slots=tuple(r.slot_id for r in requests),
            )
        )
        final = [str(c) for c in result.candidate_ids]
        resolved_pool = {aliases.get(c, c) for c in final}
        golds = [str(x) for x in (gold[case].get("fact_ids") or [])]
        resolved_gold = {aliases.get(g, g) for g in golds}
        print(f"  {case}  op={fixture['plan']['operation']}  slots={[r.slot_id for r in requests]}")
        print(f"    final pool size {len(final)}")
        print(f"    pool first 3 (raw)     : {final[:3]}")
        print(f"    pool first 3 (resolved): {sorted(resolved_pool)[:3]}")
        print(f"    gold raw               : {golds}")
        print(f"    gold resolved          : {sorted(resolved_gold)}")
        print(f"    gold in pool (raw)     : {[g in final for g in golds]}")
        print(f"    gold in pool (resolved): {[g in resolved_pool for g in resolved_gold]}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
