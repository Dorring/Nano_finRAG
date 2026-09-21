#!/usr/bin/env python3
"""C0.1: the pool-membership trace -- did each slot's gold reach the Binder's pool?

`reached.retrieval` in the sealed replay records that the stage *ran*.  It does
not say whether the gold fact got there, so `RETRIEVAL` and `BINDING` could not
be separated and every miss was attributed one stage too late.  This measures the
missing half, deterministically: retrieval, materialisation, entity/scope
ordering and the cap are all non-model steps, so this changes nothing and needs
no GPU.

    .venv/bin/python scripts/evaluation/trace_p1_8_pool_membership.py \
        --fixture <plan-fixtures-v9.jsonl> \\
        --gold <gold-evidence-v1.jsonl> \\
        --predictions <replay-predictions.jsonl> \\
        --out-dir <dir>

The canonical identity bridge
-----------------------------

Gold for the cross-entity stratum names rebuilt-iXBRL keys (`ixbrl:…`); the
retriever's pool holds legacy keys (`v2fact:…`).  `load_alias_map` resolves ids
*within* a record's namespaces -- it maps `ixbrl:X` to `ixbrl:X` -- so comparing
raw ids across the two reports every cross-entity gold as absent, and the
runtime's own successful releases prove that is wrong.

The bridge used here is the asserted quantity: **(entity, period, value)**.
That is `logical_fact_id`'s payload minus its `metric` field, and the metric is
dropped deliberately -- the two stores name it in different vocabularies (the
iXBRL row carries `canonical_concept: total_liabilities`, the legacy row carries
the printed label `Total liabilities`), and a bridge that demanded they match
would rebuild the same false negative.

**Direction of the error.**  Matching on the quantity and not the label can only
*over*-report reachability: two different metrics asserting the same value for
one filer in one period would be called one fact.  So a `RETRIEVAL` verdict here
is safe, and a `BINDING` verdict means "in the pool and still not bound", which
may in truth be a label problem.  The label question is separable and is left to
the BINDING sub-buckets.

What this cannot measure
------------------------

`gold_bound_per_slot` -- which fact the Binder put in which slot -- is not in the
sealed replay, which records only *which slot ids* bound.  It is derived here
from the bound-evidence set where that is exposed, and reported as `null` where
it is not, rather than inferred.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

BACKEND = Path("/disk/qh/nano-finrag/finquery_rag/backend")
for _p in (str(BACKEND), str(BACKEND / "scripts/evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LEGACY = Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl")
IXBRL = Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts-ixbrl-v1.jsonl")

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def quantity(text: Any) -> str | None:
    """A stated quantity as a comparable string: `$ 66,387` -> `66387`."""

    raw = str(text if text is not None else "").strip()
    if not raw or raw.casefold() in {"none", "nan", "-"}:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    match = _NUM.search(raw)
    if not match:
        return None
    digits = match.group(0).replace(",", "")
    try:
        value = float(digits)
    except ValueError:
        return None
    if negative and value > 0:
        value = -value
    return f"{value:g}"


def fold(text: Any) -> str:
    return " ".join(str(text if text is not None else "").casefold().split())


def period_of(text: Any) -> str:
    raw = fold(text)
    if not raw or raw == "none":
        return ""
    return raw.replace(" ", "")


def identity(entity: Any, period: Any, value: Any) -> str | None:
    quantity_value = quantity(value)
    if quantity_value is None:
        return None
    return f"{fold(entity)}|{period_of(period)}|{quantity_value}"


def record_identity(record: Any) -> str | None:
    get = (lambda k: record.get(k)) if isinstance(record, dict) else (lambda k: getattr(record, k, None))
    return identity(get("entity"), get("period"), get("value"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    import run_p1_2_dual_track_benchmark as runner

    runner._load_deployment_env()
    from rag_v2.contracts.plan import SupervisorPlan
    from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever
    from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader
    from src.runtime.trusted_v2_production import StructuredFactStore, _path_env
    from src.runtime.trusted_v2_r4 import (
        CandidateDirectR4Policy,
        R4RetrievalRequest,
        build_slot_retrieval_requests,
    )

    environ = dict(os.environ)
    reader = CandidateViewIndexReader(_path_env(environ, "TRUSTED_V2_R4_INDEX_DIR", directory=True))
    store = StructuredFactStore(_path_env(environ, "TRUSTED_V2_FACT_STORE_PATH", directory=False))
    policy = CandidateDirectR4Policy(CandidateDirectRetriever(reader), materializer=store.materialize)

    def load(path: Path) -> list[dict]:
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    fixture = {r["id"]: r for r in load(args.fixture)}
    gold = {r["id"]: r for r in load(args.gold)}
    replay = {r["id"]: r for r in load(args.predictions)}

    def records_by_key(path: Path) -> dict[str, dict]:
        out: dict[str, dict] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for key in (record.get("candidate_key"), record.get("fact_id"), record.get("candidate_id")):
                    if key:
                        out.setdefault(str(key), record)
        return out

    stores = records_by_key(LEGACY) | records_by_key(IXBRL)
    print(f"records indexed from both stores: {len(stores)}")

    rows: list[dict[str, Any]] = []
    for case_id, fixture_row in fixture.items():
        gold_row = gold.get(case_id) or {}
        if gold_row.get("expected_outcome") != "ANSWER":
            continue
        plan = SupervisorPlan.from_dict(fixture_row["plan"])
        requests = build_slot_retrieval_requests(plan)
        result = policy.retrieve(
            R4RetrievalRequest(
                request_id=f"trace-{case_id}",
                standalone_query=fixture_row["question"],
                plan=plan,
                reason_code="MISSING_SLOT",
                target_slots=tuple(r.slot_id for r in requests),
            )
        )
        pool_keys = [str(c) for c in result.candidate_ids]
        pool_identities = []
        for key in pool_keys:
            record = stores.get(key)
            pool_identities.append(record_identity(record) if record else None)

        slot_ids = [r.slot_id for r in requests]
        gold_ids = [str(x) for x in (gold_row.get("fact_ids") or [])]
        per_slot = []
        for position, slot_id in enumerate(slot_ids):
            gold_key = gold_ids[position] if position < len(gold_ids) else None
            gold_record = stores.get(gold_key) if gold_key else None
            gold_identity = record_identity(gold_record) if gold_record else None
            rank = None
            if gold_identity is not None and gold_identity in pool_identities:
                rank = pool_identities.index(gold_identity) + 1
            per_slot.append(
                {
                    "required_slot_id": slot_id,
                    "gold_fact_id": gold_key,
                    "gold_identity": gold_identity,
                    "gold_value": (gold_record or {}).get("value"),
                    "gold_in_final_pool": rank is not None,
                    "final_pool_rank": rank,
                }
            )

        state = replay.get(case_id) or {}
        bound_slots = set(state.get("binder_final_bound_slot_ids") or [])
        missing_slots = set(state.get("binder_final_missing_slot_ids") or [])
        for slot in per_slot:
            slot["gold_bound_per_slot"] = (
                slot["required_slot_id"] in bound_slots if bound_slots or missing_slots else None
            )
            slot["slot_complete"] = slot["required_slot_id"] in bound_slots and not missing_slots

        rows.append(
            {
                "case_id": case_id,
                "stratum": fixture_row.get("stratum"),
                "operation": plan.operation,
                "slot_count": len(slot_ids),
                "gold_fact_count": len(gold_ids),
                "final_pool_size": len(pool_keys),
                "per_slot": per_slot,
                "gold_pool_complete": all(s["gold_in_final_pool"] for s in per_slot),
                "gold_binding_complete": (
                    all(s["gold_bound_per_slot"] for s in per_slot)
                    if all(s["gold_bound_per_slot"] is not None for s in per_slot)
                    else None
                ),
                "slot_complete": all(s["slot_complete"] for s in per_slot),
                "binder_final_status": state.get("binder_final_status"),
                "reason_codes": state.get("reason_codes"),
                "validator_status": state.get("validator_status"),
                "status": state.get("status"),
                "release_status": state.get("release_status"),
                "gate_refused": bool(state.get("align_overridden")),
                "reached_binding": (state.get("reached") or {}).get("binding"),
                "reached_validation": (state.get("reached") or {}).get("validation"),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pool-membership.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    unreleased = [r for r in rows if r["release_status"] != "RELEASED"]
    released = [r for r in rows if r["release_status"] == "RELEASED"]
    print()
    print(f"traced {len(rows)} answerable cases   released {len(released)}   unreleased {len(unreleased)}")
    print()
    print("  control -- released cases whose gold is not in the pool:")
    for row in released:
        if not row["gold_pool_complete"]:
            absent = [s["required_slot_id"] for s in row["per_slot"] if not s["gold_in_final_pool"]]
            print(f"    {row['case_id']:<26} slots without gold in pool: {absent}")
    print()
    print("  the 25 unreleased, by whether the gold reached the pool at all:")
    for bucket, selector in (
        ("gold fully in pool", lambda r: r["gold_pool_complete"]),
        ("gold PARTLY in pool", lambda r: not r["gold_pool_complete"] and any(s["gold_in_final_pool"] for s in r["per_slot"])),
        ("gold absent from pool", lambda r: not any(s["gold_in_final_pool"] for s in r["per_slot"])),
    ):
        subset = [r for r in unreleased if selector(r)]
        print(f"    {bucket:<24} {len(subset):>3}")
        for row in subset:
            ranks = [s["final_pool_rank"] for s in row["per_slot"]]
            print(f"       {row['case_id']:<26} slots={row['slot_count']} pool={row['final_pool_size']:<3} ranks={ranks}")

    print(f"\nwritten to {args.out_dir / 'pool-membership.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
