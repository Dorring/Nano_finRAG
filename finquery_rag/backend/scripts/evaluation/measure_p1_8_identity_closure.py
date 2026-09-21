#!/usr/bin/env python3
"""C1.1: layered identity, so `TRUE_RETRIEVAL_MISS` stops over-reporting.

The C0 bridge used `(entity, period, value)` as *the* identity.  That was an
over-correction: it is a quantity fingerprint, not an identity, and the four
released cases whose gold it called absent proved it had false negatives.

This replaces it with a ladder ordered strongest-first, and reports which rung
each gold/pool comparison reached rather than collapsing them:

  MATCH_EXACT              the two ids resolve to the same candidate through
                           `load_alias_map` -- the runtime's own namespace bridge
  MATCH_PROVENANCE         the two records share a physical source anchor (row,
                           table fragment, citation, physical source, or
                           document+page): the same place in the filing
  MATCH_CANONICAL          entity + canonical metric + normalized period + value
                           all agree, where the canonical metric is
                           `canonical_metric_id` of the printed label on one side
                           and of the aligned concept on the other
  POSSIBLE_QUANTITY_MATCH  entity + period + value agree but the canonical metric
                           does not (or cannot be computed) -- **a possibility,
                           never an authoritative match**
  AMBIGUOUS_MATCH          the strongest rung holds for more than one distinct
                           pool candidate, so which one is "the" match is not
                           determined
  NO_MATCH                 nothing on any rung

Authoritative reachability is `MATCH_EXACT | MATCH_PROVENANCE | MATCH_CANONICAL`.
`POSSIBLE_QUANTITY_MATCH` alone is **not** reachability, which is the whole point
of splitting it out.  `AMBIGUOUS_MATCH` is reachability whose *identity* is
undetermined, and is reported separately so the two are never added together.

    .venv/bin/python scripts/evaluation/measure_p1_8_identity_closure.py \
        --fixture <plan-fixtures-v9.jsonl> --gold <gold-evidence-v1.jsonl> \
        --predictions <replay-predictions.jsonl> --out-dir <dir>
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

EXACT, PROVENANCE, CANONICAL = "MATCH_EXACT", "MATCH_PROVENANCE", "MATCH_CANONICAL"
POSSIBLE, AMBIGUOUS, NONE = "POSSIBLE_QUANTITY_MATCH", "AMBIGUOUS_MATCH", "NO_MATCH"
AUTHORITATIVE = (EXACT, PROVENANCE, CANONICAL)

#: How far up the ladder each rung sits.  Higher wins.
_RANK = {NONE: 0, POSSIBLE: 1, CANONICAL: 2, PROVENANCE: 3, EXACT: 4}


def _get(record: Any, key: str) -> Any:
    if record is None:
        return None
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key, None)


def fold(text: Any) -> str:
    return " ".join(str(text if text is not None else "").casefold().split())


def quantity(text: Any) -> str | None:
    raw = str(text if text is not None else "").strip()
    if not raw or raw.casefold() in {"none", "nan", "-"}:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    match = _NUM.search(raw)
    if not match:
        return None
    try:
        value = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    if negative and value > 0:
        value = -value
    return f"{value:g}"


def period_of(text: Any) -> str:
    raw = fold(text)
    return "" if not raw or raw == "none" else raw.replace(" ", "")


def canonical_metric(record: Any, canonical_metric_id) -> str | None:
    label = _get(record, "metric") or _get(record, "canonical_concept")
    return canonical_metric_id(label)


def provenance_anchors(record: Any) -> set[str]:
    """Physical source anchors this record claims.

    Two records that share one are the same place in the filing.  `row_id` is
    included deliberately and is the most interesting of them: the same row
    reached twice is the duplicate the store keeps on purpose, and a different
    cell of the same row is exactly the shape the column/row defect takes.
    """

    anchors: set[str] = set()
    for key in ("physical_source_id", "source_id", "citation_id", "row_id", "table_fragment_id"):
        value = _get(record, key)
        if value:
            anchors.add(f"{key}={value}")
    document = _get(record, "document_id") or _get(record, "document_name")
    page = _get(record, "page")
    if document and page is not None:
        anchors.add(f"doc={document}#p={page}")
    return anchors


def classify(gold_record: Any, pool_records: list[Any], aliases: dict[str, str], resolver, canonical_metric_id) -> dict:
    """The strongest rung any pool candidate reaches against one gold record."""

    hits: dict[str, list[int]] = {EXACT: [], PROVENANCE: [], CANONICAL: [], POSSIBLE: []}

    gold_anchor = provenance_anchors(gold_record)
    gold_key = fold(_get(gold_record, "candidate_key") or _get(gold_record, "candidate_id"))
    gold_entity = fold(_get(gold_record, "entity"))
    gold_period = period_of(_get(gold_record, "period"))
    gold_quantity = quantity(_get(gold_record, "value"))
    gold_metric = canonical_metric(gold_record, canonical_metric_id)

    for index, candidate in enumerate(pool_records):
        if candidate is None:
            continue
        candidate_key = fold(_get(candidate, "candidate_key") or _get(candidate, "candidate_id"))
        resolved_gold, resolved_candidate = resolver(gold_key, aliases), resolver(candidate_key, aliases)
        if resolved_gold and resolved_candidate and resolved_gold == resolved_candidate:
            hits[EXACT].append(index)
            continue
        if gold_anchor and (gold_anchor & provenance_anchors(candidate)):
            hits[PROVENANCE].append(index)
            continue
        same_quantity = (
            gold_quantity is not None
            and gold_quantity == quantity(_get(candidate, "value"))
        )
        if not same_quantity:
            continue
        same_frame = fold(_get(candidate, "entity")) == gold_entity and period_of(_get(candidate, "period")) == gold_period
        candidate_metric = canonical_metric(candidate, canonical_metric_id)
        if same_frame and gold_metric is not None and gold_metric == candidate_metric:
            hits[CANONICAL].append(index)
        elif same_frame:
            hits[POSSIBLE].append(index)

    for rung in (EXACT, PROVENANCE, CANONICAL):
        if hits[rung]:
            if len(hits[rung]) > 1:
                return {"match_class": AMBIGUOUS, "layer": rung, "matched_indexes": hits[rung]}
            return {"match_class": rung, "layer": rung, "matched_indexes": hits[rung]}
    if hits[POSSIBLE]:
        if len(hits[POSSIBLE]) > 1:
            return {"match_class": AMBIGUOUS, "layer": POSSIBLE, "matched_indexes": hits[POSSIBLE]}
        return {"match_class": POSSIBLE, "layer": POSSIBLE, "matched_indexes": hits[POSSIBLE]}
    return {"match_class": NONE, "layer": None, "matched_indexes": []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    import run_p1_2_dual_track_benchmark as runner

    runner._load_deployment_env()
    import score_nf_v3_final as scorer
    from src.pdf_retrieval_v4.canonical_metric_identity import canonical_metric_id
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

    aliases: dict[str, str] = {}
    records: dict[str, Any] = {}
    for path in (LEGACY, IXBRL):
        aliases.update(scorer.load_alias_map(path))
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for key in (record.get("candidate_key"), record.get("candidate_id"), record.get("fact_id")):
                    if key:
                        records.setdefault(str(key), record)

    # The pool's own candidates are what has to be materialised; `records` is the
    # lookup, and materialising through the runtime keeps the pool honest.
    def materialise(key: str) -> Any:
        try:
            return store.materialize(key)
        except Exception:  # noqa: BLE001 - a key the store cannot resolve is data
            return records.get(key)

    def resolver(identifier: str, table: dict[str, str]) -> str:
        return scorer.resolve(identifier, table)

    def load(path: Path) -> list[dict]:
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    fixture = {r["id"]: r for r in load(args.fixture)}
    gold = {r["id"]: r for r in load(args.gold)}
    replay = {r["id"]: r for r in load(args.predictions)}

    closure_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []

    for case_id, fixture_row in fixture.items():
        gold_row = gold.get(case_id) or {}
        if gold_row.get("expected_outcome") != "ANSWER":
            continue
        plan = SupervisorPlan.from_dict(fixture_row["plan"])
        requests = build_slot_retrieval_requests(plan)
        result = policy.retrieve(
            R4RetrievalRequest(
                request_id=f"closure-{case_id}",
                standalone_query=fixture_row["question"],
                plan=plan,
                reason_code="MISSING_SLOT",
                target_slots=tuple(r.slot_id for r in requests),
            )
        )
        pool_keys = [str(c) for c in result.candidate_ids]
        pool_records = [materialise(key) for key in pool_keys]
        gold_ids = [str(x) for x in (gold_row.get("fact_ids") or [])]

        per_slot = []
        for position, request in enumerate(requests):
            gold_key = gold_ids[position] if position < len(gold_ids) else None
            gold_record = records.get(gold_key) if gold_key else None
            verdict = (
                classify(gold_record, pool_records, aliases, resolver, canonical_metric_id)
                if gold_record is not None
                else {"match_class": NONE, "layer": None, "matched_indexes": []}
            )
            ranks = [i + 1 for i in verdict["matched_indexes"]]
            reachable = verdict["match_class"] in AUTHORITATIVE
            per_slot.append(
                {
                    "required_slot_id": request.slot_id,
                    "gold_fact_id": gold_key,
                    "gold_value": _get(gold_record, "value"),
                    "match_class": verdict["match_class"],
                    "match_layer": verdict["layer"],
                    "matched_pool_ranks": ranks,
                    "gold_in_final_pool": reachable,
                    "authoritative": reachable,
                }
            )
            closure_rows.append(
                {
                    "case_id": case_id,
                    "required_slot_id": request.slot_id,
                    "gold_fact_id": gold_key,
                    "match_class": verdict["match_class"],
                    "match_layer": verdict["layer"],
                    "matched_pool_ranks": ranks,
                    "pool_size": len(pool_keys),
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

        membership_rows.append(
            {
                "case_id": case_id,
                "stratum": fixture_row.get("stratum"),
                "operation": plan.operation,
                "slot_count": len(requests),
                "final_pool_size": len(pool_keys),
                "per_slot": per_slot,
                "gold_pool_complete": all(s["gold_in_final_pool"] for s in per_slot),
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
    for name, rows in (("identity-closure.jsonl", closure_rows), ("pool-membership-c1.jsonl", membership_rows)):
        (args.out_dir / name).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    counts: dict[str, int] = {}
    for row in closure_rows:
        counts[row["match_class"]] = counts.get(row["match_class"], 0) + 1
    unreleased_ids = {r["case_id"] for r in membership_rows if r["release_status"] != "RELEASED"}

    print(f"slot-level comparisons: {len(closure_rows)}")
    for name in (EXACT, PROVENANCE, CANONICAL, AMBIGUOUS, POSSIBLE, NONE):
        print(f"  {name:<26} {counts.get(name, 0)}")
    print()
    print("  by released / unreleased:")
    for label, selector in (("released", lambda r: r["case_id"] not in unreleased_ids),
                            ("unreleased", lambda r: r["case_id"] in unreleased_ids)):
        sub = [r for r in closure_rows if selector(r)]
        tally: dict[str, int] = {}
        for row in sub:
            tally[row["match_class"]] = tally.get(row["match_class"], 0) + 1
        print(f"    {label:<12} {json.dumps(tally, sort_keys=True)}")
    print()
    print("  the 25 unreleased, slot by slot:")
    for row in membership_rows:
        if row["release_status"] == "RELEASED":
            continue
        detail = " ".join(
            f"{s['required_slot_id']}:{s['match_class'].replace('MATCH_','').replace('_MATCH','')}"
            for s in row["per_slot"]
        )
        print(f"    {row['case_id']:<26} {detail}")
    print(f"\nwritten to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
