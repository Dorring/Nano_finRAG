#!/usr/bin/env python3
"""Final Seal: Benchmark V2 citation precision and recall, both ways of reading it.

A filing states one quantity more than once -- the income statement and the note
that repeats it -- and the store keeps both, correctly, because each is real
evidence with its own page.  Comparing a citation to the note against a gold id
naming the statement measures *where* a number was printed and reports it as
*what* was cited.  Both numbers are reported and neither replaces the other.

  strict ids           citations and gold compared after `load_alias_map`, which
                       resolves id namespaces within a record
  canonical identity   compared on (entity, canonical metric, normalised period,
                       value), so the same quantity filed twice counts as one

    .venv/bin/python scripts/evaluation/measure_p1_8_citation.py \
        --predictions <replay-predictions.jsonl> --gold <gold-evidence-v1.jsonl>
"""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    args = parser.parse_args(argv)

    import score_nf_v3_final as scorer
    from src.pdf_retrieval_v4.canonical_metric_identity import canonical_metric_id
    from measure_p1_8_identity_closure import canonical_metric, period_of, quantity, fold

    aliases: dict[str, str] = {}
    records: dict[str, dict] = {}
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

    def identity(record: dict | None) -> tuple | None:
        if not record:
            return None
        key = canonical_metric(record, canonical_metric_id)
        return (
            fold(record.get("entity")),
            key,
            period_of(record.get("period")),
            quantity(record.get("value")),
        )

    def load(path: Path) -> list[dict]:
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    gold = {row["id"]: row for row in load(args.gold)}
    released = [
        row
        for row in load(args.predictions)
        if row.get("release_status") == "RELEASED"
        and gold[row["id"]].get("expected_outcome") == "ANSWER"
    ]

    strict_hit = strict_total = ident_hit = ident_total = 0
    recall_strict_hit = recall_total = recall_ident_hit = 0

    for row in released:
        record = gold[row["id"]]
        wanted = {scorer.resolve(str(f), aliases) for f in (record.get("fact_ids") or [])}
        wanted_identity = {identity(records.get(w)) for w in wanted} - {None}
        cited = [str(c) for c in (row.get("citation_ids") or [])]
        cited_resolved = [scorer.resolve(c, aliases) for c in cited]

        strict_total += len(cited_resolved)
        ident_total += len(cited_resolved)
        for value in cited_resolved:
            if value in wanted:
                strict_hit += 1
            if identity(records.get(value)) in wanted_identity:
                ident_hit += 1

        recall_total += len(wanted)
        cited_set = set(cited_resolved)
        recall_strict_hit += len(wanted & cited_set)
        cited_identities = {identity(records.get(c)) for c in cited_resolved} - {None}
        recall_ident_hit += len(wanted_identity & cited_identities)

    def pct(hit: int, total: int) -> str:
        return f"{hit}/{total} = {100.0 * hit / total:.1f}%" if total else "n/a"

    print(f"released cases with citations: {len(released)}")
    print()
    print(f"{'':<20} {'strict ids':<22} canonical identity")
    print(f"{'citation precision':<20} {pct(strict_hit, strict_total):<22} {pct(ident_hit, ident_total)}")
    print(f"{'citation recall':<20} {pct(recall_strict_hit, recall_total):<22} {pct(recall_ident_hit, recall_total)}")
    print()
    print(f"  gold facts that are somewhere in the store: "
          f"{sum(1 for row in released for f in (gold[row['id']].get('fact_ids') or []) if scorer.resolve(str(f), aliases) in records)}"
          f" / {sum(len(gold[row['id']].get('fact_ids') or []) for row in released)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
