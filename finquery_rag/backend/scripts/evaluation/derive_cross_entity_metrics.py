"""P1.6-0G: find a concept every entity in a case actually reports the same way.

The cross-entity gold was built by matching a metric *string*, and most of those
strings match several different concepts across filings -- `Total` lives in every
note, `United States` is both a revenue geography and a tax jurisdiction.  The
coherence pass found only 2 of 20 cases hold one quantity for every entity.

So the stratum is re-derived rather than patched.  A replacement concept has to
satisfy two things that can be checked before anyone reads a page:

1. **The store has the metric string for every entity in the case.**  The plan
   carries one metric per slot, so a string that exists for Apple and not for
   Microsoft cannot be used however good the concept is.
2. **The coordinate is not ambiguous for any of them.**  If
   `(entity, metric, period)` holds several comparable values, the retrieval can
   land on any of them and the case is unanswerable in principle -- which is the
   defect the operand guard exists to refuse.

This ranks the candidates that survive both, and reports the cases where nothing
does.  A case with no candidate is a case whose entity set cannot be asked a
single question, and it is reported for retirement rather than repaired.

It reads the store only.  The winning candidate for each case still has to be
read back out of the filings before it becomes gold -- this narrows the search,
it does not certify anything.

  python derive_cross_entity_metrics.py --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

STRATUM_PREFIX = "tv2f01-s3-"

#: Strings that name no concept.  A replacement must not be one of these, and
#: they are listed rather than pattern-matched so the exclusion is auditable.
_NON_CONCEPTS = {
    "total",
    "other",
    "current",
    "2025",
    "2024",
    "2023",
    "united states",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, default=_BACKEND_DIR)
    parser.add_argument("--fact-store", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args(argv)

    import os

    from src.finance.operand_ambiguity import quantity_of
    from src.runtime.trusted_v2_production import StructuredFactStore

    store_path = args.fact_store or Path(os.environ["TRUSTED_V2_FACT_STORE_PATH"])
    store = StructuredFactStore(store_path)

    # metric string -> entity -> period -> facts, and the ambiguity of each
    # coordinate.  Keyed by period, not just by (entity, metric): a metric that
    # spans FY2023-FY2025 holds three different values by *design*, and folding
    # them into one bucket made every real metric read as ambiguous.  That bug
    # let only junk survive the filter -- `Thereafter`, a debt-maturity bucket
    # -- which is how a candidate list comes to look plausible and be useless.
    by_metric: dict[tuple[str, str, str], list] = collections.defaultdict(list)
    coordinate_values: dict[tuple, set] = collections.defaultdict(set)
    for record in store.iter_records():
        metric = record.get("metric")
        entity = record.get("entity")
        period = record.get("period")
        if not metric or not entity or not period:
            continue
        by_metric[(str(metric), str(entity), str(period))].append(record)
        quantity = quantity_of(record)
        if quantity is not None:
            coordinate_values[store.coordinate_key(record)].add(quantity.identity)

    def status_of(entity: str, metric: str, period: str) -> str:
        records = by_metric[(metric, entity, period)]
        if not records:
            return "absent"
        identities = set()
        for record in records:
            identities |= coordinate_values.get(store.coordinate_key(record), set())
        if not identities:
            return "unreadable"
        if len(identities) > 1:
            return "ambiguous"
        return "unique" if len(records) == 1 else "consensus"

    base = args.backend / "benchmarks/tv2_canonical_v1"
    gold = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (base / "gold-evidence-v1.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        if row["id"].startswith(STRATUM_PREFIX)
    }
    questions = {
        row["id"]: row.get("question")
        for row in (
            json.loads(line)
            for line in (base / "canonical-eval-v1.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }

    cases: list[dict] = []
    summary: collections.Counter = collections.Counter()

    for case_id in sorted(gold):
        rows = gold[case_id]
        entities = sorted(str(e) for e in (rows.get("values") or {}))
        if not entities:
            continue

        # Candidates: metrics the store holds for every entity, at this case's
        # period, without an ambiguous coordinate for any of them.
        period = str(rows.get("period") or "")
        candidates = []
        for metric in {key[0] for key in by_metric}:
            if metric.strip().casefold() in _NON_CONCEPTS:
                continue
            statuses = {entity: status_of(entity, metric, period) for entity in entities}
            if any(status in ("absent", "ambiguous") for status in statuses.values()):
                continue
            if all(status == "unreadable" for status in statuses.values()):
                continue
            values = {
                entity: by_metric[(metric, entity, period)][0].get("value")
                for entity in entities
            }
            # Prefer a concept that is exactly one stored fact per entity -- a
            # coordinate holding two *identical* values is still one reading,
            # but one row per entity is the shape a primary statement line has.
            singles = sum(1 for s in statuses.values() if s == "unique")
            candidates.append(
                {
                    "metric": metric,
                    "values": values,
                    "statuses": statuses,
                    "single_fact_entities": singles,
                    "entities": len(entities),
                }
            )

        candidates.sort(
            key=lambda c: (
                -c["single_fact_entities"],
                len(str(c["metric"])),
                str(c["metric"]),
            )
        )
        verdict = "HAS_CANDIDATES" if candidates else "NO_SHARED_CONCEPT"
        summary[verdict] += 1
        cases.append(
            {
                "case_id": case_id,
                "operation": rows.get("operation"),
                "question": questions.get(case_id),
                "current_metric": rows.get("metric"),
                "current_values": rows.get("values"),
                "verdict": verdict,
                "candidates": candidates[: args.top],
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cross-entity-metric-candidates.json").write_text(
        json.dumps(
            {"phase": "P1.6-0G", "summary": dict(summary), "cases": cases},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for case in cases:
        print(f"=== {case['case_id']}  op={case['operation']}  "
              f"current={case['current_metric']!r}  [{case['verdict']}]")
        for candidate in case["candidates"][:4]:
            values = " | ".join(
                f"{entity[:12]}={candidate['values'][entity]}" for entity in candidate["values"]
            )
            print(f"      {candidate['metric'][:40]:40} {values[:95]}")
        print()
    print(f"summary: {dict(summary)}")
    print(f"written to {args.out / 'cross-entity-metric-candidates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
