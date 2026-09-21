"""How far the store's coordinate loses the dimension that separates its facts.

P1.6-0 established that `(entity, metric, period)` does not identify a value, and
the guard now refuses operands read from a coordinate that holds several.  That
was measured on the cross-entity stratum.  This measures the whole store and the
whole benchmark, because the size of P1.6-A depends on it: if the defect is a
handful of segment tables it is a parser fix, and if it is most of the store it
is a different representation.

For every coordinate it counts the distinct **comparable** values stored under
it -- comparable, not raw distinct, because a percentage and an amount are not
competing answers to one question and counting them as such invents conflicts
that do not exist (`compare-002`'s Visa coordinate holds `1,926` and `21 %`).

It then reports, per benchmark stratum, how many cases have at least one gold
fact sitting at such a coordinate.  A case whose gold is read from an ambiguous
coordinate cannot be won by being right: the system has no way to know which of
the stored values the question wants, and the gold had to pick one.

  python survey_coordinate_ambiguity.py --out <dir>
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, default=_BACKEND_DIR)
    parser.add_argument("--fact-store", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import os

    from src.finance.operand_ambiguity import quantity_of
    from src.runtime.trusted_v2_production import StructuredFactStore

    store_path = args.fact_store or Path(os.environ["TRUSTED_V2_FACT_STORE_PATH"])
    store = StructuredFactStore(store_path)

    records = store.iter_records()

    # --- the store, coordinate by coordinate ---------------------------------

    by_coordinate: dict[tuple, list] = collections.defaultdict(list)
    for record in records:
        quantity = quantity_of(record)
        by_coordinate[store.coordinate_key(record)].append(
            (record, quantity.identity if quantity is not None else None)
        )

    ambiguous: dict[tuple, int] = {}
    for key, entries in by_coordinate.items():
        identities = {identity for _record, identity in entries if identity is not None}
        if len(identities) > 1:
            ambiguous[key] = len(identities)

    distribution = collections.Counter(ambiguous.values())
    metrics = collections.Counter(key[1] for key in ambiguous)
    entities = collections.Counter(key[0] for key in ambiguous)

    # A coordinate with no readable quantity at all is a different problem and
    # is reported rather than silently treated as unique.
    unreadable = sum(
        1
        for entries in by_coordinate.values()
        if all(identity is None for _record, identity in entries)
    )

    # --- the benchmark --------------------------------------------------------

    gold_path = args.backend / "benchmarks/tv2_canonical_v1/gold-evidence-v1.jsonl"
    eval_path = args.backend / "benchmarks/tv2_canonical_v1/canonical-eval-v1.jsonl"
    gold = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in gold_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    stratum_of = {
        row["id"]: row.get("stratum")
        for row in (
            json.loads(line)
            for line in eval_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    by_stratum: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    affected_cases: list[dict] = []
    for case_id, row in sorted(gold.items()):
        stratum = str(stratum_of.get(case_id))
        bucket = by_stratum[stratum]
        bucket["cases"] += 1
        hits = []
        for fact_id in row.get("fact_ids") or ():
            record = store._by_candidate.get(fact_id)
            if record is None:
                continue
            key = store.coordinate_key(record)
            if key in ambiguous:
                hits.append(
                    {
                        "entity": record.get("entity"),
                        "metric": record.get("metric"),
                        "period": record.get("period"),
                        "value": record.get("value"),
                        "distinct_values": ambiguous[key],
                    }
                )
        if hits:
            bucket["cases_with_ambiguous_gold"] += 1
            affected_cases.append(
                {"case_id": case_id, "stratum": stratum, "slots": hits}
            )

    report = {
        "fact_store": str(store_path),
        "store": {
            "facts": len(records),
            "coordinates": len(by_coordinate),
            "ambiguous_coordinates": len(ambiguous),
            "ambiguous_rate": round(len(ambiguous) / len(by_coordinate), 4)
            if by_coordinate
            else None,
            "coordinates_with_no_readable_quantity": unreadable,
            "distinct_value_distribution": {
                str(k): v for k, v in sorted(distribution.items())
            },
            "most_affected_metrics": metrics.most_common(20),
            "most_affected_entities": entities.most_common(20),
            "worst_coordinates": [
                {
                    "entity": key[0],
                    "metric": key[1],
                    "period": key[2],
                    "distinct_values": count,
                }
                for key, count in sorted(
                    ambiguous.items(), key=lambda item: -item[1]
                )[:25]
            ],
        },
        "benchmark": {
            stratum: dict(sorted(counter.items()))
            for stratum, counter in sorted(by_stratum.items())
        },
        "affected_cases": affected_cases,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "coordinate-ambiguity-survey.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    store_report = report["store"]
    print("=== coordinate ambiguity survey ===")
    print(f"  facts {store_report['facts']}  coordinates "
          f"{store_report['coordinates']}  ambiguous "
          f"{store_report['ambiguous_coordinates']} "
          f"({store_report['ambiguous_rate']})")
    print(f"  coordinates with no readable quantity: "
          f"{store_report['coordinates_with_no_readable_quantity']}")
    print()
    print("  distinct values per ambiguous coordinate:")
    for count, n in sorted(distribution.items()):
        print(f"      {count:>4} values  {n:>6} coordinates")
    print()
    print("  benchmark cases whose gold sits at an ambiguous coordinate:")
    for stratum, counter in sorted(by_stratum.items()):
        total = counter["cases"]
        hit = counter["cases_with_ambiguous_gold"]
        print(f"      {stratum:26} {hit:>3}/{total:<3} "
              f"({round(hit / total, 3) if total else 0})")
    print()
    print("  metrics most often ambiguous:")
    for metric, count in metrics.most_common(12):
        print(f"      {str(metric)[:46]:46} {count}")
    print()
    print(f"  written to {args.out / 'coordinate-ambiguity-survey.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
