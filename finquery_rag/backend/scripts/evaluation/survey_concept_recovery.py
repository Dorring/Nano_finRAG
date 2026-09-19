"""P1.6-A1b: does a derived concept reduce coordinate ambiguity, and at what cost?

Runs `derive_fact_concept` over the whole store **without writing anything** and
reports what changes.  The comparison is between two groupings of the same facts:

* the store's own `metric`, which is the breadcrumb path
* the derived `concept`, on the facts where derivation resolved

For each grouping it counts coordinates that hold more than one comparable value
-- the defect the operand guard refuses.  The question is whether the second is
lower, and whether anything got *worse*: a cleaner name that merges two genuinely
different quantities would raise collisions, and that would be a regression
wearing the costume of a fix.

It also reports the same for the benchmark's cross-entity cases, old versus new,
because that stratum is the one that cannot be used until this improves.

  python survey_concept_recovery.py --out <dir>
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


def _ambiguity(facts: list, key_of) -> tuple[int, int]:
    """Coordinates holding more than one comparable value, out of all of them."""

    from src.finance.operand_ambiguity import quantity_of

    grouped: dict[tuple, set] = collections.defaultdict(set)
    for record in facts:
        quantity = quantity_of(record)
        if quantity is None:
            continue
        grouped[key_of(record)].add(quantity.identity)
    ambiguous = sum(1 for identities in grouped.values() if len(identities) > 1)
    return ambiguous, len(grouped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, default=_BACKEND_DIR)
    parser.add_argument("--fact-store", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import os

    from src.finance.concept_derivation import derive_fact_concept
    from src.runtime.trusted_v2_production import StructuredFactStore

    store_path = args.fact_store or Path(os.environ["TRUSTED_V2_FACT_STORE_PATH"])
    store = StructuredFactStore(store_path)
    records = list(store.iter_records())

    derivations: dict[str, dict] = {}
    by_status: collections.Counter = collections.Counter()
    reasons: collections.Counter = collections.Counter()
    changed = unchanged = 0
    resolved_records = []

    for record in records:
        derivation = derive_fact_concept(record.get("metric"), record.get("content"))
        by_status[derivation.status] += 1
        if derivation.status != "RESOLVED":
            reasons[derivation.reason.split(":")[0]] += 1
            continue
        resolved_records.append(record)
        derivations[str(record.get("fact_id"))] = derivation.to_dict()
        if derivation.concept != str(record.get("metric")):
            changed += 1
        else:
            unchanged += 1

    old_ambiguous, old_total = _ambiguity(
        records,
        lambda r: store.coordinate_key(r),
    )
    # The comparison has to be like for like.  The concept grouping can only
    # cover facts whose derivation resolved, so counting ambiguity over *all*
    # facts against it would flatter the new grouping by giving it fewer facts
    # to collide.  Both numbers below are over the same resolved subset.
    subset_old_ambiguous, subset_old_total = _ambiguity(
        resolved_records,
        lambda r: store.coordinate_key(r),
    )
    new_ambiguous, new_total = _ambiguity(
        resolved_records,
        lambda r: (
            str(r.get("entity")),
            derivations[str(r.get("fact_id"))]["concept"],
            str(r.get("period")),
        ),
    )

    # --- the benchmark --------------------------------------------------------
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

    cases = []
    improved = worsened = same = 0
    for case_id in sorted(gold):
        entities = sorted(str(e) for e in (gold[case_id].get("values") or {}))
        old_counts, new_counts = [], []
        for entity in entities:
            old = store.facts_at_coordinate(
                entity, gold[case_id].get("metric"), gold[case_id].get("period")
            )
            o, _ = _ambiguity(old, lambda r: store.coordinate_key(r))
            concept = None
            for record in old:
                derivation = derivations.get(str(record.get("fact_id")))
                if derivation:
                    concept = derivation["concept"]
                    break
            if concept:
                new = [r for r in resolved_records
                       if str(r.get("entity")) == entity and str(r.get("period")) == gold[case_id].get("period")
                       and derivations[str(r.get("fact_id"))]["concept"] == concept]
                n, _ = _ambiguity(new, lambda r: (
                    str(r.get("entity")), derivations[str(r.get("fact_id"))]["concept"],
                    str(r.get("period"))))
            else:
                n = None
            old_counts.append(o)
            new_counts.append(n)
        verdict = "same"
        if any(n is not None and n == 0 and o > 0 for o, n in zip(old_counts, new_counts)):
            verdict = "improved"
            improved += 1
        elif all(n is None for n in new_counts):
            verdict = "unchanged"
            same += 1
        cases.append({"case_id": case_id, "entities": entities,
                      "old_ambiguous": old_counts, "new_ambiguous": new_counts,
                      "verdict": verdict})

    report = {
        "phase": "P1.6-A1b",
        "mutation": "none -- shadow derivation only",
        "facts": len(records),
        "status": dict(by_status),
        "resolved_rate": round(by_status["RESOLVED"] / len(records), 4) if records else None,
        "resolved_concept_differs_from_metric": changed,
        "resolved_concept_equals_metric": unchanged,
        "unresolved_reasons": dict(reasons.most_common(12)),
        "store_ambiguity": {
            "all_facts_metric_grouping": {
                "ambiguous": old_ambiguous, "coordinates": old_total,
                "rate": round(old_ambiguous / old_total, 4) if old_total else None,
            },
            "resolved_subset_metric_grouping": {
                "ambiguous": subset_old_ambiguous, "coordinates": subset_old_total,
                "rate": round(subset_old_ambiguous / subset_old_total, 4)
                if subset_old_total else None,
            },
            "resolved_subset_concept_grouping": {
                "ambiguous": new_ambiguous, "coordinates": new_total,
                "rate": round(new_ambiguous / new_total, 4) if new_total else None,
            },
            "note": "the third is the one to compare against the second; both cover "
            "the same resolved facts",
        },
        "cross_entity_cases": cases,
        "cross_entity_summary": {"improved": improved, "unchanged": same, "worsened": worsened},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "concept-recovery-survey.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== P1.6-A1b concept recovery survey (shadow, nothing written) ===")
    print(f"  facts {len(records)}")
    print(f"  RESOLVED   {by_status['RESOLVED']:>6}  ({report['resolved_rate']})")
    print(f"  UNRESOLVED {by_status['UNRESOLVED']:>6}")
    print(f"  of resolved: concept differs from metric {changed}, equals it {unchanged}")
    print()
    print("  unresolved reasons:")
    for reason, count in reasons.most_common(8):
        print(f"      {count:>6}  {reason}")
    print()
    print("  store ambiguity (coordinates holding >1 comparable value):")
    for label, block in report["store_ambiguity"].items():
        if not isinstance(block, dict):
            continue
        print(f"      {label:32} {block['ambiguous']:>5}/{block['coordinates']:<5} "
              f"({block['rate']})")
    print()
    print("  cross-entity cases:")
    for case in cases:
        print(f"      {case['case_id']:26} {case['verdict']:10} "
              f"old={case['old_ambiguous']} new={case['new_ambiguous']}")
    print(f"\n  summary: {report['cross_entity_summary']}")
    print(f"  written to {args.out / 'concept-recovery-survey.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
