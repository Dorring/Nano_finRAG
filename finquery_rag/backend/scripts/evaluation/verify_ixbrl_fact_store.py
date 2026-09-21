"""P1.6-A8: does the rebuilt store answer the cross-entity stratum?

Reads the file that was written, not the reasoning that produced it.  A store
that resolves correctly in memory and is written wrongly is a real failure mode,
and the only way to catch it is to load what is on disk.

For each entity of each of the 20 re-derived cases, it asks the file for the
company-level fact of the case's canonical quantity and requires exactly one
value.  Company level means `dimension_count == 0` -- which the store records
explicitly, rather than leaving it to be inferred.

  python verify_ixbrl_fact_store.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

DEFAULT_STORE = Path(
    "/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts-ixbrl-v1.jsonl"
)

#: canonical quantity -> entities, per case.  The re-derived stratum.
CASES = {
    "compare-001": ("net_income", ["The Coca-Cola Company", "Tesla"]),
    "compare-002": ("net_income", ["Apple", "Visa"]),
    "compare-003": ("total_assets", ["Tesla", "The Coca-Cola Company"]),
    "compare-004": ("net_income", ["Apple", "Microsoft"]),
    "compare-005": ("diluted_eps", ["JPMorganChase", "Apple"]),
    "compare-006": ("comprehensive_income", ["JPMorganChase", "Tesla"]),
    "compare-007": ("net_income", ["JPMorganChase", "Pfizer"]),
    "compare-008": ("diluted_eps", ["JPMorganChase", "Apple"]),
    "compare-009": ("net_income", ["Apple", "JPMorganChase"]),
    "compare-010": ("operating_income", ["Apple", "Microsoft"]),
    "crossdiff-001": ("net_income", ["NVIDIA", "Tesla"]),
    "crossdiff-002": ("net_income", ["Apple", "Microsoft"]),
    "crossdiff-003": ("total_liabilities", ["Apple", "Tesla"]),
    "crossdiff-004": ("long_term_debt", ["The Coca-Cola Company", "Visa"]),
    "crossdiff-005": ("total_assets", ["NVIDIA", "The Coca-Cola Company"]),
    "rank-001": ("net_income", ["Tesla", "The Coca-Cola Company", "Visa"]),
    "rank-002": ("research_and_development", ["Apple", "Microsoft", "Tesla"]),
    "rank-003": ("comprehensive_income",
                 ["JPMorganChase", "Microsoft", "Tesla", "Visa"]),
    "rank-004": ("interest_expense",
                 ["JPMorganChase", "The Coca-Cola Company", "Visa", "Microsoft"]),
    "rank-005": ("operating_income",
                 ["Apple", "Microsoft", "The Coca-Cola Company"]),
}

#: (case, entity) -> the value verified against the filing in P1.6-A6.
EXPECTED = {
    ("compare-001", "The Coca-Cola Company"): "13137", ("compare-001", "Tesla"): "3855",
    ("compare-009", "Apple"): "112010", ("compare-009", "JPMorganChase"): "57048",
    ("compare-005", "JPMorganChase"): "20.02", ("compare-005", "Apple"): "7.46",
    ("compare-006", "JPMorganChase"): "65214", ("compare-006", "Tesla"): "4886",
    ("compare-010", "Apple"): "133050", ("compare-010", "Microsoft"): "128528",
    ("crossdiff-003", "Apple"): "285508", ("crossdiff-003", "Tesla"): "54941",
    ("crossdiff-004", "The Coca-Cola Company"): "42119",
    ("crossdiff-004", "Visa"): "19602",
    ("crossdiff-005", "NVIDIA"): "111601",
    ("crossdiff-005", "The Coca-Cola Company"): "104816",
    ("rank-002", "Apple"): "34550", ("rank-002", "Microsoft"): "32488",
    ("rank-002", "Tesla"): "6411",
    ("rank-004", "JPMorganChase"): "97898",
    ("rank-004", "The Coca-Cola Company"): "1654", ("rank-004", "Visa"): "589",
    ("rank-004", "Microsoft"): "2385",
    ("rank-005", "Apple"): "133050", ("rank-005", "Microsoft"): "128528",
    ("rank-005", "The Coca-Cola Company"): "13762",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = parser.parse_args(argv)

    if not args.store.is_file():
        print(f"store not found: {args.store}")
        return 1

    index: dict[tuple, set] = defaultdict(set)
    total = 0
    with args.store.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            total += 1
            if record.get("dimension_count") != 0:
                continue
            index[
                (record.get("entity"), record.get("concept"), record.get("period"))
            ].add(record.get("value"))

    print(f"loaded {total} records from {args.store.name}")

    from src.finance.concept_alignment import CONCEPT_ALIGNMENT

    checked = matched = 0
    problems = []
    for case, (canonical, entities) in sorted(CASES.items()):
        for entity in entities:
            # The alignment's ordering is applied **here**, at query time.
            # Storing `canonical_concept` alone collapses `ProfitLoss` and
            # `NetIncomeLoss` into `net_income` and loses what separates them;
            # keying on it alone reports Tesla as holding both 3,794 and 3,855,
            # which is the ambiguity this whole line of work removed.
            for concept in CONCEPT_ALIGNMENT[canonical]:
                hits = {
                    period: values
                    for (e, c, period), values in index.items()
                    if e == entity and c == concept
                }
                if not hits:
                    continue
                # Prefer a fiscal-year label: an instant context resolves to
                # ASOF<date> only when it is not the reporting date.
                fiscal = {p: v for p, v in hits.items() if str(p).startswith("FY")}
                period, values = max((fiscal or hits).items(), key=lambda i: i[0])
                if len(values) != 1:
                    problems.append((case, entity, concept, "AMBIGUOUS",
                                     sorted(values)[:4]))
                    break
                value = next(iter(values))
                expected = EXPECTED.get((case, entity))
                if expected is not None:
                    checked += 1
                    if value == expected:
                        matched += 1
                    else:
                        problems.append((case, entity, concept, "VALUE_MISMATCH",
                                         [value, f"expected {expected}"]))
                print(f"  ok  {case:14} {entity[:22]:22} {canonical[:20]:20} "
                      f"{concept.replace('us-gaap:', '')[:26]:26} {str(period):8} {value}")
                break
            else:
                problems.append((case, entity, "-", "NO_COMPANY_LEVEL_FACT", []))

    print()
    if problems:
        print("=== problems ===")
        for case, entity, concept, kind, detail in problems:
            print(f"  {kind:22} {case:14} {entity[:22]:22} {detail}")
    else:
        print("=== no problems ===")
    print(f"\n  expected-value checks: {matched}/{checked} matched")
    print(f"  entity resolutions:    {sum(len(e) for _c, e in CASES.values())} attempted")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
