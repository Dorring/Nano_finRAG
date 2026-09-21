"""P1.6-A9: does retrieval resolve the stratum through the canonical store?

The wrapper's job is to take a slot's metric name and return the company-level
fact, falling back to the legacy store for anything it cannot map.  This runs the
20 re-derived cases through `CanonicalFactStore.facts_at_coordinate` -- the same
call retrieval makes -- and compares what comes back to the values verified
against the filings.

It uses the **re-derived** metric names, which is what step 3 will write into the
fixtures.  The fixtures today still carry the old strings (`United States`,
`Current`, `Total`), which are deliberately absent from the metric map, so a run
against them would show the fallback working and nothing else.

  python ab_canonical_retrieval.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

LEGACY_STORE = Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl")
IXBRL_STORE = Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts-ixbrl-v1.jsonl")

#: case -> (metric name as the re-derived fixture will carry it, entities).
CASES = {
    "compare-001": ("Net income", ["The Coca-Cola Company", "Tesla"]),
    "compare-002": ("Net income", ["Apple", "Visa"]),
    "compare-003": ("Total assets", ["Tesla", "The Coca-Cola Company"]),
    "compare-004": ("Net income", ["Apple", "Microsoft"]),
    "compare-005": ("Diluted earnings per share", ["JPMorganChase", "Apple"]),
    "compare-006": ("Comprehensive income", ["JPMorganChase", "Tesla"]),
    "compare-007": ("Net income", ["JPMorganChase", "Pfizer"]),
    "compare-008": ("Diluted earnings per share", ["JPMorganChase", "Apple"]),
    "compare-009": ("Net income", ["Apple", "JPMorganChase"]),
    "compare-010": ("Operating income", ["Apple", "Microsoft"]),
    "crossdiff-001": ("Net income", ["NVIDIA", "Tesla"]),
    "crossdiff-002": ("Net income", ["Apple", "Microsoft"]),
    "crossdiff-003": ("Total liabilities", ["Apple", "Tesla"]),
    "crossdiff-004": ("Long-term debt", ["The Coca-Cola Company", "Visa"]),
    "crossdiff-005": ("Total assets", ["NVIDIA", "The Coca-Cola Company"]),
    "rank-001": ("Net income", ["Tesla", "The Coca-Cola Company", "Visa"]),
    "rank-002": ("Research and development", ["Apple", "Microsoft", "Tesla"]),
    "rank-003": ("Comprehensive income",
                 ["JPMorganChase", "Microsoft", "Tesla", "Visa"]),
    "rank-004": ("Interest expense",
                 ["JPMorganChase", "The Coca-Cola Company", "Visa", "Microsoft"]),
    "rank-005": ("Operating income",
                 ["Apple", "Microsoft", "The Coca-Cola Company"]),
}

#: (case, entity) -> the value verified against the filing in P1.6-A6.
EXPECTED = {
    ("compare-001", "The Coca-Cola Company"): "13137", ("compare-001", "Tesla"): "3855",
    ("compare-002", "Apple"): "112010", ("compare-002", "Visa"): "20058",
    ("compare-003", "Tesla"): "137806", ("compare-003", "The Coca-Cola Company"): "104816",
    ("compare-004", "Apple"): "112010", ("compare-004", "Microsoft"): "101832",
    ("compare-005", "JPMorganChase"): "20.02", ("compare-005", "Apple"): "7.46",
    ("compare-006", "JPMorganChase"): "65214", ("compare-006", "Tesla"): "4886",
    ("compare-007", "JPMorganChase"): "57048", ("compare-007", "Pfizer"): "8062",
    ("compare-009", "Apple"): "112010", ("compare-009", "JPMorganChase"): "57048",
    ("compare-010", "Apple"): "133050", ("compare-010", "Microsoft"): "128528",
    ("crossdiff-001", "NVIDIA"): "72880", ("crossdiff-001", "Tesla"): "3855",
    ("crossdiff-002", "Apple"): "112010", ("crossdiff-002", "Microsoft"): "101832",
    ("crossdiff-003", "Apple"): "285508", ("crossdiff-003", "Tesla"): "54941",
    ("crossdiff-004", "The Coca-Cola Company"): "42119",
    ("crossdiff-004", "Visa"): "19602",
    ("crossdiff-005", "NVIDIA"): "111601",
    ("crossdiff-005", "The Coca-Cola Company"): "104816",
    ("rank-001", "Tesla"): "3855", ("rank-001", "The Coca-Cola Company"): "13137",
    ("rank-001", "Visa"): "20058",
    ("rank-002", "Apple"): "34550", ("rank-002", "Microsoft"): "32488",
    ("rank-002", "Tesla"): "6411",
    ("rank-003", "JPMorganChase"): "65214", ("rank-003", "Microsoft"): "104075",
    ("rank-003", "Tesla"): "4886", ("rank-003", "Visa"): "20614",
    ("rank-004", "JPMorganChase"): "97898",
    ("rank-004", "The Coca-Cola Company"): "1654", ("rank-004", "Visa"): "589",
    ("rank-004", "Microsoft"): "2385",
    ("rank-005", "Apple"): "133050", ("rank-005", "Microsoft"): "128528",
    ("rank-005", "The Coca-Cola Company"): "13762",
}

FISCAL_YEAR = {"Pfizer": "FY2024"}


def _digits(text: object) -> str:
    import re

    cleaned = re.sub(r"[^\d.]", "", str(text or ""))
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    return cleaned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from src.runtime.trusted_v2_canonical_store import CanonicalFactStore
    from src.runtime.trusted_v2_production import StructuredFactStore

    legacy = StructuredFactStore(LEGACY_STORE)
    store = CanonicalFactStore(legacy, IXBRL_STORE)
    print(f"iXBRL store available: {store.canonical_ready} "
          f"({len(store._records)} records)")

    rows = []
    tally = {"RESOLVED": 0, "AMBIGUOUS": 0, "EMPTY": 0}
    matched = checked = 0
    for case, (metric, entities) in sorted(CASES.items()):
        for entity in entities:
            period = FISCAL_YEAR.get(entity, "FY2025")
            facts = store.facts_at_coordinate(entity, metric, period)
            values = sorted({_digits(f.get("value")) for f in facts})
            if not facts:
                status = "EMPTY"
            elif len(values) == 1:
                status = "RESOLVED"
            else:
                status = "AMBIGUOUS"
            tally[status] += 1
            expected = EXPECTED.get((case, entity))
            if expected is not None:
                checked += 1
                if values == [expected]:
                    matched += 1
            rows.append({
                "case": case, "entity": entity, "metric": metric,
                "canonical": store.canonical_quantity(metric),
                "status": status, "values": values[:4], "expected": expected,
                "source": facts[0].get("source") if facts else None,
            })

    report = {"phase": "P1.6-A9", "tally": tally,
              "expected_matched": matched, "expected_checked": checked,
              "rows": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "canonical-retrieval-ab.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\n=== retrieval through CanonicalFactStore ===")
    for row in rows:
        mark = "ok " if row["status"] == "RESOLVED" else "!! "
        print(f"  {mark}{row['case']:14} {row['entity'][:22]:22} "
              f"{str(row['canonical'])[:22]:22} {str(row['values'])[:16]:16} "
              f"via={row['source']}")
    print(f"\n  {tally}")
    print(f"  expected values: {matched}/{checked} matched")
    print(f"  written to {args.out / 'canonical-retrieval-ab.json'}")
    return 0 if tally["EMPTY"] == 0 and tally["AMBIGUOUS"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
