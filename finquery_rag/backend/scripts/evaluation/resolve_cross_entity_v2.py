"""P1.6-A2B-13: resolve every entity of every cross-entity case against Store V2.

The stratum-level check.  With eight filings built, every case's entities exist in
one store for the first time, so the question the stratum has been unable to ask can
be asked: for each case, does each entity's slot resolve to a company-level figure,
or does it refuse?

Refusal is reported as a result, not a failure.  A slot whose candidates disagree on
scope and cannot be ordered is one the store cannot settle, and picking the best
available would be the confident-wrong-answer this line of work exists to remove.

  python resolve_cross_entity_v2.py --out <dir>
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

STORE = Path("/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a2b12-store-v2/store-v2.jsonl")
KO, JPM = "The Coca-Cola Company", "JPMorganChase"

#: case -> (metric as the re-derived fixture names it, entities).  From
#: docs/evaluation/p1-6-0g-rederivation.md.
CASES = {
    "compare-001": ("Net income", [KO, "Tesla"]),
    "compare-002": ("Net income", ["Apple", "Visa"]),
    "compare-003": ("Total assets", ["Tesla", KO]),
    "compare-004": ("Net income", ["Apple", "Microsoft"]),
    "compare-005": ("Diluted earnings per share", [JPM, "Apple"]),
    "compare-006": ("Comprehensive income", [JPM, "Tesla"]),
    "compare-007": ("Net income", [JPM, "Pfizer"]),
    "compare-008": ("Diluted earnings per share", [JPM, "Apple"]),
    "compare-009": ("Net income", ["Apple", JPM]),
    "compare-010": ("Operating income", ["Apple", "Microsoft"]),
    "crossdiff-001": ("Net income", ["NVIDIA", "Tesla"]),
    "crossdiff-002": ("Net income", ["Apple", "Microsoft"]),
    "crossdiff-003": ("Total liabilities", ["Apple", "Tesla"]),
    "crossdiff-004": ("Long-term debt", [KO, "Visa"]),
    "crossdiff-005": ("Total assets", ["NVIDIA", KO]),
    "rank-001": ("Net income", ["Tesla", KO, "Visa"]),
    "rank-002": ("Research and development", ["Apple", "Microsoft", "Tesla"]),
    "rank-003": ("Comprehensive income", [JPM, "Microsoft", "Tesla", "Visa"]),
    "rank-004": ("Interest expense", [JPM, KO, "Visa", "Microsoft"]),
    "rank-005": ("Operating income", ["Apple", "Microsoft", KO]),
}

#: Values verified against the filings earlier in this work, where we have them.
EXPECTED = {
    ("compare-009", JPM): "57,048", ("compare-009", "Apple"): "112,010",
    ("rank-002", "Tesla"): "6,411", ("rank-002", "Apple"): "34,550",
    ("compare-005", JPM): "20.02", ("compare-005", "Apple"): "7.46",
}

PERIOD = {"Pfizer": "FY2024"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "resolver", _BACKEND_DIR / "scripts/evaluation/resolve_store_v2_slot.py"
    )
    resolver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resolver)

    records = [
        json.loads(line)
        for line in args.store.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"=== cross-entity slots against Store V2 ({len(records)} records) ===")
    print()

    rows = []
    statuses = collections.Counter()
    matched = checked = 0
    for case_id, (metric, entities) in sorted(CASES.items()):
        slots = []
        for entity in entities:
            result = resolver.resolve(
                records, entity, metric, PERIOD.get(entity, "FY2025")
            )
            statuses[result["status"]] += 1
            if result["status"] == "RESOLVED_COMPANY_LEVEL":
                expected = EXPECTED.get((case_id, entity))
                if expected is not None:
                    checked += 1
                    if str(result.get("value_raw")) == expected:
                        matched += 1
            slots.append({"entity": entity, **{k: v for k, v in result.items()
                                              if k != "status"},
                          "status": result["status"]})
        resolved = sum(1 for s in slots if s["status"] == "RESOLVED_COMPANY_LEVEL")
        case_state = "FULLY_RESOLVED" if resolved == len(slots) else (
            "PARTIALLY_RESOLVED" if resolved else "UNRESOLVED")
        statuses[case_state] += 1
        rows.append({"case_id": case_id, "metric": metric,
                     "case_state": case_state, "slots": slots})
        print(f"  {case_id:14} {metric[:26]:26} {case_state}")
        for slot in slots:
            if slot["status"] == "RESOLVED_COMPANY_LEVEL":
                extra = ""
                expected = EXPECTED.get((case_id, slot["entity"]))
                if expected is not None:
                    ok = str(slot.get("value_raw")) == expected
                    extra = f"  [{'matches' if ok else 'DIFFERS from'} the verified {expected}]"
                print(f"      ok  {slot['entity'][:22]:22} {str(slot.get('value_raw')):>10}"
                      f"  under {str(slot.get('column_header'))[:44]!r}{extra}")
            else:
                print(f"      --  {slot['entity'][:22]:22} {slot['status']}")
    print()

    report = {"phase": "P1.6-A2B-13", "mutation": "none",
              "store": str(args.store), "records": len(records),
              "statuses": dict(statuses),
              "verified_matches": matched, "verified_checked": checked,
              "cases": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cross-entity-v2-resolution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"  slot statuses: {dict(statuses)}")
    print(f"  verified values: {matched}/{checked} matched")
    print(f"  written to {args.out / 'cross-entity-v2-resolution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
