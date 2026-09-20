"""P1.6-A3-W4-B2 acceptance: did the bridge persist the gains without loss or invention?

Three claims, in order.  The first is that the 6,386 facts the new admission decision adds
survive persistence at all; the second is that every one of them carries a *legitimate*
identity rather than a rendered one; the third is that the identities the line spent four
phases establishing come through unchanged.

  python accept_period_identity.py --store <store-v2.jsonl> --shadow <json> --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

BUILDER = _BACKEND_DIR / "scripts/evaluation/build_store_v2.py"

#: The one fact this acceptance is really about.  If a bridge can turn `YEAR(2025)` into
#: `2025-12-31` anywhere, it will do it here first: Coca-Cola's equity statement is the case
#: that made `PARTIAL` a status of its own.
KO_EQUITY = ("ko_fy2025", 12533)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    spec = importlib.util.spec_from_file_location("store_builder", BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    from src.runtime.trusted_v2_canonical_fact_store import build_canonical_fact_store

    records = [json.loads(line) for line in args.store.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    by_cell = {str(r.get("cell_id")): r for r in records if r.get("cell_id")}
    shadow = json.loads(args.shadow.read_text(encoding="utf-8"))
    failures: list[str] = []
    report: dict = {"phase": "P1.6-A3-W4-B2-accept", "store_records": len(records)}

    # --- 1. the gains survive, and the gate that ate them now reads the identity -------
    print("=== 1. did the 6,386 gains survive persistence? ===")
    summary = collections.Counter()
    for document_id, (ticker, accession) in sorted(builder.DOCUMENTS.items()):
        _records, s = build_canonical_fact_store(
            [builder.parse_filing(ticker, accession, document_id)])
        d = s.to_dict()
        for key in ("atomic_facts_seen", "emitted_facts", "skipped_missing_period",
                    "skipped_missing_metric", "skipped_missing_value",
                    "skipped_missing_provenance"):
            summary[key] += d.get(key, 0)
    for key in ("atomic_facts_seen", "emitted_facts", "skipped_missing_period",
                "skipped_missing_metric", "skipped_missing_value",
                "skipped_missing_provenance"):
        print(f"    {key:<28} {summary[key]}")
    report["canonical_summary"] = dict(summary)
    if summary["skipped_missing_period"] != 0:
        failures.append(f"skipped_missing_period = {summary['skipped_missing_period']}")
    if summary["atomic_facts_seen"] != summary["emitted_facts"]:
        failures.append("facts admitted and not emitted "
                        f"{summary['atomic_facts_seen'] - summary['emitted_facts']}")
    print()

    # --- 2. the gains' identities, one by one -----------------------------------------
    print("=== 2. the newly-admitted facts, and what identity they carry ===")
    gains = []
    for name in ("rule_gain", "producer_gain"):
        for entry in shadow.get(name, []):
            if entry["v2_outcome"] != "WITHHOLD":
                gains.append(entry)
    tally = collections.Counter()
    for entry in gains:
        record = by_cell.get(str(entry["cell_id"]))
        if record is None:
            tally["NOT IN THE STORE"] += 1
            continue
        status = str(record.get("period_binding_status") or "")
        granularity = str(record.get("period_granularity") or "")
        period = record.get("normalized_period")
        if status not in ("RESOLVED", "PARTIAL"):
            tally[f"status {status!r}"] += 1
        elif not period:
            tally["no normalized_period"] += 1
        elif granularity in ("", "UNKNOWN"):
            tally[f"granularity {granularity!r}"] += 1
        else:
            tally[f"ok: {status} / {granularity}"] += 1
    print(f"    gains the shadow says the migration adds: {len(gains)}")
    for key, count in tally.most_common():
        print(f"    {count:>6}  {key}")
    report["gains"] = {"expected": len(gains), "by_identity": dict(tally)}
    ok = tally.get("ok: RESOLVED / DAY", 0) + tally.get("ok: PARTIAL / YEAR", 0)
    if ok != len(gains):
        failures.append(f"only {ok} of {len(gains)} gains carry a legitimate identity")
    print()

    # --- 3. the identity this line spent four phases establishing ----------------------
    print("=== 3. Coca-Cola's equity statement, after persistence ===")
    document_id, order = KO_EQUITY
    parsed = builder.parse_filing(*builder.DOCUMENTS[document_id], document_id)
    table = next((t for t in parsed["tables"] if t.get("source_order") == order), None)
    shown = 0
    for cell in (table or {}).get("cells", []):
        record = by_cell.get(str(cell.get("cell_id")))
        if not record:
            continue
        if record.get("period_binding_status") != "PARTIAL":
            continue
        print(f"    {record.get('row_label')!r}")
        print(f"        normalized_period {record.get('normalized_period')!r}  "
              f"status {record.get('period_binding_status')!r}  "
              f"granularity {record.get('period_granularity')!r}  "
              f"period {record.get('period')!r}")
        shown += 1
        if shown >= 3:
            break
    report["ko_partial_shown"] = shown
    if not shown:
        failures.append("no PARTIAL fact found in KO's equity statement after persistence")
    print()

    print(f"=== store ===")
    print(f"    records {len(records)}")
    print(f"    predicted 26311 = 26977 + 6386 - 7052 (a prediction, not a target)")
    print()
    print(f"  failures: {failures or 'none'}")
    report["failures"] = failures
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "period-identity-acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'period-identity-acceptance.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
