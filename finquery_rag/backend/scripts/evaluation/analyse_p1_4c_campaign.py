"""P1.4c: the repaired fixture (v4) against the frozen one (v3), code held still.

Both columns are three ALIGNED-REPLAY runs at commit 8551135.  The "before"
column is the tail of the P1.4 campaign -- it was taken at this same commit on
``plan-fixtures-v3`` -- so the only variable across the columns is the fixture,
and every other seal field must come out identical.  If one does not, the two
columns are not one experiment and nothing below is a fixture effect.

Answers, in order:
1. The seal, column against column.
2. Whether the eight percentage_share cases now bind and calculate, and what
   ``pctshare-003`` computes.
3. The MISSING_OPERAND delta, attributed case by case.
4. The cases that must not move.
5. The other 112, against the repeat spread.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

BEFORE_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else
                  "/disk/qh/nano-finrag/artifacts/evaluation/p1-4-percentage-campaign")
AFTER_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else
                 "/disk/qh/nano-finrag/artifacts/evaluation/p1-4c-percentage-fixture")

BEFORE = ["p14b-run1", "p14b-run2", "p14b-run3"]
AFTER = ["v4-run1", "v4-run2", "v4-run3"]

HELD = {
    "tv2f01-s2-average-002": "average-002",
    "tv2f01-s1-aapl-003": "aapl-003",
    "tv2f01-s1-jpm-010": "jpm-010",
    "tv2f01-s1-nvda-024": "nvda-024",
    "tv2f01-s3-compare-004": "compare-004",
    "tv2f01-s3-compare-007": "compare-007",
}

SEAL_KEYS = ("commit_sha", "config_fingerprint", "eval_set_sha256",
             "gold_sha256", "question_count")


def _rows(directory: Path, name: str) -> list[dict[str, Any]]:
    path = directory / name / "replay-predictions.jsonl"
    if not path.exists():
        raise SystemExit(f"missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _seal(directory: Path, name: str) -> dict[str, Any]:
    path = directory / name / "seal.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in rows}


def _cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return "absent"
    if row.get("release_status") == "RELEASED":
        return "REL"
    reasons = row.get("reason_codes") or []
    return "not/" + (reasons[0] if reasons else "-")


def _calculated(row: dict[str, Any]) -> str:
    calcs = row.get("calculations") or []
    if not calcs:
        return "-"
    first = calcs[0] if isinstance(calcs[0], dict) else {}
    for key in ("value", "points_value"):
        if first.get(key) is not None:
            return str(first[key])
    return str(first)[:40]


def main() -> int:
    before = {name: _rows(BEFORE_DIR, name) for name in BEFORE}
    after = {name: _rows(AFTER_DIR, name) for name in AFTER}
    all_ids = [row["id"] for row in before[BEFORE[0]]]

    print("=" * 78)
    print("1. SEAL — one variable, the fixture")
    print("=" * 78)
    for key in SEAL_KEYS:
        b = {str(_seal(BEFORE_DIR, n).get(key)) for n in BEFORE}
        a = {str(_seal(AFTER_DIR, n).get(key)) for n in AFTER}
        same = b == a and len(b) == 1
        print(f"  [{'OK  ' if same else 'DIFF'}] {key:<20} {sorted(b)[0] if len(b)==1 else sorted(b)}"
              f"{'' if same else '   vs   ' + str(sorted(a))}")
    bf = {str(_seal(BEFORE_DIR, n).get("fixture_sha256")) for n in BEFORE}
    af = {str(_seal(AFTER_DIR, n).get("fixture_sha256")) for n in AFTER}
    print(f"  [{'OK  ' if bf != af and len(af)==1 else 'DIFF'}] fixture_sha256       "
          f"v3={sorted(bf)[0][:16]}…  v4={sorted(af)[0][:16]}…")

    # The eight percentage_share cases, found from the fixture rather than listed.
    fixture = Path("/disk/qh/nano-finrag/finquery_rag/backend/benchmarks/"
                   "tv2_canonical_v1/plan-fixtures-v4.jsonl")
    share_ids = sorted(
        json.loads(line)["id"]
        for line in fixture.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line)["plan"].get("operation") == "percentage_share"
    )

    print()
    print("=" * 78)
    print(f"2. percentage_share — {len(share_ids)} cases, binder → release → value")
    print("=" * 78)
    for case_id in share_ids:
        line = [f"  {case_id:<26}"]
        for label, store in (("v3", before), ("v4", after)):
            rows = [_by_id(store[n]).get(case_id) for n in (BEFORE if label == "v3" else AFTER)]
            binder = {r.get("binder_final_status") for r in rows if r}
            released = {r.get("release_status") for r in rows if r}
            values = {_calculated(r) for r in rows if r}
            line.append(f"{label}:{sorted(binder)} {'REL' if released == {'RELEASED'} else 'not'}"
                        f"/{sorted(values)}")
        print("  ".join(line))

    print()
    print("=" * 78)
    print("3. MISSING_OPERAND, attributed")
    print("=" * 78)
    for label, store, names in (("v3", before, BEFORE), ("v4", after, AFTER)):
        counter: Counter[str] = Counter()
        for name in names:
            for row in store[name]:
                if "MISSING_OPERAND" in (row.get("reason_codes") or []):
                    counter[row["id"]] += 1
        print(f"\n  {label}: {sum(counter.values())} occurrences across {len(counter)} cases")
        for case_id, count in sorted(counter.items()):
            print(f"     {case_id:<28} x{count}")

    print()
    print("=" * 78)
    print("4. CASES THAT MUST NOT MOVE")
    print("=" * 78)
    for case_id, label in HELD.items():
        line = [f"  {label:<14}"]
        for tag, store, names in (("v3", before, BEFORE), ("v4", after, AFTER)):
            rows = [_by_id(store[n]).get(case_id) for n in names]
            line.append(f"{tag}:{sorted({_cell(r) for r in rows})}")
        print("  ".join(line))

    print()
    print("=" * 78)
    print("5. THE OTHER CASES — what must stay in the repeat spread")
    print("=" * 78)
    fixed = set(share_ids)
    for label, store, names in (("v3", before, BEFORE), ("v4", after, AFTER)):
        subsets = [[r for r in store[n] if r["id"] not in fixed] for n in names]
        bound = [sum(1 for r in s if r.get("binder_final_status") == "BOUND") for s in subsets]
        rel = [sum(1 for r in s if r.get("release_status") == "RELEASED") for s in subsets]
        print(f"  {label}  n={len(subsets[0])}  bound {bound} spread {max(bound)-min(bound)}"
              f"   released {rel} spread {max(rel)-min(rel)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
