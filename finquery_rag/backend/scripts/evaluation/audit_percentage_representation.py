"""Emit the canonical identity of every quantity the fact store holds.

This exists to make one claim checkable rather than asserted: that teaching the
canonicaliser to read a percentage off the value text changes *only* the
quantities that state a percentage.

A test suite cannot establish that.  After a change this wide, a green suite is
equally consistent with "nothing broke" and with "nothing ran" -- the new
behaviour simply may not be reached.  So this emits one line per distinct
coordinate tuple in the store, and the proof is a diff:

    git stash                                    # the code before the change
    python scripts/evaluation/audit_percentage_representation.py --out /tmp/before.tsv
    git stash pop
    python scripts/evaluation/audit_percentage_representation.py --out /tmp/after.tsv
    diff /tmp/before.tsv /tmp/after.tsv

Every line that differs must name a value carrying a representation token.  A
line that differs without one is a regression, and it is visible without reading
the implementation.

Distinct tuples rather than records: identity is a pure function of the four
coordinates, so the tuples cover every record in the store and are ~30% the size.
The ``count`` column carries how many records each tuple stands for, so the diff
can report how many records a change actually reaches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from rag_v2.contracts.financial_semantics import quantity_identity  # noqa: E402

DEFAULT_FACT_STORE = Path(
    "/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl"
)
DEFAULT_COORDINATES = _BACKEND_DIR / "benchmarks/tv2_canonical_v1/quantity-coordinates-v1.jsonl"

#: The token a value carries when it states its own representation.  Named here
#: rather than imported so the audit does not depend on the behaviour it audits:
#: if the split regressed, a check built from the split would not notice.
_REPRESENTATION_MARKS = ("%", "percent", "percentage", "ratio")


def _coordinates_from_store(path: Path) -> Iterator[Mapping[str, Any]]:
    seen: set[tuple[Any, Any, Any, Any]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("fact_type") != "atomic":
                continue
            value = record.get("value")
            if not isinstance(value, str) or not value.strip():
                continue
            key = (value, record.get("unit"), record.get("currency"), record.get("scale"))
            if key in seen:
                continue
            seen.add(key)
            yield {
                "value": value,
                "unit": record.get("unit"),
                "currency": record.get("currency"),
                "scale": record.get("scale"),
                "count": 1,
            }


def _read_coordinates(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _cell(value: Any) -> str:
    return "" if value is None else str(value).replace("\t", " ").replace("\n", " ")


def _states_a_representation(value: Any) -> bool:
    text = str(value).strip().casefold()
    return any(text.endswith(mark) for mark in _REPRESENTATION_MARKS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fact-store", type=Path, default=None)
    source.add_argument("--coordinates", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.coordinates is not None:
        rows = list(_read_coordinates(args.coordinates))
        origin = str(args.coordinates)
    else:
        store = args.fact_store or DEFAULT_FACT_STORE
        if not store.exists():
            parser.error(f"fact store not found: {store}")
        rows = list(_coordinates_from_store(store))
        origin = str(store)

    lines: list[str] = []
    marked_records = 0
    literal_records = 0
    for row in rows:
        identity = quantity_identity(
            row.get("value"),
            scale=row.get("scale"),
            unit=row.get("unit"),
            currency=row.get("currency"),
        )
        count = int(row.get("count") or 1)
        if _states_a_representation(row.get("value")):
            marked_records += count
        if identity.startswith("literal:"):
            literal_records += count
        lines.append(
            "\t".join(
                (
                    _cell(row.get("value")),
                    _cell(row.get("unit")),
                    _cell(row.get("currency")),
                    _cell(row.get("scale")),
                    identity,
                    str(count),
                )
            )
        )

    lines.sort()
    text = "\n".join(lines) + "\n"
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    total = sum(int(row.get("count") or 1) for row in rows)
    print(
        f"{origin}: {len(rows)} distinct tuples, {total} records, "
        f"{marked_records} stating a representation, {literal_records} literal",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
