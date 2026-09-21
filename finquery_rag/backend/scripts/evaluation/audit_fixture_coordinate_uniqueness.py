"""How many facts each fixture slot's coordinate actually selects.

The builder validates that a coordinate *exists* -- every operand fact is found
in the store and the coordinates are pairwise distinct.  It does not validate
that a coordinate *locates one fact*, and those are different properties: a
metric like ``iPhone`` is a real row in the corpus and is carried by many of
them, so a slot naming it is truthful and still cannot be bound.

This measures the second property over every slot of every fixture, without
calling a model: index the store once, then count the facts each slot's
coordinate matches.  Zero candidates and several candidates are both failures,
and both are invisible to an existence check.

    python audit_fixture_coordinate_uniqueness.py --fixtures <path> [--out <json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

DEFAULT_STORE = Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl")
DEFAULT_FIXTURES = _BACKEND_DIR / "benchmarks/tv2_canonical_v1/plan-fixtures-v4.jsonl"

#: The slot fields that make up a coordinate, in the order they narrow it.
_COORDINATE_FIELDS = ("entity", "metric", "period", "scope")


def _fold(value: Any) -> str:
    """Casefold and collapse whitespace, as the shared text identity does."""

    if value is None:
        return ""
    return " ".join(str(value).casefold().split())


def _key(record: Mapping[str, Any], fields: Iterable[str]) -> tuple[str, ...]:
    return tuple(_fold(record.get(field)) for field in fields)


def _load_fixtures(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _store_index(store: Path) -> tuple[dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]], int]:
    """Index every atomic fact under every *subset* of the coordinate fields.

    Subsets rather than prefixes: a slot that names a metric and a period but no
    entity does not constrain the entity, so it must match facts of every
    company.  A prefix key would silently restrict it to facts whose entity is
    empty -- which is not a stricter check, it is a different and wrong one.
    """

    subsets = [
        combo
        for width in range(1, len(_COORDINATE_FIELDS) + 1)
        for combo in combinations(_COORDINATE_FIELDS, width)
    ]
    index: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = defaultdict(list)
    total = 0
    with store.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("fact_type") != "atomic":
                continue
            total += 1
            fact_id = str(record.get("fact_id") or record.get("candidate_key") or "")
            if not fact_id:
                continue
            for combo in subsets:
                index[(combo, _key(record, combo))].append(fact_id)
    return index, total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.store.exists():
        print(f"fact store not found: {args.store}", file=sys.stderr)
        return 2

    fixtures = _load_fixtures(args.fixtures)
    index, total = _store_index(args.store)
    print(f"store: {total} atomic facts, {len(index)} coordinate prefixes")

    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        for slot in fixture["plan"]["required_slots"]:
            # Only the fields the slot actually carries, in coordinate order.  A
            # slot that names no entity is not entity-constrained, and saying so
            # is the finding rather than an excuse for it.
            present = tuple(
                field for field in _COORDINATE_FIELDS if _fold(slot.get(field))
            )
            matches = index.get((present, _key(slot, present)), []) if present else []
            rows.append(
                {
                    "case_id": fixture["id"],
                    "stratum": fixture["stratum"],
                    "operation": fixture["plan"].get("operation"),
                    "slot_id": slot["slot_id"],
                    "role": slot["role"],
                    "fields_used": list(present),
                    "coordinate": {field: slot.get(field) for field in present},
                    "carries_entity": bool(_fold(slot.get("entity"))),
                    "candidates": len(matches),
                    "candidate_fact_ids": matches[:5],
                }
            )

    zero = [row for row in rows if row["candidates"] == 0]
    many = [row for row in rows if row["candidates"] > 1]
    unique = [row for row in rows if row["candidates"] == 1]

    print()
    print(f"slots checked        : {len(rows)}")
    print(f"  unique (1 candidate): {len(unique)}")
    print(f"  ZERO candidates      : {len(zero)}")
    print(f"  SEVERAL candidates   : {len(many)}")

    # The same coordinate can be shared by several slots; report by coordinate
    # so the fix is designed once per coordinate rather than once per slot.
    by_coordinate: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in many + zero:
        key = tuple(str(row["coordinate"].get(f)) for f in row["fields_used"])
        entry = by_coordinate.setdefault(
            key,
            {
                "coordinate": row["coordinate"],
                "fields_used": row["fields_used"],
                "candidates": row["candidates"],
                "carries_entity": row["carries_entity"],
                "cases": [],
            },
        )
        entry["cases"].append(f"{row['case_id']}:{row['slot_id']}({row['role']})")

    print()
    print(f"distinct problem coordinates: {len(by_coordinate)}")
    for key, entry in sorted(by_coordinate.items(), key=lambda item: -item[1]["candidates"]):
        flag = "ZERO " if entry["candidates"] == 0 else f"{entry['candidates']:>5}"
        entity = "entity" if entry["carries_entity"] else "NO-ENTITY"
        print(f"  [{flag}] [{entity}] {entry['coordinate']}")
        print(f"          {', '.join(entry['cases'][:6])}")

    if args.out is not None:
        args.out.write_text(
            json.dumps({"slots": rows, "summary": {
                "total": len(rows), "unique": len(unique),
                "zero": len(zero), "many": len(many),
            }}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
