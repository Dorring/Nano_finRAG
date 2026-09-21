"""Whether a conflicting coordinate has a total-and-components structure.

The cases the Binder refuses for `EVIDENCE_CONFLICT` share one shape: several
stored facts at one `(entity, metric, period)` with different values.  Sixteen
of the twenty-two the gate still blocks are refused that way, and the question
is whether the values are *arbitrary* or whether one of them is the consolidated
total of the others.

That distinction decides what the fix is.  If a total exists, the coordinate is
not ambiguous -- it holds a hierarchy, and the sum is a source-grounded
discriminator a reader would apply.  If no total exists, the values are
genuinely unrelated and refusing is the only honest answer.

  python audit_coordinate_structure.py --cases <cases.jsonl> --pinned <pinned.jsonl> \\
      --v2-fact-store <store.jsonl>
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _number(value: object) -> float | None:
    text = str(value or "").replace(",", "").replace("$", "").strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    if text.endswith("%"):
        return None
    match = re.fullmatch(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    number = float(match.group(0))
    return -number if negative else number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--pinned", type=Path, required=True)
    parser.add_argument("--v2-fact-store", type=Path, required=True)
    args = parser.parse_args(argv)

    by_coordinate: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    for record in _load(args.v2_fact_store):
        number = _number(record.get("value"))
        if number is None:
            continue
        key = (str(record.get("entity") or "").casefold(),
               str(record.get("metric") or "").casefold(),
               str(record.get("period") or "").casefold())
        by_coordinate[key].append(number)
    for key, values in by_coordinate.items():
        by_coordinate[key] = sorted(set(values))

    pinned = {row["case_id"]: row for row in _load(args.pinned)}
    rows = [r for r in _load(args.cases)
            if not r.get("must_refuse") and not r.get("released")]

    tally: collections.Counter = collections.Counter()
    for row in rows:
        if not row.get("gate_blocked") and not (row.get("gold_in_pool") and not row.get("gold_bound")):
            continue
        entry = pinned.get(row["case_id"]) or {}
        for slot in (entry.get("plan") or {}).get("required_slots") or []:
            key = (str(slot.get("entity") or "").casefold(),
                   str(slot.get("metric") or "").casefold(),
                   str(slot.get("period") or "").casefold())
            values = by_coordinate.get(key) or []
            if len(values) < 2:
                continue
            total = _has_total(values)
            tally["HAS_TOTAL" if total else "NO_TOTAL"] += 1
            print("  %-26s %-44s n=%d %s  %s"
                  % (row["case_id"], str(slot.get("metric"))[:44], len(values),
                     "TOTAL" if total else "     ", values[:5]))
            break
    print()
    print("  %s" % dict(tally))
    return 0


def _has_total(values: list[float]) -> bool:
    """Whether one value is the sum of a subset of the others."""

    for size in range(2, min(5, len(values)) + 1):
        for combination in itertools.combinations(values, size):
            rest = [value for value in values if value not in combination]
            for candidate in rest:
                if abs(sum(combination) - candidate) < 1e-6 * max(1.0, abs(candidate)):
                    return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
