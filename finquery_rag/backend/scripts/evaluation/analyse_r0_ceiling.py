"""R0 companion: how much of the retrieval miss is retrieval, and how much is gold?

`audit_retrieval_ceiling.py` answers where gold is lost.  This asks the question
that decides whether any of it is *winnable*: the coordinate-ambiguity survey
(P1.6-A scoping) already established that a large share of the benchmark's gold
sits at coordinates holding more than one distinct value -- 21/40 factual_lookup
and 9/35 arithmetic_calculation.  A gold fact at such a coordinate is one of
several facts the store cannot distinguish; a retriever that returns a sibling
is not obviously wrong, and a recall number computed over it is partly a
statement about the gold.

So the miss list is split two ways and reported together:

  retrieval-side   the fact is in the index and ranked; depth or ordering lost it
  gold-side        the fact's own coordinate does not identify it

Neither half excuses the other.  The point is that they need different work: the
first is a ranking problem and can be improved here, the second cannot be
improved by any retriever and belongs to the fixture, not to this sprint.

Read-only.  Joins two existing artifacts and writes nothing else.

  python analyse_r0_ceiling.py --ceiling <dir>/retrieval-ceiling.json \\
      --ambiguity <survey>.json --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ceiling", type=Path, required=True)
    parser.add_argument("--ambiguity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    ceiling = _load(args.ceiling)
    survey = _load(args.ambiguity)

    ambiguous = {str(c["case_id"]) for c in survey.get("affected_cases", [])}
    by_stratum = collections.Counter(
        str(c.get("stratum")) for c in survey.get("affected_cases", []))

    print("=" * 78)
    print("R0 -- WHERE THE MISS SITS: RANKING, OR THE GOLD ITSELF")
    print("=" * 78)
    print(f"  ceiling artifact  {args.ceiling}")
    print(f"  ambiguity survey  {args.ambiguity}")
    print(f"  cases the survey flags as ambiguous-coordinate: {len(ambiguous)}")
    for stratum, count in by_stratum.most_common():
        print(f"    {count:>4}  {stratum}")
    print()

    control = ceiling.get("control", {})
    print(f"  control ok: {control.get('ok')}")
    if not control.get("ok"):
        print("  !! The ceiling control did not reproduce the published numbers.")
        print("  !! Every split below inherits that and must not be quoted.")
    print()

    taxonomy = ceiling.get("taxonomy", {})
    examples = taxonomy.get("examples", [])
    print("=" * 78)
    print(f"MISS SPLIT -- {taxonomy.get('missed', 0)} gold ids absent from every "
          f"lane's top-20 at Q1")
    print("=" * 78)

    split = collections.Counter()
    by_stratum_split: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for entry in examples:
        case_id = str(entry["case_id"])
        stratum = str(entry.get("stratum"))
        gold_side = case_id in ambiguous
        oracle = bool(entry.get("oracle_reachable"))
        bucket = ("gold-side (ambiguous coordinate)" if gold_side
                  else "retrieval-side (coordinate identifies one value)")
        bucket += " + oracle-closable" if oracle else " + oracle cannot close"
        split[bucket] += 1
        by_stratum_split[stratum][bucket] += 1

    for bucket, count in split.most_common():
        print(f"    {count:>4}  {bucket}")
    print()
    for stratum, counter in sorted(by_stratum_split.items()):
        total = sum(counter.values())
        print(f"  {stratum} ({total} missed)")
        for bucket, count in counter.most_common():
            print(f"    {count:>4}  {bucket}")
    print()

    retrieval_side = sum(c for b, c in split.items() if b.startswith("retrieval-side"))
    gold_side = sum(c for b, c in split.items() if b.startswith("gold-side"))
    print(f"  retrieval-side {retrieval_side}   gold-side {gold_side}")
    print()

    report = {
        "analysis": "R0-miss-split",
        "control_ok": bool(control.get("ok")),
        "ambiguous_cases": len(ambiguous),
        "ambiguous_by_stratum": dict(by_stratum),
        "missed": taxonomy.get("missed", 0),
        "split": dict(split),
        "split_by_stratum": {k: dict(v) for k, v in by_stratum_split.items()},
        "retrieval_side": retrieval_side,
        "gold_side": gold_side,
        "ceiling_summary": {
            "formulations": ceiling.get("formulations"),
            "union_coverage_q1": ceiling.get("union_coverage_q1"),
            "lane_recall_q1": ceiling.get("lane_recall_q1"),
            "overlap_buckets": ceiling.get("overlap_buckets"),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "r0-miss-split.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(f"  written to {args.out / 'r0-miss-split.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
