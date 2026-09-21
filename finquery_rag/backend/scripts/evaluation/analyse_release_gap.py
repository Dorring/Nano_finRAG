"""Where the answerable cases are lost, and what class each loss belongs to.

The release target is a coverage number, so the first thing to know is which
stage each unreleased answerable case died at, and -- for the ones the semantic
gate refused -- whether the refusal is a vocabulary gap or a period/scope one.
Those are different problems with different fixes, and the aggregate
"gate blocked = 48" hides the split.

Recomputed from the case rows; nothing is read from a summary.

  python analyse_release_gap.py --cases <A-cases.jsonl>
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, action="append", required=True)
    args = parser.parse_args(argv)

    for path in args.cases:
        rows = _load(path)
        answerable = [r for r in rows if not r.get("must_refuse")]
        released = [r for r in answerable if r.get("released")]
        print("=" * 78)
        print(str(path))
        print("=" * 78)
        print("  answerable %d   released %d   = %.1f%%"
              % (len(answerable), len(released),
                 100.0 * len(released) / max(1, len(answerable))))

        buckets: collections.Counter = collections.Counter()
        for r in answerable:
            if r.get("released"):
                buckets["RELEASED"] += 1
                continue
            if r.get("gate_blocked"):
                unknown = r.get("computed_unknown_query_fields") or []
                if not r.get("comparable"):
                    buckets["GATE_BLOCKED_NONCOMPARABLE"] += 1
                elif unknown:
                    buckets["GATE_BLOCKED_UNKNOWN"] += 1
                else:
                    buckets["GATE_BLOCKED_OTHER"] += 1
                continue
            if not r.get("comparable"):
                # A case whose gold carries a relation rather than a fact id has
                # an empty gold set, so `gold_in_pool` is 0 by construction.
                # Bucketing it as a retrieval miss reported eleven cases as the
                # retrieval layer's failure when the metric had no meaning for
                # them at all.
                buckets["NONCOMPARABLE_UNRELEASED"] += 1
                continue
            if not r.get("retrieval_rounds") and not r.get("pool_size"):
                buckets["NOT_REACHED_RETRIEVAL"] += 1
            elif not r.get("gold_in_pool"):
                buckets["GOLD_NOT_IN_POOL"] += 1
            elif not r.get("gold_bound"):
                buckets["GOLD_NOT_BOUND"] += 1
            elif r.get("missing_slot_ids"):
                buckets["SLOTS_INCOMPLETE:" + ",".join(r["missing_slot_ids"])] += 1
            elif not r.get("validation_passed"):
                buckets["VALIDATION_FAILED"] += 1
            else:
                buckets["LOST_AFTER_VALIDATION"] += 1

        for key, count in buckets.most_common():
            print("    %-42s %4d" % (key, count))
        print()

        # The gate-blocked cases, by the phrase that would not resolve.
        blocked = [r for r in answerable
                   if not r.get("released") and r.get("gate_blocked")]
        if blocked:
            print("  -- gate-blocked, by terminal status")
            for key, count in collections.Counter(
                    (r.get("computed_status"), bool(
                        r.get("computed_unknown_query_fields")))
                    for r in blocked).most_common():
                print("    status=%-10s unknown=%s  %4d" % (key[0], key[1], count))
            print()
            print("  -- gate-blocked cases")
            for r in sorted(blocked, key=lambda x: str(x.get("case_id"))):
                print("    %-26s %-22s comp=%-5s %s"
                      % (r.get("case_id"), str(r.get("computed_status"))[:22],
                         r.get("comparable"), str(r.get("question"))[:58]))
                print("        plan_metrics=%s" % (r.get("plan_metrics"),))
                if r.get("computed_unknown_query_fields"):
                    print("        unknown=%s" % (r["computed_unknown_query_fields"],))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
