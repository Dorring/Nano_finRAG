#!/usr/bin/env python3
"""P1.8-D1-C5: verify a frozen fixture's operand order against the benchmark.

    python scripts/evaluation/verify_fixture_integrity.py \
        --eval-set benchmarks/tv2_canonical_v1/canonical-eval-v1.jsonl \
        <fixture.jsonl> [...]

Exits non-zero if any `difference ... between A and B` plan names the two
companies in the order the *canonical question* does not.  The rule, and why it
lives on the fixture rather than in the runtime, are in `fixture_integrity`.

The eval set is required, not optional: a stale fixture is internally
consistent, so checking it against its own recorded question finds nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixture_integrity as integrity  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-set",
        type=Path,
        required=True,
        help="the canonical questions; the authority the fixture is checked against",
    )
    parser.add_argument("fixtures", nargs="+", type=Path)
    args = parser.parse_args(argv)

    questions = integrity.questions_by_id(args.eval_set)

    failed = False
    for path in args.fixtures:
        rows = integrity.load_rows(path)
        verdicts = integrity.check_rows(rows, questions)
        bound = [v for v in verdicts if v.get("binds")]
        bad = integrity.violations(verdicts)
        drift = [v for v in verdicts if v.get("question_drift")]

        print(f"{path}")
        print(
            f"  rows {len(rows)}   operand-ordered and checkable {len(bound)}   "
            f"violations {len(bad)}   recorded-question drift {len(drift)}"
        )
        for verdict in bad:
            print(f"    VIOLATION {verdict['id']}")
            print(f"      canonical question : {verdict['question']}")
            print(f"      expected slot order: {verdict['expected_slot_entities']}")
            print(f"      fixture slot order : {verdict['observed_slot_entities']}")
        for verdict in bound:
            mark = "ok  " if verdict["satisfied"] else "FAIL"
            print(
                f"    {mark} {verdict['id']:<28} "
                f"{verdict['observed_slot_entities']} == {verdict['expected_slot_entities']}"
            )
        if drift:
            print("  recorded question differs from the canonical question (not fatal):")
            for verdict in drift[:10]:
                print(
                    f"    {verdict['id']:<28} "
                    f"fixture={verdict['question_drift']['fixture']!r}"
                )
            if len(drift) > 10:
                print(f"    ... and {len(drift) - 10} more")
        failed = failed or bool(bad)
        print()

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
