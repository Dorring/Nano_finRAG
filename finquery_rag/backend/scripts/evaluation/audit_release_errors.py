"""The released cases that are wrong, and what distinguishes them.

A gate that admits more labels is only defensible if it keeps the wrong ones
out, so the first thing to know is which releases are wrong and whether they
share a cause.  `release coverage` alone cannot say that: 42 released with 3
wrong and 16 released with 0 wrong are different systems, and only one of them
is a candidate for production.

Reported per case with the plan slot, the gold fact and the bound facts, so the
cause can be read rather than guessed.

  python audit_release_errors.py --cases <cases.jsonl> --gold <gold.jsonl> \\
      --fact-store <financial-facts.jsonl>
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import score_nf_v3_final as scorer  # noqa: E402


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, action="append", required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--fact-store", type=Path, required=True)
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args(argv)

    gold = {row["id"]: row for row in _load(args.gold)}
    aliases = scorer.load_alias_map(args.fact_store)

    for path in args.cases:
        rows = _load(path)
        verdicts: collections.Counter = collections.Counter()
        wrong_rows = []
        for row in rows:
            if row.get("must_refuse"):
                continue
            if not row.get("released"):
                continue
            verdict = scorer.score_released(row, gold.get(row["case_id"]) or {})
            verdicts[verdict] += 1
            if verdict != "correct":
                wrong_rows.append(row)
        print("=" * 78)
        print("%s   released=%d  %s" % (path, sum(verdicts.values()), dict(verdicts)))
        print("=" * 78)
        for row in wrong_rows:
            record = gold.get(row["case_id"]) or {}
            print("  %s  verdict=%s" % (row["case_id"],
                                         scorer.score_released(row, record)))
            print("    q            %s" % row["question"])
            print("    plan_metrics %s" % (row.get("plan_metrics"),))
            print("    gold fact    %s  metric=%s period=%s"
                  % (record.get("fact_ids"), record.get("metric"), record.get("period")))
            print("    bound        %s" % (row.get("bound_evidence_ids"),))
            print("    answer       %s" % str(row.get("answer"))[:200])
            print()
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
