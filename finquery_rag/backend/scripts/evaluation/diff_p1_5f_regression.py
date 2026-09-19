"""P1.5-F: the case-level behaviour diff between the two campaigns.

Aggregates cannot answer the question this phase is closing on.  A wrong release
becoming a correct fail-closed *lowers* the release count, and a summary that
reports only rates reads that as a regression.  So each case is classified on
two axes -- was the answer right, and did it release -- and the movement between
them is named.

    UNCHANGED_CORRECT      right in both
    IMPROVED               wrong -> right
    REGRESSED              right -> wrong
    SAFER_FAIL_CLOSED      released wrong -> did not release
    NEW_FALSE_RELEASE      released wrong where it did not before
    NEW_EXECUTION_ERROR    EXECUTION_ERROR where it was not
    FIXED_EXECUTION_ERROR  no longer EXECUTION_ERROR
    GROUND_TRUTH_AMBIGUOUS moved, but the case's gold cannot judge the system
    UNCHANGED_OTHER        equally wrong, or equally uninformative, both sides

`SAFER_FAIL_CLOSED` is the one that needs an explicit name.  It is an
improvement and it looks like a loss in any count of releases.

`GROUND_TRUTH_AMBIGUOUS` is the other.  Without it `compare-009` reads as a
regression: it released a correct answer and now fail-closes.  But its gold took
one cell of a flattened segment table (`P1.6-0C`), so the case was never
measuring the system -- it was measuring the fixture.  A diff that cannot say
that reports a caught defect as a lost release.

A case's verdict is taken from its **modal** status across the runs, so a single
provider flake does not read as a behavioural change.  Cases whose runs disagree
are reported separately rather than folded in.

  python diff_p1_5f_regression.py --baseline <dir> --after <dir>
      [--gold-validity <P1.6-0C artifact>]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR),):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load(directory: Path) -> dict[str, dict[str, dict]]:
    runs: dict[str, dict[str, dict]] = {}
    for run_dir in sorted(directory.glob("run*")):
        for line in (run_dir / "replay-predictions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            runs.setdefault(row["id"], {})[run_dir.name] = row
    return runs


def _modal(values: list[str]) -> tuple[str, bool]:
    """The most common value, and whether the runs agreed."""

    counts = collections.Counter(values)
    winner, count = counts.most_common(1)[0]
    return winner, count == len(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument(
        "--gold-validity",
        type=Path,
        default=None,
        help="P1.6-0C artifact; cases whose gold it discredited or could not "
        "settle are classified GROUND_TRUTH_AMBIGUOUS instead of REGRESSED",
    )
    args = parser.parse_args(argv)

    from src.evaluation.grounded_release import GoldValidity, load_gold_validity

    validity = load_gold_validity(args.gold_validity) if args.gold_validity else {}

    from src.evaluation.p1_2_dual_track import _correct_by_stratum

    gold = {
        json.loads(line)["id"]: json.loads(line)
        for line in args.gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    before, after = _load(args.baseline), _load(args.after)

    verdicts = collections.Counter()
    by_stratum: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    unstable: list[str] = []
    rows: list[dict] = []

    for case_id in sorted(set(before) | set(after)):
        b_runs, a_runs = before.get(case_id, {}), after.get(case_id, {})
        if not b_runs or not a_runs:
            verdicts["MISSING_FROM_ONE_SIDE"] += 1
            continue
        b_status, b_stable = _modal([str(r.get("status")) for r in b_runs.values()])
        a_status, a_stable = _modal([str(r.get("status")) for r in a_runs.values()])
        if not (b_stable and a_stable):
            unstable.append(case_id)

        gold_row = gold.get(case_id, {})
        exemplar = {name: row for name, row in sorted(a_runs.items())}
        b_exemplar = {name: row for name, row in sorted(b_runs.items())}
        b_correct = _correct_by_stratum(next(iter(b_exemplar.values())), gold_row)
        a_correct = _correct_by_stratum(next(iter(exemplar.values())), gold_row)
        b_released = b_status == "READY_FOR_RELEASE"
        a_released = a_status == "READY_FOR_RELEASE"

        if b_status == a_status and b_correct == a_correct:
            verdict = "UNCHANGED_CORRECT" if a_correct else "UNCHANGED_OTHER"
        elif validity.get(case_id) in (
            GoldValidity.WRONG_SCOPE,
            GoldValidity.UNRESOLVED,
        ):
            # Checked before every directional verdict.  The case moved, but the
            # gold cannot say whether the movement helped, so calling it either
            # way would be an attribution the evidence does not support.
            verdict = "GROUND_TRUTH_AMBIGUOUS"
        elif a_status == "EXECUTION_ERROR" and b_status != "EXECUTION_ERROR":
            verdict = "NEW_EXECUTION_ERROR"
        elif b_status == "EXECUTION_ERROR" and a_status != "EXECUTION_ERROR":
            verdict = "FIXED_EXECUTION_ERROR"
        elif b_released and not b_correct and not (a_released and not a_correct):
            verdict = "SAFER_FAIL_CLOSED"
        elif a_released and not a_correct and not (b_released and not b_correct):
            verdict = "NEW_FALSE_RELEASE"
        elif a_correct and not b_correct:
            verdict = "IMPROVED"
        elif b_correct and not a_correct:
            verdict = "REGRESSED"
        else:
            verdict = "UNCHANGED_OTHER"

        verdicts[verdict] += 1
        # The stratum lives on the prediction row, not on the gold row -- the
        # gold corpus carries no `stratum` field at all, so reading it from
        # there put every case under "?" and made the per-stratum table unable
        # to attribute anything.
        stratum = str(
            next(iter(exemplar.values())).get("stratum") or "?"
        )
        by_stratum[stratum][verdict] += 1
        if verdict not in ("UNCHANGED_CORRECT", "UNCHANGED_OTHER"):
            rows.append({"case": case_id, "verdict": verdict,
                         "baseline": b_status, "after": a_status,
                         "baseline_correct": b_correct, "after_correct": a_correct})

    print("=== P1.5-F behaviour diff ===")
    for verdict, count in verdicts.most_common():
        print(f"  {verdict:22} {count}")
    print()
    for stratum, counts in sorted(by_stratum.items()):
        print(f"  {stratum:26} {dict(counts)}")
    print()
    if rows:
        print("=== every case that moved ===")
        for row in rows:
            print(f"  {row['case']:30} {row['verdict']:20} "
                  f"{row['baseline']} -> {row['after']}  "
                  f"correct {row['baseline_correct']} -> {row['after_correct']}")
    else:
        print("  nothing moved")
    if unstable:
        print()
        print(f"=== cases whose runs disagreed (reported, not classified) ===")
        for case_id in unstable:
            print(f"  {case_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
