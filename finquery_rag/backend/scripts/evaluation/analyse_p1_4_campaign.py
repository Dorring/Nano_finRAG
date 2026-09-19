"""P1.4 campaign analysis: three ALIGNED-REPLAY runs at each of two commits.

Reads the output of ``run_p1_4_percentage_campaign.sh`` and answers, in order:

1. Did any variable move between runs?  A seal that disagrees on commit, fixture,
   gold or config means the six runs are not one experiment and the rest of this
   report is not about anything.
2. The four named percentage cases, per run, against what P1.4a and P1.4b were
   each supposed to do.
3. The non-percent population: did binding, release or the reason distribution
   move, per commit, against the repeat spread.
4. Whether any movement exceeds the spread the same configuration already shows.

The percentage questions are derived from the gold file rather than listed by
hand, so the split between "the cases the change is about" and "the cases it
must not touch" cannot drift as the set is edited.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

CAMPAIGN = Path(sys.argv[1] if len(sys.argv) > 1 else
                "/disk/qh/nano-finrag/artifacts/evaluation/p1-4-percentage-campaign")
GOLD = Path("/disk/qh/nano-finrag/finquery_rag/backend/benchmarks/"
            "tv2_canonical_v1/gold-evidence-v1.jsonl")

GROUPS = {"p14a": ["p14a-run1", "p14a-run2", "p14a-run3"],
          "p14b": ["p14b-run1", "p14b-run2", "p14b-run3"]}

NAMED = {
    "tv2f01-s2-pctshare-003": "pctshare-003  (P1.4b target: was 100x out)",
    "tv2f01-s2-average-002": "average-002   (must hold)",
    "tv2f01-s1-aapl-003": "aapl-003      (factual, must hold)",
    "tv2f01-s1-jpm-010": "jpm-010       (factual, must hold)",
    "tv2f01-s1-nvda-024": "nvda-024      (factual, must hold)",
    "tv2f01-s3-compare-007": "compare-007   (must stay 0/3, incomparable)",
}

SEAL_KEYS = ("config_fingerprint", "eval_set_sha256", "gold_sha256",
             "fixture_sha256", "question_count")


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _gold() -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in _rows(GOLD)}


def _percent_ids(gold: dict[str, dict[str, Any]]) -> set[str]:
    """Cases whose gold states a percentage, found rather than listed."""

    found = set()
    for case_id, row in gold.items():
        blobs = [row.get("expected_value")]
        blobs += list((row.get("values") or {}).values())
        blobs += list((row.get("operands") or {}).values())
        if any(isinstance(b, str) and "%" in b for b in blobs):
            found.add(case_id)
    return found


def _counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    rows = list(rows)
    return {
        "n": len(rows),
        "bound": sum(1 for r in rows if r.get("binder_final_status") == "BOUND"),
        "released": sum(1 for r in rows if r.get("release_status") == "RELEASED"),
        "error": sum(1 for r in rows if r.get("status") == "ERROR"),
    }


def _spread(values: list[int]) -> str:
    return f"{values} (spread {max(values) - min(values)})"


def main() -> int:
    runs: dict[str, list[dict[str, Any]]] = {}
    seals: dict[str, dict[str, Any]] = {}
    for group, names in GROUPS.items():
        for name in names:
            directory = CAMPAIGN / name
            preds = directory / "replay-predictions.jsonl"
            seal = directory / "seal.json"
            if not preds.exists():
                print(f"MISSING: {preds}")
                return 1
            runs[name] = _rows(preds)
            seals[name] = json.loads(seal.read_text(encoding="utf-8")) if seal.exists() else {}

    print("=" * 78)
    print("1. SEAL — every variable except the commit must be identical")
    print("=" * 78)
    for key in SEAL_KEYS:
        seen = {name: seals[name].get(key) for name in runs}
        distinct = set(map(str, seen.values()))
        flag = "OK  " if len(distinct) == 1 else "DIFF"
        print(f"  [{flag}] {key:<22} {sorted(distinct)[0] if len(distinct) == 1 else sorted(distinct)}")
    for group, names in GROUPS.items():
        commits = {seals[n].get("commit_sha") for n in names}
        flag = "OK  " if len(commits) == 1 else "DIFF"
        print(f"  [{flag}] {group} commit_sha      {sorted(map(str, commits))}")

    gold = _gold()
    percent = _percent_ids(gold)
    print(f"\n  percentage cases in the gold ({len(percent)}): {sorted(percent)}")

    print()
    print("=" * 78)
    print("2. THE NAMED CASES")
    print("=" * 78)
    for case_id, label in NAMED.items():
        print(f"\n  {label}")
        for group, names in GROUPS.items():
            cells = []
            for name in names:
                row = next((r for r in runs[name] if r["id"] == case_id), None)
                if row is None:
                    cells.append("absent")
                    continue
                reasons = row.get("reason_codes") or []
                marker = "REL" if row.get("release_status") == "RELEASED" else "not"
                cells.append(f"{marker}{'/' + reasons[0] if reasons else ''}")
            print(f"    {group}: " + " | ".join(cells))

    print()
    print("=" * 78)
    print("3. THE NON-PERCENT POPULATION — what the change must not move")
    print("=" * 78)
    for group, names in GROUPS.items():
        subsets = [[r for r in runs[n] if r["id"] not in percent] for n in names]
        counts = [_counts(s) for s in subsets]
        print(f"\n  {group}  (n={counts[0]['n']} non-percent cases)")
        for key in ("bound", "released", "error"):
            print(f"    {key:<9} {_spread([c[key] for c in counts])}")

    print()
    print("=" * 78)
    print("4. REASON DISTRIBUTION — non-percent, per run")
    print("=" * 78)
    for group, names in GROUPS.items():
        print(f"\n  {group}")
        per_run = []
        for name in names:
            counter = Counter()
            for row in runs[name]:
                if row["id"] in percent:
                    continue
                for reason in row.get("reason_codes") or ["<none>"]:
                    counter[reason] += 1
            per_run.append(counter)
        keys = sorted({k for c in per_run for k in c})
        for key in keys:
            values = [c.get(key, 0) for c in per_run]
            flag = "" if max(values) - min(values) <= 1 else "   <-- moved"
            print(f"    {key:<34} {_spread(values)}{flag}")

    print()
    print("=" * 78)
    print("5. PERCENTAGE CASES — release state, all runs")
    print("=" * 78)
    for case_id in sorted(percent):
        line = [f"  {case_id:<26}"]
        for group, names in GROUPS.items():
            states = []
            for name in names:
                row = next((r for r in runs[name] if r["id"] == case_id), None)
                if row is None:
                    states.append("-")
                else:
                    states.append("R" if row.get("release_status") == "RELEASED" else ".")
            line.append(f"{group}:[{''.join(states)}]")
        print(" ".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
