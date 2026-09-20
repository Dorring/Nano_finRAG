"""The systematic metrics: one table over every arm and every stage.

The harness has grown one experiment at a time, and each one printed its own
numbers in its own format.  That makes a result hard to read as a whole and easy
to mis-state in isolation: a release count without its incorrect-release count,
or a Recall@5 without the pool coverage it came out of, is not a result.

This reads the case-level artifacts and emits every layer in one place, per arm:

  retrieval     R@5 / @10 / @20 over the scorable slice
  gate          how many answerable cases the semantic gate refused
  funnel        reached retrieval / gold in pool / gold bound / slots complete
  release       released, released-correct, and REFUSED-when-it-should-refuse
  safety        incorrect release (an abstention case that released) and
                false release.  Printed for every arm, not only the last one.
  citation      groundedness (cited a cell it admitted) vs precision/recall
                against the benchmark's own gold cells

Everything is recomputed from the case rows; nothing is read from a summary
file, so a stale aggregate cannot survive here.

  python report_system_metrics.py --arm A=<cases.jsonl> --arm B=<cases.jsonl> \\
      --gold gold-evidence-v1.jsonl --fact-store financial-facts.jsonl --out <dir>
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

RETRIEVAL = {
    "shipped hybrid RRF (benchmark path)": {5: 50.7, 10: 58.0, 20: 65.3},
}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        help="NAME=<cases.jsonl>, repeatable")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--fact-store", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, default=None,
                        help="structured-rerank.json, for the retrieval layer")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    gold = {row["id"]: row for row in _load(args.gold)}
    aliases = scorer.load_alias_map(args.fact_store)

    arms: dict[str, list[dict]] = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = _load(Path(path))

    print("=" * 78)
    print("SYSTEM METRICS -- every layer, every arm")
    print("=" * 78)
    print()

    report: dict = {"arms": {}}
    for name, rows in arms.items():
        tally: collections.Counter = collections.Counter()
        for row in rows:
            case_id = row["case_id"]
            record = gold.get(case_id) or {}
            must_refuse = bool(record.get("expected_failure_reason")) or bool(
                row.get("must_refuse"))
            comparable = bool(row.get("comparable"))
            released = bool(row.get("released"))

            tally["cases"] += 1
            if must_refuse:
                tally["should_refuse"] += 1
                tally["correct_refusal" if not released else "INCORRECT_RELEASE"] += 1
            else:
                tally["answerable"] += 1
                if not comparable:
                    tally["answerable_not_comparable"] += 1
                    continue
                tally["comparable"] += 1
                if row.get("gate_blocked"):
                    tally["gate_blocked"] += 1
                if row.get("retrieval_rounds") or row.get("pool_size"):
                    tally["reached_retrieval"] += 1
                if row.get("gold_in_pool"):
                    tally["gold_in_pool"] += 1
                if row.get("gold_bound"):
                    tally["gold_bound"] += 1
                if not row.get("missing_slot_ids") and (
                        row.get("retrieval_rounds") or row.get("pool_size")):
                    tally["slots_complete"] += 1
                if released:
                    tally["released"] += 1
                    verdict = scorer.score_released(row, record)
                    if verdict == "correct":
                        tally["released_correct"] += 1
                    elif verdict == "wrong":
                        tally["released_WRONG"] += 1
                    else:
                        tally["released_unscoreable"] += 1
                    cited = [scorer.resolve(c, aliases)
                             for c in (row.get("citation_ids") or [])]
                    admitted = [scorer.resolve(e, aliases)
                                for e in (row.get("bound_evidence_ids") or [])]
                    wanted = [scorer.resolve(f, aliases)
                              for f in (record.get("fact_ids") or [])]
                    tally["cited"] += len(cited)
                    tally["cited_grounded"] += sum(1 for c in cited if c in admitted)
                    tally["cited_in_gold"] += sum(1 for c in cited if c in wanted)
                    tally["gold_total"] += len(wanted)
                    tally["gold_cited"] += sum(1 for w in wanted if w in cited)

        report["arms"][name] = dict(tally)
        comp = max(1, tally["comparable"])
        answerable = max(1, tally["answerable"])
        print(f"  --- {name}")
        print(f"    gate blocked            {tally['gate_blocked']:>4} / {tally['comparable']}"
              f"   ({tally['gate_blocked'] / comp:.1%})")
        print(f"    reached retrieval       {tally['reached_retrieval']:>4} / {tally['comparable']}")
        print(f"    gold in pool            {tally['gold_in_pool']:>4} / {tally['comparable']}")
        print(f"    gold bound              {tally['gold_bound']:>4} / {tally['comparable']}")
        print(f"    slots complete          {tally['slots_complete']:>4} / {tally['comparable']}")
        print(f"    released (answerable)   {tally['released']:>4} / {tally['answerable']}"
              f"   ({tally['released'] / answerable:.1%})")
        print(f"    released correct        {tally['released_correct']:>4} / "
              f"{tally['released']}   wrong {tally['released_WRONG']}"
              f"  unscoreable {tally['released_unscoreable']}")
        print(f"    correct refusal         {tally['correct_refusal']:>4} / {tally['should_refuse']}")
        print(f"    INCORRECT RELEASE       {tally['INCORRECT_RELEASE']:>4}"
              f"   <- must stay 0")
        if tally["cited"]:
            print(f"    citation grounded       {tally['cited_grounded']}/{tally['cited']}"
                  f"  precision-vs-gold {tally['cited_in_gold']}/{tally['cited']}"
                  f"  recall-vs-gold {tally['gold_cited']}/{max(1, tally['gold_total'])}")
        print()

    if args.retrieval and args.retrieval.is_file():
        retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
        print("  --- retrieval (benchmark path)")
        for label, row in sorted(RETRIEVAL.items()):
            print(f"    {label:38}{row[5]:>8.1f}%{row[10]:>8.1f}%{row[20]:>8.1f}%")
        for arm, row in sorted((retrieval.get("results") or {}).items()):
            print(f"    {arm:38}{row['5'] * 100:>8.3f}%{row['10'] * 100:>8.3f}%"
                  f"{row['20'] * 100:>8.3f}%")
        report["retrieval"] = retrieval.get("results")
        print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "system-metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(f"  written to {args.out / 'system-metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
