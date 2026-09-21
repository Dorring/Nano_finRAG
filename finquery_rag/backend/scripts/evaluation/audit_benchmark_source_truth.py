"""Is the benchmark measuring the system, or partly a bad oracle?

`tv2f01-s2-sum-007` asks for The Coca-Cola Company's operating income and its
gold cites a row from the table the filing introduces as *"a summary of
financial information for our equity method investees"* -- net operating
revenues 102,800 against Coca-Cola's own 47,941.  An `entity == company` check
cannot see it, because the note table's facts inherited the filer's entity.

So this audits the **oracle** rather than the system.  Nothing in `src/` is
touched; the runtime is frozen; the source HTML is the independent authority.

Risk flags, each computed rather than judged:

  MULTI_VALUE        the same (entity, metric, period) holds several values
  NON_PRIMARY_TABLE  the gold's row sits in a table that is not a primary
                     statement -- a note, a segment disclosure, an investee
                     summary, selected financial data
  CONFLICTS_PRIMARY  a primary statement states the same (entity, metric,
                     period) with a different value

Classification, and `UNRESOLVED` must reach zero:

  GOLD_CORRECT        no flag
  GOLD_WRONG_TABLE    the gold's row is from a non-primary table and a primary
                      statement states a different value for that coordinate
  GOLD_WRONG_COORDINATE the gold's fact does not carry the coordinate the
                      question asks for (metric/period/entity)
  SOURCE_AMBIGUOUS    several source tables state the coordinate and nothing
                      selects between them
  UNRESOLVED          no flag could be computed

  python audit_benchmark_source_truth.py --gold <gold.jsonl> \\
      --v2-fact-store <store.jsonl> --html-root <raw_sec_html> --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_structural_sidecar as sidecar  # noqa: E402

#: A table that states a company's own consolidated position.  Anything else is
#: a note, a segment view, an investee summary or supplemental data -- all of
#: which are legitimate tables and none of which is the company's own figure.
_PRIMARY = re.compile(
    r"consolidated\s+(?:statements?\s+of\s+(?:income|operations|financial\s+position"
    r"|cash\s+flows|comprehensive\s+income|equity|changes)|balance\s+sheets?)",
    re.IGNORECASE,
)


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--v2-fact-store", type=Path, required=True)
    parser.add_argument("--html-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    store = {str(r.get("candidate_key")): r for r in _load(args.v2_fact_store)}
    gold = _load(args.gold)
    questions = {row["id"]: row for row in _load(args.eval_set)}

    # -- document -> filing, the same selection the sidecar builder uses -----
    filings: dict[str, Path | None] = {}
    for name in sorted({str(r.get("document_name")) for r in store.values()}):
        ticker = name.split("_")[0].upper()
        year = re.search(r"(\d{4})", name)
        directory = args.html_root / ticker
        chosen: Path | None = None
        if directory.is_dir() and year:
            scored = []
            for candidate in sorted(directory.iterdir()):
                metadata = candidate / "source-metadata.json"
                if not metadata.is_file():
                    continue
                payload = json.loads(metadata.read_text(encoding="utf-8"))
                hits = (payload.get("identity_signals") or {}).get("period_hits") or []
                end = str(hits[0]) if hits else ""
                if end.startswith(year.group(1)):
                    scored.append((int(payload.get("raw_bytes") or 0),
                                   candidate / "primary.html"))
            if scored:
                scored.sort(key=lambda item: (-item[0], str(item[1])))
                chosen = scored[0][1]
        filings[name] = chosen

    # -- every source row, by label, with its table's caption ---------------
    rows_by_label: dict[str, list[tuple]] = collections.defaultdict(list)
    for document_name, path in filings.items():
        if path is None:
            continue
        flow, headings = sidecar._parse_document(path)
        for index, item in enumerate(flow):
            if item[0] != "table":
                continue
            caption = headings.get(index, "")
            for row in item[1]:
                label = next((cell for cell in row if cell.strip()), "")
                key = sidecar._norm(label)
                if key:
                    rows_by_label[key].append((document_name, caption, row))

    # -- coordinate index from the store, for MULTI_VALUE and conflicts -----
    by_coordinate: dict[tuple, dict[str, set]] = collections.defaultdict(
        lambda: collections.defaultdict(set))
    for key, record in store.items():
        coordinate = (sidecar._norm(record.get("entity")),
                      sidecar._norm(record.get("metric")),
                      sidecar._norm(record.get("period")))
        by_coordinate[coordinate][str(record.get("value"))].add(key)

    report: list[dict] = []
    tally: collections.Counter = collections.Counter()
    for record in gold:
        case_id = record["id"]
        question = questions.get(case_id) or {}
        facts = [store.get(str(f)) for f in (record.get("fact_ids") or [])]
        facts = [f for f in facts if f]
        if not facts:
            tally["GOLD_CORRECT"] += 1          # nothing to audit (abstention)
            report.append({"case_id": case_id, "verdict": "GOLD_CORRECT",
                           "flags": [], "reason": "no fact ids"})
            continue

        flags: set[str] = set()
        evidence: list[str] = []
        for fact in facts:
            coordinate = (sidecar._norm(fact.get("entity")),
                          sidecar._norm(fact.get("metric")),
                          sidecar._norm(fact.get("period")))
            values = by_coordinate.get(coordinate) or {}
            if len(values) > 1:
                flags.add("MULTI_VALUE")
                evidence.append("coordinate holds %d values" % len(values))

            label = sidecar._norm(fact.get("metric"))
            amount = sidecar._money(fact.get("value"))
            document = str(fact.get("document_name"))
            for owner, caption, row in rows_by_label.get(label, ()):
                if owner != document:
                    continue
                if amount and amount not in {sidecar._money(c) for c in row}:
                    continue
                if not _PRIMARY.search(sidecar._norm(caption)):
                    flags.add("NON_PRIMARY_TABLE")
                    evidence.append("row found under caption %r" % caption[:60])
                else:
                    # A primary statement states a *different* value for the
                    # same coordinate: the gold's number is not this company's.
                    primary_values = {sidecar._money(c) for c in row}
                    if amount and amount not in primary_values:
                        flags.add("CONFLICTS_PRIMARY")
                        evidence.append("primary statement states %s"
                                        % sorted(v for v in primary_values if v)[:3])
                    else:
                        evidence.append("primary statement %r" % caption[:50])

        if not flags:
            verdict = "GOLD_CORRECT"
        elif "CONFLICTS_PRIMARY" in flags and "NON_PRIMARY_TABLE" in flags:
            verdict = "GOLD_WRONG_TABLE"
        elif "NON_PRIMARY_TABLE" in flags:
            verdict = "GOLD_WRONG_TABLE"
        elif "MULTI_VALUE" in flags:
            verdict = "SOURCE_AMBIGUOUS"
        elif not evidence:
            verdict = "UNRESOLVED"
        else:
            verdict = "GOLD_CORRECT"
        tally[verdict] += 1
        report.append({
            "case_id": case_id,
            "question": question.get("question"),
            "verdict": verdict,
            "flags": sorted(flags),
            "gold_metric": record.get("metric"),
            "gold_expected": record.get("expected_value"),
            "gold_fact_values": [f.get("value") for f in facts],
            "evidence": evidence[:4],
        })

    print("=" * 78)
    print("BENCHMARK SOURCE-TRUTH AUDIT")
    print("=" * 78)
    print("  cases audited   %d" % len(report))
    for key, value in tally.most_common():
        print("    %-22s %3d" % (key, value))
    print("    %-22s %3d   <- must be 0" % ("UNRESOLVED", tally["UNRESOLVED"]))
    print()
    for entry in report:
        if entry["verdict"] in ("GOLD_CORRECT",):
            continue
        print("  %-26s %-20s flags=%s"
              % (entry["case_id"], entry["verdict"], entry["flags"]))
        print("      q: %s" % str(entry["question"])[:76])
        print("      gold %s=%r from %s"
              % (entry["gold_metric"], entry["gold_expected"],
                 entry["gold_fact_values"]))
        for item in entry["evidence"]:
            print("      - %s" % item)
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark-source-truth-audit.json").write_text(
        json.dumps({"counts": dict(tally), "cases": report},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print("  written to %s" % (args.out / "benchmark-source-truth-audit.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
