"""P1.6-A2B-17: why did each unresolved slot not resolve?

**Classification only. No resolver change, no classifier change, no fixture
change.** The point is a priority matrix, so that the next step is chosen from a
root-cause distribution rather than from the case that happened to be looked at
last.

The buckets, and why `B` and `C` are separate: a table wrongly labelled a primary
statement and a company figure that only exists in the notes look identical from a
refusal -- both say "no company-level fact" -- and want opposite fixes. The first is
the statement classifier admitting something it should not; the second is the
resolver being conservative where it should perhaps not be.

    A  METRIC_ALIAS / LABEL_MISMATCH   the row exists under a different label
    B  STATEMENT_CLASSIFICATION_ERROR  a non-statement table labelled one
    C  NON_PRIMARY_SOURCE_ONLY         the figure exists only in notes/MDA
    D  PERIOD_MISMATCH                 the metric exists at another period
    E  PARSER_EXTRACTION_MISS          the row is in the source, no fact came out
    F  SOURCE_ABSENCE_UNVERIFIED       nothing found -- **unverified**, because a
                                       filing states `Diluted earnings per share`
                                       whether or not this pass recognised it
    G  SOURCE_AUTHORITY_UNRESOLVED     several candidates, nothing says which

**Two populations, not one.** This pass and the resolver ask different questions --
"is there a primary-statement candidate" against "do the company-level candidates
agree" -- so their unresolved sets differ.  Both are reported, with the overlap, so
that neither number is quoted as the coverage figure without the other beside it.

  python classify_unresolved_slots.py --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

CORPUS = Path("/disk/qh/nano-finrag/data/financial_corpus_v2")
PARSER = _BACKEND_DIR / "scripts/evaluation/run_nf_v2_17a4_parse.py"
STORE = Path("/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a2b12-store-v2/store-v2.jsonl")

DOCUMENTS = {
    "Apple": ("AAPL", "SEC_320193_000032019325000079", "aapl_fy2025"),
    "JPMorganChase": ("JPM", "SEC_19617_000162828026008131", "jpm_fy2025"),
    "The Coca-Cola Company": ("KO", "SEC_21344_000162828026010047", "ko_fy2025"),
    "Microsoft": ("MSFT", "SEC_789019_000095017025100235", "msft_fy2025"),
    "NVIDIA": ("NVDA", "SEC_1045810_000104581025000023", "nvda_fy2025"),
    "Pfizer": ("PFE", "SEC_78003_000007800325000054", "pfe_fy2024"),
    "Tesla": ("TSLA", "SEC_1318605_000162828026003952", "tsla_fy2025"),
    "Visa": ("V", "SEC_1403161_000140316125000089", "v_fy2025"),
}
KO, JPM = "The Coca-Cola Company", "JPMorganChase"

CASES = {
    "compare-001": ("Net income", [KO, "Tesla"]),
    "compare-002": ("Net income", ["Apple", "Visa"]),
    "compare-003": ("Total assets", ["Tesla", KO]),
    "compare-004": ("Net income", ["Apple", "Microsoft"]),
    "compare-005": ("Diluted earnings per share", [JPM, "Apple"]),
    "compare-006": ("Comprehensive income", [JPM, "Tesla"]),
    "compare-007": ("Net income", [JPM, "Pfizer"]),
    "compare-008": ("Diluted earnings per share", [JPM, "Apple"]),
    "compare-009": ("Net income", ["Apple", JPM]),
    "compare-010": ("Operating income", ["Apple", "Microsoft"]),
    "crossdiff-001": ("Net income", ["NVIDIA", "Tesla"]),
    "crossdiff-002": ("Net income", ["Apple", "Microsoft"]),
    "crossdiff-003": ("Total liabilities", ["Apple", "Tesla"]),
    "crossdiff-004": ("Long-term debt", [KO, "Visa"]),
    "crossdiff-005": ("Total assets", ["NVIDIA", KO]),
    "rank-001": ("Net income", ["Tesla", KO, "Visa"]),
    "rank-002": ("Research and development", ["Apple", "Microsoft", "Tesla"]),
    "rank-003": ("Comprehensive income", [JPM, "Microsoft", "Tesla", "Visa"]),
    "rank-004": ("Interest expense", [JPM, KO, "Visa", "Microsoft"]),
    "rank-005": ("Operating income", ["Apple", "Microsoft", KO]),
}
PERIOD = {"Pfizer": "FY2024"}

_PRIMARY = ("INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW")
#: A table that calls itself the statement of record.
_SELF_NAMING = re.compile(
    r"\b(consolidated\s+)?(statements?|balance sheets?)\s+of\b|\bconsolidated balance", re.I)
_CAPTION = re.compile(r"\s*\((?:in|except|dollars in|amounts in)[^)]*\)\s*$", re.I)


def _norm(text: object) -> str:
    return " ".join(str(text or "").split()).casefold()


def _label_matches(label: object, metric: str) -> bool:
    return _CAPTION.sub("", _norm(label)).strip() == _norm(metric)


def _filing_tables(ticker: str, accession: str, doc_id: str) -> dict[str, dict]:
    from lxml import etree, html

    spec = importlib.util.spec_from_file_location("nf17a4", PARSER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    raw = CORPUS / "raw/SEC" / ticker / accession / "primary.html"
    root = html.parse(
        str(raw),
        etree.HTMLParser(recover=True, no_network=True, huge_tree=True,
                         remove_comments=True),
    ).getroot()
    doc = {"document_id": doc_id, "ticker": ticker, "role": "ANNUAL"}
    blocks, lookup, prior = module.make_blocks(root, doc)

    out: dict[str, dict] = {}
    for block in blocks:
        if block["block_type"] != "TABLE":
            continue
        text = module.ws(str(block.get("text") or ""))
        out[block["table_id"]] = {
            "section_type": block["section_type"],
            "text_head": text[:160],
            "self_naming": bool(_SELF_NAMING.search(text[:200])),
            "source_order": block["source_order"],
        }
    return out


def classify(records, tables, entity, metric, period) -> dict:
    year = "".join(ch for ch in period if ch.isdigit())[:4]

    same_entity = [r for r in records if str(r.get("entity")) == entity]
    exact = [r for r in same_entity
             if _label_matches(r.get("row_label"), metric)
             and str(r.get("period_end") or "")[:4] == year]
    near = [r for r in same_entity
            if _norm(metric) in _norm(r.get("row_label"))
            and str(r.get("period_end") or "")[:4] == year]
    other_period = [r for r in same_entity if _label_matches(r.get("row_label"), metric)]

    def evidence(rows, limit=4):
        out = []
        for r in rows[:limit]:
            table = tables.get(str(r.get("table_fragment_id")), {})
            out.append({
                "value": r.get("value"), "label": r.get("row_label"),
                "column_header": str(r.get("column_header"))[:70],
                "statement_type": r.get("statement_type"),
                "table_self_naming": table.get("self_naming"),
                "table_head": str(table.get("text_head"))[:90],
            })
        return out

    # G first: several primary-statement candidates that disagree, with nothing
    # naming one of them as the statement of record.
    primary = [r for r in exact if str(r.get("statement_type")) in _PRIMARY]
    values = {str(r.get("value")) for r in primary}
    if len(values) > 1:
        named = [r for r in primary
                 if tables.get(str(r.get("table_fragment_id")), {}).get("self_naming")]
        if not named:
            return {"taxonomy": "G_SOURCE_AUTHORITY_UNRESOLVED",
                    "detail": f"{len(values)} primary candidates, none self-naming",
                    "evidence": evidence(primary)}

    if primary:
        # A primary-statement candidate that is not really a statement is B.
        mislabelled = [r for r in primary
                       if not tables.get(str(r.get("table_fragment_id")), {}).get(
                           "self_naming", False)]
        if mislabelled and len(mislabelled) == len(primary):
            return {"taxonomy": "B_STATEMENT_CLASSIFICATION_ERROR",
                    "detail": "every primary-statement candidate sits in a table "
                              "that does not name itself as a statement",
                    "evidence": evidence(primary)}
        return {"taxonomy": "RESOLVED_ELSEWHERE",
                "detail": "a primary-statement candidate exists", "evidence": evidence(primary)}

    if exact:
        kinds = {str(r.get("statement_type")) for r in exact}
        if kinds <= {"UNKNOWN", "NOTES", "MDA", None, "None"}:
            return {"taxonomy": "C_NON_PRIMARY_SOURCE_ONLY",
                    "detail": f"the row exists only in {sorted(kinds)}",
                    "evidence": evidence(exact)}
        return {"taxonomy": "G_SOURCE_AUTHORITY_UNRESOLVED",
                "detail": f"candidates in {sorted(kinds)}", "evidence": evidence(exact)}

    if near:
        return {"taxonomy": "A_METRIC_ALIAS",
                "detail": "a row carries the metric's words under another label",
                "evidence": evidence(near)}

    if other_period:
        return {"taxonomy": "D_PERIOD_MISMATCH",
                "detail": "the metric exists at another period",
                "evidence": evidence(other_period)}

    # Nothing in the store.  The parsed tables still know whether a row was there.
    for table in tables.values():
        if _norm(metric) in _norm(table.get("text_head")):
            return {"taxonomy": "E_PARSER_EXTRACTION_MISS",
                    "detail": "the metric appears in a parsed table with no fact",
                    "evidence": [{"table_head": table["text_head"],
                                  "statement_type": table["section_type"]}]}
    return {"taxonomy": "SOURCE_ABSENCE_UNVERIFIED",
            "detail": "no row, fact or table in the source carries this metric; "
                      "not checked exhaustively, so this is unverified absence "
                      "rather than an established one",
            "evidence": []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    records = [
        json.loads(line)
        for line in args.store.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tables_by_entity: dict[str, dict] = {}

    rows = []
    tally = collections.Counter()
    classified_slots: set[tuple[str, str]] = set()
    for case_id, (metric, entities) in sorted(CASES.items()):
        for entity in entities:
            period = PERIOD.get(entity, "FY2025")
            if entity not in tables_by_entity:
                ticker, accession, doc_id = DOCUMENTS[entity]
                tables_by_entity[entity] = _filing_tables(ticker, accession, doc_id)
            result = classify(records, tables_by_entity[entity], entity, metric, period)
            if result["taxonomy"] == "RESOLVED_ELSEWHERE":
                continue
            tally[result["taxonomy"]] += 1
            classified_slots.add((case_id, entity))
            rows.append({"case_id": case_id, "entity": entity, "metric": metric,
                         "period": period, **result})

    # The resolver's own unresolved set, so the two denominators are named rather
    # than conflated.  They answer different questions and their difference is
    # reported instead of reconciled away.
    resolver_unresolved: set[tuple[str, str]] = set()
    resolver_path = Path(
        "/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a2b15-cross-v2/"
        "cross-entity-v2-resolution.json"
    )
    if resolver_path.is_file():
        artifact = json.loads(resolver_path.read_text(encoding="utf-8"))
        for case in artifact.get("cases") or ():
            for slot in case.get("slots") or ():
                if slot.get("status") != "RESOLVED_COMPANY_LEVEL":
                    resolver_unresolved.add((case["case_id"], slot["entity"]))

    populations = {
        "taxonomy_population": len(classified_slots),
        "resolver_unresolved_population": len(resolver_unresolved),
        "intersection": len(classified_slots & resolver_unresolved),
        "taxonomy_only": sorted(f"{c}/{e}" for c, e in classified_slots - resolver_unresolved),
        "resolver_only": sorted(f"{c}/{e}" for c, e in resolver_unresolved - classified_slots),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "unresolved-slot-taxonomy.json").write_text(
        json.dumps({"phase": "P1.6-A2B-17", "mutation": "none",
                    "tally": dict(tally), "populations": populations,
                    "rows": rows},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print("=== unresolved slot taxonomy ===")
    for name, count in tally.most_common():
        print(f"  {count:>3}  {name}")
    print()
    by_bucket: dict[str, list] = collections.defaultdict(list)
    for row in rows:
        by_bucket[row["taxonomy"]].append(row)
    for name in sorted(by_bucket):
        print(f"--- {name}")
        for row in by_bucket[name]:
            print(f"    {row['case_id']:14} {row['entity'][:22]:22} "
                  f"{row['metric'][:26]:26}")
            print(f"        {row['detail']}")
    print()
    print("=== the two populations, kept apart ===")
    for key in ("taxonomy_population", "resolver_unresolved_population",
                "intersection"):
        print(f"  {key:32} {populations[key]}")
    print(f"  taxonomy only ({len(populations['taxonomy_only'])})  "
          f"{populations['taxonomy_only'][:6]}")
    print(f"  resolver only ({len(populations['resolver_only'])})  "
          f"{populations['resolver_only'][:6]}")
    print()
    print(f"  written to {args.out / 'unresolved-slot-taxonomy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
