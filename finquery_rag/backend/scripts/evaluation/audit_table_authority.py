"""P1.6-A2B-16: does the SEC source name the consolidated statement?

Three slots resolve to several company-level values at once:

    compare-007 / compare-009   JPMorganChase  Net income      6 values
    crossdiff-004               Coca-Cola      Long-term debt  3 values

Each candidate sits in a table the parser classified `INCOME_STATEMENT` (or a
balance sheet), under the same label and period.  So the question is not which
value is right; it is whether the filing carries a **source-derived** signal that
says which of those tables is the consolidated statement.

**This audit picks nothing.**  It dumps, per candidate, the evidence such a signal
would have to come from -- table identity, statement classification, position in
the filing, the caption and heading near it, the column headers, the row label and
the inline-XBRL concept -- so the decision can be made from the filing rather than
from a tie-break rule.  If no signal exists, the slot stays AMBIGUOUS and fails
closed, which is the correct outcome and not a shortfall.

  python audit_table_authority.py --out <dir>
"""

from __future__ import annotations

import argparse
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

#: (document_id, ticker, accession, entity, metric, period) for the ambiguous slots.
TARGETS = (
    ("jpm_fy2025", "JPM", "SEC_19617_000162828026008131", "JPMorganChase",
     "Net income", "FY2025"),
    ("ko_fy2025", "KO", "SEC_21344_000162828026010047", "The Coca-Cola Company",
     "Long-term debt", "FY2025"),
)

#: Phrases a filing uses for the statement of record.  Reported as evidence, not
#: applied as a rule: what matters is whether such a phrase is *present and
#: attributable to one table*, which is a fact about the filing.
_CONSOLIDATED = re.compile(
    r"\bconsolidated\b|\bcombined statements?\b|\bthe firm\b", re.I)


def _filing_tables(ticker: str, accession: str, document_id: str) -> dict[str, dict]:
    """Parsed tables by id, with the caption and nearest preceding heading."""

    from lxml import etree, html

    module_spec = importlib.util.spec_from_file_location("nf17a4", PARSER)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    raw = CORPUS / "raw/SEC" / ticker / accession / "primary.html"
    root = html.parse(
        str(raw),
        etree.HTMLParser(recover=True, no_network=True, huge_tree=True,
                         remove_comments=True),
    ).getroot()
    document_id = document_id
    doc = {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"}

    blocks, lookup, prior = module.make_blocks(root, doc)
    tables: dict[str, dict] = {}
    for block in blocks:
        if block["block_type"] != "TABLE":
            continue
        preceding = [(o, t) for o, t in prior if o < block["source_order"]]
        heading = ""
        for _o, text in reversed(preceding):
            if 0 < len(text) <= 120 and not text.endswith("."):
                heading = module.ws(text)
                break
        caption = module.ws(" ".join(lookup[block["table_id"]].xpath("./caption//text()")))
        tables[block["table_id"]] = {
            "table_id": block["table_id"],
            "source_order": block["source_order"],
            "section_type": block["section_type"],
            "caption": caption,
            "nearest_heading": heading,
            "text_head": module.ws(str(block.get("text") or ""))[:220],
            "says_consolidated": bool(
                _CONSOLIDATED.search(caption) or _CONSOLIDATED.search(heading)
                or _CONSOLIDATED.search(str(block.get("text") or "")[:400])
            ),
        }
    return tables


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

    report = {"phase": "P1.6-A2B-16", "mutation": "none", "targets": []}

    for document_id, ticker, accession, entity, metric, period in TARGETS:
        year = "".join(ch for ch in period if ch.isdigit())[:4]
        candidates = [
            r for r in records
            if str(r.get("entity")) == entity
            and " ".join(str(r.get("row_label") or "").split()).casefold()
                == metric.casefold()
            and str(r.get("period_end") or "")[:4] == year
            and str(r.get("statement_type")) in
                ("INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW")
        ]
        tables = _filing_tables(ticker, accession, document_id)

        print(f"=== {document_id} / {entity} / {metric} / {period} ===")
        print(f"  {len(candidates)} company-level candidates")
        print()
        rows = []
        for record in candidates:
            table = tables.get(str(record.get("table_fragment_id")), {})
            row = {
                "value": record.get("value"),
                "value_raw": record.get("value_raw"),
                "statement_type": record.get("statement_type"),
                "table_id": record.get("table_fragment_id"),
                "cell_id": record.get("cell_id"),
                "column_header": record.get("column_header"),
                "row_label": record.get("row_label"),
                "table_source_order": table.get("source_order"),
                "table_caption": table.get("caption"),
                "table_heading": table.get("nearest_heading"),
                "table_says_consolidated": table.get("says_consolidated"),
                "table_text_head": table.get("text_head"),
                "anchors": [a.get("concept") for a in (record.get("source_anchors") or [])],
            }
            rows.append(row)
            print(f"  value {str(row['value_raw']):>10} | {row['statement_type']}")
            print(f"      table {str(row['table_id'])[:30]} order={row['table_source_order']}")
            print(f"      heading  {str(row['table_heading'])[:90]!r}")
            print(f"      caption  {str(row['table_caption'])[:90]!r}")
            print(f"      column   {str(row['column_header'])[:90]!r}")
            print(f"      text     {str(row['table_text_head'])[:110]!r}")
            print()
        report["targets"].append({
            "document_id": document_id, "entity": entity, "metric": metric,
            "period": period, "candidates": rows,
        })

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table-authority-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'table-authority-audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
