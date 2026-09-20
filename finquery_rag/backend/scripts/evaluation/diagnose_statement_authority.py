"""P1.6-A2B-18: what the statement classifier sees, and what it misses.

Diagnosis only.  **No classifier change, no resolver change, no fixture change.**
The taxonomy put 36 of 43 unresolved slots on the statement classification, failing
in both directions:

    false negative    a real primary statement classified UNKNOWN
    false positive    a note or breakdown classified as a primary statement

Relaxing the classifier cannot fix both, and would trade one for the other.

What this measures, per filing:

  * the current `statement_type` of every eligible table;
  * what `section()` says about **the table's own text** -- a signal the parser
    already holds and does not currently use for this decision;
  * how many tables are currently called primary and do **not** name themselves as
    a statement of record (the false-positive side);
  * how many are called UNKNOWN and **would** classify as primary from their own
    text (the false-negative side).

The oracle here is the table's own source evidence -- its caption, its heading, and
its opening rows -- not this script's opinion and not the classifier under test.

  python diagnose_statement_authority.py --out <dir>
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

DOCUMENTS = {
    "aapl_fy2025": ("AAPL", "SEC_320193_000032019325000079"),
    "jpm_fy2025": ("JPM", "SEC_19617_000162828026008131"),
    "ko_fy2025": ("KO", "SEC_21344_000162828026010047"),
    "msft_fy2025": ("MSFT", "SEC_789019_000095017025100235"),
    "nvda_fy2025": ("NVDA", "SEC_1045810_000104581025000023"),
    "pfe_fy2024": ("PFE", "SEC_78003_000007800325000054"),
    "tsla_fy2025": ("TSLA", "SEC_1318605_000162828026003952"),
    "v_fy2025": ("V", "SEC_1403161_000140316125000089"),
}

_PRIMARY = ("INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW")

#: A table that names itself the statement of record.  This is the source-derived
#: authority the resolver lacks: `INCOME_STATEMENT` says what the table is *about*,
#: this says whether it is the statement itself.  Reported, not applied.
_SELF_NAMING = re.compile(
    r"\bconsolidated\s+(statements?|balance\s+sheets?)\b"
    r"|\bstatements?\s+of\s+(operations|income|earnings|financial position|"
    r"cash flows|comprehensive income|shareholders)\b"
    r"|\bconsolidated\s+balance\s+sheets?\b",
    re.I,
)


def _scan(ticker: str, accession: str, document_id: str) -> dict:
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
    doc = {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"}
    blocks, _lookup, _prior = module.make_blocks(root, doc)

    tables = []
    for block in blocks:
        if block["block_type"] != "TABLE":
            continue
        text = module.ws(str(block.get("text") or ""))
        concept = module.section(text)
        tables.append({
            "table_id": block["table_id"],
            "current": block["section_type"],
            # What the table's own text says it is.  The parser computes
            # `section()` for blocks already but uses it on headings and
            # captions, never on the table body.
            "from_own_text": concept,
            "self_naming": bool(_SELF_NAMING.search(text[:300])),
            "text_head": text[:120],
        })
    return {"blocks": len(blocks), "tables": tables}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    report = {"phase": "P1.6-A2B-18", "mutation": "none", "documents": {}}
    totals = collections.Counter()

    print("=== statement authority diagnosis ===")
    print()
    for document_id, (ticker, accession) in sorted(DOCUMENTS.items()):
        scan = _scan(ticker, accession, document_id)
        tables = scan["tables"]
        current = collections.Counter(t["current"] for t in tables)

        primary_now = [t for t in tables if t["current"] in _PRIMARY]
        self_named = [t for t in tables if t["self_naming"]]
        # False positive side: called primary, does not name itself.
        fp = [t for t in primary_now if not t["self_naming"]]
        # False negative side: not called primary, but its own text says it is.
        fn = [t for t in tables
              if t["current"] not in _PRIMARY and t["from_own_text"] in _PRIMARY]

        report["documents"][document_id] = {
            "blocks": scan["blocks"],
            "tables": len(tables),
            "current_statement_type": dict(current),
            "currently_primary": len(primary_now),
            "self_naming": len(self_named),
            "false_positive_candidates": [
                {"table_id": t["table_id"], "current": t["current"],
                 "text_head": t["text_head"]} for t in fp[:6]],
            "false_negative_candidates": [
                {"table_id": t["table_id"], "from_own_text": t["from_own_text"],
                 "text_head": t["text_head"]} for t in fn[:6]],
        }
        totals["tables"] += len(tables)
        totals["primary_now"] += len(primary_now)
        totals["self_naming"] += len(self_named)
        totals["fp"] += len(fp)
        totals["fn"] += len(fn)

        print(f"--- {document_id}")
        print(f"    blocks {scan['blocks']:>5}  tables {len(tables):>5}  "
              f"current {dict(current)}")
        print(f"    currently primary {len(primary_now):>4}  "
              f"self-naming {len(self_named):>4}  "
              f"FP(primary but not self-naming) {len(fp):>4}  "
              f"FN(UNKNOWN but own text primary) {len(fn):>4}")
        for t in fn[:3]:
            print(f"        FN {t['table_id'][:26]} {t['from_own_text']:16} "
                  f"{t['text_head'][:66]!r}")
        for t in fp[:3]:
            print(f"        FP {t['table_id'][:26]} {t['current']:16} "
                  f"{t['text_head'][:66]!r}")
        print()

    report["totals"] = dict(totals)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "statement-authority-diagnosis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  totals {dict(totals)}")
    print(f"  written to {args.out / 'statement-authority-diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
