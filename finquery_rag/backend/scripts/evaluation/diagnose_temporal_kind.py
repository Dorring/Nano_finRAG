"""P1.6-A3-2: why a column that has a period still classifies as `bucket`.

Diagnosis only.  **No temporal rule change, no period change.**  A3-1a2 found that
NVIDIA's and Tesla's cash flow statements bind a period on 15 of 18 columns and still
emit nothing, because every one of those columns is classified `bucket` -- a kind
`emit_atomic_facts` refuses.  That is a second root cause sitting at the *shared
downstream gate* of every period repair, so it is worth settling before any header work
lands: repairing the headers of Visa, Pfizer and Coca-Cola and then finding the same
zero at the same gate is the outcome this pass exists to prevent.

`_classify_column_temporal` is a priority cascade over
`header_text + " " + cell_text`, where `cell_text` is the concatenation of every cell's
text in that column.  This reports, per column, which step fired and on what, beside a
control that reaches `point` or `duration` -- so the divergence is read off the cascade
rather than inferred from its output.

  python diagnose_temporal_kind.py --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

PARSER = _BACKEND_DIR / "scripts/evaluation/run_nf_v2_17a4_parse.py"
BUILDER = _BACKEND_DIR / "scripts/evaluation/build_store_v2.py"

#: The two that bind a period and still fail, and two that bind and pass.
TABLES = (
    ("nvda_fy2025", 8766, "CASH_FLOW", "fails"),
    ("tsla_fy2025", 10407, "CASH_FLOW", "fails"),
    ("aapl_fy2025", 5498, "BALANCE_SHEET", "control"),
    ("aapl_fy2025", 6383, "CASH_FLOW", "control"),
    ("msft_fy2025", 17151, "CASH_FLOW", "control"),
)


def cascade(tag, combined: str, normalized_period, period_kind, cell_text: str) -> dict:
    """Run the classifier's steps in order and name the one that fires.

    The steps and their order are the module's; this only reports which one matched
    and on what, which the function itself discards.
    """
    steps = (
        ("1 comparison", tag._COMPARISON_RE),
        ("2 bucket", tag._BUCKET_RE),
        ("3 segment", tag._SEGMENT_RE),
        ("4 category", tag._CATEGORY_RE),
        ("5 point", tag._POINT_RE),
        ("6 duration", tag._DURATION_RE),
        ("7 fiscal", tag._FISCAL_RE),
    )
    for name, pattern in steps:
        match = pattern.search(combined)
        if match:
            return {"step": name, "matched": match.group(0)}
    if normalized_period:
        return {"step": "7 normalized_period", "matched": normalized_period}
    if period_kind:
        return {"step": "8 period_kind", "matched": period_kind}
    if not cell_text.strip():
        return {"step": "9 non_temporal", "matched": ""}
    return {"step": "10 unknown", "matched": ""}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    spec = importlib.util.spec_from_file_location("nf17a4", PARSER)
    nf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nf)
    bspec = importlib.util.spec_from_file_location("store_builder", BUILDER)
    builder = importlib.util.module_from_spec(bspec)
    bspec.loader.exec_module(builder)

    from src.pdf_retrieval_v4.html_semantic_adapter import _adapt_table
    from src.pdf_retrieval_v4.temporal_axis_graph import (
        _classify_column_temporal as classify,
    )
    tag = sys.modules["src.pdf_retrieval_v4.temporal_axis_graph"]

    from lxml import etree, html

    report = {"phase": "P1.6-A3-2", "mutation": "none", "tables": {}}
    print("=== temporal kind: which cascade step fires ===")
    print()

    for document_id, order, family, role in TABLES:
        ticker, accession = builder.DOCUMENTS[document_id]
        raw = (Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")
               / ticker / accession / "primary.html")
        root = html.parse(str(raw), etree.HTMLParser(
            recover=True, no_network=True, huge_tree=True,
            remove_comments=True)).getroot()
        blocks, lookup, _prior = nf.make_blocks(
            root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})
        block = next(b for b in blocks
                     if b["block_type"] == "TABLE" and b["source_order"] == order)
        table = next(t for t in builder.parse_filing(ticker, accession, document_id)["tables"]
                     if t["table_id"] == block["table_id"])
        adapted = _adapt_table(table, {"document_id": document_id, "ticker": ticker})

        by_col: dict[int, list[dict]] = {}
        for cell in adapted["cells"]:
            by_col.setdefault(int(cell.get("column_index") or 0), []).append(cell)

        columns = []
        for col in sorted(by_col):
            cells = by_col[col]
            header_paths: list[str] = []
            texts: list[str] = []
            normalized = None
            period_kind = None
            for cell in cells:
                header_paths.extend(str(h) for h in (cell.get("header_path") or []))
                texts.append(str(cell.get("resolved_text") or ""))
                normalized = cell.get("normalized_period") or normalized
                period_kind = cell.get("period_kind") or period_kind
            header_text = " ".join(header_paths)
            cell_text = " ".join(texts)
            combined = header_text + " " + cell_text
            kind = classify(header_paths, normalized, period_kind, cell_text)[0]
            columns.append({
                "column": col,
                "kind": kind,
                "normalized_period": normalized,
                "period_kind": period_kind,
                "branch": cascade(tag, combined, normalized, period_kind, cell_text),
                "header_text": header_text[:120],
                "cell_text_head": cell_text[:90],
            })

        key = f"{document_id}#{order}"
        report["tables"][key] = {"family": family, "role": role, "columns": columns}
        kinds: dict[str, int] = {}
        for column in columns:
            kinds[column["kind"]] = kinds.get(column["kind"], 0) + 1
        print(f"--- {key}  {family}  ({role})   kinds {kinds}")
        for column in columns[:5]:
            print(f"    col {column['column']:<3} kind={column['kind']:<12} "
                  f"period={str(column['normalized_period']):<12} "
                  f"branch={column['branch']['step']:<20} on "
                  f"{column['branch']['matched'][:34]!r}")
        print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "temporal-kind.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'temporal-kind.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
