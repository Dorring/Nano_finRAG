"""P1.6-A3-3b: column-local evidence taken before assembly, and whether it needs the regex.

Diagnosis only.  **No rule change.**

A3-3 measured scoping with `column_headers[col]`, which is itself `col_headers`' assembled
string -- carry-forward plus a row-level scope phrase plus whatever the label column leaked
in.  Its failure therefore proves nothing about column-local evidence; it only proves that
a polluted string stays polluted.  This takes the evidence one layer earlier:

    raw header cells that geometrically cover this logical column

`grid[i][col]["raw_text"]` for the header rows, with no carry-forward and no assembly.  For
NVIDIA's column 3 that is `Jan 26, 2025` and, if the pollution was the assembly, no
`Cash flows from operating activities` at all.

Three questions, and none of them is "does 18/18 come back":

  1. does the raw scope actually remove the row-prose contamination?
  2. is the token-safe regex still needed on a pure column input?  If `raw + old regex` is
     already clean the two are architecture and defence-in-depth; if a collision still fires
     they are both contract.
  3. are genuine comparison / bucket / segment columns still classified as such?  A change
     that turns everything into point/duration would pass 1 and 2 and be useless.

For 3 the positive controls are found by scanning the eight filings' primary statements for
a header cell that *itself* declares one of those kinds.

  python shadow_column_local_scope.py --out <dir>
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

PARSER = _BACKEND_DIR / "scripts/evaluation/run_nf_v2_17a4_parse.py"
BUILDER = _BACKEND_DIR / "scripts/evaluation/build_store_v2.py"

TABLES = (
    ("nvda_fy2025", 8766, "CASH_FLOW", "failing"),
    ("tsla_fy2025", 10407, "CASH_FLOW", "failing"),
    ("aapl_fy2025", 6383, "CASH_FLOW", "control"),
    ("msft_fy2025", 17151, "CASH_FLOW", "control"),
)

#: Words that collide when matched as substrings, hardened to whole tokens.
_COLLIDING = ("rating", "range", "grade", "tier")


def token_safe(pattern: re.Pattern) -> re.Pattern:
    source = pattern.pattern
    for word in _COLLIDING:
        source = re.sub(rf"(?<![\\\w]){word}(?![\\\w])", rf"\\b{word}\\b", source)
    return re.compile(source, pattern.flags)


def classify_with(tag, safe: bool, column_text: str, normalized_period, period_kind):
    """The cascade over column evidence only.  `cell_text` is not consulted at all."""
    patterns = {"comparison": tag._COMPARISON_RE, "bucket": tag._BUCKET_RE,
                "segment": tag._SEGMENT_RE, "category": tag._CATEGORY_RE}
    if safe:
        patterns = {k: token_safe(v) for k, v in patterns.items()}
    for name in ("comparison", "bucket", "segment", "category"):
        match = patterns[name].search(column_text)
        if match:
            return name, match.group(0)
    if tag._POINT_RE.search(column_text):
        return "point", tag._POINT_RE.search(column_text).group(0)
    if tag._DURATION_RE.search(column_text):
        return "duration", tag._DURATION_RE.search(column_text).group(0)
    if tag._FISCAL_RE.search(column_text) or normalized_period:
        return "duration", normalized_period or ""
    if period_kind:
        return period_kind, period_kind
    return "unknown", ""


def raw_columns(nf, grid, idx) -> dict[int, dict]:
    """Per logical column, the header cells that geometrically cover it, unassembled."""
    width = max((len(r) for r in grid), default=0)
    out: dict[int, dict] = {}
    for col in range(width):
        cell_ids, texts = [], []
        for i in idx:
            if i >= len(grid) or col >= len(grid[i]) or not grid[i][col]:
                continue
            cell = grid[i][col]
            text = nf.ws(cell["raw_text"])
            if not text:
                continue
            key = (i, col)
            cell_ids.append(key)
            texts.append(text)
        out[col] = {"logical_col": col, "raw_header_cell_ids": cell_ids,
                    "raw_header_texts": texts}
    return out


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
    from src.pdf_retrieval_v4 import temporal_axis_graph as tag

    from lxml import etree, html

    report = {"phase": "P1.6-A3-3b", "mutation": "none", "tables": {}, "positive_controls": []}
    print("=== raw column-local evidence, scope x regex ===")
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
        grid = nf.grid_rows(nf.direct_rows(lookup[block["table_id"]]))
        idx = nf.header_idx(grid)
        raw_cols = raw_columns(nf, grid, idx)

        by_col: dict[int, list[dict]] = {}
        for cell in adapted["cells"]:
            by_col.setdefault(int(cell.get("column_index") or 0), []).append(cell)

        columns = []
        for col in sorted(by_col):
            cells = by_col[col]
            assembled = " ".join(str(h) for h in (cells[0].get("header_path") or []))
            normalized = period_kind = None
            for cell in cells:
                normalized = cell.get("normalized_period") or normalized
                period_kind = cell.get("period_kind") or period_kind
            raw_text = " ".join(raw_cols.get(col, {}).get("raw_header_texts") or [])
            variants = {}
            for scope, text in (("assembled", assembled), ("raw", raw_text)):
                for safe in (False, True):
                    kind, on = classify_with(tag, safe, text, normalized, period_kind)
                    variants[f"{scope}_{'safe' if safe else 'raw'}"] = {"kind": kind, "on": on}
            columns.append({
                "column": col, "normalized_period": normalized,
                "raw_header_cell_ids": raw_cols.get(col, {}).get("raw_header_cell_ids"),
                "raw_header_texts": raw_cols.get(col, {}).get("raw_header_texts"),
                "assembled_header_text": assembled[:120],
                "pollution_removed": "operating" not in raw_text.lower(),
                **variants,
            })

        key = f"{document_id}#{order}"
        report["tables"][key] = {"family": family, "role": role, "columns": columns}
        variants_order = ("assembled_raw", "assembled_safe", "raw_raw", "raw_safe")
        print(f"--- {key}  {family}  ({role})")
        for variant in variants_order:
            tally: dict[str, int] = {}
            for column in columns:
                kind = column[variant]["kind"]
                tally[kind] = tally.get(kind, 0) + 1
            print(f"    {variant:<15} {tally}")
        sample = next((c for c in columns if c["column"] > 0), columns[0])
        print(f"    column {sample['column']} raw cells {sample['raw_header_cell_ids']}")
        print(f"        raw       {sample['raw_header_texts']}")
        print(f"        assembled {sample['assembled_header_text'][:80]!r}")
        print()

    # --- 3. positive controls: header cells that themselves declare a non-temporal kind --
    vocab = {"comparison": tag._COMPARISON_RE, "bucket": tag._BUCKET_RE,
             "segment": tag._SEGMENT_RE, "category": tag._CATEGORY_RE}
    for document_id, (ticker, accession) in sorted(builder.DOCUMENTS.items()):
        blocks, lookup, _prior = None, None, None
        try:
            raw_path = (Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")
                        / ticker / accession / "primary.html")
            root = html.parse(str(raw_path), etree.HTMLParser(
                recover=True, no_network=True, huge_tree=True,
                remove_comments=True)).getroot()
            blocks, lookup, _prior = nf.make_blocks(
                root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})
        except Exception:  # noqa: BLE001 - a missing filing is not a finding
            continue
        for block in blocks:
            if block["block_type"] != "TABLE":
                continue
            grid = nf.grid_rows(nf.direct_rows(lookup[block["table_id"]]))
            idx = nf.header_idx(grid)
            for col, entry in raw_columns(nf, grid, idx).items():
                text = " ".join(entry["raw_header_texts"])
                if not text:
                    continue
                for kind, pattern in vocab.items():
                    match = pattern.search(text)
                    if match:
                        report["positive_controls"].append({
                            "document_id": document_id, "order": block["source_order"],
                            "column": col, "kind": kind, "matched": match.group(0),
                            "raw_header_texts": entry["raw_header_texts"],
                        })
                        break

    print("=== positive controls: header cells declaring a non-temporal kind ===")
    seen = set()
    for entry in report["positive_controls"]:
        sig = (entry["kind"], entry["matched"].lower())
        if sig in seen:
            continue
        seen.add(sig)
        print(f"    {entry['kind']:<11} {entry['matched'][:22]!r:<24} "
              f"{entry['document_id']} col {entry['column']}  {entry['raw_header_texts']}")
    print(f"    distinct kinds found: {sorted({e['kind'] for e in report['positive_controls']})}")
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "column-local-scope.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'column-local-scope.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
