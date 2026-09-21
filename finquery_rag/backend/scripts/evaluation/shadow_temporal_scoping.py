"""P1.6-A3-3: temporal-kind evidence scoping, as a 2x2.

Diagnosis only.  **No rule change.**

A3-2 traced why NVIDIA's and Tesla's cash flow statements classify every column
`bucket`: `_classify_column_temporal` reads `header_text + " " + cell_text`, where the
header path has absorbed the row-label column's prose, and `_BUCKET_RE`'s `rating` --
written for credit-rating buckets -- matches inside **ope·rating**.  `_COMPARISON_RE`'s
`increase` matches the row wording `Increase (decrease) in …` the same way.

Two candidate causes, and they are separable, so this measures all four combinations
rather than guessing which one carries the fix:

    scope   old  header_text + cell_text   the row prose is part of the kind
            new  column-local evidence     the column's own header cells only
    regex   old  as written
            safe `rating`/`range`/`grade`/`tier` as whole tokens, not substrings

  python shadow_temporal_scoping.py --out <dir>
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

#: The substring that started this: `rating` inside `operating`.
_TOKEN_SAFE = (
    (re.compile(r"\brating\b", re.I), re.compile(r"\brating\b", re.I)),
    (re.compile(r"|\brange\b", re.I), re.compile(r"|\brange\b", re.I)),
)


def token_safe(pattern: re.Pattern) -> re.Pattern:
    """The same alternation, with the bare words that collide made whole-token.

    Only the collateral words are hardened.  `\d\s*[-–to]+\s*\d\s*year` and the rest of
    the bucket vocabulary are structural and are left as they are, so a change in
    behaviour is attributable to the collision and not to a rewritten pattern.
    """
    source = pattern.pattern
    for word in ("rating", "range", "grade", "tier"):
        source = re.sub(rf"(?<![\\\w]){word}(?![\\\w])", rf"\\b{word}\\b", source)
    return re.compile(source, pattern.flags)


def classify_with(tag, safe: bool, header_text: str, cell_text: str,
                  normalized_period, period_kind):
    """The module's cascade, with the pattern set optionally hardened."""
    patterns = {
        "comparison": tag._COMPARISON_RE,
        "bucket": tag._BUCKET_RE,
        "segment": tag._SEGMENT_RE,
        "category": tag._CATEGORY_RE,
    }
    if safe:
        patterns = {k: token_safe(v) for k, v in patterns.items()}
    combined = header_text + " " + cell_text
    for name in ("comparison", "bucket", "segment", "category"):
        match = patterns[name].search(combined)
        if match:
            return name, match.group(0)
    if tag._POINT_RE.search(combined):
        return "point", tag._POINT_RE.search(combined).group(0)
    if tag._DURATION_RE.search(combined):
        return "duration", tag._DURATION_RE.search(combined).group(0)
    if tag._FISCAL_RE.search(combined) or normalized_period:
        return "duration", normalized_period or ""
    if period_kind:
        return period_kind, period_kind
    if not cell_text.strip():
        return "non_temporal", ""
    return "unknown", ""


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

    report = {"phase": "P1.6-A3-3", "mutation": "none", "tables": {}}
    print("=== temporal kind: scope x regex, 2x2 ===")
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
            paths: list[str] = []
            texts: list[str] = []
            normalized = period_kind = None
            for cell in cells:
                paths.extend(str(h) for h in (cell.get("header_path") or []))
                texts.append(str(cell.get("resolved_text") or ""))
                normalized = cell.get("normalized_period") or normalized
                period_kind = cell.get("period_kind") or period_kind
            old_header, old_cell = " ".join(paths), " ".join(texts)
            # Column-local: the column's own header cells, and nothing from the rows.
            new_header = " ".join(str(h) for h in (
                adapted["column_headers"][col:col + 1]
                if col < len(adapted["column_headers"]) else []))
            kinds = {}
            for scope, (head, txt) in (("old", (old_header, old_cell)),
                                       ("new", (new_header, ""))):
                for safe in (False, True):
                    kind, on = classify_with(tag, safe, head, txt, normalized, period_kind)
                    kinds[f"{scope}_{'safe' if safe else 'raw'}"] = {"kind": kind, "on": on}
            columns.append({"column": col, "normalized_period": normalized, **kinds})

        key = f"{document_id}#{order}"
        report["tables"][key] = {"family": family, "role": role, "columns": columns}

        variants = ("old_raw", "new_raw", "old_safe", "new_safe")
        print(f"--- {key}  {family}  ({role})")
        for variant in variants:
            tally: dict[str, int] = {}
            for column in columns:
                kind = column[variant]["kind"]
                tally[kind] = tally.get(kind, 0) + 1
            print(f"    {variant:<10} {tally}")
        for column in columns[:3]:
            print(f"      col {column['column']:<3} "
                  + "  ".join(f"{v}={column[v]['kind']}({column[v]['on'][:12]!r})"
                              for v in variants))
        print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "temporal-scoping.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'temporal-scoping.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
