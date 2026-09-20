"""NF-V3 Phase 0: what the `_BUCKET_RE` token-boundary fix actually moves.

W6 found that `rating` had no word boundaries and so matched inside `ope·rating·
activities`, classifying every cash-flow section as a maturity bucket, routing it to
`DISAGGREGATION_AXIS`, and withholding the table. Four verified primary statements --
JPMorganChase, Coca-Cola, NVIDIA and Tesla cash flow -- produced no records for that one
reason.

This is the single-variable check for the fix, and it is stated as a diff rather than as a
result. For every column in the corpus it reconstructs the exact text
`_classify_column_temporal` reads, applies the old and the new pattern to it, and splits
the answer three ways:

  KEPT     matched before and still does       -- real bucket semantics, must not move
  LOST     matched before, does not now        -- the false positives, listed with the token
  GAINED   does not match before, does now     -- must be zero; the fix may only narrow

A column is only considered if it *reaches* the bucket branch, since `comparison` is tested
first and a column it claims never gets that far. Counting a false positive that could not
have fired would overstate the fix.

  python audit_bucket_token_boundary.py --out <dir>
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

BUILDER = _BACKEND_DIR / "scripts/evaluation/build_store_v2.py"

#: Verbatim from the sealed revision, so the comparison is against the code that shipped
#: and not against a paraphrase of it.
OLD_BUCKET_RE = re.compile(
    r"less\s+than|more\s+than|over\s+\d|\d\s*[-–to]+\s*\d\s*year"
    r"|range|rating|grade|tier",
    re.IGNORECASE,
)

PROBES = ("Operating Activities", "Cash flows from operating activities:",
          "Weighted average rating", "Investment grade", "Tier 1 capital",
          "a maturity range", "Less than 1 year", "1-3 years", "Over 5 years")


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    builder = _module(BUILDER, "builder")
    from src.pdf_retrieval_v4.html_semantic_adapter import build_semantic_corpus
    from src.pdf_retrieval_v4.temporal_axis_graph import (
        _BUCKET_RE as NEW_BUCKET_RE, _COMPARISON_RE, _classify_column_temporal)

    failures: list[str] = []
    report: dict = {"phase": "NF-V3-Phase-0-bucket-boundary"}

    print("=== the tokens this exists for ===")
    for probe in PROBES:
        old = OLD_BUCKET_RE.search(probe)
        new = NEW_BUCKET_RE.search(probe)
        print(f"    {probe!r:42} old={str(old.group(0) if old else None):10} "
              f"new={str(new.group(0) if new else None)}")
    report["probes"] = {p: {"old": bool(OLD_BUCKET_RE.search(p)),
                            "new": bool(NEW_BUCKET_RE.search(p))} for p in PROBES}
    print()

    verdicts = collections.Counter()
    lost: list[dict] = []
    gained: list[dict] = []
    columns = 0
    for document_id in sorted(builder.DOCUMENTS):
        ticker, accession = builder.DOCUMENTS[document_id]
        parsed = builder.parse_filing(ticker, accession, document_id)
        corpus = build_semantic_corpus([{**parsed, **parsed["document"]}])
        for (doc, tid), table in corpus["table_meta"].items():
            by_col: dict[int, list[dict]] = collections.defaultdict(list)
            for cell in table["cells"]:
                by_col[int(cell.get("column_index") or 0)].append(cell)
            for col, cells in by_col.items():
                # Reconstruct exactly what `_classify_column_temporal` is handed.
                headers: list[str] = []
                texts: list[str] = []
                normalized_period = None
                period_kind = None
                for c in cells:
                    headers.extend(str(h) for h in (c.get("header_path") or []))
                    texts.append(str(c.get("resolved_text") or ""))
                    if c.get("normalized_period"):
                        normalized_period = c["normalized_period"]
                    if c.get("period_kind"):
                        period_kind = c["period_kind"]
                combined = " ".join(headers) + " " + " ".join(texts)

                # A column `comparison` claims never reaches the bucket branch.
                if _COMPARISON_RE.search(combined):
                    continue
                columns += 1
                old = OLD_BUCKET_RE.search(combined)
                new = NEW_BUCKET_RE.search(combined)
                if old and new:
                    verdicts["KEPT"] += 1
                elif old and not new:
                    # Was the occurrence that fired a whole token or a fragment of one?
                    # Tested at the match site rather than by searching the text again: a
                    # fragment match is what makes this a false positive, and a token
                    # match that stopped matching would be a real loss of bucket
                    # semantics dressed up as a fix.
                    start, end = old.start(), old.end()
                    before = combined[start - 1] if start else " "
                    after = combined[end] if end < len(combined) else " "
                    bounded = not (before.isalnum() or before == "_") and not (
                        after.isalnum() or after == "_")
                    verdicts["LOST"] += 1
                    now = _classify_column_temporal(headers, normalized_period,
                                                    period_kind, " ".join(texts))[0]
                    lost.append({"document_id": doc, "table_id": tid, "column": col,
                                 "token": old.group(0), "kind_now": now,
                                 "token_bounded": bounded, "cells": len(cells),
                                 "sample": re.sub(r"\s+", " ", combined)[:150]})
                elif new and not old:
                    verdicts["GAINED"] += 1
                    gained.append({"document_id": doc, "table_id": tid, "column": col,
                                   "token": new.group(0),
                                   "sample": re.sub(r"\s+", " ", combined)[:150]})

    print(f"=== columns that reach the bucket branch: {columns} ===")
    for name, count in verdicts.most_common():
        print(f"    {count:>6}  {name}")
    print()

    by_token = collections.Counter(entry["token"].lower() for entry in lost)
    by_kind = collections.Counter(entry["kind_now"] for entry in lost)
    lost_cells = sum(entry["cells"] for entry in lost)
    print(f"=== LOST: {len(lost)} columns, {lost_cells} cells ===")
    print(f"    by token that had matched: {dict(by_token)}")
    print(f"    what those columns classify as now: {dict(by_kind)}")
    print()
    for entry in lost[:10]:
        print(f"    {entry['document_id']} {entry['table_id']} col {entry['column']}  "
              f"token={entry['token']!r} -> {entry['kind_now']}  ({entry['cells']} cells)")
        print(f"        {entry['sample'][:130]!r}")
    print()

    if gained:
        print(f"=== GAINED: {len(gained)} columns (the fix must not widen) ===")
        for entry in gained[:10]:
            print(f"    {entry['document_id']} col {entry['column']} token={entry['token']!r}")
            print(f"        {entry['sample'][:130]!r}")
        failures.append(f"THE FIX WIDENED THE RULE: {len(gained)} columns")
    else:
        print("=== GAINED: 0 -- the fix only narrows, as intended ===")
    print()

    # Every LOST column must have matched a *fragment* rather than a whole token. A LOST
    # column whose match was token-bounded is a real loss of bucket semantics, not a fix.
    whole_word = [entry for entry in lost if entry["token_bounded"]]
    if whole_word:
        print(f"    {len(whole_word)} lost column(s) matched a whole token, not a "
              f"fragment:")
        for entry in whole_word[:5]:
            print(f"        {entry['document_id']} token={entry['token']!r} "
                  f"{entry['sample'][:100]!r}")
        failures.append(f"TOKEN-BOUNDED MATCH LOST = {len(whole_word)}")
    else:
        print("    every lost match was a fragment of a longer word, not a whole token")
    report["verdicts"] = dict(verdicts)
    report["lost"] = lost
    report["gained"] = gained
    report["lost_cells"] = lost_cells
    report["lost_by_token"] = dict(by_token)
    report["lost_kind_now"] = dict(by_kind)

    print(f"  failures: {failures or 'none'}")
    report["failures"] = failures
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bucket-token-boundary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'bucket-token-boundary.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
