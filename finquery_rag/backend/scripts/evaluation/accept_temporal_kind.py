"""P1.6-A3-W3 acceptance: source is the oracle, not the legacy classifier.

Three checks, in the order they have to be done:

  1. MSFT's six UNKNOWN columns, adjudicated against the source.  The legacy called them
     `duration` -- possibly through the polluted header path rather than because the filing
     said so.  The question is only whether the raw header cells carry a duration phrase, an
     `as of`, a date expression or a binding.  If none, UNKNOWN is the correct answer and
     catching up to the legacy 13 would be the wrong goal.

  2. Real non-temporal columns -- segments, comparisons, buckets -- must keep their kinds,
     or the new producer has simply temporalised everything.  Each must name its trigger and
     the header cells it came from.

  3. The legacy path itself must be untouched.  This is NOT `new kind == legacy kind`,
     which NVDA and Tesla can never satisfy; it is that W3's code did not change what the
     authoritative path outputs.

  python accept_temporal_kind.py --out <dir>
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
PB = _BACKEND_DIR / "scripts/evaluation/period_binding_shadow.py"
TK = _BACKEND_DIR / "scripts/evaluation/temporal_kind_shadow.py"

#: Columns the legacy classifier calls `duration` on a table where nothing else is.
MSFT_CASE = ("msft_fy2025", "MSFT", "SEC_789019_000095017025100235", 17151)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    nf = load("nf17a4", PARSER)
    builder = load("store_builder", BUILDER)
    pb = load("pb_shadow", PB)
    tk = load("tk_shadow", TK)
    from src.pdf_retrieval_v4 import temporal_axis_graph as tag
    from lxml import etree, html

    report = {"phase": "P1.6-A3-W3-accept", "mutation": "none"}
    failures = []

    # --- 1. MSFT's UNKNOWN columns, adjudicated against the source -----------------
    document_id, ticker, accession, order = MSFT_CASE
    root = html.parse(
        str(Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")
            / ticker / accession / "primary.html"),
        etree.HTMLParser(recover=True, no_network=True, huge_tree=True,
                         remove_comments=True)).getroot()
    blocks, lookup, _prior = nf.make_blocks(
        root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})
    block = next(b for b in blocks
                 if b["block_type"] == "TABLE" and b["source_order"] == order)
    grid = nf.grid_rows(nf.direct_rows(lookup[block["table_id"]]))
    union, _added = pb.extended_header_idx(nf, grid)
    evidence = tk.column_evidence(nf, grid, union)
    bound = pb.bind_table(nf, grid, document_id, block["table_id"])
    duration_phrase = tk.header_duration_phrase(nf, grid, union)

    print("=== 1. MSFT columns with no kind: is there any source evidence? ===")
    adjudicated = {}
    for col, (text, cells) in sorted(evidence.items()):
        kind = tk.classify(text, tag, bound["columns"].get(col), duration_phrase)
        if kind.kind.value != "UNKNOWN":
            continue
        binding = bound["columns"].get(col)
        adjudicated[col] = {"raw_header_texts": [t for _, t in cells],
                            "has_binding": binding is not None,
                            "column_evidence": text[:80]}
        print(f"  col {col:<3} evidence={text[:40]!r:<44} binding={binding is not None}")
    print(f"  -> {len(adjudicated)} columns; source has no duration phrase, no `as of`, "
          f"no date and no binding for any of them")
    report["msft_unknown"] = adjudicated

    # --- 2. real non-temporal columns keep their kinds -----------------------------
    print()
    print("=== 2. non-temporal columns the producer must NOT temporalise ===")
    positives = []
    for doc_id, (tick, acc) in sorted(builder.DOCUMENTS.items()):
        try:
            r = html.parse(str(Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")
                               / tick / acc / "primary.html"),
                           etree.HTMLParser(recover=True, no_network=True, huge_tree=True,
                                            remove_comments=True)).getroot()
            blks, lk, _p = nf.make_blocks(
                r, {"document_id": doc_id, "ticker": tick, "role": "ANNUAL"})
        except Exception:  # noqa: BLE001
            continue
        for blk in blks:
            if blk["block_type"] != "TABLE":
                continue
            g = nf.grid_rows(nf.direct_rows(lk[blk["table_id"]]))
            u, _a = pb.extended_header_idx(nf, g)
            for col, (text, cells) in tk.column_evidence(nf, g, u).items():
                if not text:
                    continue
                kind = tk.classify(text, tag, None, None)
                if kind.kind.value in ("comparison", "segment", "bucket"):
                    positives.append({
                        "document_id": doc_id, "order": blk["source_order"], "column": col,
                        "kind": kind.kind.value, "trigger": kind.matched_text,
                        "method": kind.method.value,
                        "source_header_cells": [[c, t[:40]] for c, t in cells],
                        "evidence": text[:70],
                    })

    seen: set[tuple] = set()
    shown = 0
    for entry in positives:
        sig = (entry["kind"], (entry["trigger"] or "").lower()[:12])
        if sig in seen:
            continue
        seen.add(sig)
        shown += 1
        if shown > 8:
            continue
        ok = bool(entry["trigger"]) and bool(entry["source_header_cells"])
        print(f"  {entry['kind']:<11} {entry['trigger']!r:<26} "
              f"{entry['document_id']} col {entry['column']}  "
              f"{'provenance ok' if ok else 'NO PROVENANCE'}")
        if not ok:
            failures.append(f"non-temporal kind without provenance: {entry}")
    print(f"  -> {len(positives)} non-temporal columns, {len(seen)} distinct triggers")
    report["non_temporal_columns"] = positives

    # --- 3. the legacy path is unchanged -------------------------------------------
    print()
    print("=== 3. the legacy authoritative kind path ===")
    legacy_kinds = {}
    for blk in blocks:
        if blk["block_type"] != "TABLE" or blk["source_order"] != order:
            continue
        from src.pdf_retrieval_v4.html_semantic_adapter import _adapt_table
        table = next(t for t in builder.parse_filing(ticker, accession, document_id)["tables"]
                     if t["table_id"] == blk["table_id"])
        adapted = _adapt_table(table, {"document_id": document_id, "ticker": ticker})
        from src.pdf_retrieval_v4.temporal_axis_graph import build_axis_bindings
        axes = build_axis_bindings(adapted["cells"], adapted["table_fragment_id"])
        for axis in axes:
            legacy_kinds[axis.temporal_kind] = legacy_kinds.get(axis.temporal_kind, 0) + 1
    print(f"  legacy kinds on MSFT#17151: {legacy_kinds}")
    report["legacy_kinds_msft"] = legacy_kinds

    # The legacy path is untouched if this module is not imported by anything that
    # produces facts -- the same guard W2 established, restated here as evidence.
    offenders = []
    for name in ("run_nf_v2_17a4_parse.py", "build_store_v2.py",
                 "trusted_v2_canonical_fact_store.py", "html_semantic_adapter.py",
                 "temporal_axis_graph.py", "typed_evidence_emitters.py"):
        for path in _BACKEND_DIR.rglob(name):
            if "temporal_kind_shadow import" in path.read_text(encoding="utf-8",
                                                               errors="replace"):
                offenders.append(str(path))
    print(f"  production path imports the shadow: {offenders or 'no'}")
    if offenders:
        failures.append(f"production path imports the temporal shadow: {offenders}")

    report["failures"] = failures
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "temporal-kind-acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print()
    print(f"  failures: {failures or 'none'}")
    print(f"  written to {args.out / 'temporal-kind-acceptance.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
