"""A3-W3: the new TemporalKind producer, as a second shadow output.

**Shadow only.**  The legacy `_classify_column_temporal` stays authoritative; nothing
consumes this.  W3 changes a temporal kind and nothing else -- emission, admission, the
store and the resolver are untouched, and the 1498-table emission check proves it.

Two things change from the legacy classifier, both measured in A3-3/A3-3b:

    input domain   raw column-local header cells only.  The legacy reads
                   `header_text + " " + cell_text`, where the header path has absorbed
                   the row-label column's prose -- which is how `Operating` reached it.
    patterns       `rating|range|grade|tier` as whole tokens.  A3-3b proved this is
                   contract and not defence in depth: the collision fires on a genuinely
                   raw header cell (`['Operating Leases'] -> bucket`), not only on
                   assembled prose.

Every non-UNKNOWN kind carries its provenance -- the trigger and the header cells it came
from -- so a later reader can ask *which HTML cell said Rating* rather than being told a
concatenated string happened to match.
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

from src.pdf_retrieval_v4.period_binding import (  # noqa: E402
    PeriodBindingV2,
    PeriodGranularity,
    SourceCell,
    TemporalKind,
    TemporalKindEvidence,
    TemporalKindMethod,
)

PARSER = _BACKEND_DIR / "scripts/evaluation/run_nf_v2_17a4_parse.py"
BUILDER = _BACKEND_DIR / "scripts/evaluation/build_store_v2.py"
SHADOW = _BACKEND_DIR / "scripts/evaluation/period_binding_shadow.py"

#: The words that collide as substrings, and the whole-token forms that do not.
_COLLIDING = ("rating", "range", "grade", "tier")

#: The tables W3's acceptance is about: two the legacy classifier gets wrong and two it
#: gets right.  Recorded beside the period oracles so one run covers both halves.
TEMPORAL_ORACLES = (
    ("nvda_fy2025", 8766, "CASH_FLOW"),
    ("tsla_fy2025", 10407, "CASH_FLOW"),
    ("aapl_fy2025", 6383, "CASH_FLOW"),
    ("msft_fy2025", 17151, "CASH_FLOW"),
)


def token_safe(pattern: re.Pattern) -> re.Pattern:
    """Harden the collateral words.  Everything structural is left alone, so a behaviour
    change is attributable to the collision and not to a rewritten pattern."""
    source = pattern.pattern
    for word in _COLLIDING:
        source = re.sub(rf"(?<![\\\w]){word}(?![\\\w])", rf"\\b{word}\\b", source)
    return re.compile(source, pattern.flags)


def classify(column_evidence: str, tag, binding=None,
             table_duration_phrase: str | None = None) -> TemporalKindEvidence:
    """The legacy cascade, over column-local evidence only, with token-safe patterns.

    `cell_text` is not consulted at all.  That is the whole change: a row label can no
    longer decide what kind a column is.

    The third input is the `PeriodBindingV2` for this column, and it is not optional.  A
    first version of this omitted it and returned `UNKNOWN` for nearly every column:
    `September 30, 2025` matches no pattern in the cascade, and the legacy classifier only
    reached `duration` for it through the polluted header path the scoping removes.
    Dropping the prose without adding the structural evidence would have replaced one
    defect with another.

    `PERIOD_BINDING` provenance says plainly that the kind came from the binding rather
    than from a matched word, so the two are never confused.
    """
    patterns = {
        TemporalKind.COMPARISON: tag._COMPARISON_RE,
        TemporalKind.BUCKET: tag._BUCKET_RE,
        TemporalKind.SEGMENT: tag._SEGMENT_RE,
        TemporalKind.CATEGORY: tag._CATEGORY_RE,
    }
    for kind, pattern in patterns.items():
        match = token_safe(pattern).search(column_evidence)
        if match:
            return TemporalKindEvidence(kind=kind,
                                        method=TemporalKindMethod.COLUMN_HEADER_CELL,
                                        matched_text=match.group(0))
    for pattern, kind in ((tag._POINT_RE, TemporalKind.POINT),
                          (tag._DURATION_RE, TemporalKind.DURATION)):
        match = pattern.search(column_evidence)
        if match:
            return TemporalKindEvidence(kind=kind,
                                        method=TemporalKindMethod.COLUMN_HEADER_CELL,
                                        matched_text=match.group(0))

    # Structural evidence: the binding says when, so the shape follows from it.  A bare
    # year is a year; a date is an instant *unless the table says the period ends there*.
    #
    # `table_duration_phrase` is header evidence, not row prose: Microsoft's cash flow
    # statement puts `Year Ended June 30,` in one header cell and the year in another, so
    # a column-scoped reading sees only the year and would call a duration statement an
    # instant.  A first version did exactly that and broke the control it was meant to
    # preserve -- 13 duration columns became 6 point.
    if binding is not None and binding.is_usable:
        if binding.granularity is PeriodGranularity.YEAR:
            return TemporalKindEvidence(kind=TemporalKind.YEAR,
                                        method=TemporalKindMethod.PERIOD_BINDING,
                                        source_cells=binding.source_cells,
                                        matched_text=binding.normalized_period)
        if binding.granularity is PeriodGranularity.DAY:
            kind = (TemporalKind.DURATION if table_duration_phrase
                    else TemporalKind.POINT)
            return TemporalKindEvidence(kind=kind,
                                        method=TemporalKindMethod.PERIOD_BINDING,
                                        source_cells=binding.source_cells,
                                        matched_text=table_duration_phrase
                                        or binding.normalized_period)

    return TemporalKindEvidence(kind=TemporalKind.UNKNOWN,
                                method=TemporalKindMethod.COLUMN_HEADER_CELL)


def column_evidence(nf, grid, union_idx: list[int]) -> dict[int, tuple[str, tuple]]:
    """Per column, the raw header cells' text and their addresses -- nothing else."""
    width = max((len(r) for r in grid), default=0)
    out: dict[int, tuple[str, tuple]] = {}
    for col in range(width):
        texts, cells = [], []
        for i in union_idx:
            if i < len(grid) and col < len(grid[i]) and grid[i][col]:
                text = nf.ws(grid[i][col]["raw_text"])
                if text:
                    texts.append(text)
                    cells.append((i, text))
        out[col] = (" ".join(texts), tuple(cells))
    return out


def header_duration_phrase(nf, grid, union_idx: list[int]) -> str | None:
    """A duration phrase anywhere in the header block -- header cells, not row prose."""
    pattern = re.compile(r"\byears?\s+ended\b|\byear\s+ended\b", re.I)
    for i in union_idx:
        if i >= len(grid):
            continue
        seen: set[int] = set()
        for cell in grid[i]:
            if not cell or id(cell) in seen:
                continue
            seen.add(id(cell))
            match = pattern.search(nf.ws(cell["raw_text"]))
            if match:
                return match.group(0)
    return None


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
    sspec = importlib.util.spec_from_file_location("pb_shadow", SHADOW)
    pb = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(pb)

    from src.pdf_retrieval_v4 import temporal_axis_graph as tag
    from lxml import etree, html

    report = {"phase": "P1.6-A3-W3", "mutation": "none", "shadow_only": True,
              "tables": {}}
    print("=== TemporalKind shadow (column-local, token-safe) ===")
    print()

    for document_id, order, family in pb.ORACLES + TEMPORAL_ORACLES:
        ticker, accession = builder.DOCUMENTS[document_id]
        raw_path = (Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")
                    / ticker / accession / "primary.html")
        root = html.parse(str(raw_path), etree.HTMLParser(
            recover=True, no_network=True, huge_tree=True,
            remove_comments=True)).getroot()
        blocks, lookup, _prior = nf.make_blocks(
            root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})
        block = next(b for b in blocks
                     if b["block_type"] == "TABLE" and b["source_order"] == order)
        grid = nf.grid_rows(nf.direct_rows(lookup[block["table_id"]]))
        union, _added = pb.extended_header_idx(nf, grid)
        evidence = column_evidence(nf, grid, union)
        bound = pb.bind_table(nf, grid, document_id, block["table_id"])
        duration_phrase = header_duration_phrase(nf, grid, union)

        columns = {}
        for col, (text, cells) in evidence.items():
            kind = classify(text, tag, bound["columns"].get(col), duration_phrase)
            columns[col] = {
                "kind": kind.kind.value,
                "trigger": kind.matched_text,
                "method": kind.method.value,
                "source_header_cells": [
                    SourceCell(document_id, block["table_id"], r, col, t).to_dict()
                    for r, t in cells],
                "column_evidence": text[:120],
            }

        key = f"{document_id}#{order}"
        report["tables"][key] = {"family": family, "columns": columns}
        tally: dict[str, int] = {}
        for entry in columns.values():
            tally[entry["kind"]] = tally.get(entry["kind"], 0) + 1
        print(f"--- {key}  {family}   {tally}")
        for col in sorted(columns)[:6]:
            entry = columns[col]
            if entry["kind"] == "unknown":
                continue
            print(f"    col {col:<3} {entry['kind']:<11} trigger={entry['trigger']!r:<22} "
                  f"cells={[c['row'] for c in entry['source_header_cells']]}"
                  f"  evidence={entry['column_evidence'][:44]!r}")
        print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "temporal-kind-shadow.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'temporal-kind-shadow.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
