"""Build a source-grounded Structural Context Sidecar over the whole corpus.

The store records which cells a row holds and not which row it is: four rows
labelled `Cost of revenue` and two labelled `Operating income` are one
coordinate, and nothing in the record says which is the consolidated statement's
and which a segment note's.  The source HTML does say, in the table's own
structure.

This derives that structure from the corpus and writes it beside the store,
never into it.  Nothing frozen is touched: no candidate key, no store row, no
gold, no binder or validator semantics.  The sidecar is a separate artifact and
is off by default.

**Nothing here reads a question, a gold label or a case id.**  The unit of work
is a `(document, table row)` pair and the inputs are the store's own records and
the filing they came from.  A fact is placed by matching its label and its value
against the source rows; when more than one source row matches, the fact is
recorded as `AMBIGUOUS` with its candidates rather than assigned one, because
choosing is the thing this must not do.

  python build_structural_sidecar.py --v2-fact-store <store.jsonl> \\
      --html-root <raw_sec_html> --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import html as html_module
import json
import re
import unicodedata
from pathlib import Path

_TABLE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
_ROW = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
_CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAGS = re.compile(r"<[^>]+>")
_HEADING = re.compile(
    r"(CONSOLIDATED\s+STATEMENTS?[^\n]{0,70}"
    r"|CONSOLIDATED\s+BALANCE\s+SHEET[^\n]{0,50}"
    r"|SEGMENT[^\n]{0,50}"
    r"|NOTES?\s+TO\s+CONSOLIDATED[^\n]{0,50})",
    re.IGNORECASE,
)


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("$", " ").replace(",", " ").replace("—", " ").replace("–", " ")
    text = re.sub(r"[^\w\s.%-]+", " ", text)
    return " ".join(text.split())


def _money(value: object) -> str | None:
    """The digits of a stated amount, so `$ 13,426` and `13,426` agree."""

    text = _norm(value).replace(" ", "")
    digits = re.sub(r"[^\d.]", "", text)
    return digits or None


def _cell_texts(row_html: str) -> list[str]:
    return [
        " ".join(html_module.unescape(_TAGS.sub(" ", cell)).split())
        for cell in _CELL.findall(row_html)
    ]


def _tables(document: Path) -> list[list[list[str]]]:
    raw = document.read_text(encoding="utf-8", errors="replace")
    tables: list[list[list[str]]] = []
    for table_html in _TABLE.findall(raw):
        rows = [_cell_texts(row) for row in _ROW.findall(table_html)]
        rows = [row for row in rows if any(cell for cell in row)]
        if rows:
            tables.append(rows)
    return tables


def _table_identity(rows: list[list[str]]) -> str:
    payload = "\n".join("|".join(cell for cell in row) for row in rows)
    return "table:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _table_role(rows: list[list[str]], preceding: str) -> str:
    """What kind of table this is, from its own header row and its caption."""

    header = " | ".join(rows[0])[:200]
    haystack = (_norm(preceding[-1500:]) + " " + _norm(header))
    if re.search(r"consolidated statement", haystack):
        return "CONSOLIDATED_STATEMENT"
    if re.search(r"consolidated balance sheet", haystack):
        return "CONSOLIDATED_BALANCE_SHEET"
    if re.search(r"\bsegment", haystack):
        return "SEGMENT_DISCLOSURE"
    if re.search(r"notes? to consolidated", haystack):
        return "NOTE"
    return "UNKNOWN"


def _header_row(rows: list[list[str]]) -> list[str]:
    """The row that looks like the column header: most non-empty short cells."""

    best: list[str] = []
    for row in rows[:4]:
        filled = [cell for cell in row if cell.strip()]
        if len(filled) > len(best):
            best = filled
    return best[:12]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-fact-store", type=Path, required=True)
    parser.add_argument("--html-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    facts = [json.loads(line)
             for line in args.v2_fact_store.read_text(encoding="utf-8").splitlines()
             if line.strip()]

    # -- document -> filing, deterministically -----------------------------
    #
    # The corpus keeps every filing for a ticker and the metadata carries no form
    # type, so the annual report has to be identified from what it is.  Two
    # source-grounded facts do it: a fiscal year's annual report is the filing
    # whose period-end *year* is that fiscal year, and it is the largest document
    # in that year -- a 10-K against three 10-Qs.  Both are read from the
    # metadata; neither consults a question or a label.
    filings: dict[str, Path | None] = {}
    filing_evidence: dict[str, str] = {}
    for document_name in sorted({str(f.get("document_name")) for f in facts}):
        ticker = document_name.split("_")[0].upper()
        year = re.search(r"(\d{4})", document_name)
        directory = args.html_root / ticker
        chosen: Path | None = None
        if directory.is_dir() and year:
            fiscal = year.group(1)
            scored: list[tuple[int, Path, str]] = []
            for candidate in sorted(directory.iterdir()):
                metadata = candidate / "source-metadata.json"
                if not metadata.is_file():
                    continue
                payload = json.loads(metadata.read_text(encoding="utf-8"))
                hits = (payload.get("identity_signals") or {}).get("period_hits") or []
                end = str(hits[0]) if hits else ""
                if not end.startswith(fiscal):
                    continue
                scored.append((int(payload.get("raw_bytes") or 0),
                               candidate / "primary.html", end))
            if scored:
                scored.sort(key=lambda item: (-item[0], str(item[1])))
                chosen = scored[0][1]
                filing_evidence[document_name] = (
                    "period_end=%s bytes=%d of %d candidates"
                    % (scored[0][2], scored[0][0], len(scored)))
        filings[document_name] = chosen

    print("=" * 78)
    print("STRUCTURAL CONTEXT SIDECAR")
    print("=" * 78)
    print("  facts             %d" % len(facts))
    for name, path in sorted(filings.items()):
        print("    %-14s -> %s" % (name, path.name if path else "UNRESOLVED"))
    print()

    # -- parse each filing once, index its rows ----------------------------
    row_index: dict[str, list[tuple[str, str, str, list[str]]]] = {}
    table_meta: dict[str, dict] = {}
    for document_name, path in filings.items():
        if path is None:
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        offset = 0
        for table_html in _TABLE.findall(raw):
            position = raw.find(table_html, offset)
            offset = max(offset, position + 1)
            preceding = _TAGS.sub(" ", raw[max(0, position - 4000):position])
            rows = [_cell_texts(row) for row in _ROW.findall(table_html)]
            rows = [row for row in rows if any(cell for cell in row)]
            if not rows:
                continue
            identity = _table_identity(rows)
            table_meta.setdefault(identity, {
                "logical_table_id": identity,
                "table_role": _table_role(rows, preceding),
                "column_semantics": _header_row(rows),
                "row_count": len(rows),
            })
            for row_number, row in enumerate(rows):
                label = next((cell for cell in row if cell.strip()), "")
                key = _norm(label)
                if not key:
                    continue
                row_index.setdefault(key, []).append(
                    (document_name, identity, label, row))

    print("  tables parsed     %d" % len(table_meta))
    print("  distinct labels   %d" % len(row_index))
    print()

    # -- place each fact ---------------------------------------------------
    sidecar: list[dict] = []
    tally: collections.Counter = collections.Counter()
    for fact in facts:
        document_name = str(fact.get("document_name"))
        key = _norm(fact.get("metric"))
        amount = _money(fact.get("value"))
        candidates = []
        for owner, identity, label, row in row_index.get(key, ()):
            if owner != document_name:
                continue
            if amount and amount not in {_money(cell) for cell in row}:
                continue
            candidates.append(identity)
        candidates = sorted(set(candidates))
        if len(candidates) == 1:
            status, chosen = "RESOLVED", candidates[0]
            tally["RESOLVED"] += 1
        elif not candidates:
            status, chosen = "UNRESOLVED", None
            tally["UNRESOLVED"] += 1
        else:
            status, chosen = "AMBIGUOUS", None
            tally["AMBIGUOUS"] += 1
        record = {
            "candidate_key": str(fact.get("candidate_key") or ""),
            "document_name": document_name,
            "status": status,
            "logical_table_id": chosen,
            "candidate_table_ids": candidates[:6],
        }
        if chosen:
            meta = table_meta[chosen]
            record.update({
                "table_role": meta["table_role"],
                "column_semantics": meta["column_semantics"],
            })
        sidecar.append(record)

    # -- audit -------------------------------------------------------------
    roles = collections.Counter(
        r.get("table_role", "NONE") for r in sidecar)
    audit = {
        "facts": len(facts),
        "resolved": tally["RESOLVED"],
        "unresolved": tally["UNRESOLVED"],
        "ambiguous": tally["AMBIGUOUS"],
        "resolution_rate": round(tally["RESOLVED"] / max(1, len(facts)), 4),
        "tables": len(table_meta),
        "by_table_role": dict(roles),
        "filings_resolved": sum(1 for p in filings.values() if p),
        "filings_total": len(filings),
        "filing_evidence": filing_evidence,
        "benchmark_specific_branches": 0,
    }
    audit["rebuild_hash"] = "sha256:" + hashlib.sha256(
        json.dumps(sidecar, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "structural-context-sidecar.jsonl").open(
            "w", encoding="utf-8", newline="\n") as handle:
        for record in sidecar:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    (args.out / "structural-context-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")

    for key, value in audit.items():
        print("  %-30s %s" % (key, value))
    print()
    print("  written to %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
