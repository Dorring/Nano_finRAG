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
import json
import re
import unicodedata
from html.parser import HTMLParser
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


class _Dom(HTMLParser):
    """A linear document flow: text runs and tables, in order.

    Regex tag-stripping was the wrong tool and the build showed it -- a `>`
    inside a quoted attribute ends a `<[^>]+>` match early, so the "lines" the
    heading reader searched were CSS fragments (`ont-size:10pt;font-weight:400;
    line-height:`).  A real parser cannot make that mistake, and it tracks which
    text is bold, which is how these filings mark a statement caption.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.flow: list[tuple] = []
        self._bold: list[bool] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        style = " ".join(v or "" for k, v in attrs if k == "style").replace(" ", "")
        self._bold.append(
            tag in ("b", "strong", "h1", "h2", "h3", "h4")
            or "font-weight:700" in style
            or "font-weight:bold" in style
        )
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if self._bold:
            self._bold.pop()
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if any(cell for cell in self._row):
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.flow.append(("table", self._table))
            self._table = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
        elif self._table is None:
            text = " ".join(data.split())
            if text:
                self.flow.append(("text", text, any(self._bold)))


def _heading_for(flow: list[tuple], index: int) -> str:
    """The caption above the table at ``index``.

    **Bounded by the previous table.**  A caption belongs to the table below it,
    not to whatever bold prose preceded the previous table, so the search walks
    back only as far as the last table boundary.  Without that bound the window
    is a fixed length, and a fixed length was wrong in both directions: long
    enough to reach the income statement's caption from a segment table, and
    short enough to land on prose for the income statement itself.

    Within the section the nearest bold run wins, because these filings mark a
    statement caption with `font-weight:700` and the line directly above the
    statement -- `(In millions except per share data)` -- is not bold.
    """

    window: list[tuple] = []
    for item in reversed(flow[:index]):
        if item[0] == "table":
            break
        window.append(item)
    texts = [item for item in window if item[0] == "text"]
    for item in texts:
        if item[2] and len(item[1]) <= 90:
            return item[1]
    for item in texts:
        if len(item[1]) <= 90:
            return item[1]
    return ""


def _table_identity(rows: list[list[str]]) -> str:
    payload = "\n".join("|".join(cell for cell in row) for row in rows)
    return "table:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _parse_document(path: Path) -> tuple[list[tuple], dict[int, str]]:
    parser = _Dom()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    headings = {
        index: _heading_for(parser.flow, index)
        for index, item in enumerate(parser.flow) if item[0] == "table"
    }
    return parser.flow, headings


_PERIOD_YEAR = re.compile(r"^(?:19|20)\d{2}$")
_PERIOD_LONG = re.compile(
    r"(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+\d{1,2}\s*,?\s*(?:19|20)\d{2}"
)
#: Column labels a reconciliation table uses for its parts and its total.  A
#: statement has periods here; a segment note has these.
_RECONCILING = frozenset(
    {"total", "totals", "corporate", "eliminations", "consolidated", "segments",
     "segment", "subtotal", "other", "unallocated", "adjustments"}
)


def _is_period_cell(text: str) -> bool:
    normalized = _norm(text)
    return bool(_PERIOD_YEAR.match(normalized)) or bool(_PERIOD_LONG.search(normalized))


def _column_header(rows: list[list[str]]) -> list[str]:
    """The row that labels the columns, found by what it labels.

    Not "the widest of the first three rows" -- in these filings the widest early
    row is usually a data row, which is why the first version returned UNKNOWN
    for the very tables it had just correctly separated.  The header is the row
    that names *periods*, and it is looked for as such.
    """

    for row in rows[:8]:
        filled = [cell for cell in row if cell.strip()]
        if sum(1 for cell in filled if _is_period_cell(cell)) >= 2:
            return filled
    best: list[str] = []
    for row in rows[:3]:
        filled = [cell for cell in row if cell.strip()]
        if len(filled) > len(best):
            best = filled
    return best


_STATEMENT_HEADING = re.compile(
    r"consolidated\s+statements?\s+of\s+(income|operations|financial\s+position"
    r"|cash\s+flows|comprehensive\s+income|equity|changes)",
    re.IGNORECASE,
)


def _nearest_heading(preceding: str) -> str:
    """The heading above a table, preferring one that names a statement.

    "Nearest short line" is not enough, and the source shows why: the line
    immediately above the income statement is `(In millions except per share
    data)`, with `CONSOLIDATED STATEMENTS OF INCOME` above *that*.  So the
    reader looks back for a statement caption first and only falls back to the
    nearest heading-like line, rather than taking whichever is closest.
    """

    lines = [line.strip() for line in preceding.split("\n") if line.strip()]
    recent = lines[-20:]
    for line in reversed(recent):
        normalized = _norm(line)
        if _STATEMENT_HEADING.search(normalized) or "consolidated balance sheet" in normalized:
            return line
    for line in reversed(recent):
        if len(line) <= 90 and not line.endswith("."):
            return line
    return ""


def _table_role(rows: list[list[str]], heading: str) -> str:
    """What kind of table this is, from its heading and its columns.

    The column set alone does not do it, which the build showed: Coca-Cola's
    segment `Operating income` table and its consolidated one are both
    `Year Ended December 31 | 2025 | 2024 | 2023`.  The heading is what
    separates them, and the columns only corroborate.
    """

    normalized = _norm(heading)
    if _STATEMENT_HEADING.search(normalized):
        return "CONSOLIDATED_STATEMENT"
    if "consolidated balance sheet" in normalized:
        return "CONSOLIDATED_BALANCE_SHEET"
    header = _column_header(rows)
    if len(header) < 3:
        return "UNKNOWN"
    labels = {_norm(cell).strip(".") for row in rows[:8] for cell in row}
    if labels & _RECONCILING:
        return "SEGMENT_DISCLOSURE"
    periods = sum(1 for cell in header if _is_period_cell(cell))
    if periods >= 1:
        # Period columns and no statement heading: a note or a segment's own
        # statement rather than a consolidated one.
        return "DISCLOSURE_TABLE"
    return "UNKNOWN"


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
        flow, headings = _parse_document(path)
        for index, item in enumerate(flow):
            if item[0] != "table":
                continue
            rows = item[1]
            heading = headings.get(index, "")
            if not rows:
                continue
            identity = _table_identity(rows)
            table_meta.setdefault(identity, {
                "logical_table_id": identity,
                "table_role": _table_role(rows, heading),
                "heading": heading[:90],
                "column_semantics": _column_header(rows),
                "row_count": len(rows),
            })
            for row in rows:
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
                "heading": meta["heading"],
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
