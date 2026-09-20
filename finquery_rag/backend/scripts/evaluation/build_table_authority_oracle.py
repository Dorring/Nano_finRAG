"""P1.6-A2B-19A: an independent oracle for whether a table may speak for the company.

The question this answers is **not** "what is this table about" but "does this
table have the standing to be the company-level source for a fact".  Those are two
properties, and `statement_type` has been carrying both:

    statement_type   INCOME_STATEMENT / BALANCE_SHEET / CASH_FLOW / NOTES / ...
                     what the table is about
    table_role       PRIMARY_FINANCIAL_STATEMENT / NON_PRIMARY / UNRESOLVED
                     whether it may speak for the company

No classifier change, no resolver change, no fixture change.  This builds the
oracle the classifier will later be *measured against*, so it must not be the
classifier: it reads evidence the classifier does not read, and it records every
piece of that evidence so a reader can go to the filing and disagree with a row.

**Why the classifier cannot see what the oracle sees.**  `section_type` reaches a
table by a running heading (`make_blocks`) or by `section(caption + prior +
headers)` (`parse_table`).  `make_blocks` emits blocks only for
`table` / `h1`-`h6` / `p` / `li` / `dt` / `dd`, so a statement title rendered as
`<div><span>CONSOLIDATED STATEMENTS OF OPERATIONS</span></div>` -- which is how
Apple, Coca-Cola, Visa, NVIDIA, Pfizer and Tesla all render theirs -- produces no
block at all and the table's own body is never consulted either.  Seven of the
eight filings therefore label nothing: only Microsoft does, and only because it
alone sets `id="income_statements"` on its title `<p>`, which `make_blocks` reads.

**The evidence the oracle uses.**

  * the standalone lines above the table -- the statement title, the registrant
    name, the units caption.  A standalone line is an element whose whole text is
    its own text node, which is what a title is and a sentence is not;
  * the iXBRL contexts the table's facts point at.  A fact in an **undimensioned**
    context states a company-level amount; a fact in a dimensioned context states
    a disaggregation.  That is the filing declaring its own scope, and it is the
    one signal here that is not a reading of prose;
  * the table's own body, its row labels and its column headers.

**Two decisions are made mechanically**, and only two, because only these two are
the source declaring the answer rather than this script opining:

    TITLE_STATEMENT_OF_RECORD     a statement title stands immediately above the
                                  table and the table carries company-level facts
    DIMENSIONED_CONTEXT_ONLY      every fact in the table is a disaggregation

Everything else is left `UNRESOLVED` and is adjudicated by reading the evidence
this script prints.  That split is reported, so the oracle's own coverage is
visible and no one quotes its decided rows as if they were the whole corpus.

  python build_table_authority_oracle.py --out <dir> [--documents a,b]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

CORPUS = Path("/disk/qh/nano-finrag/data/financial_corpus_v2")

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

#: The three statement families the resolver admits.  Kept here so the oracle can
#: say which family a primary table belongs to without that being the decision.
PRIMARY_FAMILIES = ("INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW")

#: A line that names a statement of record and nothing else.
#:
#: Whole-string, not a search: the filing says `Consolidated Statements of
#: Operations` in prose and in its own index, and a search would take both.  The
#: optional trailing caption is allowed because some filers set the units on the
#: same line as the title.
STATEMENT_TITLE = re.compile(
    r"""^\s*
    (?:consolidated\s+|combined\s+|condensed\s+)*
    (?:
        statements?\s+of\s+
        (?:
            operations | income | earnings | loss
          | cash\s*flows?
          | comprehensive\s+(?:income|loss)
          | financial\s+(?:position|condition)
          | changes\s+in\s+(?:(?:stockholders|shareholders|shareowners)['’]?\s+equity|equity)
          | (?:stockholders|shareholders|shareowners)['’]?\s+equity
          | redeemable\s+noncontrolling\s+interests\s+and\s+equity
          | equity
        )
      | balance\s+sheets?
      | income\s+statements?
      | cash\s*flows?\s+statements?
      | comprehensive\s+(?:income|loss)\s+statements?
      | (?:stockholders|shareholders|shareowners)['’]?\s+equity\s+statements?
    )
    (?:\s*[—–-]\s*\(?continued\)?)?
    (?:\s*\((?:in|except|amounts\s+in|dollars\s+in)[^)]*\))?
    \s*$
    """,
    re.I | re.X,
)

#: The registrant naming itself above its own statement.
COMPANY_LINE = re.compile(
    r"^(?:the\s+)?[A-Za-z][\w&.,'’\- ]{2,70}"
    r"(?:inc\.?|incorporated|corporation|company|companies|corp\.?|plc|ltd\.?|"
    r"group|&\s*co\.?|co\.|and\s+subsidiar(?:y|ies)(?:\s+companies)?)$",
    re.I,
)

#: The units caption, which sits between the title and the table.
UNITS_LINE = re.compile(
    r"^\(?\s*(?:in|except|amounts\s+in|dollars\s+in|millions|thousands|billions)"
    r"\b[^)]{0,80}\)?$",
    re.I,
)

#: What a line looks like when it introduces something that is *not* the
#: statement of record.  Deliberately over-broad: this selects the safety set that
#: must never be promoted, so a false inclusion costs a look and a false exclusion
#: costs the guarantee.
DANGER_LINE = re.compile(
    r"segment|geograph|by\s+product|product\s+and\s+service|disaggregat"
    r"|debt|borrow|maturit|derivative|hedg|fair\s+value|lease|pension"
    r"|share-based|stock-based|compensation|income\s+tax|tax\s+reconcil"
    r"|selected\s+quarter|quarterly\s+financial\s+data|earnings\s+per\s+share"
    r"|per\s+share|reconcil|supplement|subsequent\s+event|goodwill"
    r"|accumulated\s+other\s+comprehensive|allowance|concentration",
    re.I,
)

_NOTE_TITLE = re.compile(
    r"^\s*(?:note\s+\d+|notes?\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial)"
    r"\b",
    re.I,
)

_PAGE_NUMBER = re.compile(r"^[\d\s|.\-–—]+$")
_SECTION_HEADER = re.compile(r"^\s*(?:item\s+\d|part\s+[ivx]+\b)", re.I)


def _is_layout_line(line: str) -> bool:
    """A line that says nothing about what a table is.

    The units caption, the registrant's own name, a page number and an `Item 8`
    section header all sit above tables, and all of them contain words that the
    danger vocabulary would otherwise read as evidence -- `(In millions, except
    per share data)` contains `per share`, and `Item 8. Financial Statements and
    Supplementary Data` contains `supplement`.  Reading those as risk signals
    would fill the safety set with the primary statements themselves.
    """
    return bool(
        UNITS_LINE.match(line)
        or COMPANY_LINE.match(line)
        or _PAGE_NUMBER.match(line)
        or _SECTION_HEADER.match(line)
    )


def _lname(node) -> str:
    tag = node.tag
    if not isinstance(tag, str):
        return ""
    if ":" in tag:
        return tag.rsplit(":", 1)[-1].lower()
    return tag.rsplit("}", 1)[-1].lower()


def _ws(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def _hidden(node) -> bool:
    """The same visibility rule the parser applies, restated so that the oracle
    does not import the layer it is auditing."""
    walk = node
    while walk is not None:
        style = _ws(walk.get("style")).lower().replace(" ", "")
        if "display:none" in style or "visibility:hidden" in style:
            return True
        if _lname(walk) in {"script", "style", "noscript", "template"}:
            return True
        walk = walk.getparent()
    return False


def _text(node) -> str:
    parts = []
    for child in node.iter():
        if _hidden(child):
            continue
        if child.text:
            parts.append(child.text)
        parent = child.getparent()
        if child.tail and parent is not None and not _hidden(parent):
            parts.append(child.tail)
    return _ws(" ".join(parts))


def _join_fragments(lines: list[str], join: str) -> str:
    return _ws(join.join(lines))


def _title_run(window: list[str]) -> tuple[str, int, int] | None:
    """The nearest statement title in the lines above a table.

    A Workiva filing splits its titles into adjacent `<span>`s, so Microsoft's
    income statement is two lines reading `INC` and `OME STATEMENTS`, and
    `BALANCE` / `SHEETS` is two more.  Contiguous runs of up to three lines are
    therefore tried joined both with and without a space, and the run that matches
    is returned with its span, so the reader can see exactly which lines were read.

    Returns `(matched_text, distance_of_nearest_line, span)` or None.
    """
    recent = list(reversed(window))  # nearest first, so `distance` indexes it
    best: tuple[str, int, int] | None = None
    for distance in range(1, len(recent) + 1):
        for span in range(1, 4):
            start = distance - span
            if start < 0:
                continue
            run = list(reversed(recent[start:distance]))  # back to reading order
            for join in (" ", ""):
                candidate = _join_fragments(run, join)
                if candidate and STATEMENT_TITLE.match(candidate):
                    if best is None or distance < best[1]:
                        best = (candidate, distance, span)
    return best


def _family(title: str) -> str:
    low = title.lower()
    if "cash" in low and "flow" in low:
        return "CASH_FLOW"
    if "balance sheet" in low or "financial position" in low or "financial condition" in low:
        return "BALANCE_SHEET"
    if "comprehensive" in low:
        return "COMPREHENSIVE_INCOME"
    if "equity" in low or "noncontrolling" in low:
        return "EQUITY"
    return "INCOME_STATEMENT"


def _row_labels(table, limit: int = 14) -> list[str]:
    out: list[str] = []
    for row in table.iter():
        if _lname(row) != "tr" or _hidden(row):
            continue
        cells = [c for c in row if _lname(c) in ("td", "th") and not _hidden(c)]
        if not cells:
            continue
        label = _ws(" ".join(cells[0].itertext()))
        if label and not re.fullmatch(r"[\d.,()%$\s—–-]*", label):
            out.append(label[:80])
        if len(out) >= limit:
            break
    return out


_PURE_NUMBER = re.compile(r"^\(?\$?\s*-?[\d,]+(?:\.\d+)?\s*\)?%?$")


def _has_figures(table, minimum: int = 2) -> bool:
    """Whether any row carries a row of numbers.

    Distinguishes a two-row page-layout table (a footer, a checkbox line, an empty
    spacer) from a small table of figures whose values simply were not tagged.
    The cell must *be* a number rather than merely contain a digit -- a page footer
    reading `Pfizer Inc. 2024 Form 10-K 55` has digits in it and no figures.
    """
    for row in table.iter():
        if _lname(row) != "tr" or _hidden(row):
            continue
        cells = [c for c in row if _lname(c) in ("td", "th") and not _hidden(c)]
        numbers = sum(1 for cell in cells
                      if _PURE_NUMBER.match(_ws(" ".join(cell.itertext()))))
        if numbers >= minimum:
            return True
    return False


def _column_headers(table, limit: int = 14) -> list[str]:
    for row in table.iter():
        if _lname(row) != "tr" or _hidden(row):
            continue
        cells = [_ws(" ".join(c.itertext())) for c in row
                 if _lname(c) in ("td", "th") and not _hidden(c)]
        text = [c for c in cells if re.search(r"\d{4}|year|ended|as of", c, re.I)]
        if len(text) >= 2:
            return [c[:48] for c in cells[:limit]]
    return []


def scan(document_id: str, ticker: str, accession: str) -> dict:
    from lxml import etree, html

    raw = CORPUS / "raw/SEC" / ticker / accession / "primary.html"
    root = html.parse(
        str(raw),
        etree.HTMLParser(recover=True, no_network=True, huge_tree=True,
                         remove_comments=True),
    ).getroot()

    # context id -> axis of its first dimension member, or None when the context
    # is undimensioned.  None is the filing saying "the whole company".
    axis_of: dict[str, str | None] = {}
    for node in root.iter():
        if _lname(node) != "context":
            continue
        members = [m for m in node.iter() if _lname(m) == "explicitmember"]
        axis_of[_ws(node.get("id"))] = (
            _ws(members[0].get("dimension")) if members else None
        )

    # One pass, holding every proxy: lxml frees element proxies and reuses `id()`,
    # so a second traversal cannot look anything up by `id()`.
    sequence = list(root.iter())
    tags = [_lname(n) for n in sequence]

    standalone: list[tuple[int, str]] = []
    for index, node in enumerate(sequence):
        if not isinstance(node.tag, str) or _hidden(node):
            continue
        own = _ws(node.text)
        if not (3 <= len(own) <= 140) or _text(node) != own:
            continue
        if any(_lname(a) == "table" for a in node.iterancestors()):
            continue
        standalone.append((index, own))

    tables: list[dict] = []
    order = 0
    for index, node in enumerate(sequence):
        if tags[index] != "table" or _hidden(node):
            continue
        if any(_lname(a) == "table" for a in node.iterancestors()):
            continue  # a layout table nested in a table is not a statement

        order += 1
        facts = [f for f in node.iter()
                 if _lname(f) == "nonfraction" and not _hidden(f)
                 and str(f.get("{http://www.w3.org/2001/XMLSchema-instance}nil") or ""
                         ).lower() != "true"]
        axes = collections.Counter()
        undim = 0
        concepts = collections.Counter()
        for fact in facts:
            axis = axis_of.get(_ws(fact.get("contextref")))
            if axis is None:
                undim += 1
            else:
                axes[axis] += 1
            concepts[_ws(fact.get("name"))] += 1

        above = [line for position, line in standalone if position < index]
        window = above[-6:]
        title = _title_run(window)
        nearest = above[-1] if above else None
        labels = _row_labels(node)
        # What the filing actually calls this table: the nearest line above it that
        # is not a units caption, the registrant's name, a page number or an
        # `Item 8` header.  For a statement that line is its title; for a note it is
        # the note's heading, which is what the safety set is built from.
        label = next((line for line in reversed(window) if not _is_layout_line(line)),
                     None)

        tables.append({
            "document_id": document_id,
            "table_ordinal": order,
            "source_order": index,
            # Identity the comparison joins on, because reproducing the parser's
            # `table_id` would mean reproducing its text extraction too.
            "oracle_key": f"{document_id}#{index}",
            "facts": len(facts),
            "undimensioned_facts": undim,
            "axes": dict(axes.most_common(6)),
            "concepts": [c for c, _ in concepts.most_common(8)],
            "nearest_line": (nearest or "")[:140],
            "lines_above": window,
            "title": title[0] if title else None,
            "title_distance": title[1] if title else None,
            "title_span": title[2] if title else None,
            "company_line": next(
                (line for line in reversed(window) if COMPANY_LINE.match(line)), None),
            "units_line": next(
                (line for line in reversed(window) if UNITS_LINE.match(line)), None),
            "note_line": next(
                (line for line in reversed(window) if _NOTE_TITLE.match(line)), None),
            "label_line": label,
            # The safety set: a table the filing itself captions with a
            # disaggregation, a note subject or a summary.  Over-broad on purpose --
            # a false inclusion costs one look, a false exclusion costs the
            # guarantee that this set is never promoted.
            "danger_line": label if label and DANGER_LINE.search(label) else None,
            "row_labels": _row_labels(node),
            "column_headers": _column_headers(node),
            # The table's own leading caption, if it has one.  A statement of
            # record opens with a units caption and then its line items; a note
            # opens by naming its own subject.
            "own_caption": (labels[0] if labels else None),
            "rows": sum(1 for r in node.iter()
                        if _lname(r) == "tr" and not _hidden(r)),
            "has_figures": _has_figures(node),
            "body_head": _text(node)[:220],
        })

    return {"document_id": document_id, "ticker": ticker, "accession": accession,
            "tables": tables}


def decide(table: dict) -> dict:
    """Label one table, and say what the label rests on.

    Two mechanical decisions only.  A title above the table answers what the table
    *is*; an all-dimensioned fact set answers what it *is not*.  When neither
    answers, the table is left for adjudication rather than guessed at -- the
    oracle's value is that its decided rows are the source's own statement, and a
    third guess-derived bucket would spend that.
    """
    title = table["title"]
    distance = table["title_distance"]
    undim = table["undimensioned_facts"]
    facts = table["facts"]

    if facts == 0 and table["rows"] <= 2 and not table["has_figures"]:
        # A table with no tagged fact whose whole content is one or two rows with
        # no row of figures.  Workiva lays a page out as a stack of these -- 592 of
        # the 1498 tables here, carrying page numbers, cover-page checkboxes and
        # empty spacers -- and a page footer states nothing, tagged or not.
        return {"label": "NON_PRIMARY", "reason": "LAYOUT_SCAFFOLDING",
                "detail": f"a {table['rows']}-row table with no iXBRL fact and no row "
                          f"of figures: page layout, not a table of figures",
                "family": None}

    if facts >= 1 and undim == 0:
        # Decided before any title is read, because it is the filing stating the
        # scope of its own facts rather than this script reading a heading.  A
        # table whose every tagged fact sits in a dimensioned context reports
        # disaggregations and no company total, whatever it is called.
        return {"label": "NON_PRIMARY", "reason": "DIMENSIONED_CONTEXT_ONLY",
                "detail": f"all {facts} tagged fact(s) sit in a dimensioned context "
                          f"({', '.join(list(table['axes'])[:3])})",
                "family": None}

    own_caption = table.get("own_caption")
    if (
        own_caption
        and len(own_caption) <= 80
        and not re.search(r"\d", own_caption)
        and not own_caption.endswith(":")
        and not UNITS_LINE.match(own_caption)
        and DANGER_LINE.search(own_caption)
        and not STATEMENT_TITLE.match(own_caption)
    ):
        # The table opens by naming its own subject, and that subject is a
        # disaggregation.  Pfizer's `Supplemental Cash Flow Information` sits two
        # tables below the cash flow statement and inherits its title from the
        # page above; what separates them is that the statement opens with
        # `(MILLIONS)` and this one opens by saying what it is about.  A title
        # outside a table cannot outrank a caption inside it.
        return {"label": "NON_PRIMARY", "reason": "OWN_CAPTION_DISAGGREGATION",
                "detail": f"the table's own first row reads {own_caption!r}, which "
                          f"names a disaggregation rather than the statement",
                "family": None}

    if title and distance is not None and distance <= 3:
        if undim >= 1:
            return {"label": "PRIMARY_FINANCIAL_STATEMENT",
                    "reason": "TITLE_STATEMENT_OF_RECORD",
                    "detail": f"{title!r} stands {distance} line(s) above the table "
                              f"and the table carries {undim} company-level fact(s)",
                    "family": _family(title)}
        return {"label": "UNRESOLVED", "reason": "TITLE_WITHOUT_COMPANY_LEVEL_FACTS",
                "detail": f"{title!r} stands {distance} line(s) above the table but "
                          f"none of its {facts} fact(s) is company-level",
                "family": _family(title)}

    if facts == 0:
        return {"label": "UNRESOLVED", "reason": "NO_TAGGED_FACTS",
                "detail": f"{table['rows']} row(s) and no iXBRL fact, so the filing "
                          f"states nothing about this table's scope",
                "family": None}

    return {"label": "UNRESOLVED", "reason": "COMPANY_LEVEL_FACTS_WITHOUT_TITLE",
            "detail": f"{undim} of {facts} fact(s) are company-level but no "
                      f"statement title stands within three lines",
            "family": None}


def stratum_of(table: dict, verdict: dict) -> list[str]:
    strata = []
    if table["title"] and table["title_distance"] and table["title_distance"] <= 3:
        strata.append("TITLE_BEARING")
    if verdict["reason"] == "COMPANY_LEVEL_FACTS_WITHOUT_TITLE":
        strata.append("COMPANY_FACTS_NO_TITLE")
    if table["danger_line"]:
        strata.append("DANGER_NEIGHBOUR")
    if table["note_line"]:
        strata.append("NOTE_NEIGHBOUR")
    return strata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--documents", default="")
    args = parser.parse_args(argv)

    wanted = [d for d in args.documents.split(",") if d] or sorted(DOCUMENTS)
    report = {"phase": "P1.6-A2B-19A", "mutation": "none", "documents": {}}
    totals = collections.Counter()
    reasons = collections.Counter()
    per_document = {}

    print("=== table authority oracle ===")
    print()
    for document_id in wanted:
        ticker, accession = DOCUMENTS[document_id]
        record = scan(document_id, ticker, accession)
        for table in record["tables"]:
            table["verdict"] = decide(table)
            table["strata"] = stratum_of(table, table["verdict"])

        labels = collections.Counter(t["verdict"]["label"] for t in record["tables"])
        counts = collections.Counter(t["verdict"]["reason"] for t in record["tables"])
        primary = [t for t in record["tables"]
                   if t["verdict"]["label"] == "PRIMARY_FINANCIAL_STATEMENT"]
        danger = [t for t in record["tables"] if "DANGER_NEIGHBOUR" in t["strata"]]
        danger_primary = [t for t in danger
                          if t["verdict"]["label"] == "PRIMARY_FINANCIAL_STATEMENT"]

        print(f"--- {document_id}  tables {len(record['tables'])}")
        print(f"    {dict(labels)}")
        for reason, count in counts.most_common():
            print(f"      {count:>4}  {reason}")
        print(f"    primary with a title: "
              f"{[(t['table_ordinal'], t['verdict']['family']) for t in primary]}")
        print(f"    danger-neighbour tables {len(danger)}, "
              f"of them called primary {len(danger_primary)}")
        for table in danger_primary:
            print(f"      !! {table['oracle_key']} {table['title']!r} "
                  f"danger={table['danger_line']!r}")
        print()

        report["documents"][document_id] = {
            "ticker": ticker, "accession": accession,
            "tables": record["tables"],
            "labels": dict(labels), "reasons": dict(counts),
        }
        totals.update(labels)
        reasons.update(counts)
        per_document[document_id] = {
            "tables": len(record["tables"]),
            "primary": labels["PRIMARY_FINANCIAL_STATEMENT"],
            "non_primary": labels["NON_PRIMARY"],
            "unresolved": labels["UNRESOLVED"],
            "danger": len(danger),
            "danger_primary": len(danger_primary),
        }

    report["totals"] = dict(totals)
    report["reasons"] = dict(reasons)
    report["per_document"] = per_document
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table-authority-oracle.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    digest_rows = []
    for document_id, record in report["documents"].items():
        digest_rows.append(f"\n# {document_id}\n")
        for table in record["tables"]:
            verdict = table["verdict"]
            if verdict["reason"] == "DIMENSIONED_CONTEXT_ONLY":
                continue
            digest_rows.append(
                f"## {table['oracle_key']}  ordinal {table['table_ordinal']}  "
                f"rows {table['rows']}  facts {table['facts']} "
                f"(company-level {table['undimensioned_facts']})\n"
                f"   verdict  {verdict['label']} / {verdict['reason']}\n"
                f"   detail   {verdict['detail']}\n"
                f"   title    {table['title']!r}  (distance {table['title_distance']}, "
                f"span {table['title_span']})\n"
                f"   label    {table['label_line']!r}   danger {table['danger_line']!r}\n"
                f"   nearest  {table['nearest_line']!r}\n"
                f"   axes     {table['axes']}\n"
                f"   labels   {table['row_labels'][:6]}\n"
                f"   headers  {table['column_headers'][:6]}\n"
                f"   body     {table['body_head'][:150]!r}\n\n")
    (args.out / "adjudication-digest.md").write_text(
        "".join(digest_rows), encoding="utf-8")

    print("=== totals ===")
    print(f"  {dict(totals)}")
    print()
    print("=== reasons ===")
    for reason, count in reasons.most_common():
        print(f"  {count:>5}  {reason}")
    print()
    print(f"  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
