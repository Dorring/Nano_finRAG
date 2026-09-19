"""P1.6-0D: settle the cross-entity gold against the original filings.

P1.6-0C classified the cross-entity gold from the fact store alone and left nine
cases `UNRESOLVED_GROUND_TRUTH`: neither the gold value nor its rival is
restated elsewhere in the filing, so document recurrence cannot separate them.
It found one `WRONG_SCOPE_GOLD` (`compare-009`) on store evidence, and this
settles all ten against the PDF.

**The filings flatten two different ways, and both produce the same defect.**

*Same row, different columns.*  `compare-009` asks which company had the higher
FY2025 net income; the gold is `$4,520` for JPMorganChase.  That row's header
reads `Corporate | Reconciling Items | Total`, and the row's cells sit at:

    x=188   4,520   -> columns 165-312   Corporate
    x=444  57,048   -> columns 430-562   Total

Same row, same page, adjacent columns -- and the coordinate the store files both
under, `(JPMorganChase, Net income, FY2025)`, keeps neither.  `$4,520` is the
Corporate segment; `$57,048` is the Firm.

*Different rows, a `Total` row among them.*  `compare-003`'s gold for Tesla is
`$13,292`, whose row label is `Gross profit total automotive & services and
other segment`, while `$3,802` is `Gross profit energy generation and storage
segment` and `$17,094` is `Total gross profit`.  Here the segments are rows, not
columns, so a resolver that only reads column headers finds nothing.

The row rule is positional, not textual: `Total` *leading* a label marks the
company figure, while the `total` inside `Gross profit total automotive ...`
modifies a grouping and the row is still a segment.  Matching the word anywhere
would score that segment row as the total.

**This reads; it does not rebuild.**  The filings are opened read-only.  Nothing
here re-parses the corpus, rebuilds the fact store, or regenerates gold --
re-mixing "the benchmark is wrong", "the extractor is wrong" and "the system
behaved wrongly" is the conflation this audit line exists to undo.  Fixtures are
corrected downstream, in one batch, once every case's evidence is frozen.

**Evidence, not assertion.**  Every decision records the page, the row tokens,
the label, the column, and the x-coordinates it rests on, so a second reader can
re-derive it from the filing without trusting this script.  Where neither rule
resolves, the slot is reported `SOURCE_STILL_AMBIGUOUS` rather than guessed --
a wrong column name is worse than an admitted gap, because it turns a real
ambiguity into a confident-looking answer.

  python audit_cross_entity_source_truth.py \
      --pdf-dir /disk/.../data/raw_pdfs \
      --corpus-manifest /disk/.../data/raw_pdfs/corpus-manifest.json \
      --gold-validity <P1.6-0C artifact> --out <dir> \
      [--decisions <reviewed decisions.json>]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_TOTAL_MARKERS = ("total", "consolidated", "firm")
_NUMBER = re.compile(r"\(?\$?\s*-?[\d][\d,\.]*\)?")


def _numeric_key(text: object) -> str | None:
    """The digits of a value, so `$ 4,520` and `4,520` compare equal."""

    match = _NUMBER.search(str(text or ""))
    if not match:
        return None
    digits = re.sub(r"[^\d]", "", match.group(0))
    return digits or None


def _document_index(pdf_dir: Path, manifest_path: Path) -> dict[str, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    for document in manifest.get("documents") or ():
        document_id = str(document.get("document_id"))
        index[document_id] = {
            "document_id": document_id,
            "filename": document.get("local_filename"),
            "path": pdf_dir / str(document.get("local_filename")),
            "sha256": document.get("sha256"),
            "page_count": document.get("page_count"),
            "source_url": document.get("source_url"),
        }
    return index


def _verify_corpus(index: dict[str, dict]) -> dict[str, dict]:
    """Hash and page-count every filing before any page is cited."""

    import fitz

    report: dict[str, dict] = {}
    for document_id, entry in index.items():
        path = Path(entry["path"])
        if not path.is_file():
            report[document_id] = {"status": "MISSING", "path": str(path)}
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with fitz.open(path) as document:
            pages = document.page_count
        report[document_id] = {
            "status": "OK",
            "filename": entry["filename"],
            "sha256": digest,
            "sha256_matches_manifest": digest == entry["sha256"],
            "page_count": pages,
            "page_count_matches_manifest": pages == entry["page_count"],
            "source_url": entry["source_url"],
        }
    return report


def _header_tables(page) -> list[list[dict]]:
    """Tables on the page whose first row reads as a header, not as data.

    `find_tables` reports a `header` for every table, including one-row tables
    whose only row is data.  Accepting those produced resolutions like
    `col=62,647` for a gold of `62,647` -- naming the value instead of its
    column, and turning a real ambiguity into a confident-looking answer.  A
    header cell counts only when it contains letters.
    """

    tables: list[list[dict]] = []
    for table in page.find_tables().tables:
        names = table.header.names or []
        cells = table.header.cells or []
        if len(names) != len(cells):
            continue
        columns = []
        for name, cell in zip(names, cells):
            text = " ".join(str(name).split())
            if not cell or not any(character.isalpha() for character in text):
                continue
            columns.append(
                {
                    "name": text,
                    "x0": round(float(cell[0]), 1),
                    "x1": round(float(cell[2]), 1),
                }
            )
        if len(columns) >= 2:
            tables.append(columns)
    return tables


def _words(page) -> list[dict]:
    return [
        {"text": text, "x0": round(x0, 1), "x1": round(x1, 1),
         "x_center": round((x0 + x1) / 2, 1), "y0": round(y0, 1)}
        for x0, y0, x1, _y1, text, *_ in page.get_text("words")
    ]


def _value_tokens(words: list[dict], value: object) -> list[dict]:
    target = _numeric_key(value)
    if target is None:
        return []
    return [word for word in words if _numeric_key(word["text"]) == target]


def _row_tokens(words: list[dict], y: float, tolerance: float = 5.0) -> list[dict]:
    return sorted(
        (word for word in words if abs(word["y0"] - y) <= tolerance),
        key=lambda word: word["x0"],
    )


def _label_for(words: list[dict], token: dict) -> tuple[str, str]:
    """The row's label, and whether the rule read it off this line or above.

    A label may sit on the same line as its values ("Net income  $4,520 ...") or
    on the lines above it ("Gross profit total / automotive & services / and
    other segment" then the numbers).  Both shapes are in this corpus, so the
    in-line text is preferred and the nearest lines above are the fallback.
    """

    in_line = [w for w in _row_tokens(words, token["y0"]) if w["x1"] < token["x0"]]
    text = " ".join(w["text"] for w in in_line).strip()
    if text:
        return text, "same_line"

    above = [w for w in words if w["y0"] < token["y0"] - 1 and w["y0"] > token["y0"] - 60]
    if not above:
        return "", "none"
    lines: dict[float, list[dict]] = collections.defaultdict(list)
    for word in above:
        lines[word["y0"]].append(word)
    ordered = [lines[y] for y in sorted(lines)][-4:]
    return " ".join(
        w["text"] for line in ordered for w in sorted(line, key=lambda w: w["x0"])
    ).strip(), "lines_above"


def _scope_from_label(label: str) -> str:
    """`Total` leading the label marks the company figure.

    Positional, not textual.  `Gross profit total automotive & services and
    other segment` contains `total` and is a *segment* row -- the word modifies
    the grouping.  `Total gross profit` is the company row.  Matching the word
    anywhere scores the segment as the total.
    """

    folded = label.casefold().strip()
    return "TOTAL" if folded.startswith(("total", "consolidated", "firm")) else "SEGMENT"


def _context_blocks(page, value: object, metric: object, limit: int = 6) -> list[str]:
    """The page's text blocks that state the value or name the metric.

    This is the evidence a reader adjudicates from.  A word-level band is not
    enough: `crossdiff-003`'s Apple gold `9,683` sits in a block reading
    `Total 9,683  2,491  5,801  State:`, and only the block *above* it --
    `Federal:` -- says that `Total` is the Federal sub-total rather than the
    company's tax provision.  Block text carries that context; a row of
    tokens does not.
    """

    wanted = _numeric_key(value)
    needle = " ".join(str(metric or "").split()[:3]).casefold()
    found: list[str] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        text = " ".join(
            span["text"] for line in block["lines"] for span in line["spans"]
        ).strip()
        if not text:
            continue
        if (wanted and _numeric_key(text) == wanted) or (
            needle and needle in text.casefold()
        ):
            found.append(" ".join(text.split()))
    return found[:limit]


def _resolve_in(document, value: object, first_page: int, window: int = 3) -> dict:
    """Where a value sits: its page, its row label, and its column if named."""

    page_count = document.page_count
    for page_index in range(max(0, first_page - window), min(page_count, first_page + 1)):
        page = document[page_index]
        words = _words(page)
        tokens = _value_tokens(words, value)
        if not tokens:
            continue
        tables = _header_tables(page)
        for token in tokens:
            label, label_from = _label_for(words, token)
            column = None
            for columns in tables:
                for candidate in columns:
                    if candidate["x0"] <= token["x_center"] <= candidate["x1"]:
                        column = candidate
                        break
                if column:
                    break
            label_scope = _scope_from_label(label)
            column_scope = _scope_of(column["name"]) if column else None
            # The label wins when it names the total: a `Total` row is the
            # company figure whatever its column is headed.
            scope = "TOTAL" if "TOTAL" in (label_scope, column_scope) else "SEGMENT"
            return {
                "found": True,
                "pdf_page": page_index + 1,
                "row_label": label,
                "row_label_from": label_from,
                "column_name": column["name"] if column else None,
                "value_x_center": token["x_center"],
                "label_scope": label_scope,
                "column_scope": column_scope,
                "scope": scope,
                "row_tokens": [w["text"] for w in _row_tokens(words, token["y0"])][:20],
                "header_names": [c["name"] for columns in tables for c in columns],
            }
    return {"found": False}


def _scope_of(column_name: str) -> str:
    folded = column_name.casefold()
    return "TOTAL" if any(m in folded for m in _TOTAL_MARKERS) else "SEGMENT"


def _classify(gold: dict, rivals: list[dict]) -> tuple[str, str]:
    """Settle the case, and refuse to when the evidence does not discriminate.

    A **column** headed `Total`/`Consolidated` is strong evidence: the header
    names the scope, and a value under it is the company figure.

    A row **label** leading with `Total` is weak.  `crossdiff-003`'s Apple gold
    `9,683` reads `Total 9,683  2,491  5,801  State:` and is the *Federal*
    sub-total of Apple's tax provision -- the `Total` in its own row comes from
    the jurisdiction grouping above it, and the company figure is
    `Provision for income taxes $20,719`.  An earlier version of this rule
    scored that `VALID_GOLD`, certifying a segment as the company total, which
    is the worst outcome this audit can produce.  So a weak-total gold is only
    accepted when no rival is also weak-total: if the label does not separate
    them, it has not answered the question.
    """

    if not gold.get("found"):
        return "SOURCE_STILL_AMBIGUOUS", "the gold value was not located in the filing"

    gold_by_column = gold.get("column_scope") == "TOTAL"
    gold_by_label = gold.get("label_scope") == "TOTAL"

    if gold_by_column:
        return "VALID_GOLD", (
            f"the gold sits under the column {gold['column_name']!r}, which names "
            f"the scope"
        )

    rival_columns = [
        r for r in rivals if r.get("found") and r.get("column_scope") == "TOTAL"
    ]
    if rival_columns:
        best = rival_columns[0]
        return "WRONG_SCOPE_GOLD", (
            f"the gold's scope is {gold.get('column_name') or gold.get('row_label')!r}; "
            f"rival {best.get('value')!r} sits under the column "
            f"{best['column_name']!r}, which names the company figure"
        )

    rival_totals = [
        r for r in rivals if r.get("found") and r.get("label_scope") == "TOTAL"
    ]
    if gold_by_label and rival_totals:
        return "SOURCE_STILL_AMBIGUOUS", (
            f"the gold's row label {gold['row_label']!r} leads with a total, and so "
            f"do {len(rival_totals)} rival(s) in the same table -- the label does not "
            f"say which total is the company figure"
        )
    if gold_by_label:
        return "VALID_GOLD", (
            f"the gold's row label {gold['row_label']!r} leads with a total and no "
            f"rival does"
        )
    if rival_totals:
        best = rival_totals[0]
        return "WRONG_SCOPE_GOLD", (
            f"the gold's row label {gold['row_label']!r} is not a total; rival "
            f"{best.get('value')!r} has label {best['row_label']!r}"
        )
    return "SOURCE_STILL_AMBIGUOUS", (
        "neither the gold nor any rival resolves to a company-figure row or column"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--gold-validity", type=Path, required=True)
    parser.add_argument("--fact-store", type=Path, default=None)
    parser.add_argument("--decisions", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import os

    import fitz

    from src.runtime.trusted_v2_production import StructuredFactStore

    store_path = args.fact_store or Path(os.environ["TRUSTED_V2_FACT_STORE_PATH"])
    StructuredFactStore(store_path)  # fail early if the store is unusable

    index = _document_index(args.pdf_dir, args.corpus_manifest)
    corpus = _verify_corpus(index)

    audit = json.loads(args.gold_validity.read_text(encoding="utf-8"))
    contested = [
        case
        for case in audit.get("cases") or ()
        if case.get("case_classification")
        in ("WRONG_SCOPE_GOLD", "UNRESOLVED_GROUND_TRUTH")
    ]

    reviewed = (
        json.loads(args.decisions.read_text(encoding="utf-8"))
        if args.decisions
        else {}
    )

    results: list[dict] = []
    for case in contested:
        case_id = str(case.get("case_id"))
        slots_out: list[dict] = []
        for slot in case.get("slots") or ():
            if slot.get("classification") == "VALID_GOLD":
                continue

            record = {
                "case_id": case_id,
                "document_id": slot.get("document"),
                "entity": slot.get("entity"),
                "metric": slot.get("metric"),
                "period": slot.get("period"),
                "page": slot.get("page"),
                "gold_value": slot.get("gold_value"),
            }
            entry = index.get(str(slot.get("document")))
            if entry is None or corpus.get(str(slot.get("document")), {}).get("status") != "OK":
                record.update(decision="SOURCE_STILL_AMBIGUOUS",
                              reason="document not verified in the corpus")
                slots_out.append(record)
                continue

            record["source_file"] = entry["filename"]
            record["source_sha256"] = entry["sha256"]
            first_page = int(slot.get("page") or 1) - 1

            with fitz.open(Path(entry["path"])) as document:
                gold = _resolve_in(document, slot.get("gold_value"), first_page)
                rivals = []
                for value in slot.get("comparable_distinct_values") or ():
                    if str(value) == str(slot.get("gold_value")):
                        continue
                    rivals.append({"value": value, **_resolve_in(document, value, first_page)})
                context = []
                if gold.get("pdf_page"):
                    page = document[gold["pdf_page"] - 1]
                    context = _context_blocks(page, slot.get("gold_value"), slot.get("metric"))

            decision, reason = _classify(gold, rivals)
            record.update(
                resolved_pdf_page=gold.get("pdf_page"),
                row_label=gold.get("row_label"),
                row_label_from=gold.get("row_label_from"),
                gold_column=gold.get("column_name"),
                gold_scope=gold.get("scope"),
                row_tokens=gold.get("row_tokens"),
                header_names=gold.get("header_names"),
                page_context_blocks=context,
                competing=[
                    {"value": r.get("value"), "found": r.get("found"),
                     "row_label": r.get("row_label"), "column": r.get("column_name"),
                     "scope": r.get("scope"), "pdf_page": r.get("pdf_page")}
                    for r in rivals
                ],
                decision=decision,
                reason=reason,
            )
            # A reviewed decision overrides the rules, and says so, so that a
            # reader can tell a machine resolution from a human one.
            override = reviewed.get(f"{case_id}::{slot.get('entity')}")
            if override:
                record["decision"] = override["decision"]
                record["reason"] = override["reason"]
                record["decided_by"] = "review"
            else:
                record["decided_by"] = "rule"
            slots_out.append(record)

        labels = {s.get("decision") for s in slots_out}
        if "WRONG_SCOPE_GOLD" in labels or "INVALID_GOLD" in labels:
            case_decision = "WRONG_SCOPE_GOLD"
        elif labels == {"VALID_GOLD"}:
            case_decision = "VALID_GOLD"
        else:
            case_decision = "SOURCE_STILL_AMBIGUOUS"
        results.append({"case_id": case_id, "decision": case_decision, "slots": slots_out})

    args.out.mkdir(parents=True, exist_ok=True)
    artifact = {
        "phase": "P1.6-0D",
        "purpose": "cross-entity gold settled against the original filings",
        "read_only": True,
        "fact_store": str(store_path),
        "pdf_dir": str(args.pdf_dir),
        "corpus_verification": corpus,
        "corpus_verified": all(v.get("status") == "OK" for v in corpus.values()),
        "case_verdicts": dict(collections.Counter(r["decision"] for r in results)),
        "cases": results,
    }
    (args.out / "cross-entity-source-truth.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== P1.6-0D cross-entity source truth ===")
    print(f"  corpus verified: {artifact['corpus_verified']} "
          f"({len(corpus)} filings)")
    print(f"  cases audited: {len(results)}   verdicts: {artifact['case_verdicts']}")
    print()
    for case in results:
        print(f"  {case['case_id']:28} {case['decision']}")
        for slot in case["slots"]:
            print(f"      {str(slot.get('entity'))[:18]:18} gold={str(slot.get('gold_value'))[:10]:10} "
                  f"page={slot.get('resolved_pdf_page')} [{slot.get('decided_by')}]")
            print(f"          label={str(slot.get('row_label'))[:64]!r}")
            print(f"          col={str(slot.get('gold_column'))[:40]!r} scope={slot.get('gold_scope')}")
            print(f"          -> {slot['decision']}: {slot['reason'][:150]}")
            for rival in slot.get("competing") or ():
                if rival.get("scope") == "TOTAL":
                    print(f"          ** rival {str(rival['value'])[:10]:10} is TOTAL "
                          f"label={str(rival.get('row_label'))[:50]!r}")
    print()
    print(f"  written to {args.out / 'cross-entity-source-truth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
