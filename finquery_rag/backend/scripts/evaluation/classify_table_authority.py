"""P1.6-A2B-19B: a two-headed shadow classifier, and the authority precedence it obeys.

**Shadow only.**  The resolver keeps using the old logic and production behaviour does
not change.  This emits, per table, what a replacement would say and what it rests on.

Two heads, because A2B-19A proved they cannot be one:

    statement_type   INCOME_STATEMENT / BALANCE_SHEET / CASH_FLOW / EQUITY /
                     SEGMENT / DEBT / DERIVATIVES / ... / UNKNOWN
                     what the table is *about*

    table_role       PRIMARY_FINANCIAL_STATEMENT / NON_PRIMARY / UNKNOWN
                     whether it may speak for the company

A derivatives note is entirely about balance-sheet line items and is still not the
balance sheet.  One field cannot carry both, which is why `statement_type` alone could
not separate JPMorganChase's consolidated statement from its per-share summary.

**Authority precedence**, which the Pfizer supplement forced:

    the table's own explicit identity
      > the nearest external heading
        > body semantic inference

`Supplemental Cash Flow Information` sits two tables below the cash flow statement and
inherits its title from the page above.  What settles it is that the statement opens with
`(MILLIONS)` and the note opens by naming its own subject: **a title outside a table cannot
outrank a caption inside it.**  The same ordering governs the score below.

**Evidence aggregation, not keyword matching.**  Every signal is an item with a polarity
and a weight, the items are listed with the table, and the decision is a threshold plus a
set of blocking conditions.  No single hit makes a table PRIMARY, and the blocking
conditions mean no amount of positive evidence can override the table saying what it is.

Only tables with `table_eligibility == DATA_TABLE` are classified.  The 592 page-layout
scaffolds leave before this runs, so nothing downstream counts them.

  python classify_table_authority.py --oracle <dir>/table-authority-oracle.json --out <dir>
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

_ORACLE_SCRIPT = _BACKEND_DIR / "scripts/evaluation/build_table_authority_oracle.py"


def _vocabulary():
    """The evidence vocabulary, shared with the oracle builder.

    Imported rather than restated so a title the oracle recognises as a statement of
    record is the same title this recognises.  The decision logic below is not shared
    and is not the oracle's.
    """
    spec = importlib.util.spec_from_file_location("authority_oracle", _ORACLE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Axes that mean the table breaks the company's figures out by part of the business.
#: Narrow on purpose.  A statement may carry *any* axis -- Apple tags its Products and
#: Services revenue lines with `ProductOrServiceAxis` and they are income-statement rows,
#: Visa's balance sheet carries `PledgedStatusAxis`, and every statement of changes in
#: equity is tagged `StatementEquityComponentsAxis` -- so the presence of an axis is not
#: evidence against a table.  What is evidence is the axis appearing while most of the
#: table's facts are *not* company-level, which the fact counts already measure; the axis
#: only sharpens it to "broken out by segment, geography, product or customer".
SEGMENTATION_AXES = (
    "StatementBusinessSegmentsAxis",
    "StatementGeographicalAxis",
    "ProductOrServiceAxis",
    "MajorCustomersAxis",
)

#: Concepts that only a primary statement reports.  Used as weak corroboration, never as
#: a reason on its own -- every one of them also appears in notes.
STATEMENT_CONCEPTS = re.compile(
    r"^(?:NetIncomeLoss|ProfitLoss|Revenues|RevenueFromContractWithCustomer\w*"
    r"|OperatingIncomeLoss|CostOfRevenue|CostOfGoodsAndServicesSold"
    r"|EarningsPerShare(Basic|Diluted)|Assets|Liabilities"
    r"|StockholdersEquity|StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
    r"|CashAndCashEquivalentsAtCarryingValue"
    r"|NetCashProvidedByUsedIn(Operating|Investing|Financing)Activities"
    r"|IncomeTaxExpenseBenefit|GrossProfit|ComprehensiveIncomeNetOfTax\w*)$"
)

#: A table of contents or an index of statements.  Its cells are statement *names*, so a
#: scan for the table's own identity would read `Index to Consolidated Financial
#: Statements` as a list of primary statements rather than as a list of them.
INDEX_LINE = re.compile(
    r"index\s+to|table\s+of\s+contents|financial\s+statements?\s+and\s+schedules?"
    r"|financial\s+statement\s+schedules?",
    re.I,
)

#: A subject a note is about.  The negative half of the authority evidence.
SUBJECT_KINDS = (
    ("SEGMENT", r"segment|geograph|by reportable|operations by|information by"),
    ("DEBT", r"\bdebt\b|borrow|maturit|contractual obligation|principal payment"),
    ("DERIVATIVES", r"derivative|hedg"),
    ("FAIR_VALUE", r"fair value"),
    ("LEASES", r"\blease"),
    ("PENSION", r"pension|postretirement|benefit plan"),
    ("INCOME_TAXES", r"income tax|tax provision|effective tax|deferred tax"),
    ("SHARE_BASED_COMPENSATION", r"share-based|stock-based|stock option|compensation expense"),
    ("EPS", r"earnings per share|per share of common stock|per-share"),
    ("QUARTERLY_DATA", r"quarterly financial data|selected quarterly|quarter"),
    ("NON_GAAP", r"non[\s-]*gaap|adjusted"),
    ("SUPPLEMENTARY", r"supplement"),
    ("RECONCILIATION", r"reconcil"),
    ("ALLOWANCE", r"allowance for|credit loss|credit exposure"),
    ("GOODWILL", r"goodwill|intangible"),
    ("SUBSEQUENT_EVENTS", r"subsequent event"),
    ("CONCENTRATION", r"concentration"),
)

#: Weights.  Positive evidence for authority, negative evidence against.  These are read
#: as an ordering of trust, not as calibrated probabilities: the own identity outweighs
#: any external heading, and an all-dimensioned fact set outweighs the own identity in the
#: other direction, because that is the filing's own tagging rather than a reading of prose.
WEIGHTS = {
    "OWN_STATEMENT_TITLE": 6,
    "OWN_UNITS_CAPTION": 1,
    "EXTERNAL_STATEMENT_TITLE": 5,
    "EXTERNAL_COMPANY_LINE": 0,
    "COMPANY_LEVEL_FACTS": 2,
    "STATEMENT_FAMILY_CONCEPTS": 1,
    "OWN_SUBJECT_CAPTION": -5,
    "EXTERNAL_DANGER_HEADING": -3,
    "EXTERNAL_NOTE_HEADING": -2,
    "ALL_FACTS_DIMENSIONED": -6,
    "MOSTLY_DIMENSIONED": -2,
    "SEGMENTATION_AXIS": -2,
}

#: Evidence that decides the outcome regardless of the score, because it is the source
#: stating the answer rather than this script inferring one.
BLOCKING = ("OWN_SUBJECT_CAPTION", "ALL_FACTS_DIMENSIONED")

#: Role is PRIMARY only above this score, and NON_PRIMARY at or below the lower one.
#: Between the two the answer is UNKNOWN, which stays a valid output: a table the evidence
#: does not settle must not be forced into an answer it cannot support.
PRIMARY_THRESHOLD = 5
NON_PRIMARY_THRESHOLD = 1

#: A statement of changes in equity is tagged by equity component, so most of its facts are
#: dimensioned and none of the companies here is an exception.  The dimensionality penalty
#: exists to catch breakdowns; the equity statement is not one, and penalising it would
#: throw away eight statements for a property of the statement itself.
DIMENSIONALITY_EXEMPT_FAMILIES = ("EQUITY",)


def _subject_kind(text: str | None) -> str | None:
    if not text:
        return None
    for name, pattern in SUBJECT_KINDS:
        if re.search(pattern, text, re.I):
            return name
    return None


def classify(table: dict, vocab) -> dict:
    """One table, two heads, and the evidence in the order it is trusted."""
    title = table.get("title")
    distance = table.get("title_distance")
    label_line = table.get("label_line")
    own_caption = table.get("own_caption")
    facts = table.get("facts") or 0
    undim = table.get("undimensioned_facts") or 0
    axes = table.get("axes") or {}
    concepts = table.get("concepts") or []

    evidence: list[dict] = []

    def add(code: str, detail: str) -> None:
        evidence.append({"code": code, "weight": WEIGHTS[code], "detail": detail})

    # --- the table's own explicit identity, which outranks everything else ---------
    is_index = bool(label_line and INDEX_LINE.search(label_line))
    own_title = None
    if not is_index:
        for row in (table.get("row_labels") or [])[:6]:
            candidate = vocab._ws(row)  # noqa: SLF001 - the vocabulary's own normaliser
            if candidate and vocab.STATEMENT_TITLE.match(candidate):
                own_title = candidate
                break
    own_subject = _subject_kind(own_caption)
    if own_title:
        add("OWN_STATEMENT_TITLE", f"the table's own first rows read {own_title!r}")
    elif own_subject and own_caption and not vocab.UNITS_LINE.match(own_caption):
        add("OWN_SUBJECT_CAPTION",
            f"the table's own first row reads {own_caption!r} ({own_subject})")
    elif own_caption and vocab.UNITS_LINE.match(own_caption):
        add("OWN_UNITS_CAPTION", f"the table opens with the units caption {own_caption!r}")

    # --- the nearest external heading ---------------------------------------------
    external_title = title if (title and distance is not None and distance <= 3) else None
    if external_title:
        add("EXTERNAL_STATEMENT_TITLE",
            f"{external_title!r} stands {distance} line(s) above the table")
    external_subject = _subject_kind(label_line)
    if external_subject and not external_title:
        add("EXTERNAL_DANGER_HEADING",
            f"the nearest line above reads {label_line!r} ({external_subject})")
    if label_line and vocab._NOTE_TITLE.match(label_line):  # noqa: SLF001
        # Only when the note heading is the *nearest* line.  Reading every note heading in
        # the window punished Visa's statement of comprehensive income and Pfizer's income
        # statement, each of which has a note heading from the previous page still inside
        # the six lines the evidence pack carries.
        add("EXTERNAL_NOTE_HEADING", f"the nearest line above is a note heading: {label_line!r}")
    if table.get("company_line"):
        add("EXTERNAL_COMPANY_LINE",
            f"the registrant names itself: {table['company_line']!r}")

    # --- the second head, resolved first because the first head depends on it --------
    # What the table is about, on the same precedence: the table's own identity -- its
    # title *or* the subject it names -- then the external heading, then the body.  Pfizer's
    # `Supplemental Cash Flow Information` inherits `Consolidated Statements of Cash Flows`
    # from the page above; the role head already refuses to be overridden by that title, and
    # the type head must refuse too, or the table is called a cash flow statement while
    # being denied the authority of one.
    if own_title:
        statement_type, from_what = vocab.family_of(own_title), "OWN_STATEMENT_TITLE"
    elif own_subject:
        statement_type, from_what = own_subject, "OWN_SUBJECT_CAPTION"
    elif external_title:
        statement_type, from_what = vocab.family_of(external_title), "EXTERNAL_STATEMENT_TITLE"
    elif external_subject:
        statement_type, from_what = external_subject, "EXTERNAL_DANGER_HEADING"
    else:
        body = " ".join(table.get("row_labels") or []) + " " + (table.get("body_head") or "")
        statement_type, from_what = _subject_kind(body), "BODY_INFERENCE"

    # --- what the table's own numbers are -----------------------------------------
    share = (undim / facts) if facts else None
    exempt = statement_type in DIMENSIONALITY_EXEMPT_FAMILIES
    if facts and undim == 0:
        add("ALL_FACTS_DIMENSIONED",
            f"all {facts} tagged fact(s) sit in a dimensioned context")
    elif facts and share >= 0.5:
        add("COMPANY_LEVEL_FACTS",
            f"{undim} of {facts} tagged fact(s) are company-level")
    elif facts and share < 0.15 and not exempt:
        add("MOSTLY_DIMENSIONED",
            f"only {undim} of {facts} tagged fact(s) are company-level")
    segmenting = [a for a in axes if any(s in a for s in SEGMENTATION_AXES)]
    if segmenting and share is not None and share < 0.5:
        add("SEGMENTATION_AXIS",
            f"the facts are broken out by {segmenting[:3]} and are not mostly company-level")
    core = [c for c in concepts if STATEMENT_CONCEPTS.match(c.rsplit(":", 1)[-1])]
    if core:
        add("STATEMENT_FAMILY_CONCEPTS", f"{core[:3]}")

    codes = {item["code"] for item in evidence}
    score = sum(item["weight"] for item in evidence)
    blocked = [code for code in BLOCKING if code in codes]

    # A table is not PRIMARY on score alone.  It needs a positive identity, so that every
    # predicted primary can be pointed at the words in the filing that say so, and it needs
    # to be a table of figures carrying at least one company-level fact -- which is not a
    # keyword test but the precondition for the question even arising: a table with no
    # tagged fact states no company-level amount for a resolver to use.
    identity = codes & {"OWN_STATEMENT_TITLE", "EXTERNAL_STATEMENT_TITLE"}
    eligible = facts >= 1 and undim >= 1

    if blocked:
        role = "NON_PRIMARY"
        status = f"BLOCKED_BY_{blocked[0]}"
    elif score >= PRIMARY_THRESHOLD and identity and eligible:
        role = "PRIMARY_FINANCIAL_STATEMENT"
        status = "SOURCE_SUPPORTED"
    elif score <= NON_PRIMARY_THRESHOLD:
        role = "NON_PRIMARY"
        status = "EVIDENCE_AGAINST"
    else:
        role = "UNKNOWN"
        status = "INSUFFICIENT_EVIDENCE"

    return {
        "oracle_key": table["oracle_key"],
        "table_eligibility": table["table_eligibility"],
        "new_statement_type": statement_type or "UNKNOWN",
        "new_statement_type_source": from_what,
        "new_table_role": role,
        "status": status,
        "score": score,
        "evidence": evidence,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    vocab = _vocabulary()
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))

    report = {"phase": "P1.6-A2B-19B", "mutation": "none", "shadow": True,
              "documents": {}}
    classified = 0
    skipped = 0

    print("=== table authority classifier (shadow) ===")
    print()
    for document_id in sorted(oracle["documents"]):
        rows = []
        for table in oracle["documents"][document_id]["tables"]:
            if table["table_eligibility"] != "DATA_TABLE":
                skipped += 1
                continue
            rows.append(classify(table, vocab))
            classified += 1
        report["documents"][document_id] = {"tables": rows}
        roles = {}
        for row in rows:
            roles[row["new_table_role"]] = roles.get(row["new_table_role"], 0) + 1
        print(f"--- {document_id}  classified {len(rows)} data tables  {roles}")

    report["classified"] = classified
    report["scaffolds_not_classified"] = skipped
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "shadow-classification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print()
    print(f"  classified {classified} data tables; "
          f"{skipped} layout scaffolds not classified")
    print(f"  written to {args.out / 'shadow-classification.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
