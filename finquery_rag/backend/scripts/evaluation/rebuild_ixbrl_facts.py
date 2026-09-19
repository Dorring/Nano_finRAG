"""P1.6-A3: rebuild the eight filings from iXBRL and reconcile against known values.

The store's `metric` is a flattened breadcrumb string, which is what makes a
cross-filing comparison impossible and a coordinate ambiguous.  The filings
themselves carry `us-gaap:`/`dei:` concepts -- the same string at every filer --
and XBRL contexts whose dimensions name the scope (`ConsolidatedEntitiesAxis`,
`StatementEquityComponentsAxis`).  This emits facts as

    (document, concept, dimensions, period, unit, value)

and reconciles them against values read out of the filings by hand during P1.6-0D
and the stratum re-derivation.  Nothing is written to the fact store: the point
is to measure the reconciliation rate before any production behaviour changes,
so that "the store was rebuilt" and "the system reads a different store" stay
two separately measurable deltas.

The reconciliation is deliberately *not* concept-driven.  It asks only: for this
verified number, is there a fact carrying it, and if so what concept and context?
Deciding in advance which concept *should* hold a value would assume the answer.

  python rebuild_ixbrl_facts.py --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

CORPUS = Path("/disk/qh/nano-finrag/data/financial_corpus_v2")

#: store document_id -> (ticker, fiscal year).  Taken from the fact store's own
#: document names, so the mapping is the store's, not one invented here.
DOCUMENTS = {
    "aapl_fy2025": ("AAPL", "2025"),
    "jpm_fy2025": ("JPM", "2025"),
    "ko_fy2025": ("KO", "2025"),
    "msft_fy2025": ("MSFT", "2025"),
    "nvda_fy2025": ("NVDA", "2025"),
    "pfe_fy2024": ("PFE", "2024"),
    "tsla_fy2025": ("TSLA", "2025"),
    "v_fy2025": ("V", "2025"),
}

#: (document_id, label, value) for every figure read out of a filing by hand in
#: this work -- P1.6-0D's source audit and the P1.6-0G re-derivation.  Each one
#: was quoted from a page at the time; this checks whether iXBRL reproduces it.
VERIFIED = [
    ("aapl_fy2025", "Net income", "112010"),
    ("aapl_fy2025", "Diluted EPS", "7.46"),
    ("aapl_fy2025", "Operating income", "133050"),
    ("aapl_fy2025", "Total assets", "359241"),
    ("aapl_fy2025", "Total liabilities", "285508"),
    ("aapl_fy2025", "Research and development", "34550"),
    ("msft_fy2025", "Net income", "101832"),
    ("msft_fy2025", "Operating income", "128528"),
    ("msft_fy2025", "Total assets", "619003"),
    ("msft_fy2025", "Total liabilities", "275524"),
    ("msft_fy2025", "Research and development", "32488"),
    ("msft_fy2025", "Comprehensive income", "104075"),
    ("msft_fy2025", "Interest expense", "2385"),
    ("tsla_fy2025", "Net income", "3855"),
    ("tsla_fy2025", "Operating income", "4355"),
    ("tsla_fy2025", "Total assets", "137806"),
    ("tsla_fy2025", "Total liabilities", "54941"),
    ("tsla_fy2025", "Research and development", "6411"),
    ("ko_fy2025", "Net income", "13137"),
    ("ko_fy2025", "Operating income", "13762"),
    ("ko_fy2025", "Total assets", "104816"),
    ("ko_fy2025", "Long-term debt", "42119"),
    ("ko_fy2025", "Interest expense", "1654"),
    ("jpm_fy2025", "Net income", "57048"),
    ("jpm_fy2025", "Diluted EPS", "20.02"),
    ("jpm_fy2025", "Comprehensive income", "65214"),
    ("jpm_fy2025", "Interest expense", "97898"),
    ("jpm_fy2025", "Total assets", "4424900"),
    ("jpm_fy2025", "Total liabilities", "4062462"),
    ("jpm_fy2025", "Total net revenue", "182447"),
    ("v_fy2025", "Net income", "20058"),
    ("v_fy2025", "Comprehensive income", "20614"),
    ("v_fy2025", "Long-term debt", "19602"),
    ("v_fy2025", "Interest expense", "589"),
    ("nvda_fy2025", "Net income", "72880"),
    ("nvda_fy2025", "Operating income", "81453"),
    ("nvda_fy2025", "Total assets", "111601"),
    ("nvda_fy2025", "Total liabilities", "32274"),
    ("pfe_fy2024", "Net income", "8062"),
]

_CONTEXT = re.compile(
    r"<(?:xbrli:)?context[^>]*id=\"(c-\d+)\"[^>]*>(.*?)</(?:xbrli:)?context>", re.S
)
_MEMBER = re.compile(
    r"<xbrldi:explicitMember[^>]*dimension=\"([^\"]+)\"[^>]*>([^<]*)<", re.S
)


def _digits(text: object) -> str | None:
    cleaned = re.sub(r"[^\d.]", "", str(text or ""))
    if not cleaned:
        return None
    # `20.02` and `20.0200` are the same number; `2,385` is 2385.
    cleaned = cleaned.rstrip("0").rstrip(".") if "." in cleaned else cleaned
    return cleaned or None


def _filings() -> dict[str, dict]:
    """store document_id -> {document.json, primary.html} for the right filing."""

    found: dict[str, dict] = {}
    for document_id, (ticker, fiscal_year) in DOCUMENTS.items():
        root = CORPUS / "normalized/SEC" / ticker
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/document.json")):
            meta = json.loads(path.read_text(encoding="utf-8")).get("document", {})
            if str(meta.get("form_type")) != "10-K":
                continue
            if str(meta.get("fiscal_year")) != fiscal_year:
                continue
            accession = path.parent.name
            html = CORPUS / "raw/SEC" / ticker / accession / "primary.html"
            found[document_id] = {
                "accession": accession, "json": path, "html": html,
                "html_present": html.is_file(),
            }
            break
    return found


def _contexts(html_path: Path) -> dict[str, list[tuple[str, str]]]:
    """context id -> the dimensions it carries, as (axis, member) pairs."""

    if not html_path.is_file():
        return {}
    text = html_path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, list[tuple[str, str]]] = {}
    for context_id, body in _CONTEXT.findall(text):
        members = [
            (axis.split(":")[-1], member.strip().split(":")[-1])
            for axis, member in _MEMBER.findall(body)
        ]
        out[context_id] = members
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    filings = _filings()
    print(f"=== filings located: {len(filings)} / {len(DOCUMENTS)} ===")
    for document_id in sorted(DOCUMENTS):
        entry = filings.get(document_id)
        print(f"  {document_id:14} {entry['accession'] if entry else 'NOT FOUND':40} "
              f"html={'yes' if entry and entry['html_present'] else 'no'}")

    facts_by_document: dict[str, list[dict]] = {}
    context_counts: dict[str, int] = {}
    for document_id, entry in filings.items():
        contexts = _contexts(entry["html"])
        context_counts[document_id] = len(contexts)
        raw_facts = json.loads(entry["json"].read_text(encoding="utf-8"))["ixbrl_facts"]
        emitted = []
        for fact in raw_facts:
            value = _digits(fact.get("raw_value"))
            if value is None:
                continue
            context_ref = str(fact.get("context_ref") or "")
            context = fact.get("context") or {}
            emitted.append({
                "concept": str(fact.get("concept") or ""),
                "context_ref": context_ref,
                "dimensions": contexts.get(context_ref, []),
                "period_start": context.get("period_start"),
                "period_end": context.get("period_end"),
                "unit": fact.get("unit"),
                "value": value,
            })
        facts_by_document[document_id] = emitted

    # --- reconciliation -------------------------------------------------------
    rows = []
    unresolved = collections.Counter()
    undimensioned_hits = 0
    dimensioned_only = 0
    for document_id, label, expected in VERIFIED:
        facts = facts_by_document.get(document_id, [])
        want = _digits(expected)
        hits = [f for f in facts if f["value"] == want]
        # Prefer the fiscal-year period the case asks about.
        year = DOCUMENTS[document_id][1]
        period_hits = [f for f in hits if str(f.get("period_end") or "").startswith(year)]
        pool = period_hits or hits
        undimensioned = [f for f in pool if not f["dimensions"]]

        if not pool:
            status = "NOT_FOUND"
        elif undimensioned:
            status = "RESOLVED_UNDIMENSIONED"
            undimensioned_hits += 1
        else:
            status = "FOUND_ONLY_WITH_DIMENSIONS"
            dimensioned_only += 1
        unresolved[status] += 1

        concepts = sorted({f["concept"] for f in (undimensioned or pool)})
        rows.append({
            "document_id": document_id, "label": label, "expected": expected,
            "status": status,
            "undimensioned_concepts": sorted({f["concept"] for f in undimensioned}),
            "all_concepts": concepts[:4],
            "period_hits": len(period_hits), "total_hits": len(hits),
            "dimensions_seen": sorted({d[0] for f in pool for d in f["dimensions"]})[:4],
        })

    report = {
        "phase": "P1.6-A3",
        "mutation": "none -- iXBRL facts are emitted in memory only",
        "filings": {k: {"accession": v["accession"], "html_present": v["html_present"],
                        "contexts": context_counts.get(k, 0)}
                    for k, v in filings.items()},
        "fact_counts": {k: len(v) for k, v in facts_by_document.items()},
        "reconciliation": dict(unresolved),
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ixbrl-reconciliation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\n=== reconciliation ===")
    for row in rows:
        mark = {"RESOLVED_UNDIMENSIONED": "ok ",
                "FOUND_ONLY_WITH_DIMENSIONS": "dim",
                "NOT_FOUND": "MISS"}[row["status"]]
        concepts = ", ".join(c.replace("us-gaap:", "") for c in row["undimensioned_concepts"]) \
            or ", ".join(c.replace("us-gaap:", "") for c in row["all_concepts"])
        print(f"  {mark} {row['document_id']:14} {row['label'][:26]:26} "
              f"{row['expected']:>10}  {concepts[:52]}")
    print(f"\n  {dict(unresolved)}")
    print(f"  written to {args.out / 'ixbrl-reconciliation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
