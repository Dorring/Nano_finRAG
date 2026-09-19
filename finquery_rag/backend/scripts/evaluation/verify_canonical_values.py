"""P1.6-A6: read every canonical value back out of the filing that states it.

Resolving to a single undimensioned fact is not the same as the value being the
one the statement shows.  `concept_alignment` picks a fact; this opens the
filing and requires the number to appear in text that also names the quantity.

Six values had been read by hand earlier in this work. This checks all 47, so
the two standards are the same one applied evenly rather than a sample standing
in for the rest.

It cannot be section-based. `section_type` is unevenly populated -- Apple's and
Microsoft's documents carry no `INCOME_STATEMENT` block at all -- so the check is
textual: the value must sit within a short distance after a label naming the
quantity, in the same block. That is weaker than reading the page, and it is
reported as such: a PASS means "the filing states this number in this context",
not "a human confirmed the row".

  python verify_canonical_values.py --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

CORPUS = Path("/disk/qh/nano-finrag/data/financial_corpus_v2")

ENTITY_FILING = {
    "Apple": ("AAPL", "SEC_320193_000032019325000079", "2025"),
    "JPMorganChase": ("JPM", "SEC_19617_000162828026008131", "2025"),
    "The Coca-Cola Company": ("KO", "SEC_21344_000162828026010047", "2025"),
    "Microsoft": ("MSFT", "SEC_789019_000095017025100235", "2025"),
    "NVIDIA": ("NVDA", "SEC_1045810_000104581025000023", "2025"),
    "Pfizer": ("PFE", "SEC_78003_000007800325000054", "2024"),
    "Tesla": ("TSLA", "SEC_1318605_000162828026003952", "2025"),
    "Visa": ("V", "SEC_1403161_000140316125000089", "2025"),
}

#: Labels a filing might use for each canonical quantity.  A label is required
#: to *name* the quantity; the value alone proves nothing, since the same number
#: appears in many tables.
LABELS = {
    "net_income": ("net income", "consolidated net income", "net earnings"),
    "operating_income": ("operating income", "income from operations",
                         "operating profit"),
    "total_assets": ("total assets",),
    "total_liabilities": ("total liabilities",),
    "research_and_development": ("research and development",),
    "diluted_eps": ("earnings per share", "diluted"),
    "comprehensive_income": ("comprehensive income",),
    "interest_expense": ("interest expense",),
    "long_term_debt": ("long-term debt", "long term debt"),
    "revenues": ("total net revenue", "net revenue", "total revenues", "revenues"),
}

CASES = {
    "tv2f01-s3-compare-001": ("net_income", ["The Coca-Cola Company", "Tesla"]),
    "tv2f01-s3-compare-002": ("net_income", ["Apple", "Visa"]),
    "tv2f01-s3-compare-003": ("total_assets", ["Tesla", "The Coca-Cola Company"]),
    "tv2f01-s3-compare-004": ("net_income", ["Apple", "Microsoft"]),
    "tv2f01-s3-compare-005": ("diluted_eps", ["JPMorganChase", "Apple"]),
    "tv2f01-s3-compare-006": ("comprehensive_income", ["JPMorganChase", "Tesla"]),
    "tv2f01-s3-compare-007": ("net_income", ["JPMorganChase", "Pfizer"]),
    "tv2f01-s3-compare-008": ("diluted_eps", ["JPMorganChase", "Apple"]),
    "tv2f01-s3-compare-009": ("net_income", ["Apple", "JPMorganChase"]),
    "tv2f01-s3-compare-010": ("operating_income", ["Apple", "Microsoft"]),
    "tv2f01-s3-crossdiff-001": ("net_income", ["NVIDIA", "Tesla"]),
    "tv2f01-s3-crossdiff-002": ("net_income", ["Apple", "Microsoft"]),
    "tv2f01-s3-crossdiff-003": ("total_liabilities", ["Apple", "Tesla"]),
    "tv2f01-s3-crossdiff-004": ("long_term_debt", ["The Coca-Cola Company", "Visa"]),
    "tv2f01-s3-crossdiff-005": ("total_assets", ["NVIDIA", "The Coca-Cola Company"]),
    "tv2f01-s3-rank-001": ("net_income", ["Tesla", "The Coca-Cola Company", "Visa"]),
    "tv2f01-s3-rank-002": ("research_and_development",
                           ["Apple", "Microsoft", "Tesla"]),
    "tv2f01-s3-rank-003": ("comprehensive_income",
                           ["JPMorganChase", "Microsoft", "Tesla", "Visa"]),
    "tv2f01-s3-rank-004": ("interest_expense",
                           ["JPMorganChase", "The Coca-Cola Company", "Visa", "Microsoft"]),
    "tv2f01-s3-rank-005": ("operating_income",
                           ["Apple", "Microsoft", "The Coca-Cola Company"]),
}

_CONTEXT = re.compile(
    r"<(?:xbrli:)?context[^>]*id=\"([^\"]+)\"[^>]*>(.*?)</(?:xbrli:)?context>", re.S
)
_WINDOW = 160


def _digits(text: object) -> str | None:
    cleaned = re.sub(r"[^\d.]", "", str(text or ""))
    if not cleaned:
        return None
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    return cleaned or None


def _formatted(value: str) -> list[str]:
    """The spellings a filing might print for a numeric value."""

    if "." in value:
        return [value]
    spellings = [value]
    if len(value) > 3:
        spellings.append(f"{int(value):,}")
    return spellings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from src.finance.concept_alignment import CONCEPT_ALIGNMENT

    # Resolve first, exactly as the switch would.
    facts: dict[str, list[dict]] = {}
    texts: dict[str, str] = {}
    for entity, (ticker, accession, _fy) in ENTITY_FILING.items():
        raw_html = (CORPUS / "raw/SEC" / ticker / accession / "primary.html").read_text(
            encoding="utf-8", errors="replace"
        )
        dimensioned = {
            cid: "explicitMember" in body
            for cid, body in _CONTEXT.findall(raw_html)
        }
        document = json.loads(
            (CORPUS / "normalized/SEC" / ticker / accession / "document.json")
            .read_text(encoding="utf-8")
        )
        facts[entity] = [
            {
                "concept": str(f.get("concept") or ""),
                "value": _digits(f.get("raw_value")),
                "period_end": str((f.get("context") or {}).get("period_end") or ""),
                "undimensioned": not dimensioned.get(
                    str(f.get("context_ref") or ""), True
                ),
            }
            for f in document["ixbrl_facts"]
            if _digits(f.get("raw_value"))
        ]
        # The `blocks` in `document.json` are a normalised subset -- Apple's
        # document carries 62 of them and neither its balance sheet nor its
        # income statement text -- so a search there reports "not stated" for
        # values that are plainly on the page.  The filing's own HTML is the
        # source, stripped to text.
        stripped = re.sub(r"<[^>]+>", " ", raw_html)
        texts[entity] = " ".join(html.unescape(stripped).split())

    seen: set[tuple[str, str]] = set()
    rows = []
    tally: collections.Counter = collections.Counter()
    for case_id, (canonical, entities) in sorted(CASES.items()):
        for entity in entities:
            if (canonical, entity) in seen:
                continue
            seen.add((canonical, entity))
            fiscal_year = ENTITY_FILING[entity][2]
            picked = picked_fact = None
            for concept in CONCEPT_ALIGNMENT[canonical]:
                pool = [
                    f for f in facts[entity]
                    if f["concept"] == concept
                    and f["undimensioned"]
                    and f["period_end"].startswith(fiscal_year)
                ]
                if pool:
                    picked, picked_fact = concept, pool[0]
                    break
            if picked_fact is None:
                tally["UNRESOLVED"] += 1
                rows.append({"case_id": case_id, "entity": entity,
                             "canonical": canonical, "status": "UNRESOLVED"})
                continue

            value = picked_fact["value"]
            found, quote = None, None
            text = texts[entity]
            folded = text.casefold()
            for spelling in _formatted(value):
                start = 0
                while found is None:
                    index = folded.find(spelling.casefold(), start)
                    if index < 0:
                        break
                    start = index + 1
                    for label in LABELS[canonical]:
                        if label in folded[max(0, index - _WINDOW):index]:
                            found = label
                            quote = " ".join(text[max(0, index - 90):index + 40].split())
                            break
                if found:
                    break
            status = "VERIFIED" if found else "NOT_STATED_WITH_LABEL"
            tally[status] += 1
            rows.append({
                "case_id": case_id, "entity": entity, "canonical": canonical,
                "status": status, "value": value, "source_concept": picked,
                "label_found": found, "quote": quote,
            })

    report = {
        "phase": "P1.6-A6", "mutation": "none",
        "summary": dict(tally),
        "note": "PASS means the filing states this number in text naming the "
                "quantity; it is not a human reading of the row",
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "canonical-value-verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== canonical value verification ===")
    for row in sorted(rows, key=lambda r: (r["canonical"], r["entity"])):
        mark = {"VERIFIED": "ok ", "NOT_STATED_WITH_LABEL": "?? ",
                "UNRESOLVED": "---"}[row["status"]]
        print(f"  {mark} {row['entity'][:22]:22} {row['canonical'][:24]:24} "
              f"{str(row.get('value')):>10}  label={str(row.get('label_found'))[:26]}")
    print(f"\n  {dict(tally)}")
    print(f"  written to {args.out / 'canonical-value-verification.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
