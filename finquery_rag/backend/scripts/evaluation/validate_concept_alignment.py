"""P1.6-A4: does the concept alignment reproduce every verified value?

The acceptance test for `concept_alignment.py`, and the gate on keying retrieval
on a canonical quantity: take the 39 values read out of the eight filings by hand
during P1.6-0D and the stratum re-derivation, resolve each through the alignment,
and require the canonical quantity to be the same across filers and the value to
match the filing.

It reports context-parse coverage per filing, because the resolution rule depends
on knowing which contexts carry dimensions.  A filing whose contexts did not parse
cannot be filtered, and every one of its facts looks undimensioned -- which would
let a segment figure through dressed as the company total.  That is a failure to
report, not to paper over.

  python validate_concept_alignment.py --out <dir>
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

#: label as the benchmark names it -> canonical quantity in the alignment.
LABEL_TO_CANONICAL = {
    "Net income": "net_income",
    "Diluted EPS": "diluted_eps",
    "Operating income": "operating_income",
    "Total assets": "total_assets",
    "Total liabilities": "total_liabilities",
    "Research and development": "research_and_development",
    "Comprehensive income": "comprehensive_income",
    "Interest expense": "interest_expense",
    "Long-term debt": "long_term_debt",
    "Total net revenue": "revenues",
}

#: (document_id, label, value) read from a filing by hand in this work.
VERIFIED = [
    ("aapl_fy2025", "Net income", "112010"), ("aapl_fy2025", "Diluted EPS", "7.46"),
    ("aapl_fy2025", "Operating income", "133050"), ("aapl_fy2025", "Total assets", "359241"),
    ("aapl_fy2025", "Total liabilities", "285508"),
    ("aapl_fy2025", "Research and development", "34550"),
    ("msft_fy2025", "Net income", "101832"), ("msft_fy2025", "Operating income", "128528"),
    ("msft_fy2025", "Total assets", "619003"), ("msft_fy2025", "Total liabilities", "275524"),
    ("msft_fy2025", "Research and development", "32488"),
    ("msft_fy2025", "Comprehensive income", "104075"),
    ("msft_fy2025", "Interest expense", "2385"),
    ("tsla_fy2025", "Net income", "3855"), ("tsla_fy2025", "Operating income", "4355"),
    ("tsla_fy2025", "Total assets", "137806"), ("tsla_fy2025", "Total liabilities", "54941"),
    ("tsla_fy2025", "Research and development", "6411"),
    ("ko_fy2025", "Net income", "13137"), ("ko_fy2025", "Operating income", "13762"),
    ("ko_fy2025", "Total assets", "104816"), ("ko_fy2025", "Long-term debt", "42119"),
    ("ko_fy2025", "Interest expense", "1654"),
    ("jpm_fy2025", "Net income", "57048"), ("jpm_fy2025", "Diluted EPS", "20.02"),
    ("jpm_fy2025", "Comprehensive income", "65214"),
    ("jpm_fy2025", "Interest expense", "97898"), ("jpm_fy2025", "Total assets", "4424900"),
    ("jpm_fy2025", "Total liabilities", "4062462"), ("jpm_fy2025", "Total net revenue", "182447"),
    ("v_fy2025", "Net income", "20058"), ("v_fy2025", "Comprehensive income", "20614"),
    ("v_fy2025", "Long-term debt", "19602"), ("v_fy2025", "Interest expense", "589"),
    ("nvda_fy2025", "Net income", "72880"), ("nvda_fy2025", "Operating income", "81453"),
    ("nvda_fy2025", "Total assets", "111601"), ("nvda_fy2025", "Total liabilities", "32274"),
    ("pfe_fy2024", "Net income", "8062"),
]

# Context ids are not one scheme.  Filings produced through one agent use
# `c-1`, `c-30`; Microsoft's (DFIN ActiveDisclosure) uses `C_<uuid>`.  Matching
# only the first shape parsed 441 of Microsoft's contexts as zero, which left
# every one of its facts looking undimensioned -- the failure mode that would
# let a segment figure through as the company total.
_CONTEXT = re.compile(
    r"<(?:xbrli:)?context[^>]*id=\"([^\"]+)\"[^>]*>(.*?)</(?:xbrli:)?context>", re.S
)


def _digits(text: object) -> str | None:
    cleaned = re.sub(r"[^\d.]", "", str(text or ""))
    if not cleaned:
        return None
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    return cleaned or None


def _context_dimensions(html_path: Path) -> tuple[dict[str, bool], int]:
    """context id -> has dimensions, plus the number of context tags seen."""

    if not html_path.is_file():
        return {}, 0
    text = html_path.read_text(encoding="utf-8", errors="replace")
    seen = len(re.findall(r"<(?:xbrli:)?context\b", text))
    out = {}
    for context_id, body in _CONTEXT.findall(text):
        out[context_id] = "explicitMember" in body
    return out, seen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from src.finance.concept_alignment import align_concept, resolve

    facts_by_document: dict[str, list[dict]] = {}
    coverage: dict[str, dict] = {}
    for document_id, (ticker, accession) in DOCUMENTS.items():
        dimensions, seen = _context_dimensions(
            CORPUS / "raw/SEC" / ticker / accession / "primary.html"
        )
        raw = json.loads(
            (CORPUS / "normalized/SEC" / ticker / accession / "document.json")
            .read_text(encoding="utf-8")
        )["ixbrl_facts"]
        facts = []
        for fact in raw:
            value = _digits(fact.get("raw_value"))
            if value is None:
                continue
            context_ref = str(fact.get("context_ref") or "")
            facts.append({
                "concept": str(fact.get("concept") or ""),
                "value": value,
                "period_end": (fact.get("context") or {}).get("period_end"),
                # Unknown contexts count as dimensioned, not undimensioned:
                # assuming the opposite would let a segment figure through.
                "dimensions": [context_ref] if dimensions.get(context_ref, True) else [],
            })
        facts_by_document[document_id] = facts
        coverage[document_id] = {
            "context_tags_in_html": seen,
            "contexts_parsed": len(dimensions),
            "facts": len(facts),
            "dimension_filter_reliable": seen > 0 and len(dimensions) > 0,
        }

    rows = []
    summary: collections.Counter = collections.Counter()
    canonical_seen: dict[str, set] = collections.defaultdict(set)

    for document_id, label, expected in VERIFIED:
        canonical = LABEL_TO_CANONICAL[label]
        got = resolve(facts_by_document.get(document_id, []), canonical)
        matched = bool(got) and got["value"] == _digits(expected)
        status = "MATCH" if matched else ("MISMATCH" if got else "UNRESOLVED")
        summary[status] += 1
        if got:
            canonical_seen[canonical].add(got["source_concept"])
        rows.append({
            "document_id": document_id, "label": label, "canonical": canonical,
            "expected": expected, "status": status,
            "resolved_value": got["value"] if got else None,
            "source_concept": got["source_concept"] if got else None,
            "period_end": got["period_end"] if got else None,
        })

    report = {
        "phase": "P1.6-A4",
        "mutation": "none",
        "coverage": coverage,
        "summary": dict(summary),
        "source_concepts_per_canonical": {
            k: sorted(v) for k, v in canonical_seen.items()
        },
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "concept-alignment-validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== context-parse coverage ===")
    for document_id, block in sorted(coverage.items()):
        print(f"  {document_id:14} tags={block['context_tags_in_html']:>5} "
              f"parsed={block['contexts_parsed']:>5} facts={block['facts']:>6} "
              f"filter={'ok' if block['dimension_filter_reliable'] else 'UNRELIABLE'}")
    print("\n=== alignment validation ===")
    for row in rows:
        mark = {"MATCH": "ok ", "MISMATCH": "BAD", "UNRESOLVED": "---"}[row["status"]]
        print(f"  {mark} {row['document_id']:14} {row['canonical'][:24]:24} "
              f"want={row['expected']:>10} got={str(row['resolved_value']):>10} "
              f"via={str(row['source_concept']).replace('us-gaap:','')[:26]}")
    print(f"\n  {dict(summary)}")
    print("  source concepts per canonical quantity:")
    for canonical, concepts in sorted(canonical_seen.items()):
        print(f"      {canonical:26} {[c.replace('us-gaap:','') for c in concepts]}")
    print(f"\n  written to {args.out / 'concept-alignment-validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
