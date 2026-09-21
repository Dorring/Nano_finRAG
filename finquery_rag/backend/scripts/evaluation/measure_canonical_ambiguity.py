"""P1.6-A5: do canonical concepts make the cross-entity stratum answerable?

The measurement the switch decision rests on.  For each of the 20 re-derived cases,
resolve every entity's value through `concept_alignment` and ask two questions:

1. **Does it resolve at all?**  The store could not answer 13 of the 20 -- concepts
   absent (`n_distinct=0`) or holding segment values (`n_distinct=3`).  A canonical
   concept either finds the fact or it does not.
2. **Is the result ambiguous?**  More than one undimensioned fact at the same
   concept and period, carrying different values, means the coordinate still does
   not identify one value and the operand guard would be right to refuse it.

The comparison is against the store's own numbers, so the delta is a measurement
rather than a claim: 1,433 of 7,887 coordinates ambiguous (18.2%), 13 of 20
cross-entity cases unusable.

Nothing is written.  This decides whether the switch is worth making, not whether
it has been made.

  python measure_canonical_ambiguity.py --out <dir>
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

ENTITY_FILING = {
    "Apple": ("AAPL", "SEC_320193_000032019325000079", "FY2025"),
    "JPMorganChase": ("JPM", "SEC_19617_000162828026008131", "FY2025"),
    "The Coca-Cola Company": ("KO", "SEC_21344_000162828026010047", "FY2025"),
    "Microsoft": ("MSFT", "SEC_789019_000095017025100235", "FY2025"),
    "NVIDIA": ("NVDA", "SEC_1045810_000104581025000023", "FY2025"),
    "Pfizer": ("PFE", "SEC_78003_000007800325000054", "FY2024"),
    "Tesla": ("TSLA", "SEC_1318605_000162828026003952", "FY2025"),
    "Visa": ("V", "SEC_1403161_000140316125000089", "FY2025"),
}

#: case -> (canonical quantity, entities).  From docs/evaluation/p1-6-0g-rederivation.md.
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

#: The values the re-derivation verified, for a resolution to be checked against.
EXPECTED = {
    ("tv2f01-s3-compare-001", "The Coca-Cola Company"): "13137",
    ("tv2f01-s3-compare-001", "Tesla"): "3855",
    ("tv2f01-s3-rank-004", "JPMorganChase"): "97898",
    ("tv2f01-s3-rank-004", "Visa"): "589",
    ("tv2f01-s3-rank-005", "Apple"): "133050",
    ("tv2f01-s3-crossdiff-003", "Apple"): "285508",
}

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from src.finance.concept_alignment import CONCEPT_ALIGNMENT

    facts_by_entity: dict[str, list[dict]] = {}
    for entity, (ticker, accession, _fy) in ENTITY_FILING.items():
        html = CORPUS / "raw/SEC" / ticker / accession / "primary.html"
        text = html.read_text(encoding="utf-8", errors="replace")
        dimensioned = {
            context_id: "explicitMember" in body
            for context_id, body in _CONTEXT.findall(text)
        }
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
            context = fact.get("context") or {}
            facts.append({
                "concept": str(fact.get("concept") or ""),
                "value": value,
                "period_end": str(context.get("period_end") or ""),
                "undimensioned": not dimensioned.get(context_ref, True),
            })
        facts_by_entity[entity] = facts

    rows = []
    tally: collections.Counter = collections.Counter()
    disagreements = []
    for case_id, (canonical, entities) in sorted(CASES.items()):
        candidates = CONCEPT_ALIGNMENT[canonical]
        entity_rows = []
        for entity in entities:
            fiscal_year = ENTITY_FILING[entity][2][2:]
            # Candidates are applied **in order**, first present wins -- pooling
            # them instead would report Tesla as ambiguous because `ProfitLoss`
            # (3,855, consolidated) and `NetIncomeLoss` (3,794, parent-only) are
            # both present undimensioned, when the alignment exists precisely to
            # choose between them.  The pooled count is kept alongside, because
            # it is the measure of *why* the ordering is load-bearing.
            unpooled: list[dict] = []
            picked: str | None = None
            for concept in candidates:
                unpooled = [
                    f for f in facts_by_entity[entity]
                    if f["concept"] == concept
                    and f["undimensioned"]
                    and f["period_end"].startswith(fiscal_year)
                ]
                if unpooled:
                    picked = concept
                    break
            pooled = [
                f for f in facts_by_entity[entity]
                if f["concept"] in candidates
                and f["undimensioned"]
                and f["period_end"].startswith(fiscal_year)
            ]
            values = sorted({f["value"] for f in unpooled})
            pooled_values = sorted({f["value"] for f in pooled})
            if not unpooled:
                status = "NO_COMPANY_LEVEL_FACT"
            elif len(values) == 1:
                status = "RESOLVED"
            else:
                status = "AMBIGUOUS"
                disagreements.append((case_id, entity, canonical, values[:5]))
            tally[status] += 1
            expected = EXPECTED.get((case_id, entity))
            entity_rows.append({
                "entity": entity, "status": status,
                "source_concept": picked,
                "distinct_values": values[:5], "facts": len(unpooled),
                "pooled_values": pooled_values[:5],
                "pooled_would_be_ambiguous": len(pooled_values) > 1,
                "expected": expected,
                "matches_expected": (expected in values) if expected else None,
            })
        if all(r["status"] == "RESOLVED" for r in entity_rows):
            case_status = "ANSWERABLE"
        elif any(r["status"] == "NO_COMPANY_LEVEL_FACT" for r in entity_rows):
            case_status = "CONCEPT_ABSENT"
        else:
            case_status = "AMBIGUOUS"
        tally[case_status] += 1
        rows.append({"case_id": case_id, "canonical": canonical,
                     "case_status": case_status, "entities": entity_rows})

    checks = [r for row in rows for r in row["entities"] if r["expected"]]
    report = {
        "phase": "P1.6-A5",
        "mutation": "none",
        "entity_resolution": {
            k: v for k, v in tally.items()
            if k in ("RESOLVED", "AMBIGUOUS", "NO_COMPANY_LEVEL_FACT")
        },
        "case_status": {
            k: v for k, v in tally.items()
            if k in ("ANSWERABLE", "AMBIGUOUS", "CONCEPT_ABSENT")
        },
        "expected_value_checks": {
            "checked": len(checks),
            "matched": sum(1 for r in checks if r["matches_expected"]),
        },
        "remaining_ambiguities": disagreements,
        "cases": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "canonical-ambiguity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== entity-level resolution ===")
    for key in ("RESOLVED", "AMBIGUOUS", "NO_COMPANY_LEVEL_FACT"):
        print(f"  {key:24} {tally[key]}")
    print("\n=== case status ===")
    for key in ("ANSWERABLE", "AMBIGUOUS", "CONCEPT_ABSENT"):
        print(f"  {key:24} {tally[key]}")
    print(f"\n=== expected-value checks: "
          f"{report['expected_value_checks']['matched']}/"
          f"{report['expected_value_checks']['checked']} matched ===")
    print("\n=== per case ===")
    for row in rows:
        print(f"  {row['case_id']:26} {row['canonical'][:22]:22} {row['case_status']}")
        for e in row["entities"]:
            flag = "" if e["status"] == "RESOLVED" else "  <<<"
            print(f"      {e['entity'][:22]:22} {e['status']:24} "
                  f"{str(e['distinct_values'])[:34]}{flag}")
    if disagreements:
        print("\n=== remaining ambiguities ===")
        for case_id, entity, canonical, values in disagreements:
            print(f"  {case_id} / {entity} / {canonical}: {values}")
    print(f"\n  written to {args.out / 'canonical-ambiguity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
