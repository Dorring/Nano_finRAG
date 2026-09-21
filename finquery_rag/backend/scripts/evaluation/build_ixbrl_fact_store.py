"""P1.6-A7: rebuild the fact store from iXBRL.

The store being replaced files each fact under a flattened breadcrumb string
(`'Operating expenses: / Research and development'`), which is why a coordinate
holds several values and the operand guard has to refuse them.  The flattening
is a property of the table-parsing path, not of the filings: the filings state
their facts with a standard concept and an XBRL context whose dimensions name
the scope.

So each record here carries

    concept        us-gaap:NetIncomeLoss        the same string at every filer
    canonical      net_income                   via concept_alignment
    dimensions     [axis=member, ...]           [] means company level
    period, unit, value

`dimensions` being an explicit list is the whole point: `[]` is the company
figure and a non-empty list is a breakdown, so "the coordinate identifies one
value" becomes a property of the data rather than a hope.

**This writes a new file and does not touch the existing store.**  Rebuilding is
one delta and switching retrieval to the rebuilt store is another, and they are
kept separable so each can be measured on its own.

  python build_ixbrl_fact_store.py --apply
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

CORPUS = Path("/disk/qh/nano-finrag/data/financial_corpus_v2")
OUTPUT = Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts-ixbrl-v1.jsonl")

#: store document_id -> (ticker, accession).  The names are kept so the rebuilt
#: store lines up with the fixtures and the existing one.
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

# Contexts are not one scheme: `c-1` for most filers, `C_<uuid>` for the ones
# produced by DFIN ActiveDisclosure.
_CONTEXT = re.compile(
    r"<(?:xbrli:)?context[^>]*id=\"([^\"]+)\"[^>]*>(.*?)</(?:xbrli:)?context>", re.S
)
_MEMBER = re.compile(
    r"<xbrldi:explicitMember[^>]*dimension=\"([^\"]+)\"[^>]*>([^<]*)<", re.S
)


def _digits(text: object) -> str | None:
    """The value as a signed number string, or ``None``.

    Sign is preserved, and it is load-bearing.  A filing writes an outflow as
    `(2,385)` and XBRL does the same, so stripping non-digits turns Microsoft's
    interest expense into `2385` -- and `rank-004` ranks four filers by interest
    expense, where the difference between `-2,385` and `2,385` is the difference
    between last place and second.  `1,042` of the store's records carry a sign or
    parentheses; dropping them would have been invisible in every count and wrong
    in every ranking that includes a negative.
    """

    raw = str(text or "").strip()
    if not raw:
        return None
    negative = raw.startswith("(") or raw.lstrip().startswith("-")
    cleaned = re.sub(r"[^\d.]", "", raw)
    if not cleaned:
        return None
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    if not cleaned:
        return None
    return f"-{cleaned}" if negative else cleaned


def _contexts(html_path: Path) -> dict[str, dict]:
    if not html_path.is_file():
        return {}
    text = html_path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, dict] = {}
    for context_id, body in _CONTEXT.findall(text):
        members = [
            {"axis": axis.split(":")[-1], "member": member.strip().split(":")[-1]}
            for axis, member in _MEMBER.findall(body)
        ]
        out[context_id] = {"dimensions": members, "has_segment": bool(members)}
    return out


def _fiscal_period(context: dict, report_period_end: str) -> str:
    """The fiscal label for a fact's context.

    An **instant** context has no duration -- it is a balance-sheet date -- and
    its period end is the fiscal year end itself.  Labelling those `ASOF2025-09-27`
    while the income statement's facts are `FY2025` puts one company's balance
    sheet and its income statement under different periods, so a query for
    `FY2025` finds the income statement and misses the balance sheet.  When the
    instant coincides with the document's reporting date it is that fiscal year.
    """

    semantics = str(context.get("period_semantics") or "").upper()
    end = str(context.get("period_end") or "")
    year = end[:4]
    if not year:
        return "UNKNOWN"
    is_instant = not context.get("period_start")
    if is_instant:
        if report_period_end and end == report_period_end:
            return f"FY{year}"
        return f"ASOF{end}"
    if semantics == "ANNUAL" or context.get("duration_days") in (364, 365, 366):
        return f"FY{year}"
    if semantics in ("QUARTER", "QUARTERLY"):
        month = end[5:7]
        quarter = {"03": "Q1", "06": "Q2", "09": "Q3", "12": "Q4"}.get(month, "Q?")
        return f"FY{year}{quarter}"
    return f"FY{year}" if end[5:7] == "12" else f"ASOF{end}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--corpus-manifest", type=Path,
                        default=Path("/disk/qh/nano-finrag/data/raw_pdfs/corpus-manifest.json"))
    args = parser.parse_args(argv)

    from src.finance.concept_alignment import align_concept

    # entity comes from the corpus manifest rather than being invented here.
    company_of: dict[str, str] = {}
    if args.corpus_manifest.is_file():
        manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
        for document in manifest.get("documents") or ():
            name = str(document.get("document_id") or "")
            if name:
                company_of[name] = str(document.get("company") or name)

    records: list[dict] = []
    stats: collections.Counter = collections.Counter()
    per_document: dict[str, int] = {}

    for document_name, (ticker, accession) in sorted(DOCUMENTS.items()):
        root = CORPUS / "normalized/SEC" / ticker / accession
        if not (root / "document.json").is_file():
            stats["document_missing"] += 1
            continue
        document = json.loads((root / "document.json").read_text(encoding="utf-8"))
        meta = document.get("document") or {}
        report_period_end = str(meta.get("report_period_end") or "")
        contexts = _contexts(CORPUS / "raw/SEC" / ticker / accession / "primary.html")
        company = company_of.get(document_name) or str(meta.get("company") or ticker)
        emitted = 0

        for fact in document.get("ixbrl_facts") or ():
            value = _digits(fact.get("raw_value"))
            if value is None:
                stats["skipped_no_value"] += 1
                continue
            concept = str(fact.get("concept") or "").strip()
            if not concept:
                stats["skipped_no_concept"] += 1
                continue
            context_ref = str(fact.get("context_ref") or "")
            context = fact.get("context") or {}
            known = contexts.get(context_ref)
            # An unknown context is treated as *dimensioned*, never as company
            # level: the opposite assumption makes a breakdown indistinguishable
            # from a total, which is the defect being repaired.
            dimensions = (known or {}).get("dimensions", [])
            unknown_context = known is None
            if unknown_context:
                stats["unknown_context"] += 1

            alignment = align_concept(concept)
            digest = hashlib.sha256(
                f"{document_name}|{concept}|{context_ref}|{fact.get('fact_id')}".encode()
            ).hexdigest()[:32]
            records.append({
                "candidate_key": f"ixbrl:{digest}",
                "fact_id": f"ixbrl:{digest}",
                "provenance_complete": True,
                "source": "ixbrl",
                "document_name": document_name,
                "document_id": str(meta.get("document_id") or accession),
                "ticker": ticker,
                "entity": company,
                "concept": concept,
                "canonical_concept": alignment.canonical,
                "dimensions": dimensions,
                "dimension_count": len(dimensions),
                "unknown_context": unknown_context,
                "period": _fiscal_period(context, report_period_end),
                "period_start": context.get("period_start"),
                "period_end": context.get("period_end"),
                "unit": fact.get("unit"),
                "value": value,
                "raw_value": fact.get("raw_value"),
                "context_ref": context_ref,
                "ixbrl_fact_id": fact.get("fact_id"),
            })
            emitted += 1
            if alignment.canonical:
                stats["aligned_to_canonical"] += 1
            if not dimensions and not unknown_context:
                stats["company_level"] += 1
            if dimensions:
                stats["dimensioned"] += 1
        per_document[document_name] = emitted

    by_canonical = collections.Counter(
        r["canonical_concept"] for r in records if r["canonical_concept"]
    )
    concepts = {r["concept"] for r in records}
    text = "\n".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records
    ) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    print(f"=== iXBRL fact store {'WRITTEN' if args.apply else 'DRY RUN'} ===")
    print(f"  documents        {len(per_document)} / {len(DOCUMENTS)}")
    for name, count in sorted(per_document.items()):
        print(f"      {name:14} {count:>6} facts")
    print(f"  records          {len(records)}")
    print(f"  distinct concepts {len(concepts)}")
    print(f"  company level    {stats['company_level']}")
    print(f"  dimensioned      {stats['dimensioned']}")
    print(f"  unknown context  {stats['unknown_context']}")
    print(f"  aligned          {stats['aligned_to_canonical']}")
    print(f"  sha256           {digest[:16]}")
    print()
    print("  aligned records per canonical quantity:")
    for canonical, count in by_canonical.most_common():
        print(f"      {canonical:28} {count}")

    if args.apply:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        (args.output.parent / (args.output.name + ".sha256")).write_text(
            digest + "\n", encoding="utf-8"
        )
        (args.output.parent / (args.output.name + ".manifest.json")).write_text(
            json.dumps({
                "source": "SEC EDGAR primary.html iXBRL, eight filings",
                "documents": {k: v for k, v in sorted(per_document.items())},
                "records": len(records),
                "distinct_concepts": len(concepts),
                "company_level": stats["company_level"],
                "dimensioned": stats["dimensioned"],
                "aligned_to_canonical": stats["aligned_to_canonical"],
                "sha256": digest,
            }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\n  written to {args.output}")
    else:
        print("\n  (dry run -- pass --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
