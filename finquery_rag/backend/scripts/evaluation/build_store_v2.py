"""P1.6-A2B-9: build a Canonical Fact Store V2 for one filing, and check it.

The vertical slice. One filing, built by a producer that lives in this repository,
from a source it can point at, into a store whose structural fields survive and
whose audited facts come back the same.

What V2 must satisfy is **not** "resembles the legacy store".  The legacy artifact
has no producer here, no parsed source and no reproducible corpus, so it cannot
define correctness; its record count is explicitly not a target.  What V2 must
satisfy is:

  producer in this repository         the parse and the build are both here
  source locatable                    every record names its filing and cell
  structural fields not lost          column_header, row_label, ids, anchors
  audited facts rebuild correctly     checked against values read from the filing
  same source + same code -> same bytes

  python build_store_v2.py --ticker JPM --accession <acc> --apply
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

CORPUS = Path("/disk/qh/nano-finrag/data/financial_corpus_v2")
PARSER = _BACKEND_DIR / "scripts/evaluation/run_nf_v2_17a4_parse.py"

#: Values read out of JPMorganChase's filing by hand during P1.6-A2B-7, with the
#: column header each sits under. The acceptance set: the slice must reproduce
#: these, and the point of using them is that they came from the filing.
#:
#: Matched on `value_raw` -- the filing's own string -- not on `value`, which is
#: the normalized number and would have made every check fail for a formatting
#: difference rather than a missing fact. The period is matched by year, because
#: a balance-sheet instant carries a date and an income-statement row a year.
AUDITED = (
    ("Net income", "57,048", "Total", "2025"),
    ("Net income", "58,471", "Total", "2024"),
    ("Net income", "49,552", "Total", "2023"),
    ("Net income", "4,520", "Corporate", "2025"),
    ("Net income", "10,601", "Corporate", "2024"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_filing(ticker: str, accession: str, document_id: str) -> dict:
    """The parsed document, in the shape the builder consumes."""

    from lxml import etree, html

    module_spec = importlib.util.spec_from_file_location("nf17a4", PARSER)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    raw = CORPUS / "raw/SEC" / ticker / accession / "primary.html"
    root = html.parse(
        str(raw),
        etree.HTMLParser(recover=True, no_network=True, huge_tree=True,
                         remove_comments=True),
    ).getroot()
    doc = {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"}

    blocks, lookup, prior = module.make_blocks(root, doc)
    contexts = module.ix_contexts(root)
    facts = module.ix_facts(root, contexts)

    tables = []
    for block in blocks:
        if block["block_type"] != "TABLE":
            continue
        preceding = [text for order, text in prior if order < block["source_order"]]
        prior_text = ""
        for text in reversed(preceding):
            if 0 < len(text) <= 260:
                prior_text = module.ws(text)
                break
        tables.append(
            module.parse_table(
                lookup[block["table_id"]], block["table_id"], doc,
                block["section_type"], block["source_order"], prior_text,
            )
        )

    return {
        "schema_version": "NormalizedFinancialDocumentV2",
        "document": doc,
        "blocks": blocks,
        "tables": tables,
        "ixbrl_context_count": len(contexts),
        "ixbrl_fact_count": len(facts),
        "ixbrl_facts": facts,
    }


def to_v2(records: list[dict], source: dict) -> list[dict]:
    """Reshape builder records into the V2 contract.

    Explicit rather than pass-through: the point of V2 is that the structural
    fields are *named in the schema*, so a later stage cannot drop one silently.
    A field that is absent here is absent on purpose and visible as such.
    """

    out = []
    for record in records:
        anchors = record.get("ixbrl_anchors") or []
        out.append({
            "fact_id": f"v2:{record['candidate_key']}",
            "entity": record.get("entity"),
            "metric": record.get("metric"),
            "period": record.get("period"),
            "period_start": record.get("period_start"),
            "period_end": record.get("period_end"),
            "value": record.get("value"),
            "value_raw": record.get("raw_value"),
            "unit": record.get("unit"),
            "scale": record.get("scale"),
            "currency": record.get("currency"),
            # The structure the legacy store did not carry.
            "table_fragment_id": record.get("table_fragment_id"),
            "row_id": record.get("row_id"),
            "cell_id": record.get("cell_id"),
            "row_label": record.get("row_label"),
            "column_header": record.get("column_header"),
            # The filing's own tags, where the filer supplied them.  Absence
            # means the cell was untagged, never that it is not a fact.
            "source_anchors": [
                {"fact_id": a.get("fact_id"), "concept": a.get("concept"),
                 "context_ref": a.get("context_ref")}
                for a in anchors
            ],
            "source": {
                "document_id": source["document_id"],
                "accession": source["accession"],
                "primary_html_sha256": source["primary_html_sha256"],
                "producer": "scripts/evaluation/run_nf_v2_17a4_parse.py"
                            " + src/runtime/trusted_v2_canonical_fact_store.py",
            },
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="JPM")
    parser.add_argument("--accession", default="SEC_19617_000162828026008131")
    parser.add_argument("--document-id", default="jpm_fy2025")
    parser.add_argument("--entity", default="JPMorganChase")
    parser.add_argument("--out", type=Path,
                        default=Path("/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a2b9-store-v2"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    from src.runtime.trusted_v2_canonical_fact_store import build_canonical_fact_store

    raw_path = CORPUS / "raw/SEC" / args.ticker / args.accession / "primary.html"
    source = {
        "document_id": args.document_id,
        "accession": args.accession,
        "primary_html_sha256": _sha256(raw_path),
    }

    document = parse_filing(args.ticker, args.accession, args.document_id)
    records, _summary = build_canonical_fact_store([document])
    v2 = to_v2(records, source)

    text = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in v2) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Second build, same inputs, to show the producer is deterministic.
    rebuild = to_v2(build_canonical_fact_store(
        [parse_filing(args.ticker, args.accession, args.document_id)]
    )[0], source)
    rebuild_text = "\n".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rebuild
    ) + "\n"

    structured = sum(1 for r in v2 if r["column_header"] and r["row_id"] and r["cell_id"])
    anchored = sum(1 for r in v2 if r["source_anchors"])

    print(f"=== Canonical Fact Store V2 — {args.document_id} ===")
    print(f"  source        {args.accession}")
    print(f"  source sha256 {source['primary_html_sha256'][:16]}")
    print(f"  records       {len(v2)}")
    print(f"  with column_header + row_id + cell_id   {structured} / {len(v2)}")
    print(f"  with source anchors                     {anchored} / {len(v2)}")
    print(f"  sha256        {digest[:16]}")
    print(f"  reproducible  {text == rebuild_text}")
    print()

    print("  audited facts rebuilt from the filing's own row:")
    misses = []
    for metric, value, column, year in AUDITED:
        hits = [
            r for r in v2
            # `row_label` is the row as the filing labels it; `metric` is the
            # breadcrumb path, which for this filing is a different string.
            if metric in str(r["row_label"])
            and str(r["value_raw"]) == value
            and column in str(r["column_header"])
            and str(r["period_end"])[:4] == year
        ]
        mark = "ok " if hits else "MISS"
        if not hits:
            misses.append((metric, value, column, year))
        print(f"    {mark} {metric:12} {value:>9}  under {column:10} at {year}")
    print()

    if args.apply:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "store-v2.jsonl").write_text(text, encoding="utf-8")
        (args.out / "store-v2.manifest.json").write_text(
            json.dumps({
                "phase": "P1.6-A2B-9",
                "scope": "one filing; not a benchmark migration",
                "source": source,
                "records": len(v2),
                "structured_records": structured,
                "anchored_records": anchored,
                "sha256": digest,
                "deterministic": text == rebuild_text,
                "audited_checks": len(AUDITED),
                "audited_misses": misses,
                "not_the_legacy_store": (
                    "record count is not compared to financial-facts.jsonl; that "
                    "artifact has no producer, parsed source or reproducible corpus "
                    "here and cannot define correctness"
                ),
            }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"  written to {args.out / 'store-v2.jsonl'}")
    else:
        print("  (dry run -- pass --apply to write)")
    return 0 if not misses else 1


if __name__ == "__main__":
    raise SystemExit(main())
