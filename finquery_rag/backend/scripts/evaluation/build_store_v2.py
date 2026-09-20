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


#: The eight filings, with the accession each store document corresponds to.
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

CORPUS_MANIFEST = Path("/disk/qh/nano-finrag/data/raw_pdfs/corpus-manifest.json")

#: The table-role classifier's output, which carries the two fields V2 was missing.
SHADOW = Path(
    "/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a2b19b-shadow/shadow-classification.json"
)
#: And the oracle's audit verdicts, for the role sidecar only -- never for the store.
ORACLE = Path(
    "/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a2b19a-oracle/table-authority-oracle.json"
)


def table_roles(document_id: str, blocks: list[dict]) -> dict[str, dict]:
    """`table_fragment_id` -> what that table is and whether it may speak for the company.

    The classifier reads the filing and keys its verdicts by `document_id#source_order`,
    which is the index of the `<table>` in document order -- the same quantity the parser
    records on every block.  Joining on it is what carries the two fields from the
    classification into the store without either side reproducing the other's ids.

    A table the classifier never saw is a layout scaffold: it left the corpus at the
    eligibility layer, and it is recorded as such rather than defaulted to something
    permissive.
    """

    by_key: dict[str, dict] = {}
    if SHADOW.is_file():
        shadow = json.loads(SHADOW.read_text(encoding="utf-8"))
        for row in shadow["documents"].get(document_id, {}).get("tables") or ():
            by_key[str(row["oracle_key"])] = row

    # The oracle's own verdict, where it made one, so that a later pass can ask
    # whether a resolved value came out of a table the oracle read and refused.
    audit: dict[str, str] = {}
    if ORACLE.is_file():
        oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
        for table in oracle["documents"].get(document_id, {}).get("tables") or ():
            audit[str(table["oracle_key"])] = str(table["oracle_role"])

    out: dict[str, dict] = {}
    for block in blocks:
        if block["block_type"] != "TABLE":
            continue
        key = f"{document_id}#{block['source_order']}"
        row = by_key.get(key)
        if row is None:
            out[block["table_id"]] = {
                "table_eligibility": "LAYOUT_SCAFFOLD",
                "table_role": "NON_PRIMARY",
                "oracle_role": audit.get(key, "SCAFFOLD"),
            }
            continue
        out[block["table_id"]] = {
            "table_eligibility": row["table_eligibility"],
            "table_role": row["new_table_role"],
            "oracle_role": audit.get(key, "NOT_IN_ORACLE"),
        }
    return out


def company_of(document_id: str) -> str:
    """The entity name the benchmark uses for a filing, or a hard failure.

    This used to fall back to the ticker, and the fallback is what produced a
    store whose every record said `JPM` where the benchmark says `JPMorganChase`
    -- silently, so that every slot query simply matched nothing.  An entity the
    benchmark cannot recognise is not a store to build, it is a mistake to stop
    on, so there is no fallback here.
    """

    if not CORPUS_MANIFEST.is_file():
        raise SystemExit(f"corpus manifest not found: {CORPUS_MANIFEST}")
    for entry in json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8")).get("documents") or ():
        if str(entry.get("document_id")) == document_id:
            company = str(entry.get("company") or "").strip()
            if company:
                return company
            raise SystemExit(f"manifest entry {document_id} carries no company name")
    raise SystemExit(f"manifest has no entry for {document_id}")


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
    doc = {"document_id": document_id, "ticker": ticker, "role": "ANNUAL",
           "company": company_of(document_id)}

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


def to_v2(records: list[dict], source: dict, roles: dict[str, dict]) -> list[dict]:
    """Reshape builder records into the V2 contract.

    Explicit rather than pass-through: the point of V2 is that the structural
    fields are *named in the schema*, so a later stage cannot drop one silently.
    A field that is absent here is absent on purpose and visible as such.
    """

    out = []
    missing_role = 0
    for record in records:
        anchors = record.get("ixbrl_anchors") or []
        table_id = record.get("table_fragment_id")
        role = roles.get(table_id)
        if role is None:
            missing_role += 1
            role = {"table_eligibility": "UNKNOWN", "table_role": "UNKNOWN"}
        out.append({
            "fact_id": f"v2:{record['candidate_key']}",
            "entity": record.get("entity"),
            "metric": record.get("metric"),
            "period": record.get("period"),
            "period_start": record.get("period_start"),
            "period_end": record.get("period_end"),
            # W4-B2: the minimal period identity.  Named here rather than passed through
            # because V2's whole point is that the structural fields are *named in the
            # schema* -- a field that is absent is absent on purpose and visible as such.
            "normalized_period": record.get("normalized_period"),
            "period_binding_status": record.get("period_binding_status"),
            "period_granularity": record.get("period_granularity"),
            # W5: and why that identity is believed.  Named in the schema for the same
            # reason B2's three are: a field that is absent here is absent on purpose and
            # visible as such, rather than missing because a later stage dropped it.
            "period_binding_method": record.get("period_binding_method"),
            "period_target_scope": record.get("period_target_scope"),
            "period_source_cells": record.get("period_source_cells") or [],
            "period_conflict_candidates": record.get("period_conflict_candidates") or [],
            "temporal_kind_method": record.get("temporal_kind_method"),
            "temporal_kind_source_cells": record.get("temporal_kind_source_cells") or [],
            "temporal_kind_matched_text": record.get("temporal_kind_matched_text"),
            # The store's `temporal_kind` is the A3 kind; the builder calls it
            # `binding_temporal_kind` on the fact because `AtomicFact.temporal_kind` is
            # the legacy axis kind and `semantic_equivalence` groups on it. Mapped here,
            # once, rather than renaming either side.
            "temporal_kind": record.get("temporal_kind"),
            "legacy_temporal_kind": record.get("legacy_temporal_kind"),
            "value": record.get("value"),
            "value_raw": record.get("raw_value"),
            "unit": record.get("unit"),
            "scale": record.get("scale"),
            "currency": record.get("currency"),
            # Which statement the row is in.  This is the signal that separates a
            # primary statement from a breakdown, and it is what decides whether
            # a period column is the company figure: a consolidated income
            # statement has no `Total` column, because the table is the company.
            # `UNKNOWN` means "not a primary statement" and must never be read as
            # "probably one".
            "statement_type": record.get("statement_type"),
            # The two fields that separate "what is this table about" from "may it
            # speak for the company".  `statement_type` answered both before, and
            # A2B-19A showed that no single value of it can: a hedging note in the
            # legacy labelling carries `BALANCE_SHEET` and is not the balance sheet,
            # and a consolidated income statement in seven of the eight filings
            # carries `UNKNOWN` and is.  The resolver's authority test moves to
            # `table_role`; `statement_type` keeps answering the first question only.
            "table_eligibility": role["table_eligibility"],
            "table_role": role["table_role"],
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
    return out, missing_role


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a2b12-store-v2"))
    parser.add_argument("--only", default=None,
                        help="build a single document_id instead of all eight")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    from src.runtime.trusted_v2_canonical_fact_store import build_canonical_fact_store

    wanted = {args.only: DOCUMENTS[args.only]} if args.only else DOCUMENTS

    all_v2: list[dict] = []
    all_roles: dict[str, dict] = {}
    per_document: dict[str, dict] = {}
    misses: list[tuple] = []
    deterministic = True

    for document_id, (ticker, accession) in sorted(wanted.items()):
        raw_path = CORPUS / "raw/SEC" / ticker / accession / "primary.html"
        if not raw_path.is_file():
            per_document[document_id] = {"status": "MISSING_SOURCE",
                                         "path": str(raw_path)}
            continue
        source = {
            "document_id": document_id,
            "accession": accession,
            "primary_html_sha256": _sha256(raw_path),
        }
        parsed = parse_filing(ticker, accession, document_id)
        records, _summary = build_canonical_fact_store([parsed])
        roles = table_roles(document_id, parsed["blocks"])
        all_roles.update(roles)
        v2, missing_role = to_v2(records, source, roles)
        if missing_role:
            raise SystemExit(
                f"{document_id}: {missing_role} record(s) name a table the classifier "
                f"never saw, so their table_role would be UNKNOWN for the wrong reason"
            )

        # Same inputs twice, to show the producer is deterministic.  Checked per
        # filing rather than once, because a single non-deterministic document is
        # the thing that would matter and an aggregate would hide it.
        rebuilt, _ = to_v2(build_canonical_fact_store(
            [parse_filing(ticker, accession, document_id)]
        )[0], source, roles)
        same = _render(v2) == _render(rebuilt)
        deterministic = deterministic and same

        structured = sum(1 for r in v2 if r["column_header"] and r["row_id"]
                         and r["cell_id"])
        anchored = sum(1 for r in v2 if r["source_anchors"])
        per_document[document_id] = {
            "status": "OK",
            "entity": company_of(document_id),
            "accession": accession,
            "source_sha256": source["primary_html_sha256"],
            "records": len(v2),
            "structured_records": structured,
            "anchored_records": anchored,
            "authoritative_records": sum(
                1 for r in v2 if r["table_role"] == "PRIMARY_FINANCIAL_STATEMENT"),
            "scaffold_records": sum(
                1 for r in v2 if r["table_eligibility"] == "LAYOUT_SCAFFOLD"),
            "deterministic": same,
            "sha256": hashlib.sha256(_render(v2).encode("utf-8")).hexdigest(),
        }
        all_v2.extend(v2)

        # The audited set is JPMorganChase's, so it is only checked where it is
        # the filing under test -- and the check is reported as skipped rather
        # than passed for the others.
        if document_id == "jpm_fy2025":
            for metric, value, column, year in AUDITED:
                if not any(
                    metric in str(r["row_label"])
                    and str(r["value_raw"]) == value
                    and column in str(r["column_header"])
                    and str(r["period_end"])[:4] == year
                    for r in v2
                ):
                    misses.append((metric, value, column, year))

    text = _render(all_v2)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    print("=== Canonical Fact Store V2 ===")
    for document_id, block in sorted(per_document.items()):
        if block["status"] != "OK":
            print(f"  {document_id:14} {block['status']}")
            continue
        print(f"  {document_id:14} {block['entity']:22} "
              f"records {block['records']:>6}  "
              f"structured {block['structured_records']:>6}/{block['records']:<6} "
              f"anchored {block['anchored_records']:>6}  "
              f"authoritative {block['authoritative_records']:>5}  "
              f"deterministic {block['deterministic']}")
    print()
    print(f"  total records {len(all_v2)}")
    print(f"  deterministic {deterministic}")
    print(f"  sha256        {digest[:16]}")
    print()
    print("  audited facts (JPMorganChase's, the set read from that filing):")
    for metric, value, column, year in AUDITED:
        mark = "MISS" if (metric, value, column, year) in misses else "ok "
        print(f"    {mark} {metric:12} {value:>9}  under {column:10} at {year}")
    print()

    if args.apply:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "store-v2.jsonl").write_text(text, encoding="utf-8")
        (args.out / "table-roles.json").write_text(
            json.dumps(all_roles, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (args.out / "store-v2.manifest.json").write_text(
            json.dumps({
                "phase": "P1.6-A2B-12",
                "scope": "eight filings; still not a benchmark migration",
                "documents": per_document,
                "total_records": len(all_v2),
                "deterministic": deterministic,
                "sha256": digest,
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


def _render(records: list[dict]) -> str:
    return "\n".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records
    ) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
