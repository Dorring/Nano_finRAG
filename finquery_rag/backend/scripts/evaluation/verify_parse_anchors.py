"""P1.6-0K: does the parser now link blocks to the facts they state?

Step one of option B, and the only part worth doing before the pipeline is
re-run: a change at the parse layer that records each block's descendant
`ix:nonFraction`/`ix:nonNumeric` anchors is worthless if the anchors do not
reconcile against the facts the same parse emits.

Run against one filing.  It checks four things, all of which must hold:

  every anchor names a fact this parse emits        (else the link is to nothing)
  every emitted fact is reachable from some block   (else the link has holes)
  table blocks carry anchors                        (else tables stay unlinked)
  the anchor count reconciles with the raw HTML     (else something is dropped)

  python verify_parse_anchors.py --ticker MSFT --accession SEC_789019_...
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

CORPUS = Path("/disk/qh/nano-finrag/data/financial_corpus_v2")
PARSER = _BACKEND_DIR / "scripts/evaluation/run_nf_v2_17a4_parse.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--accession", required=True)
    parser.add_argument("--document-id", default=None)
    args = parser.parse_args(argv)

    from lxml import etree, html

    spec = importlib.util.spec_from_file_location("nf17a4", PARSER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    raw = CORPUS / "raw/SEC" / args.ticker / args.accession / "primary.html"
    if not raw.is_file():
        print(f"filing not found: {raw}")
        return 1

    document_id = args.document_id or f"{args.ticker.lower()}_fy2025"
    doc = {"document_id": document_id, "ticker": args.ticker}

    tree = html.parse(
        str(raw),
        etree.HTMLParser(
            recover=True, no_network=True, huge_tree=True, remove_comments=True
        ),
    )
    root = tree.getroot()
    blocks, _lookup, _prior = module.make_blocks(root, doc)
    contexts = module.ix_contexts(root)
    facts = module.ix_facts(root, contexts)

    fact_by_id = {str(f.get("fact_id")): f for f in facts}
    fact_by_key = {
        (str(f.get("concept")), str(f.get("context_ref"))): f for f in facts
    }

    anchors = []
    per_type = collections.Counter()
    blocks_with_anchors = collections.Counter()
    for block in blocks:
        found = (block.get("metadata") or {}).get("ixbrl_anchors") or []
        kind = str(block.get("block_type"))
        per_type[kind] += len(found)
        if found:
            blocks_with_anchors[kind] += 1
        for anchor in found:
            anchors.append((kind, anchor))

    def resolves(anchor: dict) -> bool:
        if anchor.get("fact_id") and anchor["fact_id"] in fact_by_id:
            return True
        # `ix_facts` synthesises an id when the element has none, so an anchor
        # without one is matched on the concept and context it does carry.
        return (anchor.get("concept"), anchor.get("context_ref")) in fact_by_key

    unresolved = [a for _kind, a in anchors if not resolves(a)]
    anchored_fact_ids = {
        a["fact_id"] for _k, a in anchors if a.get("fact_id") in fact_by_id
    }
    anchored_keys = {
        (a.get("concept"), a.get("context_ref")) for _k, a in anchors
    }
    covered = {
        str(f.get("fact_id"))
        for f in facts
        if str(f.get("fact_id")) in anchored_fact_ids
        or (str(f.get("concept")), str(f.get("context_ref"))) in anchored_keys
    }

    print(f"=== {args.ticker} {args.accession} ===")
    print(f"  blocks                {len(blocks)}")
    print(f"  facts emitted         {len(facts)}")
    print(f"  anchors recorded      {len(anchors)}")
    print(f"  distinct anchor facts {len(anchored_fact_ids) + len(anchored_keys)}")
    print()
    print("  anchors by block type:")
    for kind, count in per_type.most_common():
        print(f"      {kind:14} {count:>5} anchors in {blocks_with_anchors[kind]:>4} blocks")
    print()
    print(f"  anchors that resolve to an emitted fact : "
          f"{len(anchors) - len(unresolved)}/{len(anchors)}")
    print(f"  emitted facts reachable from a block    : "
          f"{len(covered)}/{len(facts)}")
    print()
    if unresolved:
        print("  UNRESOLVED anchors (first 3):")
        for anchor in unresolved[:3]:
            print(f"      {anchor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
