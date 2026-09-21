"""P1.6-0F: does every entity in a cross-entity case carry the same quantity?

The P1.6-0D verification classified only the slots P1.6-0C had flagged, and a
comparison is coherent only when *every* entity is right.  Two cases were already
found broken in the half nobody looked at (`compare-010`'s Apple is a component;
`rank-002`'s Apple carries a sign that flips the expected ranking), and one
verdict was withdrawn outright (`compare-001`, where both golds are the same
jurisdiction figure and the gold is correct).

So this walks all 20 cross-entity cases and **every entity in each**, not the
flagged subset, and prints for each value the evidence a reader needs to judge
its scope: the row label the value sits under, and the text block around it.

It classifies nothing.  The rule-based classifiers tried earlier in this phase
produced both a false `VALID_GOLD` (a jurisdiction sub-total certified as the
company figure) and a false ungrounded release (display units read as ratios),
and a wrong column name is worse than an admitted gap because it looks
confident.  The output is a workbook to read.

  python audit_cross_entity_coherence.py --out <dir>
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

STRATUM_PREFIX = "tv2f01-s3-"
_NUMBER = re.compile(r"\(?\$?\s*-?[\d][\d,\.]*\)?")


def _numeric_key(text: object) -> str | None:
    match = _NUMBER.search(str(text or ""))
    if not match:
        return None
    digits = re.sub(r"[^\d]", "", match.group(0))
    return digits or None


def _blocks(page) -> list[dict]:
    """Text blocks with their boxes, in reading order."""

    out = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        text = " ".join(
            span["text"] for line in block["lines"] for span in line["spans"]
        )
        text = " ".join(text.split())
        if text:
            out.append({"text": text, "bbox": block["bbox"]})
    return out


def _locate(document, value: object, window: int = 40) -> dict:
    """Where the value sits, and what the text around it says.

    The row label is taken from the value's own block when the block carries one
    -- filings put "Net income  $4,520 …" on one line -- and from the block above
    when the block is numbers only, which is how a segment table lays out.
    """

    target = _numeric_key(value)
    if target is None:
        return {"found": False}

    for page_index in range(document.page_count):
        page = document[page_index]
        spans = page.get_text("words")
        anchors = [w for w in spans if _numeric_key(w[4]) == target]
        if not anchors:
            continue
        blocks = _blocks(page)

        def block_of(y: float) -> int | None:
            for index, block in enumerate(blocks):
                if block["bbox"][1] - 1 <= y <= block["bbox"][3] + 1:
                    return index
            return None

        anchor = anchors[0]
        index = block_of(anchor[1])
        context = []
        if index is not None:
            if index > 0:
                context.append(blocks[index - 1]["text"][:160])
            context.append(blocks[index]["text"][:200])
        # The heading above the table, which names the concept being broken down.
        heading = ""
        if index is not None:
            for back in range(index - 1, max(-1, index - 6), -1):
                candidate = blocks[back]["text"]
                if len(candidate) > 70 and candidate.endswith(":"):
                    heading = candidate[:200]
                    break

        return {
            "found": True,
            "pdf_page": page_index + 1,
            "context": context,
            "heading": heading,
        }
    return {"found": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, default=_BACKEND_DIR)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--fact-store", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import os

    import fitz

    from src.runtime.trusted_v2_production import StructuredFactStore

    store_path = args.fact_store or Path(os.environ["TRUSTED_V2_FACT_STORE_PATH"])
    store = StructuredFactStore(store_path)

    # entity -> filing, taken from the store rather than hand-written.
    entity_document: dict[str, str] = {}
    for record in store.iter_records():
        entity, document = record.get("entity"), record.get("document_name")
        if entity and document:
            entity_document.setdefault(str(entity), str(document))

    manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    filename = {
        str(d.get("document_id")): d.get("local_filename")
        for d in manifest.get("documents") or ()
    }

    base = args.backend / "benchmarks/tv2_canonical_v1"
    gold = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (base / "gold-evidence-v1.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        if row["id"].startswith(STRATUM_PREFIX)
    }
    questions = {
        row["id"]: row.get("question")
        for row in (
            json.loads(line)
            for line in (base / "canonical-eval-v1.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }

    cases: list[dict] = []
    tally: collections.Counter = collections.Counter()

    for case_id in sorted(gold):
        row = gold[case_id]
        entities: list[dict] = []
        for entity, value in (row.get("values") or {}).items():
            document_id = entity_document.get(str(entity))
            path = args.pdf_dir / str(filename.get(document_id))
            entry = {"entity": entity, "value": value, "document_id": document_id}
            if not path.is_file():
                entry["found"] = False
                entry["note"] = "filing not found"
            else:
                with fitz.open(path) as document:
                    entry.update(_locate(document, value))
            entry["found"] = bool(entry.get("found"))
            tally["found" if entry["found"] else "not_found"] += 1
            entities.append(entry)

        cases.append(
            {
                "case_id": case_id,
                "question": questions.get(case_id),
                "metric": row.get("metric"),
                "operation": row.get("operation"),
                "expected_ranking": row.get("expected_ranking"),
                "expected_higher": row.get("expected_higher"),
                "expected_value": row.get("expected_value"),
                "entities": entities,
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cross-entity-coherence.json").write_text(
        json.dumps(
            {"phase": "P1.6-0F", "cases": cases, "tally": dict(tally)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for case in cases:
        print(f"=== {case['case_id']}   metric={case['metric']!r}")
        print(f"    Q: {case['question']}")
        for entity in case["entities"]:
            page = entity.get("pdf_page")
            print(f"    - {str(entity['entity'])[:22]:22} {str(entity['value']):12} p{page}")
            if entity.get("heading"):
                print(f"        heading: {entity['heading'][:150]}")
            for line in entity.get("context") or ():
                print(f"        ctx: {line[:165]}")
        print()
    print(f"located {tally.get('found', 0)} / "
          f"{tally.get('found', 0) + tally.get('not_found', 0)} values")
    print(f"written to {args.out / 'cross-entity-coherence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
