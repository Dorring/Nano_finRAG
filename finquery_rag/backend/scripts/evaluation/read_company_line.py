"""Read one company-level line out of a filing, by its label.

The re-derivation cannot take values from the fact store.  The store's
`(entity, metric, period)` coordinate is exactly what P1.6-A has to repair, and
the coherence pass showed the damage: `rank-001`'s coordinate is *unique* for all
three entities and all three values are the wrong line.  A candidate picked from
the store therefore inherits the extraction defect it is meant to avoid.

So this finds a line by its **label** in the filing and reads the number beside
it.  It is deliberately literal: it looks for the label text, takes the first
number after it in the same block, and reports the page.  It refuses rather than
guesses -- a label that is absent, or present without an adjacent number, comes
back as not found with the block it did see.

  python read_company_line.py --pdf-dir ... --corpus-manifest ... \
      --request requests.json --out <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# A number as filings print them: `$ 112,010`, `(34,550)`, `7.46`, `15.6%`.
_NUMBER = re.compile(r"\(?\$?\s*-?\d[\d,]*(?:\.\d+)?\s*%?\)?")

#: Headings that mark a *primary* statement.  A label is only read from a page
#: carrying one, which is what makes the read reliable: `Net income` appears in
#: prose, in notes and in segment tables as well as on the income statement, and
#: a search that accepts any of them returns the wrong number as often as not.
#: The first version of this accepted any page and reported Tesla's net income
#: as `$3.79` -- the diluted EPS from the row beside it.
_STATEMENT_MARKERS = (
    "consolidated statements of operations",
    "consolidated statements of income",
    "consolidated statements of earnings",
    "consolidated statements of comprehensive income",
    "consolidated balance sheet",
    "consolidated statements of financial position",
    "statements of operations",
    "statements of income",
    "statements of earnings",
    "statements of comprehensive income",
    "balance sheets",
    "statements of financial position",
)


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def _is_statement_page(text: str) -> bool:
    folded = text.casefold()
    return any(marker in folded for marker in _STATEMENT_MARKERS)


def _first_number_after(block_text: str, label: str) -> str | None:
    """The first number following the label when the label *leads* the block.

    A statement row reads `Net income $ 112,010 $ 93,736 $ 96,995`: the label
    opens the block and the numbers follow, current year first.  Requiring the
    label to lead is what separates that from prose that merely mentions the
    term somewhere and has an unrelated figure further along.
    """

    folded = block_text.casefold()
    index = folded.find(label.casefold())
    if index < 0 or index > 12:
        return None
    tail = block_text[index + len(label):]
    match = _NUMBER.search(tail)
    return match.group(0).strip() if match else None


def read_line(document, label: str) -> dict:
    """Read the label's line from a primary statement, or refuse.

    Reports the first match found in page order, and the block it came from, so
    a reader can check the row rather than trust the number.
    """

    for page_index in range(document.page_count):
        page = document[page_index]
        page_text = page.get_text()
        if not _is_statement_page(page_text):
            continue
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            text = _clean(
                " ".join(
                    span["text"] for line in block["lines"] for span in line["spans"]
                )
            )
            if not text:
                continue
            value = _first_number_after(text, label)
            if value is None:
                continue
            return {"found": True, "value": value, "pdf_page": page_index + 1,
                    "block": text[:200]}
    return {"found": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True,
                        help='JSON: [{"label": "...", "document": "aapl_fy2025"}, ...]')
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import fitz

    manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    filename = {
        str(d.get("document_id")): d.get("local_filename")
        for d in manifest.get("documents") or ()
    }

    requests = json.loads(args.request.read_text(encoding="utf-8"))
    results = []
    for item in requests:
        label = item["label"]
        document_id = item["document"]
        path = args.pdf_dir / str(filename.get(document_id))
        entry = {"label": label, "document": document_id}
        if not path.is_file():
            entry["found"] = False
            entry["note"] = "filing not found"
        else:
            with fitz.open(path) as document:
                entry.update(read_line(document, label))
        entry["found"] = bool(entry.get("found"))
        results.append(entry)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "company-lines.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for entry in results:
        if entry["found"]:
            print(f"  {entry['document']:16} {entry['label'][:34]:34} "
                  f"= {entry['value']:14} p{entry['pdf_page']}")
        else:
            print(f"  {entry['document']:16} {entry['label'][:34]:34} = NOT FOUND")
    print(f"\nwritten to {args.out / 'company-lines.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
