"""B1 -- can the source say which table a row belongs to?

The twelve `SEVERAL_ROWS` cases are the whole of the Go margin for a fact
representation migration, and the question they turn on is narrow: for a row
whose label is shared with another row somewhere else in the filing, does the
source identify the table it sits in?

Coca-Cola is the worked example. Two rows are labelled `Operating income` and
both are real:

    Operating Income | 13,762 |  9,992 | 11,311
    Operating income | 13,426 | 12,536 | 11,868

The first is a segment table's, the second the consolidated income statement's,
and nothing in the store says so. This reads the filing and prints what actually
precedes each occurrence, so the answer is read rather than assumed.

  python audit_b1_source_caption.py --html <primary.html> --label "Operating income"
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

_TAG = re.compile(r"<[^>]+>")
#: Headings a filing uses to title a statement or a table.  All-caps lines are
#: how every one of these filings writes them.
_CAPTION = re.compile(
    r"(CONSOLIDATED\s+STATEMENTS?[^\n]{0,80}"
    r"|CONSOLIDATED\s+BALANCE\s+SHEET[^\n]{0,60}"
    r"|SEGMENT[^\n]{0,60}"
    r"|NOTES?\s+TO[^\n]{0,60})",
    re.IGNORECASE,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--window", type=int, default=2600)
    args = parser.parse_args(argv)

    raw = args.html.read_text(encoding="utf-8", errors="replace")
    text = html.unescape(_TAG.sub(" ", raw))
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)

    print("=" * 78)
    print("%s  --  %r" % (args.html.name, args.label))
    print("=" * 78)
    hits = [m.start() for m in re.finditer(re.escape(args.label), text)]
    print("  occurrences: %d" % len(hits))
    for index, position in enumerate(hits, 1):
        before = text[max(0, position - args.window):position]
        captions = _CAPTION.findall(before)
        print()
        print("  -- occurrence %d" % index)
        print("     nearest captions before it: %s"
              % ([c.strip()[:60] for c in captions[-3:]] or "NONE FOUND"))
        tail = text[position:position + 140].replace("\n", " ")
        print("     %s" % tail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
