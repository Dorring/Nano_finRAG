"""SEC 10-K Footnote Anchor and Superscript Disclosure Linking.

Detects footnote markers (such as '(1)', '(2)', '*', '[a]') embedded in
financial statements and table labels, extracts the corresponding footnote
explanations from the bottom of the page or notes disclosure, and attaches
institutional-grade explanatory context to evidence citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class FootnoteAnchor:
    """A linked footnote explanation for a financial metric or table row."""

    marker: str
    metric: str
    annotation: str
    page: int | str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FootnoteAnchorDetector:
    """Extract and link footnote markers from financial table rows and pages."""

    # Matches footnote markers like (1), (2), [a], [1], *, †, ‡ attached to labels
    MARKER_REGEX = re.compile(
        r"(?:\((?P<paren>\d{1,2}|[a-z])\)|\[(?P<bracket>\d{1,2}|[a-z])\]|(?P<symbol>[*†‡]))\s*$",
        re.IGNORECASE,
    )

    # Matches footnote definition headers at bottom of tables: e.g. "(1) Includes...", "(*) Consists of..."
    DEF_REGEX = re.compile(
        r"(?:^|\n)\s*(?P<marker>\(\d{1,2}\)|\([a-z]\)|\[\d{1,2}\]|\[[a-z]\]|[*†‡])\s+(?P<text>[^\n]+(?:\n(?!\s*(?:\(\d|\([a-z]|\[|[*†‡]))[^\n]+)*)",
        re.MULTILINE,
    )

    @classmethod
    def extract_marker(cls, label: str) -> str | None:
        """Extract the trailing footnote marker from a metric label if present."""
        if not label:
            return None
        m = cls.MARKER_REGEX.search(label.strip())
        if not m:
            return None
        if m.group("paren"):
            return f"({m.group('paren')})"
        if m.group("bracket"):
            return f"[{m.group('bracket')}]"
        return m.group("symbol")

    @classmethod
    def strip_marker(cls, label: str) -> str:
        """Return the clean label without trailing footnote marker."""
        if not label:
            return ""
        return cls.MARKER_REGEX.sub("", label).strip()

    @classmethod
    def extract_page_footnote_definitions(cls, page_text: str) -> dict[str, str]:
        """Extract all footnote marker -> explanation mappings from page text."""
        if not page_text:
            return {}
        definitions: dict[str, str] = {}
        for m in cls.DEF_REGEX.finditer(page_text):
            marker = m.group("marker").strip()
            text = " ".join(m.group("text").split()).strip()
            if text:
                definitions[marker] = text
        return definitions

    @classmethod
    def link_metric_footnotes(
        cls,
        metric_label: str,
        context_or_page_text: str,
        page: int | str | None = None,
    ) -> FootnoteAnchor | None:
        """Link a metric's footnote marker to its explanation if found."""
        marker = cls.extract_marker(metric_label)
        if not marker or not context_or_page_text:
            return None

        defs = cls.extract_page_footnote_definitions(context_or_page_text)
        annotation = defs.get(marker)
        if not annotation:
            # Fallback: search for inline marker explanation
            m_inline = re.search(
                rf"(?:^|\n)\s*{re.escape(marker)}\s*([^\n]+)",
                context_or_page_text,
            )
            if m_inline:
                annotation = " ".join(m_inline.group(1).split()).strip()

        if not annotation:
            return None

        clean_metric = cls.strip_marker(metric_label)
        return FootnoteAnchor(
            marker=marker,
            metric=clean_metric,
            annotation=f"{marker} {annotation}",
            page=page,
        )


__all__ = [
    "FootnoteAnchor",
    "FootnoteAnchorDetector",
]
