"""Diagnostic-only canonical metric identity for Gate09 R5.2-R0."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

STOPWORDS = {"the", "of", "for"}
MORPHOLOGY = {
    "assets": "asset",
    "operations": "operation",
    "operating": "operation",
    "revenues": "revenue",
    "sales": "sale",
    "services": "service",
}


def canonical_metric_tokens(value: Any) -> tuple[str, ...]:
    """Normalize structure and a closed morphology map; never infer synonyms."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[/\-:()]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [
        MORPHOLOGY.get(token, token)
        for token in text.split()
        if token and token not in STOPWORDS
    ]
    return tuple(sorted(tokens))


def canonical_metric_id(value: Any) -> str | None:
    tokens = canonical_metric_tokens(value)
    if not tokens:
        return None
    payload = "\x1f".join(tokens).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
