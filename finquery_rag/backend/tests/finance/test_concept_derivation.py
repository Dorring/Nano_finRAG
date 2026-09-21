"""A concept resolver that refuses is worth more than one that guesses.

`build_canonical_fact_store` files facts under their breadcrumb path, so a
cross-filing comparison cannot key on `metric`.  The obvious fix -- take the
path's last segment -- files `/s/ Arthur D. Levi` as a concept for NVIDIA's
compensation table, whose "path" is really a run of column headers.  That is how
`NVIDIA / Colette M. Kress` became a coordinate, and repeating it one level up
would be the same defect wearing a new label.

So resolution needs the path's leaf and the content's row label to agree, and
everything that does not is reported UNRESOLVED with the reason.  These tests
pin both halves: what resolves, and -- more of them -- what must not.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from src.finance.concept_derivation import derive_fact_concept  # noqa: E402


def test_two_agreeing_signals_resolve() -> None:
    derivation = derive_fact_concept(
        "Operating expenses: / Research and development",
        "Research and development | 34,550 | 31,370 | 29,915",
    )
    assert derivation.status == "RESOLVED"
    assert derivation.concept == "Research and development"
    assert derivation.source == "metric_path_leaf+content_label"


def test_a_single_segment_path_is_its_own_leaf() -> None:
    derivation = derive_fact_concept(
        "Total liabilities", "Total liabilities | 275,524 | 243,686"
    )
    assert derivation.status == "RESOLVED"
    assert derivation.concept == "Total liabilities"


def test_a_column_header_path_is_refused() -> None:
    """The NVIDIA counterexample: the path is column headers, not a concept.

    Its leaf is a signature.  A resolver that took the last segment would return
    `/s/ Arthur D. Levi` as the concept for a compensation fact.
    """

    derivation = derive_fact_concept(
        "KEVAN PAREKH / CHRIS KONDO / WANDA AUSTIN / ALEX GORSKY / ANDREA JUNG / /s/ Arthur D. Levi",
        "/s/ Wanda AustinDirector \nOctober 31, 2025 \nWANDA AUSTIN",
    )
    assert derivation.status == "UNRESOLVED"
    assert derivation.concept is None
    assert "/s/" in derivation.reason


def test_disagreeing_signals_are_refused() -> None:
    """One signal agreeing with itself is not evidence; two must agree."""

    derivation = derive_fact_concept(
        "Operating expenses: / Research and development",
        "Selling, general and administrative | 27,601 | 26,097",
    )
    assert derivation.status == "UNRESOLVED"
    assert "disagree" in derivation.reason


def test_a_signature_in_the_path_is_refused_even_when_the_label_matches() -> None:
    derivation = derive_fact_concept(
        "Signatures / /s/ Timothy D. Cook", "/s/ Timothy D. Cook | 2025"
    )
    assert derivation.status == "UNRESOLVED"


def test_a_label_longer_than_a_name_is_refused() -> None:
    long_label = (
        "Assets / Liabilities / Total interest-bearing liabilities and "
        "shareholders equity attributable to the parent company and others"
    )
    derivation = derive_fact_concept(long_label, f"{long_label} | 1,234")
    assert derivation.status == "UNRESOLVED"
    assert "longer than" in derivation.reason


def test_an_all_caps_multiword_label_is_refused() -> None:
    """`WANDA AUSTIN` is a name; `GROSS PROFIT` is a heading, not a metric."""

    derivation = derive_fact_concept("WANDA AUSTIN", "WANDA AUSTIN | 6,830,072")
    assert derivation.status == "UNRESOLVED"
    assert "upper case" in derivation.reason


def test_a_label_with_no_letters_is_refused() -> None:
    derivation = derive_fact_concept("2025", "2025 | 1")
    assert derivation.status == "UNRESOLVED"


def test_an_empty_path_is_refused() -> None:
    derivation = derive_fact_concept("", "Net income | 1")
    assert derivation.status == "UNRESOLVED"
    assert "empty" in derivation.reason


def test_content_without_a_row_label_is_refused() -> None:
    derivation = derive_fact_concept("Operating expenses: / Net income", "")
    assert derivation.status == "UNRESOLVED"


def test_a_trailing_caption_is_stripped_from_the_concept() -> None:
    derivation = derive_fact_concept(
        "Net interest income (in millions)", "Net interest income (in millions) | 6,114"
    )
    assert derivation.status == "RESOLVED"
    assert derivation.concept == "Net interest income"


def test_the_raw_path_is_always_preserved() -> None:
    """Provenance survives refusal, so a reader can see what was rejected."""

    raw = "Selling, general and administrative expenses: / Other operating charges"
    derivation = derive_fact_concept(raw, "Something else | 1")
    assert derivation.raw_metric_path == raw
    assert derivation.content_label == "Something else"
