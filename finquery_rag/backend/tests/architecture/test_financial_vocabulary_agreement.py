"""The V1 scale vocabulary cannot import the shared semantics, so it is guarded.

H2A-2B consolidated the three scale vocabularies that existed in different
layers -- retrieval, the generator's validator, the runtime claim verifier --
onto one authority in `rag_v2/contracts/financial_semantics.py`.  Two more
tables name the same magnitudes from the V1 side:

    src/finance/primitive_tools.py       _SCALE_FACTORS
    src/validation/claim_extractor.py    _SCALE_MAP
    src/finance/unit_normalizer.py       _SCALE_WORDS
    src/finance/structured_operand_binding.py  _SCALE_WORDS

They are not delegated, and the reason is a layer contract rather than an
oversight.  `src/finance` and `src/validation` are pure V1 modules -- the first
documents itself "intentionally pure and dependency-free", the second "imports
only from ``src.domain`` and stdlib" -- and neither package imports `rag_v2`
anywhere today.  Pointing them at it would make V1 production code depend on the
development-shadow V2 package, which is the same class of inversion the audit
closed for `rag_v2 -> src/`, and the brief does not authorise it.

What it does authorise is that a duplicated authority must not be left able to
drift.  So the duplication is guarded instead of collapsed: every table here is
checked against the shared semantics, and a word the shared layer defines may
not mean anything else in V1.  A drift fails a test rather than shipping.

The expectations are read from the shared module and compared, never restated --
a second literal table in this file would just be a third authority.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from rag_v2.contracts import financial_semantics as fs


def _shared_multiplier(word: str) -> Decimal | None:
    scale = fs.magnitude_of(word)
    return None if scale is None else fs.magnitude_multiplier(scale)


# --- multiplier tables -------------------------------------------------------


def _v1_multiplier_tables() -> list[tuple[str, dict]]:
    from src.finance.primitive_tools import _SCALE_FACTORS
    from src.validation.claim_extractor import _SCALE_MAP

    return [("primitive_tools._SCALE_FACTORS", _SCALE_FACTORS),
            ("claim_extractor._SCALE_MAP", _SCALE_MAP)]


@pytest.mark.parametrize("name,table", _v1_multiplier_tables())
def test_a_v1_multiplier_table_agrees_with_the_shared_semantics(
    name: str, table: dict
) -> None:
    """Every magnitude word the shared layer defines means the same here."""

    for word, multiplier in table.items():
        expected = _shared_multiplier(word)
        if expected is None:
            # A word only this table knows -- a bare suffix, a Chinese unit, the
            # deliberate "zillion" sentinel.  Out of the shared vocabulary, and
            # out of scope.
            continue
        assert Decimal(multiplier) == expected, (
            f"{name}[{word!r}] is {multiplier}, shared semantics say {expected}"
        )


@pytest.mark.parametrize("name,table", _v1_multiplier_tables())
def test_a_v1_multiplier_table_holds_magnitudes_and_nothing_else(
    name: str, table: dict
) -> None:
    """A table of scale factors may not list a non-magnitude.

    ``percent`` and ``ratio`` are how a number is read and the unit words are
    what it is measured in; neither is a size.  Putting them in a scale table is
    the conflation this phase removed from the other three vocabularies, and it
    must not survive in these two.
    """

    misplaced = [
        word
        for word in table
        if fs.representation_of(word) is not None
        or fs.measurement_unit_of(word) is not None
    ]
    assert misplaced == [], f"{name} lists {misplaced} as a magnitude"


# --- known-scale word lists --------------------------------------------------


def _v1_scale_word_lists() -> list[tuple[str, tuple[str, ...]]]:
    from src.finance.structured_operand_binding import _SCALE_WORDS as binding_words
    from src.finance.unit_normalizer import _SCALE_WORDS as normalizer_words

    return [("unit_normalizer._SCALE_WORDS", normalizer_words),
            ("structured_operand_binding._SCALE_WORDS", binding_words)]


@pytest.mark.parametrize("name,words", _v1_scale_word_lists())
def test_a_v1_scale_word_list_covers_the_shared_vocabulary(
    name: str, words: tuple[str, ...]
) -> None:
    """A magnitude the shared layer defines is a magnitude V1 recognises.

    The list may hold words the shared layer does not (bare suffixes, Chinese
    units); it may not be missing one, which is how a consumer silently stops
    seeing a magnitude the rest of the repository still knows.
    """

    missing = [word for word in fs.magnitude_words() if word not in words]
    assert not missing, f"{name} does not know {missing}"


@pytest.mark.parametrize("name,words", _v1_scale_word_lists())
def test_a_v1_scale_word_list_has_no_word_the_shared_layer_calls_text(
    name: str, words: tuple[str, ...]
) -> None:
    """No representation kind or measurement unit may live in a scale list."""

    misplaced = [
        word
        for word in words
        if fs.representation_of(word) is not None
        or fs.measurement_unit_of(word) is not None
    ]
    assert misplaced == [], f"{name} classifies {misplaced} as a scale"


# --- lexical scans -----------------------------------------------------------


def test_the_offline_scale_scan_recognises_the_shared_magnitudes() -> None:
    """``nf_opt_07``'s audit pattern is a lexical scan, not a multiplier table.

    It answers "does this chunk mention a scale", so what it must agree on is
    membership: every magnitude the repository defines, and nothing it does not.
    """

    from src.evaluation.nf_opt_07 import _SCALE

    for word in fs.magnitude_words():
        assert _SCALE.search(word) is not None, f"{word} is not recognised"
        assert _SCALE.search(f"{word}s") is not None, f"{word}s is not recognised"

    assert _SCALE.search("zillion") is None
    assert _SCALE.search("millionaire") is None
