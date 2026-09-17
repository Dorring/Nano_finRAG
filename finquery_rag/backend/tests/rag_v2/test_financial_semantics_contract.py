"""H2A-2B-0: the shared financial-semantic contract, pinned from outside.

Every expectation here is a literal authored in this file, or a relation between
two literals.  Nothing is computed with the implementation it checks -- a test
that derived its expected value by calling ``canonical_decimal`` would pass for
any implementation, including one that returns the same number for every input.
The literal assertions on magnitude are there for the same reason: an
equivalence-only suite is satisfied by a canonicaliser that maps everything to
one constant.

The contract these tests freeze:

    magnitude, measurement unit and representation kind are three dimensions,
    not one vocabulary; 1,500 is 1500 and 1,5 is nothing at all; an unknown
    scale stays unknown rather than becoming a magnitude we do know.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from rag_v2.contracts import financial_semantics as fs


# --- the numeric lexical contract -------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1500", Decimal("1500")),
        ("1500.0", Decimal("1500.0")),
        ("1,500", Decimal("1500")),
        ("1,500.25", Decimal("1500.25")),
        ("12,345,678", Decimal("12345678")),
        # Behaviour that already existed and is relied on, kept under test so it
        # cannot be lost while the strict forms are tightened.
        ("+1500", Decimal("1500")),
        ("-1500", Decimal("-1500")),
        ("(1,500)", Decimal("-1500")),
        ("$1,500", Decimal("1500")),
        ("1.5e3", Decimal("1500")),
        ("1,500(a)", Decimal("1500")),
    ],
)
def test_accepted_numeric_forms(text: str, expected: Decimal) -> None:
    assert fs.canonical_decimal(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "1,5",  # a comma followed by one digit is not a group separator
        "1,50",
        "1,5000",
        "12,34,567",
        "1,,500",
        "1,500,",  # trailing separator with nothing after it
        ",1500",
        "1500.",
        "1.5.3",
        "abc",
        "",
        "1,5 million",
    ],
)
def test_rejected_numeric_forms_are_not_canonicalizable(text: str) -> None:
    assert fs.canonical_decimal(text) is None


@pytest.mark.parametrize("text", ["1,5", "1,50", "1,5000", "12,34,567"])
def test_a_rejected_form_does_not_keep_the_digits(text: str) -> None:
    """The defect this contract exists for, stated as its own assertion.

    ``1,5`` describes 1.5 or 15 or something the author meant by a comma we do
    not share.  Deleting the comma answers 15.  Neither answer is recoverable
    from the text, so the text keeps its literal identity and the two are not
    equated with the number that comma-deletion would have produced.
    """

    stripped = Decimal(text.replace(",", ""))
    assert fs.quantity_identity(text) != fs.quantity_identity(str(stripped))
    assert not fs.quantity_identity(text).startswith(str(stripped))


def test_grouping_is_a_separator_and_not_a_decimal_point() -> None:
    assert fs.canonical_decimal("1,500") == Decimal("1500")
    assert fs.canonical_decimal("1,500") != Decimal("1.5")


# --- magnitude ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value,scale,expected",
    [
        ("1000", "million", Decimal("1000000000")),
        ("1", "billion", Decimal("1000000000")),
        ("1000000", "thousand", Decimal("1000000000")),
        ("391", None, Decimal("391")),
        ("1", "thousands", Decimal("1000")),
        ("1,500", "million", Decimal("1500000000")),
    ],
)
def test_magnitude_is_folded_to_a_literal_quantity(
    value: str, scale: str | None, expected: Decimal
) -> None:
    """Literal, not relational: a constant canonicaliser fails all of these."""

    quantity = fs.canonical_quantity(value, scale=scale)
    assert quantity is not None
    assert quantity.value == expected


def test_representation_equivalent_magnitudes_are_one_quantity() -> None:
    """Present, but never the only assertion about magnitude."""

    assert fs.quantity_identity("1000", scale="million") == fs.quantity_identity(
        "1", scale="billion"
    )


def test_decimal_precision_is_not_a_difference_in_quantity() -> None:
    assert fs.quantity_identity("1.000", scale="billion") == fs.quantity_identity(
        "1000.0", scale="million"
    )
    assert fs.quantity_identity("100", scale=None) == fs.quantity_identity(
        "100.0", scale=None
    )


# --- unknown vocabulary stays unknown ---------------------------------------


@pytest.mark.parametrize(
    "token", ["zillion", "adjusted-billion", "bazillion", "", "  ", "milliard"]
)
def test_an_unknown_scale_names_no_magnitude(token: str) -> None:
    assert fs.magnitude_of(token) is None


@pytest.mark.parametrize("known", ["billion", "million", "thousand", "trillion"])
def test_an_unknown_scale_is_not_a_known_magnitude(known: str) -> None:
    """Not a fallback to base, and not equal to a neighbouring magnitude."""

    unknown = fs.quantity_identity("1", scale="zillion")
    assert unknown != fs.quantity_identity("1", scale=known)
    assert unknown != fs.quantity_identity("1", scale=None)
    assert "zillion" in unknown


def test_an_unknown_scale_does_not_receive_a_multiplier() -> None:
    quantity = fs.canonical_quantity("1", scale="zillion")
    assert quantity is None


def test_base_magnitude_is_reachable_only_by_an_absent_scale() -> None:
    """The distinction between "no scale" and "a scale we do not know"."""

    assert fs.canonical_quantity("1", scale=None) is not None
    assert fs.canonical_quantity("1", scale="") is not None
    assert fs.canonical_quantity("1", scale="zillion") is None


# --- measurement unit --------------------------------------------------------


def test_a_different_currency_is_a_different_quantity() -> None:
    assert fs.quantity_identity("1", scale="billion", currency="USD") != (
        fs.quantity_identity("1", scale="billion", currency="EUR")
    )


def test_a_different_unit_is_a_different_quantity() -> None:
    assert fs.quantity_identity("1", scale="billion", unit="shares") != (
        fs.quantity_identity("1", scale="billion", unit="USD")
    )


def test_a_known_unit_is_named_and_an_unknown_one_is_not() -> None:
    assert fs.measurement_unit_of("USD") is fs.MeasurementUnit.USD
    assert fs.measurement_unit_of("shares") is fs.MeasurementUnit.SHARES
    assert fs.measurement_unit_of("furlongs") is None


def test_currency_symbols_are_not_resolved_to_an_iso_code() -> None:
    """``¥`` means CNY in one module and JPY in another; this layer picks neither.

    Silently choosing would hide a real disagreement between two existing
    conventions, and would make the choice look like a fact.
    """

    assert fs.measurement_unit_of("¥") is None
    assert fs.measurement_unit_of("$") is None


# --- representation kind -----------------------------------------------------


def test_percent_and_ratio_are_distinct_representations() -> None:
    assert fs.representation_of("percent") is fs.RepresentationKind.PERCENT
    assert fs.representation_of("%") is fs.RepresentationKind.PERCENT
    assert fs.representation_of("ratio") is fs.RepresentationKind.RATIO
    assert fs.RepresentationKind.PERCENT is not fs.RepresentationKind.RATIO


@pytest.mark.parametrize(
    "token", ["percent", "percentage", "%", "ratio"]
)
def test_a_representation_kind_is_not_a_magnitude(token: str) -> None:
    """The conflation the old vocabularies carried, pinned as a rejection."""

    assert fs.magnitude_of(token) is None
    assert fs.representation_of(token) is not None


def test_a_ratio_is_not_its_percentage_form() -> None:
    """0.12 and 12% are convertible, and are still not the same claim here.

    A conversion exists; no contract in this repository says the two are the
    same fact, and folding them would merge a ratio with a percentage point.
    """

    assert fs.quantity_identity("0.12", unit="ratio") != fs.quantity_identity(
        "12", unit="%"
    )


def test_an_absolute_quantity_carries_no_representation() -> None:
    quantity = fs.canonical_quantity("100", unit="USD")
    assert quantity is not None
    assert quantity.representation is fs.RepresentationKind.ABSOLUTE


# --- the three dimensions are three dimensions -------------------------------


def test_the_vocabularies_do_not_overlap() -> None:
    magnitude = set(fs.MAGNITUDE_SYNONYMS)
    unit = set(fs.MEASUREMENT_UNIT_SYNONYMS)
    representation = set(fs.REPRESENTATION_SYNONYMS)

    assert magnitude & unit == set()
    assert magnitude & representation == set()
    assert unit & representation == set()


def test_every_synonym_resolves_through_its_own_dimension_only() -> None:
    for token in fs.MAGNITUDE_SYNONYMS:
        assert fs.magnitude_of(token) is not None
        assert fs.measurement_unit_of(token) is None
        assert fs.representation_of(token) is None
    for token in fs.MEASUREMENT_UNIT_SYNONYMS:
        assert fs.measurement_unit_of(token) is not None
        assert fs.magnitude_of(token) is None
    for token in fs.REPRESENTATION_SYNONYMS:
        assert fs.representation_of(token) is not None
        assert fs.magnitude_of(token) is None


# --- the multipliers themselves ---------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("BASE", Decimal(1)),
        ("THOUSAND", Decimal(1000)),
        ("MILLION", Decimal(1000000)),
        ("BILLION", Decimal(1000000000)),
        ("TRILLION", Decimal(1000000000000)),
    ],
)
def test_magnitude_multipliers_are_exact(name: str, expected: Decimal) -> None:
    assert fs.magnitude_multiplier(getattr(fs.MagnitudeScale, name)) == expected


def test_a_magnitude_is_not_another_magnitude() -> None:
    scales = list(fs.MagnitudeScale)
    multipliers = {fs.magnitude_multiplier(scale) for scale in scales}
    assert len(multipliers) == len(scales)


# --- the bare words, and the delegation consumers go through -----------------


def test_the_bare_words_are_the_vocabulary_in_singular() -> None:
    assert fs.magnitude_words() == ("thousand", "million", "billion", "trillion")
    assert fs.MAGNITUDE_WORDS == {
        fs.MagnitudeScale.THOUSAND: "thousand",
        fs.MagnitudeScale.MILLION: "million",
        fs.MagnitudeScale.BILLION: "billion",
        fs.MagnitudeScale.TRILLION: "trillion",
    }
    for scale, word in fs.MAGNITUDE_WORDS.items():
        assert fs.magnitude_of(word) is scale


def test_a_consumer_table_cannot_disagree_with_the_shared_magnitudes() -> None:
    """The delegation used by the tables that keep a vocabulary of their own.

    ``million`` is written wrong on purpose: a local table keeps the words the
    shared semantics has never heard of, and cannot keep a different meaning for
    a word it has heard of.
    """

    resolved = fs.with_shared_magnitudes(
        {
            "million": Decimal("1"),
            "thousand": Decimal("1000"),
            "k": Decimal("1000"),
            "万": Decimal("10000"),
        }
    )

    assert resolved["million"] == Decimal("1000000")
    assert resolved["thousand"] == Decimal("1000")
    assert resolved["k"] == Decimal("1000")
    assert resolved["万"] == Decimal("10000")
    assert set(resolved) == {"million", "thousand", "k", "万"}


# --- textual identity --------------------------------------------------------


def test_text_identity_does_not_touch_commas() -> None:
    """The helper that was applied to every field, pinned to text only."""

    assert fs.text_identity("Revenue, net") != fs.text_identity("Revenue net")


def test_text_identity_folds_only_case_and_whitespace() -> None:
    assert fs.text_identity("  Net   Income ") == fs.text_identity("net income")


def test_a_numeric_field_does_not_use_the_text_helper() -> None:
    """1,500 and 1500 are one quantity; the text helper would say so too.

    The distinction is that the quantity path says so *because of the grouping
    rule*, and the text path is never consulted -- which is why 1,5 and 15 are
    not folded the same way.
    """

    assert fs.quantity_identity("1,500") == fs.quantity_identity("1500")
    assert fs.quantity_identity("1,5") != fs.quantity_identity("15")
