from decimal import Decimal

from src.evaluation.nf41_numeric_identity import normalize_numeric_identity, value_matches_expected


def test_million_scale_is_normalized_without_changing_label_compatibility():
    identity = normalize_numeric_identity("$42.2 million")
    assert identity is not None
    assert identity.canonical_value == Decimal("42200000")
    assert identity.currency == "USD"
    assert value_matches_expected("$42.2 million", ("42.2",))


def test_parentheses_preserve_negative_sign():
    identity = normalize_numeric_identity("(1.5 million)")
    assert identity is not None
    assert identity.canonical_value == Decimal("-1500000")


def test_percentage_is_not_plain_number():
    identity = normalize_numeric_identity("15%")
    assert identity is not None
    assert identity.canonical_value == Decimal("0.15")
    assert identity.value_type == "percentage"
    assert not value_matches_expected("15%", ("15",))


def test_currency_mismatch_does_not_match_by_canonical_value_alone():
    usd = normalize_numeric_identity("USD 42.2 million")
    chf = normalize_numeric_identity("CHF 42.2 million")
    assert usd is not None and chf is not None
    assert usd.currency != chf.currency
