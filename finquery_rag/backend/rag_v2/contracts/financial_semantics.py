"""The frozen financial semantics every canonicalisation site has to agree on.

H2A-2B found three vocabularies for the same words -- one in the retrieval
layer (``src/pdf_retrieval_v4/semantic_scale_resolver.py``), one in the
generator's validator (``rag_v2/generation/validator.py``), one in the runtime
claim verifier (``rag_v2/runtime/semantic_claims.py``) -- with no shared source.
They agreed by luck, and they classified ``percent`` and ``ratio`` as the same
kind of token as ``million``.  Two consumers could therefore drift into
disagreeing about what a word means while both looking correct in isolation.

This module is the single authority for what those words mean.  It is not a
single parser: retrieval still resolves scale keywords from table captions, and
the validator still scans answer text with its own regex.  What is shared is the
*semantics* -- which words exist, which dimension each belongs to, and what
multiplier a magnitude carries -- so a consumer may choose its own lexical form
without choosing its own meaning.

Three dimensions, deliberately separate
---------------------------------------

``MagnitudeScale``     how large is the represented quantity?      million -> 1e6
``MeasurementUnit``    what is measured or denominated?            USD, shares
``RepresentationKind`` how should the number be read?              percent, ratio

They answer different questions and are not interchangeable.  ``percent`` and
``ratio`` are *not* magnitude scales -- they say how a number is expressed, not
how big it is -- and they are not each other: ``12%`` and ``ratio 0.12`` are
different claims.  A conversion between them exists mathematically, and that is
exactly why folding them here would be wrong: this layer's job is to say what a
representation *means*, not to decide when two representations may be treated as
one.  Only an explicit domain contract may do that, and none does today.

Dependency direction
--------------------

Imports stdlib only.  Every layer that needs these meanings -- retrieval,
generation, the adaptive runtime, ``src/runtime`` -- depends on this module;
this module depends on none of them.  A reverse edge here would put the shared
authority above one of its consumers and make the next consolidation a cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Iterable, Mapping


# ---------------------------------------------------------------------------
# Dimension 1: magnitude
# ---------------------------------------------------------------------------


class MagnitudeScale(str, Enum):
    """How large a represented quantity is, independent of how it is written."""

    BASE = "base"
    THOUSAND = "thousand"
    MILLION = "million"
    BILLION = "billion"
    TRILLION = "trillion"


#: The exact multiplier for each known magnitude.  Decimal rather than float
#: because these multiply typed financial values and the result is compared for
#: equality: ``1.0e9`` and ``Decimal("1000000000")`` agree today, but a
#: magnitude that is not a power of ten would not, and there is no reason for
#: the shared authority to be the one place that loses exactness.
MAGNITUDE_MULTIPLIERS: Mapping[MagnitudeScale, Decimal] = {
    MagnitudeScale.BASE: Decimal(1),
    MagnitudeScale.THOUSAND: Decimal(1000),
    MagnitudeScale.MILLION: Decimal(1000000),
    MagnitudeScale.BILLION: Decimal(1000000000),
    MagnitudeScale.TRILLION: Decimal(1000000000000),
}

#: The stable name for a magnitude, as a table's caption names it.  Retrieval
#: records this alongside the factor and compares two captions by it, so it is
#: part of the semantics rather than a presentation detail.
MAGNITUDE_CANONICAL_NAMES: Mapping[MagnitudeScale, str] = {
    MagnitudeScale.BASE: "base",
    MagnitudeScale.THOUSAND: "thousands",
    MagnitudeScale.MILLION: "millions",
    MagnitudeScale.BILLION: "billions",
    MagnitudeScale.TRILLION: "trillions",
}

#: The bare singular word for each magnitude that has one, in declaration order.
#:
#: Distinct from :data:`MAGNITUDE_CANONICAL_NAMES`: that one is a caption's
#: plural ("dollars in millions") and is compared as a label, this one is the
#: word a consumer names a magnitude by.  A consumer that matches text uses it
#: to state *which* magnitudes it looks for without restating what they are --
#: the words here are checked against the vocabulary and the multipliers below,
#: so a list built from them cannot name a magnitude the shared semantics does
#: not define.
MAGNITUDE_WORDS: Mapping[MagnitudeScale, str] = {
    MagnitudeScale.THOUSAND: "thousand",
    MagnitudeScale.MILLION: "million",
    MagnitudeScale.BILLION: "billion",
    MagnitudeScale.TRILLION: "trillion",
}

#: Every word or phrasing that names a magnitude, lower-cased and whitespace
#: normalised.
#:
#: The multi-word forms are table-caption idioms ("revenue, in millions",
#: "amounts in thousands") and are lexical variants of the same magnitude, not a
#: different one -- which is why they live in one mapping with the bare words
#: rather than in a retrieval-only table.  Retrieval scans the *keys* of this
#: mapping as substrings, so their order is load-bearing: longer phrasings
#: precede the bare words they contain, so a caption is reported as the phrase
#: that actually appears.
MAGNITUDE_SYNONYMS: Mapping[str, MagnitudeScale] = {
    "in millions": MagnitudeScale.MILLION,
    "in million": MagnitudeScale.MILLION,
    "millions": MagnitudeScale.MILLION,
    "million": MagnitudeScale.MILLION,
    "in billions": MagnitudeScale.BILLION,
    "in billion": MagnitudeScale.BILLION,
    "billions": MagnitudeScale.BILLION,
    "billion": MagnitudeScale.BILLION,
    "in thousands": MagnitudeScale.THOUSAND,
    "in thousand": MagnitudeScale.THOUSAND,
    "thousands": MagnitudeScale.THOUSAND,
    "thousand": MagnitudeScale.THOUSAND,
    "dollars in millions": MagnitudeScale.MILLION,
    "dollars in thousands": MagnitudeScale.THOUSAND,
    "dollars in billions": MagnitudeScale.BILLION,
    "amounts in millions": MagnitudeScale.MILLION,
    "amounts in thousands": MagnitudeScale.THOUSAND,
    "amounts in billions": MagnitudeScale.BILLION,
    "in trillions": MagnitudeScale.TRILLION,
    "in trillion": MagnitudeScale.TRILLION,
    "trillions": MagnitudeScale.TRILLION,
    "trillion": MagnitudeScale.TRILLION,
    "dollars in trillions": MagnitudeScale.TRILLION,
    "amounts in trillions": MagnitudeScale.TRILLION,
}


def magnitude_of(token: Any) -> MagnitudeScale | None:
    """The magnitude a token names, or ``None`` if it names none we know.

    ``None`` is not ``BASE``.  An unrecognised scale word is an uninterpreted
    qualifier -- "adjusted-billion" is not a billion, and a caller that folded
    it to the base magnitude would silently equate two different quantities.
    Callers keep the literal token as part of the identity instead.
    """

    if token is None:
        return None
    key = " ".join(str(token).casefold().split())
    if not key:
        return None
    return MAGNITUDE_SYNONYMS.get(key)


def magnitude_multiplier(scale: MagnitudeScale) -> Decimal:
    return MAGNITUDE_MULTIPLIERS[scale]


def magnitude_canonical_name(scale: MagnitudeScale) -> str:
    return MAGNITUDE_CANONICAL_NAMES[scale]


def magnitude_words() -> tuple[str, ...]:
    """The bare singular word of every magnitude that has one, in order."""

    return tuple(
        MAGNITUDE_WORDS[scale] for scale in MagnitudeScale if scale in MAGNITUDE_WORDS
    )


def with_shared_magnitudes(local: Mapping[str, Decimal]) -> dict[str, Decimal]:
    """Re-point a consumer's own scale table at the shared multipliers.

    A consumer that recognises words this layer does not -- bare suffixes
    (``k``, ``bn``), Chinese magnitude units -- keeps its own table, because
    that is its lexical vocabulary.  It does not keep its own *numbers*: every
    word the shared semantics knows takes the shared multiplier here, so a table
    can hold a vocabulary the shared layer has never heard of and still be
    incapable of disagreeing with it about what ``million`` means.

    The replacement is unconditional rather than a comparison.  A local table
    that disagrees is the defect this exists to remove, and it should not be
    allowed to win because it was written first.
    """

    resolved: dict[str, Decimal] = {}
    for word, fallback in local.items():
        scale = magnitude_of(word)
        resolved[word] = (
            magnitude_multiplier(scale) if scale is not None else fallback
        )
    return resolved


# ---------------------------------------------------------------------------
# Dimension 2: measurement unit
# ---------------------------------------------------------------------------


class MeasurementUnit(str, Enum):
    """What a quantity is measured in -- a denomination, not a size."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"
    SHARES = "shares"


#: Words that name a measurement unit.
#:
#: Currency *symbols* are deliberately absent: the repository has two
#: conventions for ``¥`` (``src/validation/claim_extractor.py`` reads it as CNY,
#: ``src/finance/unit_normalizer.py`` as JPY) and this layer must not silently
#: pick a winner.  Symbols stay with the consumers that already decided, and the
#: disagreement stays visible until a contract resolves it.
MEASUREMENT_UNIT_SYNONYMS: Mapping[str, MeasurementUnit] = {
    "usd": MeasurementUnit.USD,
    "us dollar": MeasurementUnit.USD,
    "us dollars": MeasurementUnit.USD,
    "dollar": MeasurementUnit.USD,
    "dollars": MeasurementUnit.USD,
    "eur": MeasurementUnit.EUR,
    "euro": MeasurementUnit.EUR,
    "euros": MeasurementUnit.EUR,
    "gbp": MeasurementUnit.GBP,
    "pound": MeasurementUnit.GBP,
    "pounds": MeasurementUnit.GBP,
    "jpy": MeasurementUnit.JPY,
    "yen": MeasurementUnit.JPY,
    "cny": MeasurementUnit.CNY,
    "rmb": MeasurementUnit.CNY,
    "yuan": MeasurementUnit.CNY,
    "share": MeasurementUnit.SHARES,
    "shares": MeasurementUnit.SHARES,
}


def measurement_unit_of(token: Any) -> MeasurementUnit | None:
    """The measurement unit a token names, or ``None`` if we do not know it.

    An unknown unit is not a known one: callers keep the literal token.
    """

    if token is None:
        return None
    key = " ".join(str(token).casefold().split())
    if not key:
        return None
    return MEASUREMENT_UNIT_SYNONYMS.get(key)


# ---------------------------------------------------------------------------
# Dimension 3: representation kind
# ---------------------------------------------------------------------------


class RepresentationKind(str, Enum):
    """How a number should be read, as distinct from how large it is.

    ``PERCENT`` and ``RATIO`` stay distinct.  They are convertible, and that is
    precisely the reason not to convert them here: whether ``12%`` and
    ``ratio 0.12`` are the same claim is a domain decision, and this layer has
    been given no contract that says they are.
    """

    ABSOLUTE = "absolute"
    PERCENT = "percent"
    RATIO = "ratio"


REPRESENTATION_SYNONYMS: Mapping[str, RepresentationKind] = {
    "percent": RepresentationKind.PERCENT,
    "percentage": RepresentationKind.PERCENT,
    "%": RepresentationKind.PERCENT,
    "ratio": RepresentationKind.RATIO,
}


def representation_of(token: Any) -> RepresentationKind | None:
    """The representation kind a token names, or ``None`` if it names none."""

    if token is None:
        return None
    key = " ".join(str(token).casefold().split())
    if not key:
        return None
    return REPRESENTATION_SYNONYMS.get(key)


# ---------------------------------------------------------------------------
# Lexical forms, built from the definitions above
# ---------------------------------------------------------------------------
#
# A consumer's regex is a lexical choice -- which of these words it scans for,
# and in what context -- and stays with the consumer.  What it may not do is
# re-type the words: a consumer that lists them again has its own vocabulary
# again, and the drift this module exists to stop starts over.


def magnitude_tokens() -> tuple[str, ...]:
    """The bare magnitude words, longest first so a regex prefers the longer."""

    return _word_tokens(MAGNITUDE_SYNONYMS)


def representation_tokens() -> tuple[str, ...]:
    """The bare representation words.  ``%`` is a symbol, not matched by ``\\b``."""

    return _word_tokens(REPRESENTATION_SYNONYMS)


def measurement_unit_tokens() -> tuple[str, ...]:
    """The bare measurement-unit words."""

    return _word_tokens(MEASUREMENT_UNIT_SYNONYMS)


def _word_tokens(synonyms: Mapping[str, Any]) -> tuple[str, ...]:
    words = {token for token in synonyms if token.isalpha()}
    return tuple(sorted(words, key=len, reverse=True))


def token_pattern(tokens: Iterable[str]) -> str:
    """A regex alternation for ``tokens``, longest first.

    Ordering matters for an alternation as much as for retrieval's substring
    scan: without it, ``millions`` could be matched as ``million`` plus a stray
    ``s``, and a caller reading the token back would see a word that is not in
    the vocabulary.
    """

    return "|".join(re.escape(token) for token in sorted(tokens, key=len, reverse=True))


# ---------------------------------------------------------------------------
# The numeric lexical contract
# ---------------------------------------------------------------------------
#
# Strict English/US financial syntax.  The project supports no locale
# alternates, and the rule that decides the grammar is this: an unrecognised
# representation must never be *guessed* into a number.
#
#     1500        -> 1500          1,500     -> 1500
#     1500.0      -> 1500          1,500.25  -> 1500.25
#     1,5         -> not canonicalizable
#     1,50        -> not canonicalizable
#     12,34,567   -> not canonicalizable
#
# ``1,5`` is the case that matters.  Removing punctuation -- the obvious
# implementation, and the one this repository had -- turns it into ``15``: a
# number ten times larger, arrived at by deleting a character the author wrote
# deliberately.  In a locale that uses the comma as a decimal separator it means
# ``1.5``, which is a third number again.  Neither is recoverable from the text,
# so neither is chosen.  The value keeps a literal identity and the callers
# above stay conservative.
#
# Only *strict* grouping is accepted: a separator may only appear between digit
# groups, and every group after the first is exactly three digits.
#
# Existing behaviour that is preserved because it is already relied on:
# explicit signs, accounting parentheses for negatives, scientific notation,
# and a surrounding currency symbol.

_FOOTNOTE_MARKER_RE = re.compile(r"\([a-zA-Z]\)")
_CURRENCY_EDGE_RE = re.compile(r"^[\s$€£¥]+|[\s$€£¥]+$")
_ACCOUNTING_NEGATIVE_RE = re.compile(r"^\((?P<body>.*)\)$", re.DOTALL)

#: A sign, then either strictly-grouped digits or ungrouped digits, an optional
#: fractional part that must have digits on both sides of the point, and an
#: optional exponent.  Anchored by ``fullmatch`` at the call site: a partial
#: match is how ``1,5000`` would become ``1500`` plus an ignored remainder.
_NUMERIC_RE = re.compile(
    r"(?P<sign>[-+])?"
    r"(?P<digits>\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.(?P<fraction>\d+))?"
    r"(?:[eE](?P<exponent>[-+]?\d+))?"
)


def canonical_decimal(value: Any) -> Decimal | None:
    """The exact number a typed numeric field states, or ``None``.

    ``None`` means "this text is not a financial number this project
    understands".  It does **not** mean zero, and it does not mean the base
    magnitude: a caller that cannot canonicalise a typed value must fall back to
    the value's literal identity, never to a parsed default.

    ``Decimal`` is already returned unchanged, so a typed field that carries a
    parsed value does not round-trip through its own string form.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        # ``str`` first: the binary value's shortest decimal form is what the
        # producer meant, and ``Decimal(0.1)`` is not.
        return Decimal(str(value)) if _is_finite_float(value) else None

    text = str(value).strip()
    if not text:
        return None

    # A footnote marker rides along with a table value and is not part of it.
    text = _FOOTNOTE_MARKER_RE.sub("", text)
    text = _CURRENCY_EDGE_RE.sub("", text)

    negative = False
    accounting = _ACCOUNTING_NEGATIVE_RE.match(text)
    if accounting is not None:
        text = accounting.group("body").strip()
        negative = True

    match = _NUMERIC_RE.fullmatch(text)
    if match is None:
        return None

    sign = match.group("sign") or ""
    if sign == "-":
        negative = True
    digits = match.group("digits").replace(",", "")
    fraction = match.group("fraction")
    exponent = match.group("exponent")

    literal = f"{digits}.{fraction}" if fraction is not None else digits
    if exponent is not None:
        literal = f"{literal}e{exponent}"
    try:
        number = Decimal(literal)
    except InvalidOperation:
        return None
    return -number if negative else number


def _is_finite_float(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


# ---------------------------------------------------------------------------
# Textual identity
# ---------------------------------------------------------------------------


def text_identity(value: Any) -> str:
    """Conservative identity for a field that is text, not a number.

    Case and whitespace runs are folded; nothing else is.  In particular commas
    are **not** removed.  The previous shared helper removed them everywhere,
    which is right for a digit-grouping separator and wrong for every other
    comma, and it was applied to metrics, periods, entities and scopes as well
    as to numbers -- so a metric that differed by a comma compared equal, and a
    typed value that differed by a comma compared equal whether or not the comma
    was a grouping separator.  Numeric fields do not use this function at all;
    they go through :func:`quantity_identity`.
    """

    if value is None:
        return ""
    return " ".join(str(value).casefold().split())


# ---------------------------------------------------------------------------
# The shared quantity primitive
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalQuantity:
    """A typed financial quantity with its magnitude folded in.

    ``scale`` is absent by design: it has been applied to ``value``, and keeping
    the name as well would make ``1000 million`` and ``1 billion`` differ on a
    label after agreeing on the quantity -- a false conflict over a word.
    """

    value: Decimal
    magnitude: MagnitudeScale
    unit: str
    representation: RepresentationKind

    @property
    def identity(self) -> str:
        """A stable, comparable rendering of the whole quantity."""

        return "|".join(
            (
                _decimal_text(self.value),
                self.unit,
                self.representation.value,
            )
        )


def canonical_quantity(
    value: Any,
    *,
    scale: Any = None,
    unit: Any = None,
    currency: Any = None,
) -> CanonicalQuantity | None:
    """Canonicalise a typed financial quantity, or ``None`` if it will not.

    ``None`` is returned when the numeric representation is not one this project
    understands **or** when the scale names a magnitude we do not know.  Both
    are the same answer for the same reason: the quantity cannot be established
    from what was written, so the caller keeps the literal form.
    """

    number = canonical_decimal(value)
    if number is None:
        return None

    magnitude = _magnitude_of_field(scale)
    if magnitude is None:
        return None

    folded = number * magnitude_multiplier(magnitude)
    unit_text = text_identity(unit)
    currency_text = text_identity(currency)
    return CanonicalQuantity(
        value=folded,
        magnitude=magnitude,
        unit=" ".join(part for part in (unit_text, currency_text) if part),
        representation=representation_of(unit) or RepresentationKind.ABSOLUTE,
    )


def quantity_identity(
    value: Any,
    *,
    scale: Any = None,
    unit: Any = None,
    currency: Any = None,
) -> str:
    """The shared identity of a typed financial quantity.

    Conflict consensus and the content fingerprint both call this, so that the
    two agree on what "the same quantity" means by construction rather than by
    maintenance.  When the quantity cannot be canonicalised the literal fields
    are used instead -- never a default, and never a value derived from them.
    """

    quantity = canonical_quantity(value, scale=scale, unit=unit, currency=currency)
    if quantity is not None:
        return quantity.identity
    return "|".join(
        (
            "literal:" + text_identity(value),
            text_identity(unit),
            text_identity(currency),
            text_identity(scale),
        )
    )


def _magnitude_of_field(scale: Any) -> MagnitudeScale | None:
    """The magnitude a typed ``scale`` field names.

    An absent or empty scale is the base magnitude -- that is a stated fact
    about the record, not a fallback for the unknown.  Anything else must name a
    magnitude we know, or the quantity is not established.
    """

    if scale is None:
        return MagnitudeScale.BASE
    text = text_identity(scale)
    if not text:
        return MagnitudeScale.BASE
    return magnitude_of(text)


def _decimal_text(number: Decimal) -> str:
    """The stable rendering of a canonical magnitude.

    ``normalize`` strips trailing zeros from the coefficient, so ``1.000
    billion`` and ``1000.0 million`` render identically -- decimal precision is
    presentation, and two sources writing one quantity at different precision
    must not read as a disagreement.
    """

    try:
        return str(number.normalize()).casefold()
    except (ArithmeticError, ValueError):  # pragma: no cover - defensive
        return str(number)
