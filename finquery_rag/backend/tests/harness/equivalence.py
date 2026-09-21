"""NF-V3 H1: the ``legacy`` <-> ``harness_v3`` decision-equivalence contract.

``harness_v3`` is an ablation, not a second product.  For one request over one
set of capability ports it must reach the same release decision, for the same
reasons, backed by the same evidence.  The two modes are allowed to differ only
in *how* the run was executed and in *what the run recorded about itself*.

Getting this comparison wrong is how H1's first two divergences survived a
review pass.  The original assertion set compared ``status`` / ``answer`` /
``citation_ids`` -- all of which agreed -- while ``route`` was silently falling
back to the plan intent in the rebuilt outcome and ``reason_codes`` was
inheriting the trace's superset, labelling a clean release with an
already-resolved recovery code such as ``WRONG_PERIOD``.  A second pass found
``calculations`` / ``calculation_result_id`` / ``claim_provenance`` missing from
the same list.

Per-test field lists are what made that possible, so the partition lives here
once and every differential test goes through :func:`decision_differences`,
:func:`decision_equivalent` or :func:`assert_decision_equivalent`.

The partition
-------------
A. :data:`DECISION_BEARING_FIELDS` plus :data:`DECISION_BEARING_METADATA` --
   must be identical.  Everything that shaped or reported the release verdict.
B. :data:`HARNESS_ONLY_METADATA_KEYS` and ``debug_metadata`` -- ``harness_v3``
   is *expected* to add these (it executes more phases, so it records more
   execution).  They are stripped, never asserted equal, and the trace is
   verified separately against its own expectations.
C. :data:`VOLATILE_METADATA_KEYS` -- measurements and per-run identifiers.
   Equal-and-required-to-stay-equal would make the comparison flaky for reasons
   that carry no decision.

Adding a field to ``V2ExecutionOutcome`` means classifying it here.  There is no
default: an unclassified field is neither compared nor explicitly excused, which
is the failure mode this module exists to prevent.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

__all__ = [
    "CONTRACT_FIELDS",
    "DECISION_BEARING_FIELDS",
    "DECISION_BEARING_METADATA",
    "HARNESS_ONLY_FIELDS",
    "HARNESS_ONLY_METADATA_KEYS",
    "VOLATILE_METADATA_KEYS",
    "assert_decision_equivalent",
    "canonicalize_decision_result",
    "decision_differences",
    "decision_equivalent",
    "unclassified_fields",
    "unknown_contract_fields",
]


# --- A. must be identical ----------------------------------------------------

DECISION_BEARING_FIELDS: tuple[str, ...] = (
    "status",
    "release_status",
    "answer",
    "route",
    "reason_codes",
    "evidence_ids",
    "citation_ids",
    "calculation_ids",
    "calculation_result_id",
    "citations",
    "calculations",
    "validator_status",
    "claim_provenance",
    # Identifiers of the artefacts the decision is about.  Deterministic from a
    # fixed request, so including them costs nothing and catches an outcome that
    # reports a different packet than the one it was built from.
    "plan_id",
    "evidence_packet_id",
    # Nothing populates this yet.  Classified as decision-bearing anyway: a
    # measurement container is exactly the kind of field that quietly grows a
    # decision payload, and the cost of comparing an empty dict is zero.
    "latency_metadata",
)

#: Keys inside ``runtime_metadata`` that must match.  ``runtime_metadata`` as a
#: whole is decision-bearing -- it carries ``release_decision``,
#: ``validation_status``, ``failed_checks`` and the terminal state -- so it is
#: compared in full, minus C and minus B.
DECISION_BEARING_METADATA = "runtime_metadata"


# --- B. harness_v3 is expected to add ---------------------------------------

#: ``harness_v3`` runs calculation as a loop phase, and records that it did.
#: ``legacy`` has no equivalent marker because its calculation is not a phase.
HARNESS_ONLY_METADATA_KEYS: tuple[str, ...] = ("calculation_in_harness",)

#: Container fields whose *contents* are execution record rather than decision.
#: ``debug_metadata`` holds the execution trace, which ``harness_v3`` is
#: expected to differ on: it runs more phases, so it records more phases.  The
#: trace is still verified -- just against its own invariants, in
#: ``h1_integration._trace_problems``, not against the legacy trace.
HARNESS_ONLY_FIELDS: tuple[str, ...] = ("debug_metadata",)


# --- C. measurements and per-run identifiers --------------------------------

VOLATILE_METADATA_KEYS: tuple[str, ...] = (
    "latency_ms",
    "duration_ms",
    "elapsed_ms",
    "run_id",
    "span_id",
)


def _jsonable(value: Any) -> Any:
    """Reduce a contract object to plain JSON-comparable data.

    ``Enum`` is checked before ``str`` on purpose.  ``V2ExecutionStatus`` and
    ``ReleaseStatus`` both subclass ``str``, so the other order makes the enum
    branch unreachable for exactly the two fields it exists for -- the payload
    then holds enum members rather than their values, which compares and
    serialises fine and quietly stops being "plain JSON data".
    """

    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


def _strip(value: Any, keys: frozenset[str]) -> Any:
    """Remove ``keys`` at any depth, leaving the shape otherwise intact."""

    if isinstance(value, Mapping):
        return {
            str(key): _strip(item, keys)
            for key, item in value.items()
            if key not in keys
        }
    if isinstance(value, (list, tuple)):
        return [_strip(item, keys) for item in value]
    return value


_VOLATILE = frozenset(VOLATILE_METADATA_KEYS)
_METADATA_CONTAINERS = (
    DECISION_BEARING_METADATA,
    "latency_metadata",
)

#: Everything the comparison reads.  ``DECISION_BEARING_METADATA`` is a field
#: name as well as a key name, and it is deliberately not repeated in
#: ``DECISION_BEARING_FIELDS`` -- so the payload has to be built from the union.
#: Building it from ``DECISION_BEARING_FIELDS`` alone silently stopped comparing
#: ``runtime_metadata``, which carries ``release_decision``,
#: ``validation_status``, ``failed_checks`` and the terminal state, and every
#: differential test stayed green.  ``test_every_contract_name_reaches_the_payload``
#: now pins that.
CONTRACT_FIELDS: tuple[str, ...] = (*DECISION_BEARING_FIELDS, DECISION_BEARING_METADATA)


def canonicalize_decision_result(outcome: Any) -> dict[str, Any]:
    """Return the decision surface both runtime modes must agree on.

    The result is plain JSON data, so it can be compared directly or written
    into an integration report without a second serializer.
    """

    ignored = _VOLATILE | frozenset(HARNESS_ONLY_METADATA_KEYS)
    payload: dict[str, Any] = {}
    for name in CONTRACT_FIELDS:
        value = _jsonable(getattr(outcome, name, None))
        if name in _METADATA_CONTAINERS:
            value = _strip(value, ignored)
        payload[name] = value
    return payload


def unclassified_fields(outcome_type: Any) -> list[str]:
    """Fields of an outcome the contract has not placed in a bucket.

    Nothing may be silently uncompared.  A new field on
    ``V2ExecutionOutcome`` has to be classified as decision-bearing,
    harness-only or volatile before this returns empty again.
    """

    classified = (
        set(DECISION_BEARING_FIELDS)
        | set(HARNESS_ONLY_FIELDS)
        | {DECISION_BEARING_METADATA}
    )
    fields = getattr(outcome_type, "__dataclass_fields__", {})
    return sorted(name for name in fields if name not in classified)


def unknown_contract_fields(outcome_type: Any) -> list[str]:
    """Names in the contract that the outcome does not actually have.

    The other direction of :func:`unclassified_fields`.  A typo in
    ``DECISION_BEARING_FIELDS`` used to be invisible -- ``getattr(outcome,
    name, None)`` returns ``None`` on both sides, the fields compare equal, and
    the contract silently stops comparing the thing it names.
    """

    fields = set(getattr(outcome_type, "__dataclass_fields__", {}))
    named = set(DECISION_BEARING_FIELDS) | set(HARNESS_ONLY_FIELDS)
    return sorted(name for name in named if name not in fields)


def _require_classified(outcome: Any, side: str) -> None:
    """Refuse to compare an outcome whose fields the contract has not placed.

    This is deliberately *not* a check that the payload contains every name in
    ``CONTRACT_FIELDS``.  That check is circular and cannot fire:
    ``canonicalize_decision_result`` builds the payload *by iterating the same
    list*, so removing a name removes it from both sides and the guard agrees
    with itself.  It was written that way once and gave false assurance.

    The non-circular question is the reverse one: does the contract classify
    every field this outcome actually has?  A field nobody classified is neither
    compared nor explicitly excused, which is the failure mode this module
    exists to prevent.
    """

    unclassified = unclassified_fields(type(outcome))
    if unclassified:
        raise AssertionError(
            f"the {side} outcome has fields the contract does not classify: "
            f"{unclassified}. Classify them as decision-bearing, harness-only "
            f"or volatile; an unclassified field is neither compared nor "
            f"excused."
        )


def _require_covers(outcome: Any, payload: Mapping[str, Any], side: str) -> None:
    """Refuse a payload that does not carry every decision-bearing outcome field.

    Together with :func:`_require_classified` this closes both ways a comparison
    can silently stop comparing, and neither check is circular:

    * contract truncated -- a name dropped from ``CONTRACT_FIELDS`` shows up as
      an *unclassified outcome field*, which the type-level check above catches;
    * builder collapsed -- a payload missing names the outcome has is caught
      here, because the expected set comes from the outcome type rather than from
      the list the builder iterates.
    """

    expected = {name for name in CONTRACT_FIELDS if hasattr(outcome, name)}
    missing = sorted(expected - set(payload))
    if missing:
        raise AssertionError(
            f"the {side} decision payload is missing fields the outcome "
            f"carries: {missing}; comparing it would report agreement about "
            f"something it never read"
        )


def decision_differences(legacy: Any, harness: Any) -> list[dict[str, Any]]:
    """Return one entry per decision-bearing field the two modes disagree on."""

    _require_classified(legacy, "legacy")
    _require_classified(harness, "harness_v3")
    left = canonicalize_decision_result(legacy)
    right = canonicalize_decision_result(harness)
    _require_covers(legacy, left, "legacy")
    _require_covers(harness, right, "harness_v3")
    return [
        {"field": field, "legacy": left.get(field), "harness_v3": right.get(field)}
        for field in sorted(set(left) | set(right))
        if left.get(field) != right.get(field)
    ]


def decision_equivalent(legacy: Any, harness: Any) -> bool:
    return not decision_differences(legacy, harness)


def assert_decision_equivalent(legacy: Any, harness: Any) -> None:
    """Fail with the full field-level diff rather than the first mismatch."""

    differences = decision_differences(legacy, harness)
    if differences:
        rendered = "\n".join(
            f"  {item['field']}:\n"
            f"    legacy    = {item['legacy']!r}\n"
            f"    harness_v3= {item['harness_v3']!r}"
            for item in differences
        )
        raise AssertionError(
            "legacy and harness_v3 disagree on "
            f"{len(differences)} decision-bearing field(s):\n{rendered}"
        )
