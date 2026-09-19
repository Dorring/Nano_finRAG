"""Did the release rest on operands the gold actually authorises?

``decision_correct`` asks whether the answer the system rendered matches the
gold.  That is necessary and not sufficient, and `rank-002` is the case that
shows why: it released ``Microsoft > Tesla > Apple`` against a gold of
``[Microsoft, Tesla, Apple]`` -- correct -- while its Tesla operand was
``1,332`` where the gold fact is ``6,411``.  The ordering survived only because
``32488 > 1332 > -34550`` has the same shape as ``32488 > 6411 > -34550``.

A scorer that reads only the final relation records that as a success.  It is
not one.  The evidence the system stated was wrong, and the answer was right by
coincidence.  Worse, the coincidence is invisible in the aggregate: it inflates
the release count in exactly the direction that flatters the system.

So this module splits the verdict into the claims it was silently conflating:

    decision_correct         the rendered relation matches the gold
    operand_grounded_correct every operand the executor used states a gold
                             operand value
    benchmark_gold_valid     the gold is itself a trustworthy oracle for this
                             case -- see `P1.6-0C`, which found gold that took
                             one cell of a flattened segment table
    trusted_release_correct  all three, together

Only the last is a claim about the system.  The first three are there so that
a failure can be attributed: an answer that is wrong, an answer that is right
for the wrong reason, and a case the benchmark cannot judge at all are three
different findings and must not collapse into one number.

**Grounding is judged on values, not on fact ids.**  The gold corpus' `fact_ids`
and the fact store's candidate keys are different id spaces -- every operand in
the clean `crossdiff-001` release fails an id-equality check -- so comparing ids
would mark correct releases ungrounded.  The operand *values* are in both, and
are compared as a multiset so that a repeated operand cannot be matched twice.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from src.evaluation.p1_2_dual_track import _decimal


class GoldValidity(str, Enum):
    """Whether a case's gold can judge the system at all."""

    #: The audit confirmed the gold value is the one the source supports.
    VALID = "VALID"
    #: The gold took one cell of a flattened multi-column table -- a segment
    #: where the question asks for the Firm.  Fixing it is a fixture change.
    WRONG_SCOPE = "WRONG_SCOPE"
    #: The source cannot separate the gold from its rivals without the original
    #: filing.  The case cannot judge the system either way until then.
    UNRESOLVED = "UNRESOLVED"
    #: Out of the audited stratum.  Nothing is claimed about it, and it is not
    #: treated as invalid -- an unaudited gold is not a discredited one.
    UNAUDITED = "UNAUDITED"


class ReleaseVerdict(str, Enum):
    """What a release actually established."""

    #: Right answer, gold-authorised operands, trustworthy gold.
    TRUSTED = "TRUSTED"
    #: Right answer from an operand the gold does not authorise.  `rank-002`.
    UNGROUNDED = "UNGROUNDED"
    #: Right answer, but the gold cannot judge the case (`P1.6-0C`).
    GOLD_UNGRADEABLE = "GOLD_UNGRADEABLE"
    #: Released and wrong.  The thing the release invariant exists to prevent.
    WRONG = "WRONG"
    #: Not released, so no claim about a release is available.
    NOT_RELEASED = "NOT_RELEASED"


def load_gold_validity(path: Path | str) -> dict[str, GoldValidity]:
    """Per-case gold validity from the P1.6-0C artifact.

    Absent or unreadable, every case is ``UNAUDITED``: a scorer that cannot find
    the audit must not silently promote its cases to valid.
    """

    try:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    mapping = {"VALID_GOLD": GoldValidity.VALID,
               "WRONG_SCOPE_GOLD": GoldValidity.WRONG_SCOPE,
               "UNRESOLVED_GROUND_TRUTH": GoldValidity.UNRESOLVED}
    validity: dict[str, GoldValidity] = {}
    for case in artifact.get("cases") or ():
        label = mapping.get(str(case.get("case_classification")))
        if label is not None:
            validity[str(case.get("case_id"))] = label
    return validity


def gold_operand_values(gold_row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """The operand values the gold authorises, or ``None`` if it states none.

    The corpus spells this two ways and both are load-bearing: a cross-entity
    gold carries ``values`` keyed by entity, an arithmetic gold carries
    ``operands`` keyed by role.  A gold carrying neither cannot ground anything.
    """

    values = gold_row.get("values")
    if isinstance(values, Mapping) and values:
        return tuple(values.values())

    operands = gold_row.get("operands")
    if isinstance(operands, Mapping) and operands:
        return tuple(operands.values())
    if isinstance(operands, Sequence) and not isinstance(operands, (str, bytes)):
        collected = tuple(
            item.get("value") for item in operands if isinstance(item, Mapping)
        )
        if collected:
            return collected
    return None


def executed_operands(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Every operand the deterministic executor actually used.

    Read from the calculation results rather than from the answer text: the
    question is what the system executed, and a prose answer does not say.
    """

    found: list[Mapping[str, Any]] = []
    for calculation in row.get("calculations") or ():
        if not isinstance(calculation, Mapping):
            continue
        for operand in calculation.get("operands") or ():
            if isinstance(operand, Mapping):
                found.append(operand)
    return tuple(found)


def operand_grounded_correct(
    row: Mapping[str, Any], gold_row: Mapping[str, Any]
) -> bool | None:
    """Whether every executed operand states a value the gold authorises.

    ``None`` means the question does not apply -- no deterministic execution
    happened, or the gold states no operands to check against.  Returning
    ``False`` there would report every prose answer as ungrounded, which is a
    different claim and not this one.
    """

    operands = executed_operands(row)
    if not operands:
        return None
    authorised = gold_operand_values(gold_row)
    if not authorised:
        return None

    # `percent_is_ratio` follows the same rule the correctness scorer uses: an
    # arithmetic gold is a computed ratio, a factual one is the display string.
    percent_is_ratio = str(row.get("stratum")) == "arithmetic_calculation"
    tolerance = gold_row.get("tolerance")

    remaining = list(authorised)
    for operand in operands:
        stated = _decimal(operand.get("value"), percent_is_ratio=percent_is_ratio)
        if stated is None:
            return False
        for index, candidate in enumerate(remaining):
            expected = _decimal(candidate, percent_is_ratio=percent_is_ratio)
            if expected is None:
                continue
            limit = abs(_decimal(tolerance) or 0) if tolerance is not None else 0
            if abs(stated - expected) <= limit:
                del remaining[index]
                break
        else:
            # The operand consumes no authorised value.  It was manufactured,
            # mis-bound, or read from a coordinate that does not identify it.
            return False
    return True


def release_verdict(
    row: Mapping[str, Any],
    gold_row: Mapping[str, Any],
    *,
    decision_correct: bool,
    gold_validity: GoldValidity = GoldValidity.UNAUDITED,
) -> ReleaseVerdict:
    """What ``row`` established, when it released anything at all."""

    if row.get("release_status") != "RELEASED":
        return ReleaseVerdict.NOT_RELEASED
    if not decision_correct:
        return ReleaseVerdict.WRONG
    # A gold the audit discredited cannot certify the release, and neither can
    # one the source could not settle.  Both are reported as ungradeable rather
    # than as failures: the case says nothing about the system.
    if gold_validity in (GoldValidity.WRONG_SCOPE, GoldValidity.UNRESOLVED):
        return ReleaseVerdict.GOLD_UNGRADEABLE
    if operand_grounded_correct(row, gold_row) is False:
        return ReleaseVerdict.UNGROUNDED
    return ReleaseVerdict.TRUSTED
