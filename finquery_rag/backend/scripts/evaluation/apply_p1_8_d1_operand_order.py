#!/usr/bin/env python3
"""NOT APPLIED, AND NOT TO BE APPLIED — see docs/evaluation/p1-8-d1-c3-fixture-regeneration.md.

This is the applier for the narrow cross-entity operand-order patch, kept as the
record of a hypothesis that was tested and rejected. D1's root cause was a *stale
pinned fixture*, not a runtime defect: the runtime executed `current - previous`
faithfully, in the order the plan gave, and the plan carried the operand order of
a question the benchmark no longer asked. Regenerating the fixture through the
authoring contract fixed both cases with this patch OFF.

It is therefore **not** part of the runtime and must not be run against it. The
`_cross_entity_difference_specs` helper below exists nowhere in `src/`; if it
appears there, someone has applied this and the live operand authority has moved
from the plan to the question's surface wording, which is the failure mode D1
exists to rule out.

---

Apply the P1.8-D1 fix: cross-entity `difference between A and B`.

Responsibility boundary, fixed:
  Planner                  identifies the subtraction / difference intent
  Structured Operand Binding  determines lhs_slot_id / rhs_slot_id and emits
                              minuend / subtrahend
  Calculator               executes lhs - rhs, infers nothing

This patch touches **only** structured_operand_binding.py.  The operand-order
rule is not duplicated into operand_planner.py.

  python apply_p1_8_d1_operand_order.py /path/to/structured_operand_binding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HELPERS = '''

#: Entity surface forms as they appear in a question -> the store's `entity`.
#:
#: A cross-entity difference must bind by *entity*, not by the order candidate
#: facts happen to arrive in.  The alias table is what lets `NVIDIA` name the
#: same entity the store files as `NVIDIA` while `Coca-Cola` names
#: `The Coca-Cola Company`.
_ENTITY_ALIASES: tuple[tuple[str, str], ...] = (
    ("the coca-cola company", "The Coca-Cola Company"),
    ("coca-cola", "The Coca-Cola Company"),
    ("coca cola", "The Coca-Cola Company"),
    ("jpmorganchase", "JPMorganChase"),
    ("jpmorgan chase", "JPMorganChase"),
    ("jpmorgan", "JPMorganChase"),
    ("nvidia", "NVIDIA"),
    ("microsoft", "Microsoft"),
    ("apple", "Apple"),
    ("tesla", "Tesla"),
    ("pfizer", "Pfizer"),
    ("visa", "Visa"),
)

#: `difference in <metric> between <A> and <B> (in|as of|for) <period>`
_CROSS_ENTITY_DIFFERENCE_RE = re.compile(
    r"difference\\s+in\\s+(?P<metric>.+?)\\s+between\\s+"
    r"(?P<lhs>.+?)\\s+and\\s+(?P<rhs>.+?)"
    r"(?=\\s+(?:in|as of|for)\\s+(?:fiscal\\s+year\\s+|fy\\s*)?\\d{4}"
    r"|\\s*[?,.]|$)",
    re.IGNORECASE,
)


def _match_entity(text: str) -> str | None:
    """Resolve one mention to the store's entity name, longest alias first."""

    normalized = _SPACE_RE.sub(" ", str(text or "").strip().casefold())
    if not normalized:
        return None
    for alias, entity in _ENTITY_ALIASES:
        if alias == normalized:
            return entity
    for alias, entity in _ENTITY_ALIASES:
        if alias in normalized:
            return entity
    return None


def _cross_entity_difference_specs(
    question: str,
    calculation_intent: CalculationIntent,
) -> tuple[OperandBindingSpec, ...]:
    """`difference between A and B` -> A - B, ordered by explicit mention.

    The question's own word order is the authority.  The first-mentioned entity
    is the minuend and the second is the subtrahend; nothing about retrieval,
    candidate order or Binder order is consulted, and if either mention will not
    resolve to an entity this returns `()` so the caller fails closed rather
    than guessing a direction.
    """

    match = _CROSS_ENTITY_DIFFERENCE_RE.search(question)
    if not match:
        return ()
    lhs = _match_entity(match.group("lhs"))
    rhs = _match_entity(match.group("rhs"))
    if lhs is None or rhs is None:
        return ()
    # A cross-entity difference compares the same measure for both sides, so one
    # metric serves both roles; the entity is what separates them.
    metric = match.group("metric").strip()
    period = normalize_period(next(iter(calculation_intent.period_terms), ""))
    return (
        OperandBindingSpec("minuend", metric, period, lhs, None, None),
        OperandBindingSpec("subtrahend", metric, period, rhs, None, None),
    )

'''

BRANCH = '''    if operation.value == "difference":
        match = re.search(r":\\s*(.+?)\\s+or\\s+(.+?)(?:,|\\?|$)", question, re.IGNORECASE)
        period = normalize_period(next(iter(calculation_intent.period_terms), ""))
        if not match:
            # Not a `... or ...` period pair.  Before giving up, try the
            # cross-entity shape `difference in <metric> between A and B`.
            # The period-pair branch above is unchanged and still wins when it
            # matches, so existing behaviour is preserved.
            return _cross_entity_difference_specs(question, calculation_intent)
        return (
            OperandBindingSpec(
                "minuend", match.group(1).strip(), period, None, None, None
            ),
            OperandBindingSpec(
                "subtrahend", match.group(2).strip(), period, None, None, None
            ),
        )
'''

OLD_BRANCH = '''    if operation.value == "difference":
        match = re.search(r":\\s*(.+?)\\s+or\\s+(.+?)(?:,|\\?|$)", question, re.IGNORECASE)
        period = normalize_period(next(iter(calculation_intent.period_terms), ""))
        if not match:
            return ()
        return (
            OperandBindingSpec(
                "minuend", match.group(1).strip(), period, None, None, None
            ),
            OperandBindingSpec(
                "subtrahend", match.group(2).strip(), period, None, None, None
            ),
        )
'''

ANCHOR = "def build_operand_specs(\n"


def main() -> int:
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    if "_cross_entity_difference_specs" in text:
        print("already applied; nothing to do")
        return 0
    if OLD_BRANCH not in text:
        raise SystemExit("difference branch not found -- file differs from expected")
    if ANCHOR not in text:
        raise SystemExit("build_operand_specs anchor not found")

    text = text.replace(ANCHOR, HELPERS.lstrip("\n") + "\n" + ANCHOR, 1)
    text = text.replace(OLD_BRANCH, BRANCH, 1)
    path.write_text(text, encoding="utf-8")
    print("applied to %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
