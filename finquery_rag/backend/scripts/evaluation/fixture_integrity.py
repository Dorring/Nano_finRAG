#!/usr/bin/env python3
"""Does a `difference ... between A and B` plan name A first?

Why this exists
---------------

`crossdiff-001` and `crossdiff-002` were pinned with the operand order of a
question the benchmark no longer asked, and the runtime executed them
*faithfully*: `current - previous`, in the order the fixture's slots gave. The
answer came out with the wrong sign, and nothing was wrong with the runtime --
a plan is supposed to be authoritative, and overriding it with the question's
wording would be the runtime inventing an operand order the plan did not state.

So the defect was never in the runtime and could not be caught there. It was
that the fixture had drifted from the question it claims to answer, and no
check compared the two. This is that check.

The rule
--------

For a question that uniquely names two companies -- `difference in <metric>
between <A> and <B>` -- the fixture's first slot must be A and its second must
be B. Where the question does *not* uniquely resolve (a period pair, a mention
the alias table does not know, both mentions resolving to one company) the rule
does not bind and nothing is reported. Silence there is not a pass; it is the
absence of a claim.

**The question checked is the canonical one, passed in by the caller.**  This is
the whole point, and getting it wrong makes the guard useless: a stale fixture
is *internally* consistent -- `crossdiff-001`'s recorded question reads
`between Tesla and NVIDIA` and its slots are `[Tesla, NVIDIA]` -- so a check
that reads the fixture's own `question` field finds nothing wrong with it.  The
drift is between the fixture and the benchmark's `canonical-eval-v1.jsonl`, and
only a check that holds the two side by side can see it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

#: A question's mention -> the store's `entity`.  The check has to compare a
#: question written in prose against a fixture written in the store's
#: vocabulary, and this table is the whole of the difference between them.  It
#: is deliberately small and closed: a mention absent from it makes the question
#: *not* uniquely parseable, and the rule only binds when it is.  A matcher
#: would guess, and a guess here is the thing being guarded against.
ALIASES: tuple[tuple[str, str], ...] = (
    ("the coca-cola company", "The Coca-Cola Company"),
    ("the coca cola company", "The Coca-Cola Company"),
    ("coca-cola", "The Coca-Cola Company"),
    ("coca cola", "The Coca-Cola Company"),
    ("ko", "The Coca-Cola Company"),
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
_DIFFERENCE_RE = re.compile(
    r"difference\s+in\s+(?P<metric>.+?)\s+between\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)"
    r"(?=\s+(?:in|as of|for)\s+(?:fiscal\s+year\s+|fy\s*)?\d{4}|\s*[?,.]|$)",
    re.IGNORECASE,
)

#: Plan operations whose answer is `slot[0] - slot[1]` between two companies.
_OPERAND_ORDERED = frozenset({"difference", "cross_entity_difference"})


def _normalise(text: Any) -> str:
    return " ".join(str(text or "").strip().casefold().split())


def resolve_mention(mention: str) -> str | None:
    """A question's mention of a company -> the store's entity name, or None."""

    folded = _normalise(mention).strip(" ?.,")
    if not folded:
        return None
    for alias, entity in ALIASES:
        if alias == folded:
            return entity
    for alias, entity in ALIASES:
        if alias in folded:
            return entity
    return None


def check_row(row: Mapping[str, Any], question: str) -> dict[str, Any]:
    """One row's verdict against the canonical `question`.

    `violation` is set only when the rule binds *and* fails.  A record is
    returned either way so a caller can tell "checked and agreed" from "not
    checkable" -- different results that should not be collapsed.

    `question_drift` is reported separately and is not a violation: a fixture
    whose recorded question is not the canonical one is stale in exactly the way
    that caused this defect, but its operand order may still be right.  Three
    cases in the stratum are in that state by design (they are not operand-order
    defects), so failing on it would fail a fixture that answers its questions
    correctly.
    """

    case = row.get("id")
    plan = row.get("plan") or {}
    operation = str(plan.get("operation") or "")
    slots = list(plan.get("required_slots") or [])

    verdict: dict[str, Any] = {"id": case, "binds": False}
    recorded = str(row.get("question") or "")
    if recorded != question:
        verdict["question_drift"] = {"fixture": recorded, "canonical": question}

    if operation not in _OPERAND_ORDERED:
        verdict["reason"] = "not an operand-ordered operation"
        return verdict
    if len(slots) < 2:
        verdict["reason"] = "fewer than two slots"
        return verdict

    match = _DIFFERENCE_RE.search(question)
    if not match:
        verdict["reason"] = "question is not `difference ... between A and B`"
        return verdict

    first, second = resolve_mention(match.group("a")), resolve_mention(match.group("b"))
    if first is None or second is None:
        verdict["reason"] = "the two mentions do not uniquely resolve to store entities"
        verdict["mentions"] = [match.group("a").strip(), match.group("b").strip()]
        return verdict
    if first == second:
        verdict["reason"] = "both mentions resolve to one entity"
        return verdict

    observed = [str(slot.get("entity") or "") for slot in slots[:2]]
    verdict.update(
        {
            "binds": True,
            "question": question,
            "expected_slot_entities": [first, second],
            "observed_slot_entities": observed,
            "satisfied": observed == [first, second],
        }
    )
    if not verdict["satisfied"]:
        verdict["violation"] = (
            f"slot order disagrees with the canonical question: slot[0] is "
            f"{observed[0]!r} but the question names {first!r} first"
        )
    return verdict


def check_rows(
    rows: Iterable[Mapping[str, Any]],
    questions: Mapping[str, str],
) -> list[dict[str, Any]]:
    return [check_row(row, questions.get(str(row.get("id")), "")) for row in rows]


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def questions_by_id(eval_set: Path) -> dict[str, str]:
    return {
        str(row["id"]): str(row.get("question") or "")
        for row in load_rows(eval_set)
    }


def violations(verdicts: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [v for v in verdicts if v.get("violation")]


def assert_plan_integrity(question: str, row: Mapping[str, Any]) -> None:
    """Raise if `row` breaks the invariant.  For use inside a generator.

    The generator is where the defect was introduced, so the guard belongs here
    as well as in the standalone verifier: a fixture that cannot be verified
    should not be written in the first place.  Inside a generator the canonical
    question is simply the one being planned for.
    """

    verdict = check_row(row, question)
    if verdict.get("violation"):
        raise ValueError(f"{row.get('id')!r}: {verdict['violation']}")
