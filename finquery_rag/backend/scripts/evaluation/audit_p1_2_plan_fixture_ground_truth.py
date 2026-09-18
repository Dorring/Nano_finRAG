"""P1.3-0B: does each frozen plan fixture actually carry its own ground truth?

The P1.2 fixture builder was allowed to *author* ground truth from the question
when the gold did not supply it, and it did so mechanically: it took the first
period the question named and copied it into every slot.  A plan whose slots are
indistinguishable cannot express what the question asks for, so a binder that
fails against it has not been shown to be wrong -- and a binder that *succeeds*
against it has not been shown to be right.

This audit does not judge the binder.  It judges the fixtures, per question,
against the question and the gold, and it is deliberately conservative: it flags
only what it can establish structurally, and sends everything else to the
review tier rather than guessing.

Three defect classes are separable, and they were found by reading the builder,
not by pattern-matching the output:

``LOST_PERIOD_OPERAND``
    The operation's arity is over periods (``sum``, ``average``), the question
    names more periods than the slots carry, and the missing ones were never in
    the gold.  Fixed in fixture v2: the periods now come from the gold operand
    facts.  ``build_p1_2_plan_fixtures.py``.

``SLOT_IDENTITY_COLLISION``
    Two slots identical on the whole coordinate.  This one was structural and
    could not be fixed in the builder: ``RequiredSlot`` had no entity field, so
    the plan contract could not express "Apple's revenue" and "Visa's revenue"
    as two different requirements at all -- and because the builder also gave
    every multi-evidence plan a literal two slots, a four-company ranking asked
    for two.  Fixed in fixture v3, on top of the contract change that added the
    coordinate.

``GARBLED_METRIC``
    The metric came from the question rather than the gold, and is not a metric.

``UNDERSPECIFIED_SLOT_COUNT``
    The plan asks for fewer facts than the gold names.

Output is written to ``--out-dir``; the script prints only the summary.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_EVAL_SET = Path(
    "benchmarks/tv2_canonical_v1/canonical-eval-v1.jsonl"
)
DEFAULT_GOLD = Path(
    "benchmarks/tv2_canonical_v1/gold-evidence-v1.jsonl"
)
DEFAULT_FIXTURES = Path(
    "benchmarks/tv2_canonical_v1/plan-fixtures-v3.jsonl"
)
DEFAULT_OUT_DIR = Path("artifacts/evaluation/p1-3-ground-truth-audit")

_PERIOD_RE = re.compile(r"\bFY\s?\d{4}\b", re.IGNORECASE)

#: Operations whose arity is over *periods* -- two operands of one metric at two
#: times.  ``percentage_share`` is two slots too, but numerator and denominator
#: are different quantities at the same time, so a shared period is correct
#: there and must not be flagged.
_MULTI_PERIOD_OPERATIONS = frozenset({"growth_rate", "difference", "sum", "average"})

#: A metric is a field name from a filing.  These are things the question-text
#: fallback produced that are plainly not: a bare year, or a truncated header.
_GARBLED_METRIC_RE = re.compile(r"^(?:FY?\s?\d{4}|\d{4}|\d+)$", re.IGNORECASE)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _slot_signature(slot: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    """What has to differ for two slots to be two requirements.

    ``slot_id`` is excluded on purpose: it is an index, and two slots that
    differ only by index are the same requirement written twice.

    ``entity`` is included because the contract added it to the coordinate -- a
    plan may now say *whose* fact it needs, and two slots naming two companies
    are two requirements however identical the rest of them is.  Reading the
    signature without it would report the repaired comparison fixtures as still
    colliding, which is the audit disagreeing with the contract it audits.
    """

    return (
        str(slot.get("role", "")),
        str(slot.get("metric", "")),
        str(slot.get("period", "")),
        str(slot.get("unit") or ""),
        str(slot.get("entity") or ""),
    )


def audit_case(
    question_record: Mapping[str, Any],
    gold: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    question = str(question_record.get("question", ""))
    plan = fixture.get("plan") or {}
    slots: Sequence[Mapping[str, Any]] = plan.get("required_slots") or ()
    operation = plan.get("operation")
    sourced = fixture.get("sourced_from") or {}

    signatures = [_slot_signature(slot) for slot in slots]
    distinct = set(signatures)

    periods_in_question = sorted(
        {match.group(0).replace(" ", "").upper() for match in _PERIOD_RE.finditer(question)}
    )
    periods_in_slots = sorted({str(slot.get("period", "")) for slot in slots})

    # How many independent pieces of evidence the gold says are needed.  This
    # is the one arity signal that does not come from the plan, the question
    # text or the binder -- an abstention has none by definition, and a ranking
    # question has one per company it names.
    gold_support = len(gold.get("fact_ids") or [])
    is_abstention = str(gold.get("expected_outcome") or "").upper() == "ABSTENTION"

    metric = str(slots[0].get("metric", "")) if slots else ""
    metric_from = str(sourced.get("metric", ""))

    findings: list[str] = []

    # Two slots that are the same requirement cannot be told apart, so no
    # binder output can say which one a fact was meant for and coverage cannot
    # be counted per slot.  This is where the plan contract itself runs out:
    # ``RequiredSlot`` has no entity field, so two companies at one period have
    # nowhere to differ except ``role``, which the builder leaves as ``value``.
    if len(signatures) > 1 and len(distinct) < len(signatures):
        findings.append("SLOT_IDENTITY_COLLISION")

    # Fewer slots than the gold has supports: the plan is asking for less than
    # the question needs, so a "miss" here is the plan's, not the binder's.
    if not is_abstention and gold_support > len(signatures):
        findings.append("UNDERSPECIFIED_SLOT_COUNT")

    # A multi-period operation that carries fewer distinct periods than the
    # question names lost an operand.  The builder duplicated the first period
    # rather than dropping it, which is why the check is on *distinct* periods
    # and not on the slot count -- the slot count is always right.
    if str(operation) in _MULTI_PERIOD_OPERATIONS:
        if len(periods_in_question) > 1 and len(periods_in_slots) < len(periods_in_question):
            findings.append("LOST_PERIOD_OPERAND")

    # Only judge a metric the builder derived itself; a gold metric is the
    # corpus's own answer and this audit has no standing to overrule it.
    if metric_from.startswith("question") and (
        _GARBLED_METRIC_RE.match(metric.strip()) or len(metric.strip()) < 3
    ):
        findings.append("GARBLED_METRIC")

    return {
        "id": question_record["id"],
        "stratum": question_record.get("stratum"),
        "question": question,
        "operation": operation,
        "intent": plan.get("intent"),
        "slot_signatures": [list(item) for item in signatures],
        "distinct_slot_count": len(distinct),
        "slot_count": len(signatures),
        "gold_support_count": gold_support,
        "is_abstention": is_abstention,
        "periods_in_question": periods_in_question,
        "periods_in_slots": periods_in_slots,
        "metric": metric,
        "metric_from": metric_from,
        "period_from": str(sourced.get("period", "")),
        "findings": findings,
        # The plan can be bound against only if its slots can be told apart
        # *and* it asks for as much as the question needs.
        "ground_truth": "SUSPECT" if findings else "OK",
    }


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_stratum: dict[str, Counter[str]] = {}
    for row in rows:
        counter = by_stratum.setdefault(str(row["stratum"]), Counter())
        counter["total"] += 1
        # A case counts once per class it carries; the class totals are
        # incidence and the verdict is not, so both are reported rather than
        # one being derived from the other.
        if not row["findings"]:
            counter["ok"] += 1
        for finding in row["findings"]:
            counter[finding] += 1

    verdicts = Counter(
        "SUSPECT" if row["findings"] else "OK" for row in rows
    )
    return {
        "total": len(rows),
        "verdicts": dict(verdicts),
        "finding_incidence": dict(
            Counter(f for row in rows for f in row["findings"])
        ),
        "by_stratum": {key: dict(value) for key, value in sorted(by_stratum.items())},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    questions = _load_jsonl(args.eval_set)
    gold = {row["id"]: row for row in _load_jsonl(args.gold)}
    fixtures = {row["id"]: row for row in _load_jsonl(args.fixtures)}

    rows = [
        audit_case(record, gold.get(record["id"], {}), fixtures[record["id"]])
        for record in questions
    ]
    summary = summarise(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fixture-ground-truth-audit.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
