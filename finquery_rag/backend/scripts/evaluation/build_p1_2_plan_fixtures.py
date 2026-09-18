#!/usr/bin/env python3
"""P1.2: author and freeze the ALIGNED-REPLAY plan fixtures.

The dual-track benchmark needs, for every question, the plan a *correct*
supervisor would have produced.  This module authors that plan from the
**question and the gold metadata only**, and freezes it with a digest before any
run.

What it deliberately does not use, and why each one would invalidate the
benchmark:

* **the current Supervisor's output** -- the fixture would then be production
  output used as both actual and expected;
* **`align_query_to_plan`'s verdict** -- authoring a plan to satisfy the gate is
  fitting the fixture to the thing under test;
* **any runtime artefact** -- retrieved evidence, bound evidence, a calculation
  or an answer.  The fixture carries *plan* fields (intent, metric, period,
  role, operation) and nothing else.  In particular the gold's `fact_ids`,
  `expected_value`, `operands` and `values` never enter a fixture, because a
  benchmark that hands the runtime the answer is not a RAG benchmark.

Every authored field records which tier produced it
(``sourced_from``), so a reader can see how much of a plan is gold and how much
is derived from the question text.  A plan is only as trustworthy as that record.

    python build_p1_2_plan_fixtures.py --out-dir artifacts/evaluation/p1-2-plan-fixtures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

DEFAULT_EVAL_SET = (
    _BACKEND_DIR
    / "artifacts/evaluation/tv2-final-01-canonical-eval-set/canonical-eval-v1.jsonl"
)
DEFAULT_GOLD = (
    _BACKEND_DIR
    / "artifacts/evaluation/tv2-final-01-canonical-eval-set/gold-evidence-v1.jsonl"
)
DEFAULT_OUT_DIR = _BACKEND_DIR / "artifacts/evaluation/p1-2-plan-fixtures"

#: Gold `operation` values that are executable plan operations.  The rest of the
#: gold vocabulary (`comparison`, `cross_entity_difference`, `ranking`) names a
#: *shape of answer* rather than an arithmetic operation, and `SupervisorPlan`
#: refuses an operation on a non-`CALCULATION` intent -- so those become
#: `MULTI_EVIDENCE` with no operation instead of being forced into a calculation
#: they are not.
_ARITHMETIC_OPERATIONS = {
    "growth_rate": "growth_rate",
    "percentage_share": "percentage_share",
    "difference": "difference",
    "sum": "sum",
    "average": "average",
    "gross_margin": "gross_margin",
    "net_margin": "net_margin",
    "debt_ratio": "debt_ratio",
    "scale_conversion": "scale_conversion",
}

#: Slots per operation, in the role vocabulary `validate_plan_v2_01` accepts.
#: The count is the operation's arity, not a guess: a `growth_rate` with one
#: operand is not a growth rate.
_OPERATION_SLOTS: dict[str, tuple[str, ...]] = {
    "growth_rate": ("current", "prior"),
    "percentage_share": ("numerator", "denominator"),
    "difference": ("minuend", "subtrahend"),
    "sum": ("operand", "operand"),
    "average": ("operand", "operand"),
    "gross_margin": ("gross_profit", "revenue"),
    "net_margin": ("net_income", "revenue"),
    "debt_ratio": ("debt", "assets"),
    "scale_conversion": ("value",),
}

_PERIOD_RE = re.compile(r"\bFY\s?\d{4}\b", re.IGNORECASE)
_ENTITY_CLAUSE_RE = re.compile(r"'s\s+(?P<metric>.+?)\s+(?:in|as of|for)\s+", re.IGNORECASE)
_REPORTED_RE = re.compile(
    r"\b(?:reported|figure for|value (?:reported )?(?:by .+? )?for)\s+(?P<metric>.+?)"
    r"\s+(?:in|as of|for)\s+",
    re.IGNORECASE,
)
_TRAILING_RE = re.compile(r"\s+(?:in|as of|for)\s+FY\s?\d{4}\b.*$", re.IGNORECASE)


def _period_from_question(question: str) -> str | None:
    match = _PERIOD_RE.search(question)
    return match.group(0).replace(" ", "").upper() if match else None


def _metric_from_question(question: str) -> tuple[str, str] | None:
    """The metric phrase the question is about, or ``None``.

    Two authored patterns cover the corpus's question shapes, and a last-resort
    trim keeps the fallback mechanical rather than clever.  The tier is returned
    so the frozen fixture can say which one fired.
    """

    for pattern, tier in (
        (_REPORTED_RE, "question_reported_clause"),
        (_ENTITY_CLAUSE_RE, "question_possessive_clause"),
    ):
        match = pattern.search(question)
        if match:
            metric = match.group("metric").strip(" ?.,")
            if metric:
                return metric, tier

    trimmed = _TRAILING_RE.sub("", question).strip(" ?.,")
    for prefix in (
        "Which company had a higher ",
        "Compare ",
        "What was the difference in ",
        "Calculate the percentage change in ",
    ):
        if trimmed.lower().startswith(prefix.lower()):
            trimmed = trimmed[len(prefix) :].strip()
    if 0 < len(trimmed) <= 80:
        return trimmed, "question_trimmed"
    return None


def author_plan(
    question: str,
    gold: dict[str, Any],
    question_record: dict[str, Any],
) -> dict[str, Any]:
    """Author one plan fixture.  Returns the JSON row, not a runtime object."""

    sourced: dict[str, str] = {}

    gold_metric = gold.get("metric")
    if isinstance(gold_metric, str) and gold_metric.strip():
        metric, sourced["metric"] = gold_metric.strip(), "gold"
    else:
        derived = _metric_from_question(question)
        if derived is None:
            raise ValueError(f"no metric could be authored for {question_record['id']!r}")
        metric, sourced["metric"] = derived

    periods: list[str] = []
    sources_period = ""
    if isinstance(gold.get("period"), str) and gold["period"].strip():
        periods = [gold["period"].strip()]
        sources_period = "gold"
    elif gold.get("period_current") and gold.get("period_previous"):
        periods = [str(gold["period_current"]).strip(), str(gold["period_previous"]).strip()]
        sources_period = "gold_operand_periods"
    else:
        derived_period = _period_from_question(question)
        if derived_period is None:
            raise ValueError(
                f"no period could be authored for {question_record['id']!r}"
            )
        periods = [derived_period, derived_period]
        sources_period = "question_text"
    sourced["period"] = sources_period

    operation = gold.get("operation")
    plan_operation = _ARITHMETIC_OPERATIONS.get(str(operation)) if operation else None

    if plan_operation is not None:
        intent = "CALCULATION"
        sourced["intent"] = f"gold_operation:{operation}"
        roles = _OPERATION_SLOTS[plan_operation]
        slots = [
            {
                "slot_id": f"s{index + 1}",
                "metric": metric,
                "period": periods[min(index, len(periods) - 1)],
                "role": role,
                "value_type": "numeric",
                "unit": None,
            }
            for index, role in enumerate(roles)
        ]
    else:
        # An abstention question is a lookup that retrieval will not satisfy --
        # authoring it as ABSTAIN would put the expected failure into the plan
        # instead of letting the pipeline reach it, and would make the gold's
        # `expected_failure_reason` unreachable by construction.
        intent = (
            "MULTI_EVIDENCE"
            if str(gold.get("operation") or "") in {"comparison", "cross_entity_difference", "ranking"}
            else "DIRECT_FACT"
        )
        sourced["intent"] = (
            f"gold_operation:{operation}"
            if operation
            else (
                "gold_outcome:abstention"
                if gold.get("expected_outcome") == "ABSTENTION"
                else "gold_intent:" + str(question_record.get("expected_intent"))
            )
        )
        # A comparison names two entities; the plan carries one slot per side so
        # the binder has something to bind against, rather than a single slot
        # that would silently answer a one-sided question.
        arity = 2 if intent == "MULTI_EVIDENCE" else 1
        slots = [
            {
                "slot_id": f"s{index + 1}",
                "metric": metric,
                "period": periods[min(index, len(periods) - 1)],
                "role": "value",
                "value_type": "numeric",
                "unit": None,
            }
            for index in range(arity)
        ]

    return {
        "id": question_record["id"],
        "stratum": question_record["stratum"],
        "question": question,
        "plan": {
            "intent": intent,
            "required_slots": slots,
            "operation": plan_operation,
            "next_action": "RETRIEVE",
        },
        "sourced_from": sourced,
    }


def build_fixtures(eval_path: Path, gold_path: Path) -> list[dict[str, Any]]:
    questions = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold_by_id = {
        json.loads(line)["id"]: json.loads(line)
        for line in gold_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return [
        author_plan(
            record["question"],
            gold_by_id.get(record["id"], {}),
            record,
        )
        for record in questions
    ]


def render_fixtures(fixtures: list[dict[str, Any]]) -> str:
    return "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) for row in fixtures
    ) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--gold-evidence", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    fixtures = build_fixtures(args.eval_set, args.gold_evidence)
    rendered = render_fixtures(fixtures)
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = args.out_dir / "plan-fixtures-v1.jsonl"
    fixture_path.write_text(rendered, encoding="utf-8", newline="\n")
    (args.out_dir / "plan-fixtures-v1.jsonl.sha256").write_text(
        digest + "\n", encoding="utf-8", newline="\n"
    )

    intents = Counter(row["plan"]["intent"] for row in fixtures)
    sources = Counter(
        f"{field}:{tier}"
        for row in fixtures
        for field, tier in row["sourced_from"].items()
    )
    manifest = {
        "stage": "P1-2-PLAN-FIXTURES",
        "total": len(fixtures),
        "fixture_sha256": digest,
        "eval_set_sha256": hashlib.sha256(args.eval_set.read_bytes()).hexdigest(),
        "gold_sha256": hashlib.sha256(args.gold_evidence.read_bytes()).hexdigest(),
        "intent_distribution": dict(sorted(intents.items())),
        "provenance": dict(sorted(sources.items())),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"wrote {len(fixtures)} fixtures to {fixture_path}")
    print(f"  sha256    : {digest}")
    print(f"  intents   : {dict(sorted(intents.items()))}")
    for key, count in sorted(sources.items()):
        print(f"    {key:<42} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
