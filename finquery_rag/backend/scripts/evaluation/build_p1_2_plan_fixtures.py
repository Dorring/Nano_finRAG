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

The one thing it does read from outside the question is a fact's **coordinate**
-- its metric and period -- and only for the operations whose operands are one
gold fact each.  That is not a contradiction of the paragraph above and the
distinction is the point:

    the builder may SERIALISE ground truth.  It may not CREATE it.

For ``sum``/``average`` the gold names two operand facts but no periods, so the
first version of this builder re-derived the periods from the question -- using
the *first* period it found, twice.  Both slots then asked for the same
quantity, which the binder correctly refused, and ten benchmark questions
scored as binder failures for a fixture defect.  A period read off a gold fact
is a coordinate the corpus states; a period inferred from question text is this
script's opinion, and it was wrong.

So when the gold cannot supply the operands, this now **fails loudly**.  A
fixture that cannot be authored is a finding about the corpus, not something to
paper over with a guess.

Every authored field still records which tier produced it (``sourced_from``),
so a reader can see how much of a plan is gold and how much is question text.

    python build_p1_2_plan_fixtures.py --fact-store <store.jsonl> \
        --out-dir benchmarks/tv2_canonical_v1 --name plan-fixtures-v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# The Harness's own canonicalisers rather than a second opinion.  A fixture that
# wrote an ``entity_id`` some other way would be authoring an identity that the
# runtime then disagrees with, which is the drift this whole set exists to catch.
from rag_v2.supervisor import canonical_entity_id, canonical_scope_id  # noqa: E402
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

DEFAULT_EVAL_SET = (
    _BACKEND_DIR
    / "benchmarks/tv2_canonical_v1/canonical-eval-v1.jsonl"
)
DEFAULT_GOLD = (
    _BACKEND_DIR
    / "benchmarks/tv2_canonical_v1/gold-evidence-v1.jsonl"
)
DEFAULT_OUT_DIR = _BACKEND_DIR / "benchmarks/tv2_canonical_v1"

#: Where the build's intermediate findings go.  Under ``artifacts/``, which is
#: gitignored: the ambiguity list is a working record that will change as the
#: store or the fixtures change, and a file that gets rewritten every build does
#: not belong in the same tree as the fixtures it is about.
DEFAULT_AUDIT_DIR = _BACKEND_DIR / "artifacts/evaluation/p1-4d-fixture-audit"
#: The authoritative fact store.  It is the deployment's, not the checkout's, so
#: a full build runs where the store lives.
DEFAULT_FACT_STORE = Path(
    "/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl"
)

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

#: Gold operations that are executable *relational* operations.  They used to
#: fall into `_MULTI_EVIDENCE_OPERATIONS` below and become plans with no
#: operation, which left their answers as prose no validator had a structured
#: result to check -- the path `compare-002` released a wrong answer on.
#:
#: `cross_entity_difference` is deliberately **not** here.  Its gold expects a
#: numeric value rather than a relation, so it is an arithmetic `difference` and
#: belongs in a migration of its own; folding it in here would change two
#: different things in one measurement.
_RELATIONAL_OPERATIONS = {
    "comparison": "comparison",
    "ranking": "ranking",
}
#: Gold operations whose answer is a numeric difference *between entities*
#: rather than between periods.
#:
#: They were planned with no operation, which left five cases answering in
#: prose that no validator had a structured result to check -- the last
#: unchecked decision path in the stratum, and the last false release in it.
#: The arithmetic is the ordinary `difference`; what differs is only where the
#: operands come from, which is why this is a mapping rather than a new
#: executor.
#:
#: The gold's `fact_ids` are in slot order and the gold's `expected_value` is
#: `operands[0] - operands[1]`: `crossdiff-001` is Tesla `1,083` minus NVIDIA
#: `1,203` = `-120`, and `crossdiff-004` is Coca-Cola `850` minus Visa `19,602`
#: = `-18,752`. Roles `minuend`/`subtrahend` normalise to `current`/`previous`,
#: which is the role pair `difference` already expects.
_CROSS_ENTITY_ARITHMETIC = {"cross_entity_difference": "difference"}

_RELATIONAL_OPERATION_NAMES = frozenset(_RELATIONAL_OPERATIONS.values())

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

#: Operations whose operands are one gold fact each, in ``gold.fact_ids`` order.
#: Their periods come from those facts and never from the question: the question
#: names the periods, but *which* period belongs to *which* operand is the
#: gold's to say, and a builder that re-derives it is authoring ground truth.
#: ``growth_rate`` and ``difference`` are not here because their gold already
#: carries ``period_current``/``period_previous``.
#: Operations whose every operand is exactly one gold fact, so the fixture can
#: read the operand's coordinates rather than derive them from the question.
#:
#: ``percentage_share`` is here for the same reason as the other two and was
#: missing from it: its operands are two facts the gold names, so a metric read
#: off the question's possessive clause was never ground truth.  The clause
#: mashing both operands together -- "Impact of the State Aid Decision was
#: Statutory federal income tax rate" -- matched no fact in the corpus, the
#: binder correctly returned MISSING on both slots, and the whole operation was
#: unreachable in the benchmark.
_FACT_PER_OPERAND_OPERATIONS = frozenset({"sum", "average", "percentage_share"})

#: Gold operations that name a *shape of answer* rather than an arithmetic one.
#: They become MULTI_EVIDENCE plans carrying one slot per gold fact -- a
#: comparison names two sides, a ranking names as many sides as it ranks.
_MULTI_EVIDENCE_OPERATIONS = frozenset(
    {"comparison", "cross_entity_difference", "ranking"}
)

#: The join key between a gold ``fact_ids`` entry and a fact-store record.
_FACT_ID_FIELDS = ("candidate_key", "candidate_id", "fact_id", "evidence_id")

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


def load_fact_coordinates(
    fact_store: Path,
    fact_ids: Iterable[str],
) -> dict[str, Mapping[str, Any]]:
    """Read the records of exactly the facts a fixture build needs.

    One pass over the store keeping only the wanted ids, so a twenty-thousand
    record store costs seconds rather than memory.  A record is indexed under
    whichever of ``_FACT_ID_FIELDS`` the gold's id matched, and the first record
    for an id wins: a store can carry several extraction rows for one candidate,
    and they agree on the coordinate that is read from here.
    """

    wanted = {str(item).strip() for item in fact_ids if str(item).strip()}
    if not wanted:
        return {}
    found: dict[str, Mapping[str, Any]] = {}
    with fact_store.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for field in _FACT_ID_FIELDS:
                key = record.get(field)
                if key is not None and str(key) in wanted:
                    found.setdefault(str(key), record)
                    break
    return found


def _operand_coordinates_from_gold(
    gold: Mapping[str, Any],
    question_record: Mapping[str, Any],
    operand_facts: Mapping[str, Mapping[str, Any]],
    *,
    arity: int,
    operation: str,
) -> list[tuple[str, str]]:
    """One ``(metric, period)`` per gold operand fact, or a failure naming what is missing.

    Every branch here raises rather than falling back.  Each one is a statement
    that the corpus cannot support this question's fixture, and a fixture
    invented to cover that would measure this script's guess instead of the
    binder -- which is precisely how both ``sum`` operands came to name the same
    period, and how every ``percentage_share`` slot came to name a metric no
    fact carries.

    What must be distinct is the *coordinate*, not the period.  ``sum`` and
    ``average`` add one metric across two periods; ``percentage_share`` divides
    two metrics at one period.  Both are two requirements, and checking periods
    alone refused the second shape outright -- a check that only ever passed for
    the operation it was written for.
    """

    case = question_record["id"]
    fact_ids = [
        str(item).strip() for item in (gold.get("fact_ids") or []) if str(item).strip()
    ]
    if len(fact_ids) != arity:
        raise ValueError(
            f"{case!r}: {operation} needs {arity} operand facts, gold names {len(fact_ids)}"
        )

    coordinates: list[tuple[str, str]] = []
    for fact_id in fact_ids:
        record = operand_facts.get(fact_id)
        if record is None:
            raise ValueError(
                f"{case!r}: gold operand fact {fact_id!r} is not in the fact store"
            )
        period = str(record.get("period") or "").strip()
        if not period:
            raise ValueError(f"{case!r}: gold operand fact {fact_id!r} states no period")
        metric = str(record.get("metric") or "").strip()
        if not metric:
            raise ValueError(f"{case!r}: gold operand fact {fact_id!r} states no metric")
        coordinates.append((metric, period))

    # Two operands with one coordinate between them is the same requirement
    # written twice -- the binder refuses it, and it is the defect this reader
    # removes.
    if len(set(coordinates)) != len(coordinates):
        raise ValueError(
            f"{case!r}: {operation} operands are not distinct on "
            f"(metric, period): {coordinates}"
        )
    return coordinates


def author_plan(
    question: str,
    gold: dict[str, Any],
    question_record: dict[str, Any],
    operand_facts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Author one plan fixture.  Returns the JSON row, not a runtime object.

    ``operand_facts`` maps a gold ``fact_ids`` entry to its fact-store record and
    is read only for the operations whose operands are one fact each.  Omitting
    it is allowed for corpora that contain none of those; a corpus that contains
    one fails here rather than inventing the period.
    """

    sourced: dict[str, str] = {}

    gold_metric = gold.get("metric")
    states_metric = isinstance(gold_metric, str) and bool(gold_metric.strip())
    if states_metric:
        metric, sourced["metric"] = gold_metric.strip(), "gold"
    else:
        derived = _metric_from_question(question)
        if derived is None:
            raise ValueError(f"no metric could be authored for {question_record['id']!r}")
        metric, sourced["metric"] = derived

    operation = gold.get("operation")
    plan_operation = (
        _ARITHMETIC_OPERATIONS.get(str(operation))
        or _RELATIONAL_OPERATIONS.get(str(operation))
        or _CROSS_ENTITY_ARITHMETIC.get(str(operation))
        if operation
        else None
    )
    # A relational operation is planned on MULTI_EVIDENCE, because that is what
    # the question is -- it names several sides.  Execution no longer keys off
    # the intent (`registry.supports(operation)` decides that), so the intent is
    # free to say what was wanted rather than what may be executed.
    is_relational = plan_operation in _RELATIONAL_OPERATION_NAMES
    roles = _OPERATION_SLOTS.get(plan_operation or "", ())

    periods: list[str] = []
    operand_coordinates: list[tuple[str, str]] = []
    if plan_operation in _FACT_PER_OPERAND_OPERATIONS:
        operand_coordinates = _operand_coordinates_from_gold(
            gold,
            question_record,
            operand_facts or {},
            arity=len(roles),
            operation=str(plan_operation),
        )
        periods = [period for _metric, period in operand_coordinates]
        sourced["period"] = "gold_operand_facts"
    elif isinstance(gold.get("period"), str) and gold["period"].strip():
        periods = [gold["period"].strip()]
        sourced["period"] = "gold"
    elif gold.get("period_current") and gold.get("period_previous"):
        periods = [str(gold["period_current"]).strip(), str(gold["period_previous"]).strip()]
        sourced["period"] = "gold_operand_periods"
    else:
        derived_period = _period_from_question(question)
        if derived_period is None:
            raise ValueError(
                f"no period could be authored for {question_record['id']!r}"
            )
        periods = [derived_period, derived_period]
        sourced["period"] = "question_text"

    if plan_operation is not None and not is_relational:
        intent = "CALCULATION"
        sourced["intent"] = f"gold_operation:{operation}"
        # Which metric each slot needs.  A gold that states one states it for the
        # whole operation and every slot shares it -- the shape ``sum`` and
        # ``average`` have, and the reason their rows do not move here.  A gold
        # that states none names its operands by fact instead, and each slot
        # takes its own.  ``percentage_share`` divides two *different* metrics,
        # and a fixture allowed only one metric for both had to read them off the
        # question at once -- which is how it came to name a phrase that matches
        # no fact, and why the whole operation was unreachable.
        if operand_coordinates and not states_metric:
            slot_metrics = [operand_metric for operand_metric, _period in operand_coordinates]
            sourced["metric"] = "gold_operand_facts"
        else:
            slot_metrics = [metric] * len(roles)
        if str(operation) in _CROSS_ENTITY_ARITHMETIC:
            # One slot per gold fact, in the gold's order, each naming its own
            # company -- the shape the multi-evidence cases already use, because
            # these operands are two *entities'* facts rather than two periods
            # of one entity's.  Building from `roles` alone would hand both
            # slots the same metric and no entity, which is precisely the defect
            # the entity fields were added to remove.
            slots = _multi_evidence_slots(
                gold,
                question_record,
                operand_facts or {},
                metric=metric,
                periods=periods,
            )
            for index, slot in enumerate(slots):
                slot["role"] = roles[index] if index < len(roles) else "operand"
        else:
            slots = [
                {
                    "slot_id": f"s{index + 1}",
                    "metric": slot_metrics[index],
                    # ``min`` only ever clamps operations whose arity exceeds the
                    # periods the gold states; for the fact-per-operand operations
                    # the two lengths are checked equal before this point.
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
        if intent == "MULTI_EVIDENCE":
            # One slot per gold fact, each naming its own company.  The arity is
            # the gold's, not a literal: a ranking names as many sides as it
            # ranks, and a plan that asked for two of four could not be
            # satisfied by any binder.
            slots = _multi_evidence_slots(
                gold,
                question_record,
                operand_facts or {},
                metric=metric,
                periods=periods,
            )
            sourced["entity"] = "gold_operand_facts"
        else:
            slots = [
                {
                    "slot_id": "s1",
                    "metric": metric,
                    "period": periods[0],
                    "role": "value",
                    "value_type": "numeric",
                    "unit": None,
                }
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


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    """The question rows by id.  The company a question names lives here."""

    return {
        json.loads(line)["id"]: json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _load_gold(path: Path) -> dict[str, dict[str, Any]]:
    return _load_records(path)


def build_fixtures(
    eval_path: Path,
    gold_path: Path,
    operand_facts: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    questions = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold_by_id = _load_gold(gold_path)
    return [
        author_plan(
            record["question"],
            gold_by_id.get(record["id"], {}),
            record,
            operand_facts,
        )
        for record in questions
    ]


def operand_fact_ids(gold_path: Path) -> list[str]:
    """Every gold fact id the build may need to read a coordinate from.

    This used to be scoped to the operations whose *slots* were authored from
    operand facts.  It is now every gold fact, because every answerable slot is
    enriched from its fact: a lookup slot that names only a metric is
    under-determined store-wide, and the entity that fixes that comes from the
    fact the gold points at, not from the question.
    """

    needed: list[str] = []
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        needed.extend(str(item) for item in (row.get("fact_ids") or []))
    return needed


def load_coordinate_index(store: Path) -> dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]]:
    """How many facts each coordinate selects, indexed by coordinate subset.

    Only ids are kept -- the validation needs counts, not content, so this stays
    small where ``load_fact_coordinates`` deliberately keeps only the records a
    build asks for.

    Subsets rather than prefixes, and for the same reason as in the audit: a slot
    that names no scope does not constrain the scope, so it must be counted
    against facts of every scope.  A prefix key would silently count it only
    against facts whose scope is empty, which is a different and wrong question.
    """

    fields = ("entity", "metric", "period", "scope")
    subsets = [
        combo
        for width in range(1, len(fields) + 1)
        for combo in _combinations(fields, width)
    ]
    index: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = {}
    with store.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("fact_type") != "atomic":
                continue
            fact_id = str(record.get("fact_id") or record.get("candidate_key") or "")
            if not fact_id:
                continue
            for combo in subsets:
                key = (combo, tuple(_fold_field(record.get(f)) for f in combo))
                index.setdefault(key, []).append(fact_id)
    return index


def _combinations(items: tuple[str, ...], width: int) -> list[tuple[str, ...]]:
    from itertools import combinations

    return list(combinations(items, width))


def _fold_field(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).casefold().split())


#: The coordinate a slot's own fields describe, in the order they narrow it.
_SLOT_COORDINATE_FIELDS = ("entity", "metric", "period", "scope")


def _slot_candidates(
    slot: Mapping[str, Any],
    index: Mapping[tuple[tuple[str, ...], tuple[str, ...]], list[str]],
) -> list[str]:
    present = tuple(
        field for field in _SLOT_COORDINATE_FIELDS if _fold_field(slot.get(field))
    )
    if not present:
        return []
    # Fold each field, not the generator producing them: a generator stringifies
    # to its repr, so folding it builds a key no record can ever match and every
    # coordinate silently counts as empty.
    key = tuple(_fold_field(slot.get(field)) for field in present)
    return index.get((present, key), [])


def _enrich_slots(
    row: dict[str, Any],
    gold: Mapping[str, Any],
    fact_coordinates: Mapping[str, Mapping[str, Any]],
    index: Mapping[tuple[tuple[str, ...], tuple[str, ...]], list[str]],
    *,
    case: str,
    entity_hint: str,
) -> None:
    """Give every slot the coordinates of the fact the gold points at, in place.

    Only fields with values are written, so a question that gains no entity does
    not gain an ``entity: null`` -- the shape that made every slot in the set
    look alike when the four coordinate fields were introduced.

    A slot is enriched from the fact at the same index, and only when the gold
    names exactly as many facts as the plan has slots.  A gold that does not
    line up is left alone rather than paired by guess: the pairing is the whole
    content of the change, and inventing one is how the operand periods came out
    duplicated the first time.
    """

    fact_ids = [
        str(item).strip() for item in (gold.get("fact_ids") or []) if str(item).strip()
    ]
    slots = row["plan"]["required_slots"]

    if len(fact_ids) != len(slots):
        # An abstention question has no fact to read -- the gold names none -- so
        # it takes the company from the question instead.  That is not a fallback
        # for a missing gold: it is the coordinate that expresses *why* the
        # question is unanswerable.  "Coca-Cola's Principal-protected debt" is
        # unanswerable because Coca-Cola does not report that metric, and a slot
        # that omits Coca-Cola matches every other company that does -- so the
        # fixture would assert an abstention it does not describe, and would do it
        # by accident of who else happens to report the metric.
        if entity_hint:
            for slot in slots:
                if not _fold_field(slot.get("entity")):
                    slot["entity"] = entity_hint
    else:
        for slot, fact_id in zip(slots, fact_ids):
            record = fact_coordinates.get(fact_id)
            if record is None:
                continue
            entity = str(record.get("entity") or "").strip()
            if entity and not _fold_field(slot.get("entity")):
                slot["entity"] = entity
            scope = str(record.get("scope") or "").strip()
            if scope and not _fold_field(slot.get("scope")):
                slot["scope"] = scope

    # Ids last, so they describe the mention actually written above -- including
    # a mention the fixture already carried.
    for slot in slots:
        entity = str(slot.get("entity") or "").strip()
        if entity:
            entity_id = canonical_entity_id(entity)
            if entity_id:
                slot["entity_id"] = entity_id
        scope = str(slot.get("scope") or "").strip()
        if scope:
            scope_id = canonical_scope_id(scope)
            if scope_id:
                slot["scope_id"] = scope_id


def enrich_and_validate(
    fixtures: list[dict[str, Any]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    fact_coordinates: Mapping[str, Mapping[str, Any]],
    index: Mapping[tuple[tuple[str, ...], tuple[str, ...]], list[str]],
    eval_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enrich every slot, count its candidates, and enforce the stratum's rule.

    The stratum decides what a correct count is, and a single rule cannot serve
    both.  An abstention question is *supposed* to select nothing -- that is what
    makes it unanswerable -- so a zero there is the design, and an answerable
    question selecting nothing can never be bound, so a zero there is a defect.
    One rule for both would either wave through every unanswerable slot or fail
    every abstention fixture.
    """

    enriched: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    eval_by_id = eval_by_id or {}

    for row in fixtures:
        case = row["id"]
        stratum = row.get("stratum")
        gold = gold_by_id.get(case, {})
        entity_hint = str((eval_by_id.get(case) or {}).get("entity") or "").strip()
        _enrich_slots(
            row,
            gold,
            fact_coordinates,
            index,
            case=case,
            entity_hint=entity_hint,
        )

        counts: dict[str, int] = {}
        for slot in row["plan"]["required_slots"]:
            candidates = _slot_candidates(slot, index)
            counts[slot["slot_id"]] = len(candidates)
            if len(candidates) > 1:
                # Recorded, not fatal.  These are not duplicates: the store
                # carries several facts under one (entity, metric, period) with
                # *different values* and no scope to separate them, so the plan
                # contract cannot express the difference and no fixture edit can
                # remove the ambiguity.
                problems.append(
                    {
                        "case_id": case,
                        "slot_id": slot["slot_id"],
                        "role": slot["role"],
                        "coordinate": {
                            f: slot.get(f) for f in _SLOT_COORDINATE_FIELDS if slot.get(f)
                        },
                        "candidates": len(candidates),
                        "candidate_fact_ids": candidates[:5],
                    }
                )
        row["coordinate_candidates"] = counts
        row["expects_no_candidate"] = stratum == "adversarial_abstention"

        if stratum == "adversarial_abstention":
            wrong = {k: v for k, v in counts.items() if v != 0}
            if wrong:
                raise ValueError(
                    f"{case!r}: an abstention plan must select no fact, got {wrong}"
                )
        else:
            empty = {k: v for k, v in counts.items() if v == 0}
            if empty:
                raise ValueError(
                    f"{case!r}: an answerable slot selects no fact: {empty}"
                )
        enriched.append(row)

    return enriched, problems



def _multi_evidence_slots(
    gold: Mapping[str, Any],
    question_record: Mapping[str, Any],
    operand_facts: Mapping[str, Mapping[str, Any]],
    *,
    metric: str,
    periods: list[str],
) -> list[dict[str, Any]]:
    """One slot per gold fact, each naming the company its fact is about.

    The arity used to be the literal 2.  That is right for a comparison and
    wrong for a ranking: `rank-003` ranks four companies and got two slots, so
    the plan asked for half the evidence the question needs and no binder could
    have satisfied it.  The gold states how many facts are required, so the gold
    decides the arity.

    Each slot carries its fact's ``entity`` because without it the slots are
    indistinguishable -- one metric, one period, one role, repeated -- and a
    plan whose slots cannot be told apart cannot say which company a fact was
    for.  That is what made all twenty comparison fixtures unanswerable.

    ``entity_id`` is deliberately absent.  The Harness derives it from the
    mention, and a fixture that wrote one would be authoring an identity rather
    than serialising a fact -- which is the mistake that produced the operand
    periods this builder had to be fixed for.
    """

    case = question_record["id"]
    fact_ids = [
        str(item).strip() for item in (gold.get("fact_ids") or []) if str(item).strip()
    ]
    if not fact_ids:
        raise ValueError(f"{case!r}: a multi-evidence plan needs gold facts, none named")

    slots: list[dict[str, Any]] = []
    for index, fact_id in enumerate(fact_ids):
        record = operand_facts.get(fact_id)
        if record is None:
            raise ValueError(
                f"{case!r}: gold fact {fact_id!r} is not in the fact store"
            )
        entity = str(record.get("entity") or "").strip()
        if not entity:
            raise ValueError(f"{case!r}: gold fact {fact_id!r} names no entity")
        slots.append(
            {
                "slot_id": f"s{index + 1}",
                "metric": metric,
                "period": periods[min(index, len(periods) - 1)],
                "role": "value",
                "value_type": "numeric",
                "unit": None,
                "entity": entity,
            }
        )
    return slots


def render_fixtures(fixtures: list[dict[str, Any]]) -> str:
    return "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) for row in fixtures
    ) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--gold-evidence", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--fact-store", type=Path, default=DEFAULT_FACT_STORE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument(
        "--name",
        default="plan-fixtures-v2",
        help="fixture version to write; an earlier version is never overwritten",
    )
    args = parser.parse_args(argv)

    needed = operand_fact_ids(args.gold_evidence)
    operand_facts: dict[str, Mapping[str, Any]] = {}
    coordinate_index: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = {}
    if needed:
        if not args.fact_store.exists():
            print(f"fact store not found: {args.fact_store}", file=sys.stderr)
            print(
                "  every answerable slot takes its coordinates from the fact the "
                "gold names, and those are not derivable from the question",
                file=sys.stderr,
            )
            return 2
        operand_facts = load_fact_coordinates(args.fact_store, needed)
        missing = sorted(set(needed) - set(operand_facts))
        if missing:
            print(
                f"{len(missing)} gold fact(s) are not in the store: {missing[:5]}",
                file=sys.stderr,
            )
            return 2
        coordinate_index = load_coordinate_index(args.fact_store)

    fixtures = build_fixtures(args.eval_set, args.gold_evidence, operand_facts)
    fixtures, ambiguous = enrich_and_validate(
        fixtures,
        _load_gold(args.gold_evidence),
        operand_facts,
        coordinate_index,
        _load_records(args.eval_set),
    )
    rendered = render_fixtures(fixtures)
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = args.out_dir / f"{args.name}.jsonl"
    fixture_path.write_text(rendered, encoding="utf-8", newline="\n")
    (args.out_dir / f"{args.name}.jsonl.sha256").write_text(
        digest + "\n", encoding="utf-8", newline="\n"
    )

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    ambiguous_path = args.audit_dir / f"{args.name}.ambiguous.json"
    ambiguous_rendered = (
        json.dumps(
            {
                "fixture": args.name,
                "fixture_sha256": digest,
                "count": len(ambiguous),
                "slots": ambiguous,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    ambiguous_path.write_text(ambiguous_rendered, encoding="utf-8", newline="\n")
    ambiguous_digest = hashlib.sha256(ambiguous_rendered.encode("utf-8")).hexdigest()
    (args.audit_dir / f"{args.name}.ambiguous.json.sha256").write_text(
        ambiguous_digest + "\n", encoding="utf-8", newline="\n"
    )

    intents = Counter(row["plan"]["intent"] for row in fixtures)
    sources = Counter(
        f"{field}:{tier}"
        for row in fixtures
        for field, tier in row["sourced_from"].items()
    )
    manifest = {
        "stage": "P1-3-PLAN-FIXTURES",
        "fixture": args.name,
        "total": len(fixtures),
        "fixture_sha256": digest,
        "eval_set_sha256": hashlib.sha256(args.eval_set.read_bytes()).hexdigest(),
        "gold_sha256": hashlib.sha256(args.gold_evidence.read_bytes()).hexdigest(),
        "fact_store_sha256": (
            hashlib.sha256(args.fact_store.read_bytes()).hexdigest()
            if operand_facts
            else None
        ),
        "operand_facts_read": len(operand_facts),
        "intent_distribution": dict(sorted(intents.items())),
        "provenance": dict(sorted(sources.items())),
        "ambiguous_slots": len(ambiguous),
        "ambiguous_sha256": ambiguous_digest,
        "coordinate_candidate_histogram": dict(
            sorted(
                Counter(
                    count
                    for row in fixtures
                    for count in row["coordinate_candidates"].values()
                ).items()
            )
        ),
        "abstention_rows_expecting_no_candidate": sum(
            1 for row in fixtures if row["expects_no_candidate"]
        ),
    }
    (args.out_dir / f"{args.name}.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"wrote {len(fixtures)} fixtures to {fixture_path}")
    print(f"  sha256    : {digest}")
    print(f"  intents   : {dict(sorted(intents.items()))}")
    print(f"  candidates: {manifest['coordinate_candidate_histogram']}")
    print(f"  ambiguous : {len(ambiguous)} slot(s)  -> {ambiguous_path}")
    for key, count in sorted(sources.items()):
        print(f"    {key:<42} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
