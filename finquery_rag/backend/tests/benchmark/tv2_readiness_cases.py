"""Sealed fixtures for all 22 TV2-07 production-readiness cases.

``tests/fixtures/tv2_07_production_readiness/`` ships 22 labelled cases and no
fact corpus, so a case could not be executed at all -- H1.1 could only use it as
a coverage checklist and reported 10 of 22 reachable.  This module supplies the
missing half: a sealed fixture spec per ``fixture_key``, driven by the labels
rather than by a copy of them.

The labels are the single source of truth.  ``cases()`` reads both JSONL files
and pairs each row with its spec, and raises if a case has no spec -- so a case
added upstream fails loudly instead of quietly going uncovered, and a label
edit cannot drift out of step with the fixture.

These are wiring cases, not natural-language benchmarks: the questions describe
the scenario ("Find the revenue evidence after the first retrieval misses it").
Some of them are expected to fail against the current runtime.  That is the
point of a benchmark -- see ``test_tv2_readiness_benchmark.py`` for which, and
``BASE_EXPECTATIONS`` for the ones whose label the runtime cannot currently
meet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_v2.contracts import Intent
from tests.harness.h1_integration import H1Fixture
from tests.test_trusted_v2_r4_binder import _fact

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "tv2_07_production_readiness"
QUESTIONS_PATH = FIXTURES_DIR / "questions.jsonl"
LABELS_PATH = FIXTURES_DIR / "labels.jsonl"


def _rev(fact_id: str, **kw: Any) -> dict[str, Any]:
    return _fact(fact_id, **kw)


# --- the 22 fixture specs, keyed by the label's fixture_key -------------------

_R = {"revenue": "Revenue", "margin": "operating_margin", "opincome": "operating_income"}

FIXTURE_SPECS: dict[str, H1Fixture] = {
    "fact_direct": H1Fixture(
        fixture_id="tv2-fact-direct",
        description="One admitted FY2024 revenue fact.",
        query="What was Apple's FY2024 revenue?",
        slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
        facts={"E1": _rev("E1", slots=("revenue",), value="100")},
        routes=(("", ("E1",)),),
    ),
    "fact_margin": H1Fixture(
        fixture_id="tv2-fact-margin",
        description="One admitted FY2024 operating-margin fact.",
        query="What was Apple's FY2024 operating margin?",
        slots=({"slot_id": "margin", "metric": "operating_margin", "period": "FY2024"},),
        facts={"E2": _rev("E2", metric="operating_margin", slots=("margin",), value="100")},
        routes=(("", ("E2",)),),
    ),
    "multi_evidence": H1Fixture(
        fixture_id="tv2-multi-evidence",
        description="Two bound facts; the routing policy selects MULTI.",
        query="What was Apple's FY2024 operating margin?",
        slots=({"slot_id": "margin", "metric": "operating_margin", "period": "FY2024"},),
        facts={
            "Q1": _rev("Q1", metric="operating_margin", slots=("margin",), value="100"),
            "Q2": _rev("Q2", metric="operating_margin", slots=("margin",), value="80"),
        },
        routes=(("", ("Q1", "Q2")),),
    ),
    "multi_evidence_two": H1Fixture(
        fixture_id="tv2-multi-evidence-two",
        description="Two candidates for one slot; the label expects no release.",
        query="Summarize the two reported revenue figures.",
        slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
        facts={
            "M1": _rev("M1", slots=("revenue",), value="100"),
            "M2": _rev("M2", slots=("revenue",), value="120"),
        },
        routes=(("", ("M1", "M2")),),
        expect_released=False,
    ),
    "calc_growth": H1Fixture(
        fixture_id="tv2-calc-growth",
        description="Growth rate across two periods.",
        query="How much did Apple's revenue grow from FY2023 to FY2024?",
        slots=(
            {"slot_id": "current", "metric": "Revenue", "period": "FY2024", "role": "current"},
            {"slot_id": "prior", "metric": "Revenue", "period": "FY2023", "role": "prior"},
        ),
        facts={
            "CURRENT": _rev("CURRENT", period="FY2024", slots=("current",), value="391"),
            "PRIOR": _rev("PRIOR", period="FY2023", slots=("prior",), value="383"),
        },
        routes=(("", ("CURRENT", "PRIOR")),),
        intent=Intent.CALCULATION.value,
        operation="growth_rate",
        calculation="deterministic",
    ),
    "calc_difference": H1Fixture(
        fixture_id="tv2-calc-difference",
        description="Difference between two periods.",
        query="What is the FY2024 minus FY2023 revenue difference?",
        slots=(
            {"slot_id": "current", "metric": "Revenue", "period": "FY2024", "role": "current"},
            {"slot_id": "prior", "metric": "Revenue", "period": "FY2023", "role": "prior"},
        ),
        facts={
            "CURRENT-D": _rev("CURRENT-D", period="FY2024", slots=("current",), value="391"),
            "PRIOR-D": _rev("PRIOR-D", period="FY2023", slots=("prior",), value="383"),
        },
        routes=(("", ("CURRENT-D", "PRIOR-D")),),
        intent=Intent.CALCULATION.value,
        operation="difference",
        calculation="deterministic",
    ),
    "qualitative": H1Fixture(
        fixture_id="tv2-qualitative",
        description="Qualitative question the label marks unanswerable.",
        query="Why did operating margin decline?",
        slots=({"slot_id": "margin", "metric": "operating_margin", "period": "FY2024"},),
        facts={
            "Q1": _rev("Q1", metric="operating_margin", slots=("margin",), value="100"),
            "Q2": _rev("Q2", metric="operating_margin", slots=("margin",), value="80"),
        },
        routes=(("", ("Q1", "Q2")),),
        expect_released=False,
    ),
    "table_heavy": H1Fixture(
        fixture_id="tv2-table-heavy",
        description="A fact carrying table provenance.",
        query="What is the revenue value in the FY2024 table row?",
        slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
        facts={"T1": _rev("T1", slots=("revenue",), value="100")},
        routes=(("", ("T1",)),),
    ),
    "cross_source": H1Fixture(
        fixture_id="tv2-cross-source",
        description="Two documents reporting the same metric; MULTI route.",
        query="Compare the annual report and filing revenue figures.",
        slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
        facts={
            "X1": _rev("X1", slots=("revenue",), value="100"),
            "X2": _rev("X2", slots=("revenue",), value="100"),
        },
        routes=(("", ("X1", "X2")),),
    ),
    "wrong_period": H1Fixture(
        fixture_id="tv2-wrong-period",
        description="A FY2023 distractor alongside the requested FY2024 fact.",
        query="What was Apple's FY2024 revenue?",
        slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
        facts={
            "WP-right": _rev("WP-right", slots=("revenue",), value="100"),
            "WP-wrong": _rev("WP-wrong", period="FY2023", slots=("revenue",), value="90"),
        },
        routes=(("", ("WP-right",)),),
    ),
    "wrong_row": H1Fixture(
        fixture_id="tv2-wrong-row",
        description="A revenue distractor alongside the requested operating income.",
        query="What was Apple's FY2024 operating income?",
        slots=({"slot_id": "opincome", "metric": "operating_income", "period": "FY2024"},),
        facts={
            "WR-right": _rev("WR-right", metric="operating_income", slots=("opincome",), value="100"),
            "WR-wrong": _rev("WR-wrong", metric="Revenue", slots=("opincome",), value="90"),
        },
        routes=(("", ("WR-right",)),),
    ),
    "unit_scale": H1Fixture(
        fixture_id="tv2-unit-scale",
        description="A fact reported at a stated scale.",
        query="Report FY2024 revenue in USD millions.",
        slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
        facts={"US1": _rev("US1", slots=("revenue",), value="100")},
        routes=(("", ("US1",)),),
    ),
    "no_answer": H1Fixture(
        fixture_id="tv2-no-answer",
        description="The requested metric is not in the corpus.",
        query="What was Apple's FY2024 invented regulatory metric?",
        slots=({"slot_id": "invented", "metric": "InventedMetric", "period": "FY2024"},),
        facts={"E1": _rev("E1", slots=("revenue",), value="100")},
        routes=(("", ()),),
        expect_released=False,
    ),
    "missing_slot": H1Fixture(
        fixture_id="tv2-missing-slot",
        description="The requested segment is not represented.",
        query="What was revenue for the unavailable segment?",
        slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
        facts={"E1": _rev("E1", slots=("revenue",), value="100")},
        routes=(("", ()),),
        expect_released=False,
    ),
    "conflict": H1Fixture(
        fixture_id="tv2-conflict",
        description="Two contradictory values for one slot.",
        query="Which conflicting FY2024 revenue figure is authoritative?",
        slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
        facts={
            "CF1": _rev("CF1", slots=("revenue",), value="100"),
            "CF2": _rev("CF2", slots=("revenue",), value="999"),
        },
        routes=(("", ("CF1", "CF2")),),
        expect_released=False,
    ),
    "unsupported_calculation": H1Fixture(
        fixture_id="tv2-unsupported-calculation",
        description="A calculation whose prior-period operand is absent.",
        query="Calculate growth when the prior-period operand is unavailable.",
        slots=(
            {"slot_id": "current", "metric": "Revenue", "period": "FY2024", "role": "current"},
            {"slot_id": "prior", "metric": "Revenue", "period": "FY2023", "role": "prior"},
        ),
        facts={"CURRENT": _rev("CURRENT", period="FY2024", slots=("current",), value="391")},
        routes=(("", ("CURRENT",)),),
        intent=Intent.CALCULATION.value,
        operation="growth_rate",
        calculation="deterministic",
        expect_released=False,
    ),
    "recovery_slot": H1Fixture(
        fixture_id="tv2-recovery-slot",
        description="The label expects a retry to recover the missed evidence.",
        query="Find the revenue evidence after the first retrieval misses it.",
        slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
        facts={"RS1": _rev("RS1", slots=("revenue",), value="100")},
        routes=(("", ("RS1",)),),
    ),
    "recovery_period": H1Fixture(
        fixture_id="tv2-recovery-period",
        description="The label expects a replan to replace a FY2023 candidate.",
        query="What was Apple's FY2024 revenue?",
        slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
        facts={"RP-right": _rev("RP-right", slots=("revenue",), value="100")},
        routes=(("", ("RP-right",)),),
    ),
    "repair_once": H1Fixture(
        fixture_id="tv2-repair-once",
        description="The label expects one repair round to reach release.",
        query="What was revenue when the first candidate needs one repair?",
        slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
        facts={"R1": _rev("R1", slots=("revenue",), value="100")},
        routes=(("", ("R1",)),),
    ),
    "validator_rejection": H1Fixture(
        fixture_id="tv2-validator-rejection",
        description="The candidate cites evidence outside admission.",
        query="What was revenue when the candidate fails validation?",
        slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
        facts={"V1": _rev("V1", slots=("revenue",), value="100")},
        routes=(("", ("V1",)),),
        generation="foreign_citation",
        expect_released=False,
    ),
    "assistant_history": H1Fixture(
        fixture_id="tv2-assistant-history",
        description="Assistant text must not become evidence.",
        query="What was the verified revenue?",
        slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
        facts={"AH1": _rev("AH1", slots=("revenue",), value="100")},
        routes=(("", ("AH1",)),),
    ),
    "unknown_citation": H1Fixture(
        fixture_id="tv2-unknown-citation",
        description="A citation id the Binder never admitted.",
        query="What was revenue with an invalid citation candidate?",
        slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
        facts={"UC1": _rev("UC1", slots=("revenue",), value="100")},
        routes=(("", ("UC1",)),),
    ),
}


@dataclass(frozen=True)
class ReadinessCase:
    """One case: its label, and the fixture that makes it executable."""

    case_id: str
    fixture_key: str
    question: str
    category: str
    fixture: H1Fixture
    answerable: bool
    expected_release: bool
    expected_route: str | None
    expected_evidence_ids: tuple[str, ...]
    expected_citation_ids: tuple[str, ...]
    expected_calculation: Any
    expected_reason_codes: tuple[str, ...]
    required_answer_terms: tuple[str, ...]
    forbidden_answer_terms: tuple[str, ...]
    forbidden_evidence_prefixes: tuple[str, ...]


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["case_id"]] = row
    return rows


def cases() -> tuple[ReadinessCase, ...]:
    """Pair every labelled case with its sealed fixture.

    Raises if a case has no spec.  That is the coverage tripwire: the H1.1 audit
    found 12 of 22 unreachable and could only report it, because there was
    nothing to run them against.
    """

    questions = _read_jsonl(QUESTIONS_PATH)
    labels = _read_jsonl(LABELS_PATH)
    missing = sorted(
        key for key in questions if questions[key]["metadata"]["fixture_key"] not in FIXTURE_SPECS
    )
    if missing:
        raise AssertionError(f"sealed cases with no fixture spec: {missing}")
    stale = sorted(set(FIXTURE_SPECS) - {q["metadata"]["fixture_key"] for q in questions.values()})
    if stale:
        raise AssertionError(f"fixture specs for unknown cases: {stale}")

    out = []
    for case_id, question in questions.items():
        label = labels[case_id]
        key = question["metadata"]["fixture_key"]
        out.append(
            ReadinessCase(
                case_id=case_id,
                fixture_key=key,
                question=question["question"],
                category=question["category"],
                fixture=FIXTURE_SPECS[key],
                answerable=bool(label["answerable"]),
                expected_release=bool(label["expected_release"]),
                expected_route=label.get("expected_route"),
                expected_evidence_ids=tuple(label.get("expected_evidence_ids") or ()),
                expected_citation_ids=tuple(label.get("expected_citation_ids") or ()),
                expected_calculation=label.get("expected_calculation"),
                expected_reason_codes=tuple(label.get("expected_reason_codes") or ()),
                required_answer_terms=tuple(label.get("required_answer_terms") or ()),
                forbidden_answer_terms=tuple(label.get("forbidden_answer_terms") or ()),
                forbidden_evidence_prefixes=tuple(
                    label.get("forbidden_evidence_prefixes") or ()
                ),
            )
        )
    return tuple(sorted(out, key=lambda case: case.case_id))


#: Cases whose label the runtime does not currently meet, with the reason.
#: Derived from an actual run, not from reading the labels: an earlier version of
#: this registry was written by inspection and five of its entries were wrong --
#: including one fixture that declared a substituted generator and silently got
#: the production one, because the dispatch was keyed by fixture id and this
#: module names its fixtures differently.  A benchmark that skips what it fails
#: is not a benchmark, so these still execute and are still scored; they are
#: listed here only so a runner can report them as known rather than new.
KNOWN_MISMATCHES: dict[str, str] = {
    "conflict": (
        "two candidates for one slot with contradictory values bind to one "
        "admitted fact and release; the label expects CONFLICT and no release"
    ),
    "multi_evidence_two": (
        "the same shape: two candidates for one slot resolve to one admitted "
        "fact, so the runtime releases where the label abstains"
    ),
    "qualitative": (
        "the label marks this unanswerable and expects MULTI; the deterministic "
        "binder admits one fact and the runtime releases STRUCTURED_SINGLE"
    ),
    "multi_evidence": (
        "release matches the label; the route does not -- one fact binds, so the "
        "routing policy selects STRUCTURED_SINGLE rather than MULTI"
    ),
    "cross_source": (
        "release matches the label; the route does not, for the same reason"
    ),
    "no_answer": (
        "the outcome matches the label -- no release -- but the reason code "
        "does not: an invented metric is rejected by the semantic-alignment gate "
        "as QUERY_PLAN_SEMANTIC_MISMATCH before the binder can report MISSING_SLOT"
    ),
}
