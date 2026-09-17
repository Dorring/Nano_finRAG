"""Scoring for the TV2-07 readiness cases, and the H1 baseline record.

Every check is driven by the label that ships with the case rather than by an
expectation written here, so the benchmark measures the runtime against the
project's own readiness contract.

Token figures use the real tokenizer when one is available and say so when they
do not.  They are never character counts presented as tokens: the H2A ablation
depends on those numbers being comparable, and the existing
``ContextBudgetManager`` estimate is a 1.15x word count that reports itself as
"context tokens", which is exactly the kind of number this must not add to.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from src.runtime.harness_runtime_mode import AgentRuntimeMode
from tests.benchmark.tv2_readiness_cases import (
    KNOWN_MISMATCHES,
    ReadinessCase,
    cases,
)
from tests.harness.h1_integration import run_fixture


@dataclass(frozen=True)
class CaseResult:
    """One case, scored against its own label."""

    case_id: str
    fixture_key: str
    category: str
    expected_release: bool
    released: bool
    status: str
    route: str | None
    expected_route: str | None
    reason_codes: tuple[str, ...]
    failures: tuple[str, ...]
    known_mismatch: str | None
    evidence_tokens: int | None
    answer_tokens: int | None
    #: Length of the reference payload the outcome carries, not of the raw
    #: evidence: by design the outcome is reference-only, so the raw source text
    #: never reaches this boundary.  Measuring raw exposure needs the loop
    #: instrumented, which is H2A-1A work.
    evidence_reference_chars: int

    @property
    def matches_label(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "matches_label": self.matches_label}


def _tokenizer() -> tuple[Any | None, str]:
    """The real tokenizer, or an explicit statement that there is none."""

    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base"), "tiktoken:cl100k_base"
    except Exception:
        return None, "unavailable"


def _count(tokenizer: Any | None, text: str) -> int | None:
    """Real token count, or ``None`` -- never a stand-in presented as tokens."""

    if tokenizer is None:
        return None
    return len(tokenizer.encode(text, disallowed_special=()))


def score_case(case: ReadinessCase, outcome: Any, tokenizer: Any | None) -> CaseResult:
    failures: list[str] = []
    released = outcome.release_status.value == "RELEASED"
    answer = outcome.answer or ""
    evidence_ids = list(outcome.evidence_ids)

    if released != case.expected_release:
        failures.append(
            f"release: label={case.expected_release} actual={released}"
        )
    if case.expected_route is not None and outcome.route != case.expected_route:
        failures.append(f"route: label={case.expected_route} actual={outcome.route}")

    missing_evidence = [e for e in case.expected_evidence_ids if e not in evidence_ids]
    if missing_evidence and released:
        failures.append(f"evidence: {missing_evidence} not in {evidence_ids}")
    missing_citations = [
        c for c in case.expected_citation_ids if c not in list(outcome.citation_ids)
    ]
    if missing_citations and released:
        failures.append(f"citations: {missing_citations} missing")

    if released:
        for term in case.required_answer_terms:
            if term not in answer:
                failures.append(f"required answer term {term!r} absent")
    for term in case.forbidden_answer_terms:
        if term in answer:
            failures.append(f"forbidden answer term {term!r} present")
    for prefix in case.forbidden_evidence_prefixes:
        leaked = [e for e in evidence_ids if e.startswith(prefix)]
        if leaked:
            failures.append(f"forbidden evidence {leaked} with prefix {prefix!r}")

    for code in case.expected_reason_codes:
        if code not in outcome.reason_codes:
            failures.append(f"reason code {code} absent from {list(outcome.reason_codes)}")

    if case.expected_calculation and not outcome.calculations:
        failures.append("calculation expected but none published")

    reference = list(getattr(outcome, "evidence_ids", []) or [])
    reference_chars = len(json.dumps(reference, ensure_ascii=False))
    return CaseResult(
        case_id=case.case_id,
        fixture_key=case.fixture_key,
        category=case.category,
        expected_release=case.expected_release,
        released=released,
        status=outcome.status.value,
        route=outcome.route,
        expected_route=case.expected_route,
        reason_codes=tuple(outcome.reason_codes),
        failures=tuple(failures),
        known_mismatch=KNOWN_MISMATCHES.get(case.fixture_key),
        evidence_tokens=_count(tokenizer, json.dumps(reference, ensure_ascii=False)),
        answer_tokens=_count(tokenizer, answer),
        evidence_reference_chars=reference_chars,
    )


def run_benchmark(
    mode: AgentRuntimeMode = AgentRuntimeMode.HARNESS_V3,
) -> dict[str, Any]:
    """Run every sealed case and summarise it against its label."""

    tokenizer, tokenizer_name = _tokenizer()
    results = [
        score_case(case, run_fixture(case.fixture, mode), tokenizer)
        for case in cases()
    ]

    released = [r for r in results if r.released]
    expected_release = [r for r in results if r.expected_release]
    return {
        "mode": mode.value,
        "tokenizer": tokenizer_name,
        "cases": len(results),
        "results": [r.to_dict() for r in results],
        "summary": {
            "task_success": sum(1 for r in results if r.matches_label),
            "known_mismatch": sum(
                1 for r in results if not r.matches_label and r.known_mismatch
            ),
            "unexplained_mismatch": sum(
                1 for r in results if not r.matches_label and not r.known_mismatch
            ),
            # Trust metrics, counted the way the label defines them.
            "false_release": sum(
                1 for r in results if r.released and not r.expected_release
            ),
            "over_conservative": sum(
                1 for r in results if r.expected_release and not r.released
            ),
            "grounded_success": sum(
                1
                for r in released
                if r.matches_label or not any("evidence" in f for f in r.failures)
            ),
            "released": len(released),
            "expected_release": len(expected_release),
            # Size, measured at the boundary that matters.
            "token_counts_available": tokenizer is not None,
            "evidence_tokens": (
                sum(r.evidence_tokens or 0 for r in results)
                if tokenizer is not None
                else None
            ),
            "answer_tokens": (
                sum(r.answer_tokens or 0 for r in results)
                if tokenizer is not None
                else None
            ),
            "evidence_reference_chars": sum(
                r.evidence_reference_chars for r in results
            ),
        },
    }


def write_baseline(path: Any) -> dict[str, Any]:
    """Record the H1 baseline so the H2A ablation has something to compare to."""

    report = run_benchmark(AgentRuntimeMode.HARNESS_V3)
    payload = {"baseline": "nf-v3-h1", **report}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return payload
