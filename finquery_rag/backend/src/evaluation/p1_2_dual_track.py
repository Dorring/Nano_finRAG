"""P1.2: scoring for the dual-track benchmark.

Two tracks, two questions, and this module keeps them apart on purpose.

**Reachability** asks only how far a query got.  It says nothing about the
financial model, because on the RAW track almost nothing reaches it.

**Conditional downstream** asks how the real chain behaves once the alignment
gate is out of the way -- real retrieval, real binding, real calculation, real
2.08B generation, real validation.  Its denominator is stated on every rate and
it is never called end-to-end accuracy: it isolates the upstream gate, and a
number that hides which gate was opened is a number nobody can use.

Three separations this module exists to make, each of which the existing
canonical scorer collapses:

* **candidate quality != released quality.**  A wrong candidate that the
  validator correctly blocked is a *Harness* success and a *model* failure, and
  those are counted separately.  A wrong answer that got released is
  ``false_release`` and nothing else.
* **a reason code has to be the right one.**  The existing scorer treats any
  non-release as a correct reason attribution, which makes the metric true for
  every failure and therefore worth nothing.
* **a value match is not a binding hit.**  The existing scorer promotes one to
  the other, which is why its four S2 counters are all 3/35.

Everything here is a pure function over prediction rows and gold rows.  It
imports nothing from ``src.runtime``, so it can be tested without a runtime, a
checkpoint or a network.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "CITATION_PLACEHOLDER",
    "PROVIDER_FRAME_TOKENS",
    "citation_diagnostics",
    "numeric_match",
    "pareto_by_stratum",
    "score_downstream",
    "score_reachability",
]

#: The four chat-frame tokens the provider wraps around a rendered prompt.
#: Named here so the token columns stay reconcilable and nobody folds them into
#: either side: the compiler's counter measures the payload it built, and the
#: provider's count measures the sequence the model received.
PROVIDER_FRAME_TOKENS = 4

#: The literal the Step-156 model emits where it wanted a citation.  A model
#: quality artefact, measured and not repaired -- see the P1.1 record.
CITATION_PLACEHOLDER = "[citation needed]"

_CURRENCY = re.compile(r"^[\s$€£¥]+|[\s$€£¥]+$")
_PAREN_NEGATIVE = re.compile(r"^\((.+)\)$")
#: A number in free text.  The parenthesised alternative matters as much as the
#: plain one: accounting negatives are written ``(8,077)``, and a pattern that
#: matched only the inner digits would read a loss as a profit whenever the
#: answer stated it the way the filing does.
_NUMBER = re.compile(
    r"\(\s*[-+]?\d[\d,]*\.?\d*%?\s*\)" r"|" r"[-+]?\d[\d,]*\.?\d*%?"
)
_BRACKET = re.compile(r"\[([^\]]+)\]")
_CANONICAL_HANDLE = re.compile(r"^[EC]\d+$")
_FACT_CITATION = re.compile(r"^citation:[0-9a-fA-F]{6,}$")


def _decimal(raw: Any, *, percent_is_ratio: bool = True) -> Decimal | None:
    """A number from a cell or an answer token, or ``None``.

    ``percent_is_ratio`` is the whole subtlety, because the gold corpus does not
    use one convention:

    * an **arithmetic** gold is a computed ratio -- ``0.3105`` for a 31.05%
      growth rate -- so a ``%`` in an answer has to be divided by 100 before the
      two can be compared.  Stripping the sign instead, which the existing
      scorer does, makes those two never match.
    * a **factual** gold is the display string the filing printed -- ``"1.83 %"``
      or ``"$ 82,056"`` -- so it is already in the units it is written in, and
      dividing it by 100 would turn an exact match into a miss.

    One ``%`` rule cannot serve both; the stratum decides which is in force.
    """

    if raw is None:
        return None
    text = _CURRENCY.sub("", str(raw).strip())
    negative = False
    parenthesised = _PAREN_NEGATIVE.match(text)
    if parenthesised:
        text, negative = parenthesised.group(1), True
    percent = text.endswith("%")
    text = text.rstrip("%").replace(",", "").strip()
    if not text:
        return None
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if percent and percent_is_ratio:
        value = value / Decimal(100)
    return -value if negative else value


def numeric_match(
    answer: str,
    expected: Any,
    *,
    tolerance: Any = None,
    structured: Iterable[Any] = (),
    percent_is_ratio: bool = True,
) -> bool:
    """Whether ``answer`` states ``expected``, within ``tolerance``.

    Candidates are read from the structured calculation results first and from
    the answer text second, because a released calculation carries the value and
    a prose answer only mentions it.
    """

    target = _decimal(expected, percent_is_ratio=percent_is_ratio)
    if target is None:
        return False
    tol = _decimal(tolerance) if tolerance is not None else Decimal(0)
    tol = abs(tol) if tol is not None else Decimal(0)

    for candidate in structured:
        value = _decimal(candidate, percent_is_ratio=percent_is_ratio)
        if value is not None and abs(value - target) <= tol:
            return True
    for token in _NUMBER.findall(answer or ""):
        value = _decimal(token, percent_is_ratio=percent_is_ratio)
        if value is not None and abs(value - target) <= tol:
            return True
    return False


def _released(row: Mapping[str, Any]) -> bool:
    return row.get("release_status") == "RELEASED"


def _correct_by_stratum(row: Mapping[str, Any], gold: Mapping[str, Any]) -> bool:
    """Whether the answer the model produced states the gold fact.

    Independent of release: this is the *candidate* verdict, and the release
    verdict is derived separately so the two can be reported apart.
    """

    stratum = row.get("stratum")
    answer = row.get("answer") or ""

    if stratum == "adversarial_abstention":
        # There is no correct answer to state; the correct behaviour is to have
        # no answer.  Scoring this here would let a released fabrication count,
        # so it is handled by the release verdict alone.
        return False

    structured = [
        item.get("value", item.get("result"))
        for item in (row.get("calculations") or [])
        if isinstance(item, Mapping)
    ]

    if stratum == "cross_entity_comparison":
        expected_higher = gold.get("expected_higher")
        if expected_higher:
            return str(expected_higher).lower() in answer.lower()
        ranking = gold.get("expected_ranking") or ()
        if ranking:
            return all(str(entity).lower() in answer.lower() for entity in ranking)
        # A cross-entity *difference* answers with a number, neither a name nor
        # an order.  This branch used to end at `bool(ranking) and ...`, so a
        # gold carrying `expected_value` and nothing else -- which is all five
        # `cross_entity_difference` cases -- returned False however right the
        # answer was, and a correct deterministic release was scored as a false
        # release.  An evaluator that cannot see the gold is not measuring the
        # system; this is the same defect as the circular release verdict
        # removed from the canonical scorer.
        return numeric_match(
            answer,
            gold.get("expected_value"),
            tolerance=gold.get("tolerance"),
            structured=structured,
            percent_is_ratio=False,
        )

    is_calculation = stratum == "arithmetic_calculation"
    return numeric_match(
        answer,
        gold.get("expected_value"),
        tolerance=gold.get("tolerance"),
        structured=structured,
        percent_is_ratio=is_calculation,
    )


def citation_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Frequency of citation problems, measured and not repaired.

    ``cited_handle_not_in_pack`` needs to know how many handles the pack issued.
    The replay track records that; the RAW track cannot see it, so the rate is
    reported over the rows that have it and the denominator says so.
    """

    total = len(rows) or 1
    placeholder = 0
    malformed = 0
    not_in_pack = 0
    pack_known = 0

    for row in rows:
        answer = row.get("answer") or ""
        if CITATION_PLACEHOLDER.lower() in answer.lower():
            placeholder += 1

        handles = [match.strip() for match in _BRACKET.findall(answer)]
        if any(
            not _CANONICAL_HANDLE.match(handle)
            and not _FACT_CITATION.match(handle)
            and handle.lower() != CITATION_PLACEHOLDER.strip("[]").lower()
            for handle in handles
        ):
            malformed += 1

        issued = row.get("pack_handle_count")
        if isinstance(issued, int) and issued >= 0:
            pack_known += 1
            if any(
                _CANONICAL_HANDLE.match(handle)
                and handle.startswith("E")
                and int(handle[1:]) > issued
                for handle in handles
            ):
                not_in_pack += 1

    return {
        "rows": len(rows),
        "literal_citation_placeholder_rate": round(placeholder / total, 4),
        "malformed_citation_rate": round(malformed / total, 4),
        "cited_handle_not_in_pack_rate": (
            round(not_in_pack / pack_known, 4) if pack_known else None
        ),
        "cited_handle_not_in_pack_denominator": pack_known,
    }


def _tokens(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def _values(name: str) -> list[int]:
        return [
            int(row["tokens"][name])
            for row in rows
            if isinstance(row.get("tokens"), Mapping)
            and isinstance(row["tokens"].get(name), int)
        ]

    payload = _values("context_payload_tokens")
    prompt = _values("actual_model_input_tokens")
    completion = _values("actual_output_tokens")

    def _stats(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"count": 0}
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "mean": round(sum(ordered) / len(ordered), 2),
            "p50": ordered[len(ordered) // 2],
            "max": ordered[-1],
        }

    return {
        "context_payload_tokens": _stats(payload),
        "actual_model_input_tokens": _stats(prompt),
        "actual_output_tokens": _stats(completion),
        "provider_frame_tokens": PROVIDER_FRAME_TOKENS,
    }


def _latency(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = sorted(
        float(row[key]) for row in rows if isinstance(row.get(key), (int, float))
    )
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_ms": round(sum(values) / len(values), 2),
        "p50_ms": round(values[len(values) // 2], 2),
        "p90_ms": round(values[min(len(values) - 1, int(len(values) * 0.9))], 2),
        "max_ms": round(values[-1], 2),
    }


def score_reachability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How far each query got.  Deliberately says nothing about answers."""

    total = len(rows)
    reasons = Counter(
        reason for row in rows for reason in (row.get("reason_codes") or [])
    )
    reached = Counter()
    for row in rows:
        stage = row.get("reached") or {}
        for name in ("retrieval", "binding", "generation", "validation"):
            if stage.get(name):
                reached[name] += 1

    aligned = sum(
        1 for row in rows if row.get("computed_align_status") == "ALIGNED"
    )
    overridden = sum(1 for row in rows if row.get("align_overridden"))
    # The RAW track reaches the question over HTTP, and the response does not
    # expose the gate's verdict.  Reporting 0.0 there would read as "no query
    # passed the gate" when the truth is "this track cannot see the gate" --
    # a measurement that is absent must be reported as absent.
    gate_observable = any(row.get("computed_align_status") for row in rows)

    return {
        "total": total,
        "align_pass_rate": (
            round(aligned / total, 4) if (total and gate_observable) else None
        ),
        "align_reject_rate": (
            round((total - aligned) / total, 4) if (total and gate_observable) else None
        ),
        "align_gate_observable": gate_observable,
        "align_override_count": overridden,
        "reject_reason_distribution": dict(sorted(reasons.items())),
        "reached_retrieval": reached["retrieval"],
        "reached_binding": reached["binding"],
        "reached_generation": reached["generation"],
        "reached_validation": reached["validation"],
    }


def score_downstream(
    rows: Sequence[Mapping[str, Any]],
    gold: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """The post-alignment chain, scored with candidate and release kept apart."""

    total = len(rows)
    counters: Counter[str] = Counter()
    # Every counter is emitted, including the ones that never fired.  A report
    # that silently omits a rate because nothing triggered it is how "we never
    # released anything" comes to read as "no false releases" -- the same shape
    # of mistake the sealed artifact makes when it prints a hard-coded 100%
    # compliance line beside a true count of zero.
    for name in (
        "candidate_correct",
        "released_correct",
        "false_release",
        "over_conservative_block",
        "blocked_and_wrong",
        "no_answer_correct",
        "no_answer_false_release",
        "no_answer_reason_attributed",
        "provider_failure",
    ):
        counters[name] = 0
    by_stratum: dict[str, Counter[str]] = {}

    for row in rows:
        stratum = str(row.get("stratum"))
        gold_row = gold.get(str(row.get("id"))) or {}
        answerable = str(gold_row.get("expected_outcome")) != "ABSTENTION"
        released = _released(row)
        candidate_correct = _correct_by_stratum(row, gold_row)
        bucket = by_stratum.setdefault(stratum, Counter())
        bucket["total"] += 1

        if _correct_by_stratum(row, gold_row) and answerable:
            counters["candidate_correct"] += 1
            bucket["candidate_correct"] += 1

        if answerable:
            if released and candidate_correct:
                counters["released_correct"] += 1
                bucket["released_correct"] += 1
            elif released and not candidate_correct:
                counters["false_release"] += 1
                bucket["false_release"] += 1
            elif not released and candidate_correct:
                counters["over_conservative_block"] += 1
                bucket["over_conservative_block"] += 1
            else:
                counters["blocked_and_wrong"] += 1
                bucket["blocked_and_wrong"] += 1
        else:
            # The gold says there is no answer.  Not releasing is the success.
            if released:
                counters["no_answer_false_release"] += 1
                bucket["no_answer_false_release"] += 1
            else:
                counters["no_answer_correct"] += 1
                bucket["no_answer_correct"] += 1
                # And the *right* refusal, which is a separate claim.  The
                # existing scorer counts any non-release here, which makes the
                # metric true for every failure; requiring the gold reason to
                # actually appear is what makes it mean something.  A blanket
                # `FAIL_CLOSED` now scores 0.
                expected_reason = str(gold_row.get("expected_failure_reason") or "")
                reason_codes = [str(code) for code in (row.get("reason_codes") or [])]
                if expected_reason and any(
                    expected_reason == code or expected_reason.lower() in code.lower()
                    for code in reason_codes
                ):
                    counters["no_answer_reason_attributed"] += 1
                    bucket["no_answer_reason_attributed"] += 1

        if row.get("provider_failure"):
            counters["provider_failure"] += 1

    def _rate(numerator: str, total_for_rate: int) -> float | None:
        return round(counters[numerator] / total_for_rate, 4) if total_for_rate else None

    answerable_count = sum(
        1
        for row in rows
        if str((gold.get(str(row.get("id"))) or {}).get("expected_outcome"))
        != "ABSTENTION"
    )
    abstention_count = total - answerable_count

    return {
        "total": total,
        "answerable": answerable_count,
        "abstention": abstention_count,
        "counts": dict(sorted(counters.items())),
        "answerable_correctness": (
            round(counters["released_correct"] / answerable_count, 4)
            if answerable_count
            else None
        ),
        "no_answer_correctness": (
            round(counters["no_answer_correct"] / abstention_count, 4)
            if abstention_count
            else None
        ),
        "no_answer_reason_attribution": (
            round(counters["no_answer_reason_attributed"] / abstention_count, 4)
            if abstention_count
            else None
        ),
        "candidate_correctness": (
            round(counters["candidate_correct"] / answerable_count, 4)
            if answerable_count
            else None
        ),
        "false_release_rate": _rate("false_release", answerable_count),
        "over_conservative_rate": _rate("over_conservative_block", answerable_count),
        "citations": citation_diagnostics(rows),
        "tokens": _tokens(rows),
        "latency": {
            "end_to_end": _latency(rows, "latency_ms"),
            "generation": _latency(rows, "generation_latency_ms"),
        },
        "by_stratum": {key: dict(sorted(value.items())) for key, value in sorted(by_stratum.items())},
    }


def pareto_by_stratum(
    rows: Sequence[Mapping[str, Any]],
    gold: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """A compact per-stratum view for the report's second table."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("stratum")), []).append(row)

    table: dict[str, Any] = {}
    for stratum, subset in sorted(grouped.items()):
        gold_subset = {str(row.get("id")): gold.get(str(row.get("id")), {}) for row in subset}
        scored = score_downstream(subset, gold_subset)
        table[stratum] = {
            "total": scored["total"],
            "answerable": scored["answerable"],
            "abstention": scored["abstention"],
            "counts": scored["counts"],
            "answerable_correctness": scored["answerable_correctness"],
            "no_answer_correctness": scored["no_answer_correctness"],
        }
    return table
