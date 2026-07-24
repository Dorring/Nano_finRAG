"""Characterization tests for the current evaluation system.

These tests document the *existing* evaluation behavior before Phase 5
introduces sealed evaluation, ablations, and threshold calibration. They
serve as a safety net: if Phase 5 refactors accidentally break the
current scoring/runner behavior, these tests will fail.

Covered areas:
- EvaluationCase / Prediction dataclass round-trips
- score_prediction: citation, number, no-answer, calculation, intent
- evaluate_predictions: aggregation and summary structure
- eval_runner: RAGEngine query signature and prediction extraction
- Label/Expected isolation: EvaluationCase carries expected_* fields
  (this is the *current* behavior; Phase 5 will split them)
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.evaluation.evaluation import (
    EvaluationCase,
    ExpectedSource,
    Prediction,
    evaluate_predictions,
    score_prediction,
)
from src.evaluation.eval_runner import run_case, validate_n_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_case(**overrides: Any) -> EvaluationCase:
    defaults: dict[str, Any] = {
        "id": "test-001",
        "question": "What is the revenue?",
    }
    defaults.update(overrides)
    return EvaluationCase.from_dict(defaults)


def _make_prediction(**overrides: Any) -> Prediction:
    base: dict[str, Any] = {
        "id": "test-001",
        "answer": "The revenue is 1000.",
        "sources": [{"filename": "report.pdf", "page": 1, "chunk_id": "report.pdf#p1"}],
        "retrieved_chunks": [{"filename": "report.pdf", "page": 1, "chunk_id": "report.pdf#p1"}],
        "calculations": [],
        "intent": "document_qa",
        "intent_confidence": 0.9,
        "latency_ms": 50.0,
    }
    base.update(overrides)
    return Prediction.from_dict(base)


# ---------------------------------------------------------------------------
# 1. Dataclass round-trips
# ---------------------------------------------------------------------------

class TestEvaluationCaseRoundTrip:
    """EvaluationCase.from_dict / to_dict must round-trip correctly."""

    def test_minimal_case(self) -> None:
        case = _make_case()
        assert case.case_id == "test-001"
        assert case.question == "What is the revenue?"
        assert case.expected_sources == ()
        assert case.expected_numbers == ()
        assert case.expected_no_answer is False

    def test_case_with_expected_sources(self) -> None:
        case = _make_case(
            expected_sources=[{"filename": "report.pdf", "page": 1}]
        )
        assert len(case.expected_sources) == 1
        assert case.expected_sources[0].filename == "report.pdf"
        assert case.expected_sources[0].page == 1

    def test_case_id_aliases(self) -> None:
        """EvaluationCase.from_dict accepts id, case_id, and trace_id."""
        for id_field in ("id", "case_id", "trace_id"):
            case = EvaluationCase.from_dict({
                id_field: "alt-001",
                "question": "q",
            })
            assert case.case_id == "alt-001"

    def test_case_to_dict_round_trip(self) -> None:
        case = _make_case(
            expected_sources=[{"filename": "r.pdf", "page": 2, "chunk_id": "r.pdf#p2"}],
            expected_numbers=["1000"],
            expected_answer_contains=["revenue"],
            tags=["number"],
        )
        d = case.to_dict()
        assert d["id"] == "test-001"
        assert d["expected_numbers"] == ["1000"]
        assert d["expected_sources"][0]["filename"] == "r.pdf"
        case2 = EvaluationCase.from_dict(d)
        assert case2 == case

    def test_case_missing_id_raises(self) -> None:
        with pytest.raises(ValueError, match="missing id/case_id"):
            EvaluationCase.from_dict({"question": "q"})

    def test_case_missing_question_raises(self) -> None:
        with pytest.raises(ValueError, match="missing question"):
            EvaluationCase.from_dict({"id": "x"})


class TestExpectedSourceMatching:
    """ExpectedSource.matches must check all set fields."""

    def test_match_by_filename_and_page(self) -> None:
        src = ExpectedSource(filename="report.pdf", page=1)
        assert src.matches({"filename": "report.pdf", "page": 1}) is True
        assert src.matches({"filename": "other.pdf", "page": 1}) is False
        assert src.matches({"filename": "report.pdf", "page": 2}) is False

    def test_match_by_chunk_id(self) -> None:
        src = ExpectedSource(chunk_id="report.pdf#p1")
        assert src.matches({"chunk_id": "report.pdf#p1"}) is True
        assert src.matches({"chunk_id": "report.pdf#p2"}) is False

    def test_empty_source_matches_anything(self) -> None:
        src = ExpectedSource()
        assert src.matches({"filename": "anything", "page": 99}) is True


class TestPredictionRoundTrip:
    """Prediction.from_dict must parse all fields."""

    def test_minimal_prediction(self) -> None:
        pred = Prediction.from_dict({"id": "p1", "answer": "hello"})
        assert pred.case_id == "p1"
        assert pred.answer == "hello"
        assert pred.sources == ()

    def test_full_prediction(self) -> None:
        pred = _make_prediction()
        assert pred.case_id == "test-001"
        assert pred.answer == "The revenue is 1000."
        assert len(pred.sources) == 1
        assert pred.latency_ms == 50.0


# ---------------------------------------------------------------------------
# 2. score_prediction behavior
# ---------------------------------------------------------------------------

class TestScorePredictionCitation:
    """Citation scoring: precision, recall, f1."""

    def test_perfect_citation(self) -> None:
        case = _make_case(
            expected_sources=[{"filename": "report.pdf", "page": 1}]
        )
        pred = _make_prediction()
        result = score_prediction(case, pred)
        assert result["citation_precision"] == 1.0
        assert result["citation_recall"] == 1.0

    def test_missing_citation(self) -> None:
        case = _make_case(
            expected_sources=[{"filename": "report.pdf", "page": 1}]
        )
        pred = _make_prediction(sources=[], retrieved_chunks=[])
        result = score_prediction(case, pred)
        assert result["citation_recall"] == 0.0

    def test_extra_citation(self) -> None:
        case = _make_case(
            expected_sources=[{"filename": "report.pdf", "page": 1}]
        )
        pred = _make_prediction(
            sources=[
                {"filename": "report.pdf", "page": 1},
                {"filename": "other.pdf", "page": 5},
            ]
        )
        result = score_prediction(case, pred)
        assert result["citation_precision"] == pytest.approx(0.5)


class TestScorePredictionNumbers:
    """Numeric accuracy scoring."""

    def test_correct_number(self) -> None:
        case = _make_case(expected_numbers=["1000"])
        pred = _make_prediction(answer="The revenue is 1000.")
        result = score_prediction(case, pred)
        assert result["number_accuracy"] == 1.0

    def test_wrong_number(self) -> None:
        case = _make_case(expected_numbers=["1000"])
        pred = _make_prediction(answer="The revenue is 2000.")
        result = score_prediction(case, pred)
        assert result["number_accuracy"] == 0.0

    def test_no_expected_numbers(self) -> None:
        case = _make_case()
        pred = _make_prediction()
        result = score_prediction(case, pred)
        # When no expected numbers, score is 1.0 (vacuously true)
        assert result["number_accuracy"] == 1.0


class TestScorePredictionNoAnswer:
    """No-answer scoring."""

    def test_correct_no_answer(self) -> None:
        case = _make_case(expected_no_answer=True)
        pred = _make_prediction(answer="I couldn't find the relevant information.")
        result = score_prediction(case, pred)
        assert result["no_answer_accuracy"] == 1.0

    def test_should_not_answer_but_did(self) -> None:
        case = _make_case(expected_no_answer=True)
        pred = _make_prediction(answer="The revenue is 1000.")
        result = score_prediction(case, pred)
        assert result["no_answer_accuracy"] == 0.0

    def test_expected_answer_vacuously_true(self) -> None:
        """When expected_no_answer is False, no_answer_score is always 1.0."""
        case = _make_case(expected_no_answer=False)
        pred = _make_prediction(answer="")
        result = score_prediction(case, pred)
        assert result["no_answer_accuracy"] == 1.0


class TestScorePredictionIntent:
    """Intent accuracy scoring."""

    def test_correct_intent(self) -> None:
        case = _make_case(expected_intent="document_qa")
        pred = _make_prediction(intent="document_qa")
        result = score_prediction(case, pred)
        assert result["intent_accuracy"] == 1.0

    def test_wrong_intent(self) -> None:
        case = _make_case(expected_intent="document_qa")
        pred = _make_prediction(intent="financial_calculation")
        result = score_prediction(case, pred)
        assert result["intent_accuracy"] == 0.0

    def test_no_expected_intent(self) -> None:
        case = _make_case()
        pred = _make_prediction()
        result = score_prediction(case, pred)
        assert result["intent_accuracy"] == 1.0


class TestScorePredictionOverallPass:
    """Overall pass flag combines all checks."""

    def test_perfect_prediction_passes(self) -> None:
        case = _make_case(
            expected_sources=[{"filename": "report.pdf", "page": 1}],
            expected_numbers=["1000"],
            expected_answer_contains=["revenue"],
        )
        pred = _make_prediction()
        result = score_prediction(case, pred)
        assert result["passed"] is True

    def test_wrong_number_fails(self) -> None:
        case = _make_case(expected_numbers=["1000"])
        pred = _make_prediction(answer="The revenue is 2000.")
        result = score_prediction(case, pred)
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# 3. evaluate_predictions aggregation
# ---------------------------------------------------------------------------

class TestEvaluatePredictionsAggregation:
    """evaluate_predictions must aggregate per-case results into a summary."""

    def test_all_pass(self) -> None:
        cases = [_make_case(id="c1"), _make_case(id="c2")]
        preds = {
            "c1": _make_prediction(id="c1"),
            "c2": _make_prediction(id="c2"),
        }
        report = evaluate_predictions(cases, preds)
        assert report["summary"]["pass_rate"] == 1.0
        assert report["summary"]["total_cases"] == 2
        assert len(report["cases"]) == 2

    def test_mixed_results(self) -> None:
        cases = [
            _make_case(id="c1", expected_numbers=["1000"]),
            _make_case(id="c2", expected_numbers=["2000"]),
        ]
        preds = {
            "c1": _make_prediction(id="c1", answer="1000"),
            "c2": _make_prediction(id="c2", answer="9999"),
        }
        report = evaluate_predictions(cases, preds)
        assert report["summary"]["pass_rate"] == 0.5
        assert report["summary"]["total_cases"] == 2

    def test_empty_input(self) -> None:
        report = evaluate_predictions([], {})
        assert report["summary"]["total_cases"] == 0
        assert len(report["cases"]) == 0

    def test_missing_prediction(self) -> None:
        cases = [_make_case(id="c1"), _make_case(id="c2")]
        preds = {"c1": _make_prediction(id="c1")}
        report = evaluate_predictions(cases, preds)
        assert report["summary"]["missing_predictions"] == 1
        assert "c2" in report["missing_case_ids"]


# ---------------------------------------------------------------------------
# 4. eval_runner behavior
# ---------------------------------------------------------------------------

class TestEvalRunnerSignature:
    """run_case must call rag_engine.query with the correct parameters."""

    def test_query_call_signature(self) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self.last_call: dict[str, Any] = {}

            async def query(
                self,
                question: str,
                doc_names: list[str] | None = None,
                user_id: int = 1,
                n_results: int = 5,
                **kwargs: Any,
            ) -> dict[str, Any]:
                self.last_call = {
                    "question": question,
                    "doc_names": doc_names,
                    "user_id": user_id,
                    "n_results": n_results,
                }
                return {"answer": "test", "sources": []}

        engine = FakeEngine()
        case = _make_case(document_names=["report.pdf"])
        pred = asyncio.run(run_case(case, engine, user_id=1, n_results=5))

        assert engine.last_call["question"] == "What is the revenue?"
        assert engine.last_call["doc_names"] == ["report.pdf"]
        assert engine.last_call["user_id"] == 1
        assert engine.last_call["n_results"] == 5
        assert pred["answer"] == "test"
        assert "latency_ms" in pred

    def test_empty_doc_names_becomes_none(self) -> None:
        class FakeEngine:
            last_call: dict[str, Any] = {}

            async def query(
                self,
                question: str,
                doc_names: list[str] | None = None,
                user_id: int = 1,
                n_results: int = 5,
                **kwargs: Any,
            ) -> dict[str, Any]:
                type(self).last_call = {
                    "doc_names": doc_names,
                }
                return {"answer": ""}

        engine = FakeEngine()
        case = _make_case()  # no document_names
        asyncio.run(run_case(case, engine, user_id=1, n_results=3))
        assert FakeEngine.last_call["doc_names"] is None

    def test_prediction_preserves_public_validation_fields(self) -> None:
        """Offline predictions preserve public grounding verdicts.

        Calculations and warnings remain trace-only here, but answerability,
        validation, and repair are needed to distinguish a safe refusal from
        a retrieval or generation regression in evaluation reports.
        """

        class FakeEngine:
            async def query(self, **kwargs: Any) -> dict[str, Any]:
                return {
                    "answer": "test",
                    "sources": [],
                    "calculations": [{"operation": "sum"}],
                    "answerability": {"status": "answerable"},
                    "validation": {"status": "passed"},
                    "repair": {"fallback_used": False},
                    "warnings": ["minor"],
                }

        engine = FakeEngine()
        case = _make_case()
        pred = asyncio.run(run_case(case, engine, user_id=1, n_results=3))
        # Trace-only fields are intentionally not copied into predictions.
        assert "calculations" not in pred
        assert "warnings" not in pred
        assert pred["answerability"] == {"status": "answerable"}
        assert pred["validation"] == {"status": "passed"}
        assert pred["repair"] == {"fallback_used": False}


class TestValidateNResults:
    """validate_n_results enforces 1..20."""

    def test_valid_values(self) -> None:
        assert validate_n_results(1) == 1
        assert validate_n_results(5) == 5
        assert validate_n_results(20) == 20

    def test_too_low(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            validate_n_results(0)

    def test_too_high(self) -> None:
        with pytest.raises(ValueError, match="<= 20"):
            validate_n_results(21)

    def test_non_integer(self) -> None:
        with pytest.raises(ValueError, match="must be an integer"):
            validate_n_results("abc")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. Current label/expected isolation (documents CURRENT behavior)
# ---------------------------------------------------------------------------

class TestCurrentLabelIsolationStatus:
    """Document the CURRENT behavior: EvaluationCase carries expected_* fields.

    Phase 5 will split these into separate EvaluationQuery (no expected_*)
    and EvaluationLabel (all expected_*) objects. These tests document the
    starting point so we can verify the split is complete.
    """

    def test_evaluation_case_carries_expected_sources(self) -> None:
        case = _make_case(
            expected_sources=[{"filename": "r.pdf", "page": 1}]
        )
        assert len(case.expected_sources) == 1

    def test_evaluation_case_carries_expected_numbers(self) -> None:
        case = _make_case(expected_numbers=["1000"])
        assert case.expected_numbers == ("1000",)

    def test_evaluation_case_carries_expected_calculations(self) -> None:
        case = _make_case(
            expected_calculations=[{
                "id": "calc1",
                "operation": "sum",
                "args": {"values": ["100", "200"]},
                "expected_value": "300",
            }]
        )
        assert len(case.expected_calculations) == 1
        assert case.expected_calculations[0].operation == "sum"

    def test_evaluation_case_carries_expected_no_answer(self) -> None:
        case = _make_case(expected_no_answer=True)
        assert case.expected_no_answer is True

    def test_evaluation_case_carries_expected_intent(self) -> None:
        case = _make_case(expected_intent="document_qa")
        assert case.expected_intent == "document_qa"
