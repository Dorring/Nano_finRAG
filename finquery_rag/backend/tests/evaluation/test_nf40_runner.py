import asyncio

import pytest

from src.application.frozen_evaluation import FrozenEvaluationContext
from src.evaluation.evaluation import EvaluationCase
from src.evaluation.nf40_frozen_context import FrozenCaseContext, FrozenContextCandidate
from src.evaluation.nf40_runner import FrozenContextEvaluationRunner, validate_labeled_cases


def test_answerable_case_requires_expected_sources():
    with pytest.raises(ValueError):
        validate_labeled_cases([EvaluationCase(case_id="a", question="q")], expected_count=1)


def test_frozen_runner_does_not_call_retrieval():
    class Engine:
        def __init__(self):
            self.called = False

        async def answer_frozen_evaluation(self, **kwargs):
            self.called = True
            context: FrozenEvaluationContext = kwargs["frozen_context"]
            observer = kwargs["evaluation_observer"]
            observer.record_raw_generation("42.2")
            observer.record_release(answer="42.2", response_type="answer")
            return {"answer": "42.2", "context": context.context, "sources": list(context.sources), "intent": None}

    case = EvaluationCase.from_dict({"id": "a", "question": "q", "expected_sources": [{"filename": "a.pdf", "page": 1}], "expected_answer_contains": ["42.2"]})
    candidate = FrozenContextCandidate("a", 1, "candidate:v1:1", "x", "[a.pdf, p1]\n42.2", "a.pdf", "e1", 1, "text")
    frozen = FrozenCaseContext("a", (candidate,), "hash")
    engine = Engine()
    runner = FrozenContextEvaluationRunner(rag_engine=engine)
    run = asyncio.run(runner.run_case(case=case, frozen=frozen, tenant_id=1))
    assert engine.called
    assert run.public_record["context_hash"] == "hash"
