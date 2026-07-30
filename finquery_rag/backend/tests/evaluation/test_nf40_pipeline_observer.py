import pytest

from src.evaluation.nf40_pipeline_observer import EvaluationExecutionContext, sha256_text


def test_frozen_execution_context_rejects_side_effects():
    with pytest.raises(ValueError):
        EvaluationExecutionContext(retrieval_enabled=True).validate()


def test_frozen_execution_context_is_side_effect_free():
    EvaluationExecutionContext().validate()
    assert sha256_text("answer") != "answer"
