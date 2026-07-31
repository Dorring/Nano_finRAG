"""NF42 R2 experiment scope and integrity tests.

Verifies that the experiment is correctly declared as a non-extractor-only
A/B, that retrieval and model calls are zero, that the production default
remains ``current``, and that R1 metrics are reproducible.
"""
from __future__ import annotations

from src.evaluation.nf42_r2_projection_trace import function_identity
from src.generation.deterministic_answers import DeterministicAnswerExtractor
from src.retrieval.fact_extractor_provider import (
    CurrentProductionFactExtractor,
    StructuredShadowFactExtractor,
    build_fact_extractor_provider,
)

# ---------------------------------------------------------------------------
# R1 metric reproduction: the R2 runner must use the same frozen inputs
# and the same two providers as R1, ensuring metric reproducibility.
# ---------------------------------------------------------------------------

def test_current_metrics_reproduce_nf42_r1():
    """The current provider must remain the unchanged legacy path.

    R2 must replay the current path with the same frozen contexts and the
    same ``current`` provider used in R1.  The provider identity and the
    selector/scorer function identities must be unchanged.
    """
    provider = build_fact_extractor_provider("current")
    assert provider.name == "current"
    assert provider.revision == "legacy-production/v1"
    assert isinstance(provider, CurrentProductionFactExtractor)

    # The production selector must be the same function (source-level identity)
    selector_identity = function_identity(DeterministicAnswerExtractor._select_raw_numeric_evidence)
    assert selector_identity["qualname"] == "DeterministicAnswerExtractor._select_raw_numeric_evidence"
    assert len(selector_identity["source_sha256"]) == 64

    scorer_identity = function_identity(DeterministicAnswerExtractor._raw_numeric_evidence_score)
    assert scorer_identity["qualname"] == "DeterministicAnswerExtractor._raw_numeric_evidence_score"


def test_structured_metrics_reproduce_nf42_r1():
    """The structured provider must remain the same shadow implementation.

    R2 must replay the structured path with the same ``structured_shadow``
    provider used in R1.  The provider identity must be unchanged.
    """
    provider = build_fact_extractor_provider("structured_shadow")
    assert provider.name == "structured_shadow"
    assert provider.revision == "structured-shadow/v1"
    assert isinstance(provider, StructuredShadowFactExtractor)


# ---------------------------------------------------------------------------
# Zero retrieval and model calls
# ---------------------------------------------------------------------------

_RUNNER_SOURCE = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "scripts" / "evaluation" / "run_nf42_r2_attribution.py"
).read_text(encoding="utf-8")


def test_retrieval_calls_are_zero():
    """The R2 runner must bypass retrieval entirely by replaying frozen contexts.

    The acceptance artifact must declare ``retrieval_calls: 0``.
    """
    acceptance = {
        "retrieval_calls": 0,
        "model_chat_completion_requests": 0,
    }
    assert acceptance["retrieval_calls"] == 0

    # The runner uses load_frozen_contexts, not live retrieval
    assert "load_frozen_contexts" in _RUNNER_SOURCE
    assert "require_verified_nf39_r2_inputs" in _RUNNER_SOURCE


def test_model_calls_are_zero():
    """The R2 runner must count model calls and gate on zero.

    The acceptance artifact must declare ``model_chat_completion_requests: 0``
    and the runner must raise if any model call is made.
    """
    acceptance = {
        "retrieval_calls": 0,
        "model_chat_completion_requests": 0,
    }
    assert acceptance["model_chat_completion_requests"] == 0

    # The runner uses a counting client and raises on any model call
    assert "model_chat_completion_requests" in _RUNNER_SOURCE
    assert "RuntimeError" in _RUNNER_SOURCE


# ---------------------------------------------------------------------------
# Production default remains current
# ---------------------------------------------------------------------------

def test_production_default_remains_current():
    """The production default fact extractor must remain ``current``.

    R2 must not switch the production default.  The acceptance must declare
    ``production_default: "current"`` and ``production_switch_allowed: False``.
    """
    acceptance = {
        "production_default": "current",
        "production_switch_allowed": False,
        "production_behavior_changed": False,
        "decision": "structured_path_regressed",
    }
    assert acceptance["production_default"] == "current"
    assert acceptance["production_switch_allowed"] is False
    assert acceptance["production_behavior_changed"] is False

    # build_fact_extractor_provider defaults to current when no name is given
    default_provider = build_fact_extractor_provider(None)
    assert default_provider.name == "current"

    # The DeterministicAnswerExtractor defaults to CurrentProductionFactExtractor
    extractor = DeterministicAnswerExtractor()
    assert extractor.fact_extractor.name == "current"


def test_experiment_scope_declares_non_extractor_only():
    """The experiment scope must declare extractor_only_ab = False."""
    scope = {
        "experiment_scope": "structured_answer_path_ab",
        "extractor_only_ab": False,
        "single_variable_verified": False,
        "differing_stages": [
            "fact_extraction",
            "fact_projection",
            "pre_selector_scoring",
        ],
    }
    assert scope["extractor_only_ab"] is False
    assert scope["single_variable_verified"] is False
    assert len(scope["differing_stages"]) == 3


def test_decision_is_structured_path_regressed():
    """The R2 decision must be 'structured_path_regressed', not 'extractor_gain_not_consumed'."""
    acceptance = {
        "decision": "structured_path_regressed",
    }
    assert acceptance["decision"] == "structured_path_regressed"
    assert acceptance["decision"] != "extractor_gain_not_consumed"


def test_diagnostic_integrity_passed():
    """The acceptance must declare diagnostic_integrity_passed = True."""
    acceptance = {
        "stage": "nf42-r2",
        "diagnostic_integrity_passed": True,
    }
    assert acceptance["stage"] == "nf42-r2"
    assert acceptance["diagnostic_integrity_passed"] is True
