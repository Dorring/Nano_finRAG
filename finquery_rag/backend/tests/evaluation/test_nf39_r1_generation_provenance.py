from src.evaluation.nf39_r1_integrity import attach_generation_outcome


def test_ranking_only_does_not_claim_answer_failure():
    result = attach_generation_outcome(
        ranking_manifest={"question_hash": "q"}, generation_artifact=None
    )
    assert result["generation_status"] == "not_evaluated"
    assert result["validation_status"] == "not_evaluated"


def test_generation_outcome_requires_matching_fingerprints():
    fields = {
        "question_hash": "q",
        "label_hash": "l",
        "final_context_hash": "c",
        "generator_model_identity": "m",
        "prompt_hash": "p",
        "generation_config_hash": "g",
        "validator_config_hash": "v",
        "calculator_config_hash": "x",
    }
    artifact = dict(fields)
    artifact["prompt_hash"] = "different"
    result = attach_generation_outcome(
        ranking_manifest=fields, generation_artifact=artifact
    )
    assert result["reason"] == "artifact_fingerprint_mismatch"
    assert result["generation_status"] == "not_evaluated"

