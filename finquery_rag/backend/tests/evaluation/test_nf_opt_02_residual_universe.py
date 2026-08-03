import pytest

from src.evaluation.nf_opt_02 import NFOpt02Error, residual_candidate_keys


def test_residual_universe_excludes_current_candidates():
    assert residual_candidate_keys(canonical_keys={"a", "b", "c"}, current_keys={"a", "b"}) == {"c"}


def test_residual_universe_is_not_built_from_gold_labels():
    keys = residual_candidate_keys(canonical_keys={"a", "b"}, current_keys={"a"})
    assert keys == {"b"}


def test_only_whitelisted_documents_enter_residual_index():
    with pytest.raises(NFOpt02Error):
        residual_candidate_keys(canonical_keys={"a"}, current_keys={"out-of-scope"})
