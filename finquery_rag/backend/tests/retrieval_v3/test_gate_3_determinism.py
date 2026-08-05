from .test_structured_lane import test_fixed_rrf_is_identity_deduplicated_and_deterministic as _contract


def test_gate_3_determinism() -> None:
    _contract()
