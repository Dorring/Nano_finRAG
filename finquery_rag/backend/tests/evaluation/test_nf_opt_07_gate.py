from src.evaluation.nf_opt_07 import Recoverability, recoverability_gate


def test_recoverability_gate_requires_eighteen_sources():
    records = [{"recoverability": Recoverability.NOT_RECOVERABLE} for _ in range(22)]
    assert not recoverability_gate(records)["gate_passed"]


def test_recoverability_gate_accepts_valid_set():
    records = [
        {"recoverability": Recoverability.SELF_CONTAINED_TABLE} for _ in range(22)
    ]
    assert recoverability_gate(records)["gate_passed"]
