from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_runtime_router_has_no_governance_or_label_dependency() -> None:
    source = (ROOT / "src/retrieval_v3/query_router.py").read_text(encoding="utf-8")
    assert "benchmark-governance" not in source
    assert "labels.golden" not in source
    assert "expected_sources" not in source
