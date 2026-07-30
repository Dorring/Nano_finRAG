import pytest

from src.evaluation.nf40_frozen_context import (
    FrozenCaseContext,
    FrozenContextCandidate,
    FrozenContextError,
    as_evaluation_context,
    load_frozen_contexts,
)


def test_frozen_snapshot_requires_complete_nf39_r2_shape(tmp_path):
    payload = tmp_path / "payload.jsonl"
    payload.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"artifact_schema":"nf39-r2/v1","cases":{}}', encoding="utf-8")
    with pytest.raises(FrozenContextError, match="135"):
        load_frozen_contexts(payload, manifest)


def test_evaluation_context_keeps_verified_rendered_order():
    first = FrozenContextCandidate("case", 1, "candidate:v1:one", "a", "[a.pdf, p1]\nA", "a.pdf", "one", 1, "text")
    second = FrozenContextCandidate("case", 2, "candidate:v1:two", "b", "[b.pdf, p2]\nB", "b.pdf", "two", 2, "text")
    frozen = FrozenCaseContext("case", (first, second), "context-hash")
    evaluation_context = as_evaluation_context(frozen)
    assert evaluation_context.context == "[a.pdf, p1]\nA\n\n---\n\n[b.pdf, p2]\nB"
    assert [chunk["doc_id"] for chunk in evaluation_context.chunks] == ["one", "two"]
