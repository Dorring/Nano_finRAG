"""Tests for the NF38 canonical evidence corpus."""
from __future__ import annotations

from src.evaluation.nf38_corpus import (
    CanonicalEvidenceRecord,
    build_corpus_manifest,
    hash_embedding_text,
)


def _make_record(
    evidence_id: str = "r1",
    document_id: str = "a.pdf",
    page: int = 1,
    block_type: str = "text",
    text: str = "revenue was 100",
    parent_id: str | None = None,
) -> CanonicalEvidenceRecord:
    return CanonicalEvidenceRecord(
        evidence_id=evidence_id,
        document_id=document_id,
        page=page,
        block_type=block_type,
        parent_id=parent_id,
        embedding_text=text,
        embedding_text_hash=hash_embedding_text(text),
    )


def test_corpus_hash_is_deterministic():
    records = [_make_record("r1"), _make_record("r2", text="cash flow")]
    manifest_a = build_corpus_manifest(records)
    manifest_b = build_corpus_manifest(records)
    assert manifest_a["corpus_hash"] == manifest_b["corpus_hash"]


def test_corpus_hash_changes_when_text_changes():
    records = [_make_record("r1", text="revenue 100")]
    original = build_corpus_manifest(records)
    changed = build_corpus_manifest([_make_record("r1", text="revenue 200")])
    assert original["corpus_hash"] != changed["corpus_hash"]


def test_corpus_hash_is_order_independent():
    r1 = _make_record("r1")
    r2 = _make_record("r2", text="different")
    assert (
        build_corpus_manifest([r1, r2])["corpus_hash"]
        == build_corpus_manifest([r2, r1])["corpus_hash"]
    )


def test_evidence_ids_hash_is_stable():
    records = [_make_record("r1"), _make_record("r2")]
    manifest = build_corpus_manifest(records)
    ids_hash = manifest["evidence_ids_hash"]
    assert ids_hash and len(ids_hash) == 64


def test_block_type_counts_are_reported():
    records = [
        _make_record("r1", block_type="text"),
        _make_record("r2", block_type="text"),
        _make_record("r3", block_type="table"),
    ]
    manifest = build_corpus_manifest(records)
    assert manifest["block_type_counts"] == {"table": 1, "text": 2}


def test_table_cell_global_count_is_reported():
    records = [
        _make_record("r1", block_type="text"),
        _make_record("r2", block_type="table_cell"),
    ]
    manifest = build_corpus_manifest(records)
    assert manifest["table_cell_global_count"] == 1


def test_duplicate_evidence_ids_are_detected():
    records = [_make_record("r1"), _make_record("r1", text="duplicate")]
    manifest = build_corpus_manifest(records)
    assert manifest["duplicate_evidence_ids"] >= 1


def test_missing_text_is_counted():
    records = [_make_record("r1", text=""), _make_record("r2", text="ok")]
    manifest = build_corpus_manifest(records)
    assert manifest["missing_text"] == 1


def test_document_count_is_unique():
    records = [
        _make_record("r1", document_id="a.pdf"),
        _make_record("r2", document_id="a.pdf"),
        _make_record("r3", document_id="b.pdf"),
    ]
    manifest = build_corpus_manifest(records)
    assert manifest["document_count"] == 2


def test_record_to_dict_excludes_embedding_text():
    record = _make_record("r1", text="sensitive content")
    d = record.to_dict()
    assert "embedding_text" not in d
    assert d["embedding_text_hash"] == hash_embedding_text("sensitive content")


def test_record_to_dict_includes_identity_fields():
    record = CanonicalEvidenceRecord(
        evidence_id="r1",
        document_id="a.pdf",
        page=3,
        block_type="table_row",
        parent_id="parent-1",
        table_id="t-1",
        section_path=("Revenue", "2024"),
        embedding_text="x",
        embedding_text_hash="abc",
    )
    d = record.to_dict()
    assert d["evidence_id"] == "r1"
    assert d["document_id"] == "a.pdf"
    assert d["page"] == 3
    assert d["block_type"] == "table_row"
    assert d["parent_id"] == "parent-1"
    assert d["table_id"] == "t-1"
    assert d["section_path"] == ["Revenue", "2024"]
