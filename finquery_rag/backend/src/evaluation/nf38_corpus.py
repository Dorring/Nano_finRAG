"""Canonical evidence corpus for NF38 Embedding A/B.

The canonical corpus is a frozen snapshot of the evidence records that
participate in dense retrieval. Both the MiniLM and BGE-M3 indexes must be
built from the same canonical corpus so that the only experimental variable
is the embedding model.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CanonicalEvidenceRecord:
    """A single dense-indexable evidence record."""

    evidence_id: str
    document_id: str
    page: int | None
    block_type: str
    parent_id: str | None = None
    table_id: str | None = None
    section_path: tuple[str, ...] = ()
    embedding_text: str = ""
    embedding_text_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "document_id": self.document_id,
            "page": self.page,
            "block_type": self.block_type,
            "parent_id": self.parent_id,
            "table_id": self.table_id,
            "section_path": list(self.section_path),
            "embedding_text_hash": self.embedding_text_hash,
        }


def hash_embedding_text(text: str) -> str:
    """Return a stable SHA-256 hash of the embedding text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_corpus_manifest(records: list[CanonicalEvidenceRecord]) -> dict[str, Any]:
    """Build a manifest summarizing a canonical corpus.

    The corpus_hash is derived from evidence_id + document_id + page +
    block_type + embedding_text_hash, so any change to the identity or text
    of a record produces a different hash.
    """
    block_type_counts: dict[str, int] = {}
    evidence_ids: list[str] = []
    missing_text = 0
    duplicate_evidence_ids = 0
    seen_ids: set[str] = set()
    document_ids: set[str] = set()

    for record in records:
        block_type_counts[record.block_type] = block_type_counts.get(record.block_type, 0) + 1
        evidence_ids.append(record.evidence_id)
        document_ids.add(record.document_id)
        if not record.embedding_text.strip():
            missing_text += 1
        if record.evidence_id in seen_ids:
            duplicate_evidence_ids += 1
        seen_ids.add(record.evidence_id)

    corpus_hash = _compute_corpus_hash(records)
    evidence_ids_hash = hashlib.sha256(
        "\n".join(sorted(evidence_ids)).encode("utf-8")
    ).hexdigest()

    return {
        "document_count": len(document_ids),
        "record_count": len(records),
        "corpus_hash": corpus_hash,
        "evidence_ids_hash": evidence_ids_hash,
        "block_type_counts": dict(sorted(block_type_counts.items())),
        "table_cell_global_count": block_type_counts.get("table_cell", 0),
        "duplicate_evidence_ids": duplicate_evidence_ids,
        "missing_document_ids": 0,
        "missing_text": missing_text,
    }


def _compute_corpus_hash(records: list[CanonicalEvidenceRecord]) -> str:
    """Hash the identity + text of every record in stable order."""
    hasher = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.evidence_id):
        identity = "|".join([
            record.evidence_id,
            record.document_id,
            str(record.page) if record.page is not None else "",
            record.block_type,
            record.embedding_text_hash,
        ])
        hasher.update(identity.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()
