from src.domain.evidence import EvidenceItem


def test_raw_child_evidence_can_be_placed_before_parent_context_evidence():
    raw = EvidenceItem.from_chunk({
        "doc_id": "doc::child-1",
        "content": "Metric A revenue was $38 million and grew 70%.",
        "metadata": {"doc_name": "report.pdf", "page": 7},
    })
    parent = EvidenceItem.from_chunk({
        "doc_id": "doc::parent-1",
        "content": "A long parent section with unrelated figures.",
        "metadata": {"doc_name": "report.pdf", "page": 7},
    })

    evidence = (raw,) + tuple(item for item in (parent,) if item.chunk_id != raw.chunk_id)

    assert [item.chunk_id for item in evidence] == ["doc::child-1", "doc::parent-1"]
