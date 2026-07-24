from src.services.retrieval import SqliteBM25Retriever


def test_document_quality_report_flags_page_one_concentration(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    chunks = []
    for index in range(8):
        chunks.append({
            "content": f"first page evidence {index}",
            "metadata": {
                "doc_id": f"user_3_report.pdf::page_1::chunk_{index}",
                "doc_name": "report.pdf",
                "page": 1,
                "type": "text",
            },
        })
    for page in (2, 3, 4):
        chunks.append({
            "content": f"page {page} evidence",
            "metadata": {
                "doc_id": f"user_3_report.pdf::page_{page}::chunk_0",
                "doc_name": "report.pdf",
                "page": page,
                "type": "table" if page == 3 else "text",
            },
        })
    retriever.add_chunks(chunks, user_id=3)

    report = retriever.document_quality_report(user_id=3)

    assert len(report) == 1
    item = report[0]
    assert item["table_chunk_count"] == 1
    assert item["indexed_page_count"] == 4
    assert item["page_1_share"] > 0.5
    assert "page_1_concentration" in item["warnings"]
