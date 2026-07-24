from types import SimpleNamespace

from src.services import mineru_parser


class _Pdf:
    def __len__(self):
        return 3

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Splitter:
    def split_documents(self, documents):
        return documents


def _hierarchy(metadata, **kwargs):
    return {
        "parent_id": f"parent-{kwargs['page']}-{kwargs['chunk_idx']}",
        "parent_page": kwargs["page"],
        "parent_excerpt": kwargs["parent_content"],
        "parent_child": True,
    }


def test_mineru_content_list_preserves_page_and_table_contract(monkeypatch):
    monkeypatch.setattr(mineru_parser.pymupdf, "open", lambda _: _Pdf())
    monkeypatch.setattr(
        mineru_parser,
        "_run_mineru",
        lambda *_: [
            {"type": "text", "text": "Revenue discussion", "text_level": 1, "page_idx": 0},
            {"type": "table", "table_caption": ["Revenue table"], "table_body": "<table><tr><td>$42</td></tr></table>", "page_idx": 2},
            {"type": "footer", "text": "confidential", "page_idx": 2},
        ],
    )

    chunks, pages = mineru_parser.process_pdf_with_mineru(
        "annual.pdf",
        user_id=7,
        recursive_splitter=_Splitter(),
        long_chunk_threshold=500,
        hierarchy_metadata_fn=_hierarchy,
        chunk_content_with_section_fn=lambda text, section: f"{section}: {text}" if section else text,
    )

    assert pages == 3
    assert [chunk["metadata"]["page"] for chunk in chunks] == [1, 3]
    assert chunks[1]["metadata"]["type"] == "table"
    assert "$42" in chunks[1]["content"]
    assert all("confidential" not in chunk["content"] for chunk in chunks)
