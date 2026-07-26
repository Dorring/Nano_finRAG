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
            {"type": "text", "text": "Annual Revenue Report", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "Revenue discussion", "page_idx": 0},
            {
                "type": "table",
                "table_caption": ["Revenue table"],
                "table_body": "<table><tr><th>Metric</th><th>Amount</th></tr><tr><td>Revenue</td><td>$42</td></tr></table>",
                "page_idx": 2,
            },
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
    titles = [chunk for chunk in chunks if chunk["metadata"]["type"] == "front_matter"]
    tables = [chunk for chunk in chunks if chunk["metadata"]["type"] == "table"]
    assert titles[0]["metadata"]["subtype"] == "title"
    assert "Annual Revenue Report" in titles[0]["content"]
    assert len(tables) == 1
    assert tables[0]["metadata"]["page"] == 3
    assert "Revenue | $42" in tables[0]["content"]
    assert all("confidential" not in chunk["content"] for chunk in chunks)


def test_mineru_groups_adjacent_text_into_one_parent_evidence_window(monkeypatch):
    monkeypatch.setattr(mineru_parser.pymupdf, "open", lambda _: _Pdf())
    monkeypatch.setattr(
        mineru_parser,
        "_run_mineru",
        lambda *_: [
            {"type": "text", "text": "Management discussion", "text_level": 1, "page_idx": 1},
            {"type": "text", "text": "Revenue increased to $42 million.", "page_idx": 1},
            {"type": "text", "text": "The increase was driven by subscriptions.", "page_idx": 1},
        ],
    )

    chunks, _ = mineru_parser.process_pdf_with_mineru(
        "annual.pdf",
        user_id=7,
        recursive_splitter=_Splitter(),
        long_chunk_threshold=500,
        hierarchy_metadata_fn=_hierarchy,
        chunk_content_with_section_fn=lambda text, _: text,
    )

    text_chunks = [chunk for chunk in chunks if chunk["metadata"]["type"] == "text"]
    assert len(text_chunks) == 1
    assert "Revenue increased to $42 million." in text_chunks[0]["content"]
    assert "subscriptions" in text_chunks[0]["metadata"]["parent_excerpt"]

def test_mineru_subprocess_can_scope_resources_and_method(monkeypatch, tmp_path):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(mineru_parser.subprocess, "run", fake_run)
    monkeypatch.setattr(mineru_parser, "_load_content_list", lambda _: [{"type": "text"}])
    monkeypatch.setenv("MINERU_CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("MINERU_METHOD", "txt")

    assert mineru_parser._run_mineru("annual.pdf", tmp_path) == [{"type": "text"}]
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert captured["args"][-1] == "txt"

    monkeypatch.setenv("MINERU_FORCE_CPU", "true")
    mineru_parser._run_mineru("annual.pdf", tmp_path)


def test_mineru_table_rows_become_retrievable_children(monkeypatch):
    monkeypatch.setattr(mineru_parser.pymupdf, "open", lambda _: _Pdf())
    monkeypatch.setattr(
        mineru_parser,
        "_run_mineru",
        lambda *_: [
            {
                "type": "table",
                "table_body": (
                    "<table><tr><th>Metric</th><th>Amount</th></tr>"
                    "<tr><td>Subscription revenue</td><td>$42</td></tr>"
                    "<tr><td>Service revenue</td><td>$9</td></tr></table>"
                ),
                "page_idx": 1,
            },
        ],
    )

    chunks, _ = mineru_parser.process_pdf_with_mineru(
        "annual.pdf",
        user_id=7,
        recursive_splitter=_Splitter(),
        long_chunk_threshold=500,
        hierarchy_metadata_fn=_hierarchy,
        chunk_content_with_section_fn=lambda text, _: text,
    )

    parents = [chunk for chunk in chunks if chunk["metadata"]["type"] == "table"]
    rows = [chunk for chunk in chunks if chunk["metadata"]["type"] == "table_row"]
    assert len(parents) == 1
    assert len(rows) == 2
    assert "Subscription revenue | $42" in rows[0]["content"]
    assert rows[0]["metadata"]["parent_table_id"] == parents[0]["metadata"]["doc_id"]
    assert rows[0]["metadata"]["doc_id"].startswith(parents[0]["metadata"]["doc_id"] + "::row_")


def test_table_row_child_preserves_caption_units_and_column_headers():
    chunks = [{
        "content": (
            "Consolidated cash flow statement (in thousands)\n"
            "For the year ended December 31, 2025\n"
            "| Activity | 2025 | 2024 |\n"
            "| --- | --- | --- |\n"
            "| Operating activities | 24,053 | 9,703 |"
        ),
        "metadata": {"type": "table", "doc_id": "user_7_report.pdf::page_8::table_1"},
    }]

    rows = [
        chunk for chunk in mineru_parser.append_table_row_children(chunks)
        if chunk["metadata"]["type"] == "table_row"
    ]

    assert len(rows) == 1
    assert "in thousands" in rows[0]["content"]
    assert "For the year ended December 31, 2025" in rows[0]["content"]
    assert "| Activity | 2025 | 2024 |" in rows[0]["content"]
    assert "Operating activities | 24,053 | 9,703" in rows[0]["content"]
    assert "in thousands" in rows[0]["metadata"]["table_header_context"]
