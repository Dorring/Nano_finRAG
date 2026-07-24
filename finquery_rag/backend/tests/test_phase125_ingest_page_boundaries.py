import sys
from unittest.mock import MagicMock


for module_name in [
    "camelot",
    "pymupdf",
    "langchain_core",
    "langchain_core.documents",
    "langchain_text_splitters",
]:
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()

sys.modules["langchain_core.documents"].Document = MagicMock()
sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = MagicMock()
sys.modules["langchain_text_splitters"].MarkdownHeaderTextSplitter = MagicMock()
sys.modules["camelot"].read_pdf = lambda *args, **kwargs: []

from src.services import ingest


class _Split:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {"Header 1": "Section"}


def test_page_boundary_is_owned_by_pdf_page_not_splitter_metadata(monkeypatch):
    """Headers can repeat across pages without collapsing their page identity."""
    monkeypatch.setattr(
        ingest,
        "_split_page_markdown",
        lambda markdown, source: [_Split(f"{markdown}:first"), _Split(f"{markdown}:second")],
    )

    output = list(
        ingest._iter_page_markdown_splits(
            [(1, "same heading"), (24, "same heading")], "report.pdf"
        )
    )

    assert [(page, split.page_content) for page, split in output] == [
        (1, "same heading:first"),
        (1, "same heading:second"),
        (24, "same heading:first"),
        (24, "same heading:second"),
    ]
