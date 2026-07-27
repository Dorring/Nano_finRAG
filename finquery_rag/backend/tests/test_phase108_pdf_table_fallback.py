import importlib
from types import SimpleNamespace


class _FakeDf:
    empty = False

    def __len__(self):
        return 1


def _load_process_tables(monkeypatch, read_pdf):
    monkeypatch.setitem(__import__("sys").modules, "camelot", SimpleNamespace(read_pdf=read_pdf))
    module = importlib.import_module("src.services.process_tables")
    return importlib.reload(module)


def test_camelot_failures_do_not_block_pdf_ingest(monkeypatch):
    def read_pdf(*args, **kwargs):
        raise RuntimeError("no table bbox")

    process_tables = _load_process_tables(monkeypatch, read_pdf)

    assert process_tables.extract_tables_with_camelot("sample.pdf") == {}


def test_table_without_bbox_is_kept_as_usable_table(monkeypatch):
    class TableWithoutBbox:
        page = "1"
        df = _FakeDf()

        @property
        def bbox(self):
            raise RuntimeError("no table bbox")

    def read_pdf(*args, **kwargs):
        if kwargs.get("flavor") == "stream":
            return [TableWithoutBbox()]
        return []

    process_tables = _load_process_tables(monkeypatch, read_pdf)
    monkeypatch.setattr(process_tables, "format_table", lambda table: "| a |\n|---|\n| 1 |")

    tables = process_tables.extract_tables_with_camelot("sample.pdf")

    assert tables == {1: [{"md": "| a |\n|---|\n| 1 |", "bbox": None}]}


def test_flattened_table_markdown_is_rejected_before_native_fallback():
    from src.services.process_tables import is_usable_table_markdown

    flattened = (
        "| Original Budget 2020 Program Title System A 110,231 109,097 98,755 "
        "10,342 Actual Expense Difference and repeated page text | Another very long "
        "flattened cell containing a visual table without usable column boundaries |\n"
        "| --- | --- |"
    )

    assert is_usable_table_markdown(flattened) is False


def test_structured_two_column_table_remains_usable():
    from src.services.process_tables import is_usable_table_markdown

    markdown = "| Metric | Value |\n| --- | --- |\n| Cash and cash equivalents | 42.2 million |"
    assert is_usable_table_markdown(markdown) is True


def test_format_table_keeps_multiline_cell_in_one_markdown_row():
    import pandas as pd
    from types import SimpleNamespace
    from src.services.process_tables import format_table

    table = SimpleNamespace(df=pd.DataFrame([
        ["Metric", "2020"],
        ["Cash and\n cash equivalents", "143,540\r\n"],
    ]))

    markdown = format_table(table)
    assert "Cash and cash equivalents" in markdown
    assert "Cash and\n cash" not in markdown
    assert "143,540" in markdown


def test_pymupdf_table_detection_failure_returns_empty_bboxes():
    from src.services.ingest import _safe_find_table_bboxes

    class Page:
        number = 0

        def find_tables(self):
            raise RuntimeError("no table bbox")

    assert _safe_find_table_bboxes(Page()) == []


def test_pymupdf_tables_can_supply_markdown_when_camelot_has_no_result():
    from src.services.ingest import _extract_pymupdf_table_entries

    class Table:
        bbox = (10, 20, 100, 200)

        def extract(self):
            return [
                ["Metric", "2025"],
                ["Cash and cash equivalents", "$42.2 million"],
            ]

    entries = _extract_pymupdf_table_entries(None, [Table()])

    assert entries == [{
        "bbox": (10, 20, 100, 200),
        "md": (
            "| Metric | 2025 |\n"
            "| --- | --- |\n"
            "| Cash and cash equivalents | $42.2 million |"
        ),
    }]


def test_layout_rows_keep_table_label_and_values_on_same_line():
    from src.services.ingest import _extract_layout_table_row_entries

    class Table:
        bbox = (0, 0, 500, 200)

    class Page:
        number = 0

        def get_text(self, mode):
            assert mode == "words"
            return [
                (10, 50, 45, 62, "Cash", 0, 0, 0),
                (48, 50, 72, 62, "and", 0, 0, 1),
                (75, 50, 100, 62, "cash", 0, 0, 2),
                (103, 50, 190, 62, "equivalents", 0, 0, 3),
                (230, 50, 245, 62, "3", 0, 0, 4),
                (330, 50, 380, 62, "143,540", 0, 0, 5),
                (400, 50, 450, 62, "206,031", 0, 0, 6),
            ]

    rows = _extract_layout_table_row_entries(Page(), [Table()])

    assert len(rows) == 1
    assert rows[0]["content"] == "Cash and cash equivalents | 3 | 143,540 | 206,031"


def test_layout_rows_include_observed_period_header_context():
    from src.services.ingest import _extract_layout_table_row_entries

    class Table:
        bbox = (0, 0, 500, 200)

    class Page:
        number = 0

        def get_text(self, mode):
            assert mode == "words"
            return [
                (10, 20, 75, 32, "December", 0, 0, 0),
                (78, 20, 95, 32, "31,", 0, 0, 1),
                (98, 20, 130, 32, "2020", 0, 0, 2),
                (180, 20, 245, 32, "December", 0, 0, 3),
                (248, 20, 265, 32, "31,", 0, 0, 4),
                (268, 20, 300, 32, "2019", 0, 0, 5),
                (10, 50, 45, 62, "Cash", 0, 1, 0),
                (48, 50, 72, 62, "and", 0, 1, 1),
                (75, 50, 100, 62, "cash", 0, 1, 2),
                (103, 50, 190, 62, "equivalents", 0, 1, 3),
                (230, 50, 245, 62, "3", 0, 1, 4),
                (330, 50, 380, 62, "143,540", 0, 1, 5),
                (400, 50, 450, 62, "206,031", 0, 1, 6),
            ]

    rows = _extract_layout_table_row_entries(Page(), [Table()])

    assert rows[1]["content"] == (
        "Cash and cash equivalents | 3 | 143,540 | 206,031\n"
        "Table column context: December 31, 2020 | December 31, 2019"
    )
    assert rows[1]["table_header_context"] == "December 31, 2020 | December 31, 2019"
