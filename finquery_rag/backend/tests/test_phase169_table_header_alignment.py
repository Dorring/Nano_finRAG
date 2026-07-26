from src.services.mineru_parser import append_table_row_children, _is_table_header_row


def _table(content):
    return {
        "content": content,
        "metadata": {"type": "table", "doc_id": "doc::table", "page": 1},
    }


def test_year_headers_with_footnote_markers_are_not_data_rows():
    assert _is_table_header_row(
        "2020 (1) | 2020 (2) | comparable basis 2020 | (3)"
    )


def test_multiline_header_uses_richest_value_column_row_for_cells():
    chunks = append_table_row_children([
        _table(
            "Original Budget | Budget Transfers | Actual Expense | Difference\n"
            "2020 (1) | 2020 (2) | comparable basis 2020 | (3)\n"
            "Program | Program Title\n"
            "5 | System A | 110,231 | 109,097 | 98,755 | 10,342"
        )
    ])

    cells = [chunk for chunk in chunks if chunk["metadata"].get("type") == "table_cell"]
    assert [(cell["metadata"]["table_column"], cell["content"].split("; ")[1]) for cell in cells] == [
        ("2020 (1)", "Value: 110,231"),
        ("2020 (2)", "Value: 109,097"),
        ("comparable basis 2020", "Value: 98,755"),
        ("(3)", "Value: 10,342"),
    ]
