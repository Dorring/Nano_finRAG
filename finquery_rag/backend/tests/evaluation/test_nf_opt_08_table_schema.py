from decimal import Decimal
from src.evaluation.nf_opt_08 import StructuredTableCell, StructuredTableRecord

def test_structured_table_preserves_cell_coordinates():
    cell = StructuredTableCell(1, 2, 1, 1, "1,234", "1,234", Decimal("1234"), "number", (0,0,1,1), .9)
    table = StructuredTableRecord("t", "d", "h", 2, 2, None, None, (cell,), None, None, "p", "v", "a", None)
    assert table.cells[0].row_index == 1 and table.cells[0].column_index == 2

def test_header_cell_is_not_numeric_fact():
    cell = StructuredTableCell(0, 1, 1, 1, "2025", "2025", None, "header", None, None)
    assert cell.numeric_value is None

def test_merged_cells_preserve_span():
    cell = StructuredTableCell(0, 0, 1, 2, "Year ended", "year ended", None, "header", None, None)
    assert cell.column_span == 2

def test_missing_cell_is_not_zero():
    cell = StructuredTableCell(1, 1, 1, 1, "-", "-", None, "missing", None, None)
    assert cell.numeric_value is None
