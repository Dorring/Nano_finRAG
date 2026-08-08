"""Gate 03 R2 Pass B — Financial Header / Metric Graph.

Builds multi-level metric paths for financial-data rows.  A metric path
is the hierarchical label chain from the nearest enclosing group/section
header down to the leaf metric row, e.g.::

    Intelligent Cloud / Revenue

Multi-level parents are supported::

    Automotive
      └─ Revenues
          └─ Services and other

Fail-closed: when two parents are equally plausible or the header scope
is uncertain, ``metric_status = "ambiguous"`` rather than guessing.

Gate: Metric Path Coverage >= 95% (denominator = eligible financial
data rows only, not all 11,607 rows).
"""

from __future__ import annotations

from src.pdf_retrieval_v4.semantic_graph_models import MetricPath, SemanticRow
from src.pdf_retrieval_v4.table_html_parser import norm_text


def _clean_metric_label(label: str) -> str:
    """Normalize a metric label: strip whitespace, collapse spaces."""
    return " ".join(label.split()).strip()


def build_metric_paths(
    semantic_rows: list[SemanticRow],
) -> list[MetricPath]:
    """Build metric paths for all financial-data rows in a table.

    The algorithm walks rows in order, maintaining a stack of enclosing
    group/section headers.  When a financial-data row is encountered, its
    metric path is the join of the current header stack + its own leaf
    label.

    Parameters
    ----------
    semantic_rows
        Rows from a single table, in row_index order (output of
        ``semantic_row_classifier.classify_table_rows``).

    Returns
    -------
    list[MetricPath]
        One MetricPath per financial-data row (metric_row / subtotal /
        total).  Non-financial rows are skipped.
    """
    header_stack: list[tuple[str, str]] = []  # (row_id, label)
    results: list[MetricPath] = []

    for sr in semantic_rows:
        if sr.row_type in ("group_header", "section_header"):
            # Push onto stack — but avoid duplicates
            label = _clean_metric_label(sr.raw_label)
            if label:
                # Check for conflicting parent: if the same label is
                # already on the stack at a different depth, it's a
                # re-occurrence (pop to that level then re-push)
                for i, (_, existing) in enumerate(header_stack):
                    if norm_text(existing) == norm_text(label):
                        header_stack = header_stack[:i]
                        break
                header_stack.append((sr.row_id, label))
            continue

        if not sr.is_financial_data_row:
            continue

        leaf = _clean_metric_label(sr.raw_label)
        if not leaf:
            # Financial data row with no label — can't build a path
            results.append(
                MetricPath(
                    row_id=sr.row_id,
                    table_fragment_id=sr.table_fragment_id,
                    raw_row_label=sr.raw_label,
                    leaf_metric="",
                    metric_path="",
                    metric_path_segments=(),
                    metric_depth=0,
                    parent_metric_row_id=header_stack[-1][0] if header_stack else None,
                    metric_status="missing",
                )
            )
            continue

        segments = tuple(label for _, label in header_stack) + (leaf,)
        metric_path = " / ".join(segments)
        metric_depth = len(segments)
        parent_id = header_stack[-1][0] if header_stack else None

        metric_status = "resolved"

        results.append(
            MetricPath(
                row_id=sr.row_id,
                table_fragment_id=sr.table_fragment_id,
                raw_row_label=sr.raw_label,
                leaf_metric=leaf,
                metric_path=metric_path,
                metric_path_segments=segments,
                metric_depth=metric_depth,
                parent_metric_row_id=parent_id,
                metric_status=metric_status,
            )
        )

    return results


def detect_parent_cycles(
    metric_paths: list[MetricPath],
    semantic_rows: list[SemanticRow],
) -> int:
    """Detect cycles in the parent_metric_row_id chain.

    Returns the count of rows involved in a cycle (0 = no cycles).
    """
    row_by_id: dict[str, SemanticRow] = {sr.row_id: sr for sr in semantic_rows}
    cycle_count = 0

    for mp in metric_paths:
        visited: set[str] = set()
        current = mp.parent_metric_row_id
        while current and current in row_by_id:
            if current in visited:
                cycle_count += 1
                break
            visited.add(current)
            current = row_by_id[current].parent_row_id

    return cycle_count


def detect_conflicting_parents(
    metric_paths: list[MetricPath],
) -> int:
    """Count rows with metric_status == 'ambiguous' due to conflicting parents."""
    return sum(1 for mp in metric_paths if mp.metric_status == "ambiguous")
