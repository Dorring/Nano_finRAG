from decimal import Decimal

from src.domain.evidence import EvidenceItem
from src.finance.structured_operand_binding import extract_financial_facts


def _item(metadata):
    return EvidenceItem(
        chunk_id="chunk-1",
        content="Revenue | 100 | 90",
        document_name="report.pdf",
        page=1,
        content_type="table_row",
        score=0.0,
        rerank_score=None,
        metadata=metadata,
    )


def test_table_row_creates_fact_per_period_column():
    facts = extract_financial_facts(
        (
            _item(
                {
                    "candidate_key": "key",
                    "row_label": "Revenue",
                    "column_headers": ["FY2025", "FY2024"],
                    "cells": ["100", "90"],
                    "scale": "million",
                }
            ),
        )
    )
    assert [(fact.period, fact.value) for fact in facts] == [
        ("FY2025", Decimal("100000000")),
        ("FY2024", Decimal("90000000")),
    ]


def test_table_year_is_not_treated_as_value():
    facts = extract_financial_facts(
        (
            _item(
                {
                    "candidate_key": "key",
                    "row_label": "Revenue",
                    "column_headers": ["FY2025"],
                    "cells": ["100"],
                    "scale": "million",
                }
            ),
        )
    )
    assert facts[0].value == Decimal("100000000")


def test_missing_scale_is_not_guessed():
    facts = extract_financial_facts(
        (
            _item(
                {
                    "candidate_key": "key",
                    "row_label": "Revenue",
                    "column_headers": ["FY2025"],
                    "cells": ["100"],
                }
            ),
        )
    )
    assert facts[0].scale is None
    assert facts[0].value == Decimal("100")
