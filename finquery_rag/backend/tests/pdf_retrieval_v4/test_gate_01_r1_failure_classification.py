import json
from decimal import Decimal
from pathlib import Path

from scripts.evaluation.run_pdf_v4_gate_01_r1 import (
    ORIGINAL_METRICS,
    normalize_financial_numeric_text,
    source_identity,
)


ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "artifacts/evaluation/nf-opt-08-r2/manual-mapping-review-package.json"


def test_numeric_normalization_representation_only() -> None:
    assert normalize_financial_numeric_text("(1,234.5)")["decimal"] == Decimal("-1234.5")
    assert normalize_financial_numeric_text("−1 234.5")["decimal"] == Decimal("-1234.5")
    assert normalize_financial_numeric_text("$ 1,234.5")["decimal"] == Decimal("1234.5")
    percent = normalize_financial_numeric_text("12.5%")
    assert percent["decimal"] == Decimal("12.5")
    assert percent["percent"] is True


def test_numeric_normalization_never_repairs_digits() -> None:
    assert normalize_financial_numeric_text("120,81")["decimal"] == Decimal("12081")
    assert not normalize_financial_numeric_text("1O0")["valid"]
    assert not normalize_financial_numeric_text("approximately 100")["valid"]


def test_oracle_denominator_deduplication_keeps_records() -> None:
    records = json.loads(ORACLE.read_text(encoding="utf-8"))["records"]
    identities = [source_identity(record) for record in records]
    assert len(records) == 22
    assert len(set(identities)) == 14
    assert len(identities) - len(set(identities)) == 8


def test_original_probe_metrics_are_frozen() -> None:
    assert ORIGINAL_METRICS == {
        "hybrid_high": {"numeric": [10, 22], "scale": [18, 22]},
        "pipeline_auto_ocr": {"numeric": [10, 22], "scale": [16, 22]},
    }


def test_r1_contract_forbids_backend_selection_per_record() -> None:
    source = (ROOT / "scripts/evaluation/run_pdf_v4_gate_01_r1.py").read_text(encoding="utf-8")
    assert '"per_record_backend_selection": False' in source
    assert '"mineru_reruns": 0' in source
    assert '"production_index_writes": 0' in source
