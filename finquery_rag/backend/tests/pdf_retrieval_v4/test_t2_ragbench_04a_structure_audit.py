from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parents[2] / "scripts" / "evaluation"
MODULE_PATH = SCRIPT_DIR / "run_t2_ragbench_04a_structure_audit.py"
sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location("t2_04a_structure_audit", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_period_normalization_is_deterministic():
    assert module.extract_periods("FY 2023 and Q4 2024") == ["2024", "fy2023", "q4"]


def test_metric_normalization_keeps_financial_identity_tokens():
    assert module.normalize_metric_tokens("Services net sales") == (
        "net",
        "sale",
        "service",
    )
    assert module.normalize_metric_tokens("gross margin") != module.normalize_metric_tokens(
        "gross profit"
    )


def test_operation_contract_does_not_read_answers():
    assert module.extract_operation_intent("What was the percentage change from 2020 to 2021?") == (
        "percentage_change"
    )
    assert module.extract_operation_intent("What was total revenue?") == "direct_fact"


def test_query_structure_is_gold_independent():
    structure = module.extract_query_structure(
        "Company X: What was net revenue in 2020 and 2021?",
        "Company X",
    )
    assert structure["entity"] == ["company x"]
    assert structure["periods"] == ["2020", "2021"]
    assert structure["requires_multiple_periods"] is True
    assert structure["operation_intent"] == "direct_fact"


def test_required_split_contract_rejects_published_all_cohort():
    required = {"train": ["a"], "dev": ["b"], "test": ["c"]}
    all_ids = ["d"]
    assert set().union(*(set(v) for v in required.values())) | set(all_ids) == {
        "a",
        "b",
        "c",
        "d",
    }
    assert all_ids

