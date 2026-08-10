from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "artifacts/evaluation/t2-ragbench-02b-dense-residual-review"
SCRIPT = ROOT / "scripts/evaluation/run_t2_ragbench_02b_dense_residual_review.py"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_t2_02b_denominator_and_empty_questions_are_frozen() -> None:
    supply = read_json(AUDIT / "candidate-supply-analysis.json")
    decision = read_json(AUDIT / "decision.json")
    assert supply["formal_query_denominator"] == 23088
    assert supply["empty_question_rows_retained"] == 11
    assert decision["formal_query_denominator"] == 23088
    assert decision["dataset_commit"] == "adf7fe1541ac37351ce1142544d8e3b43010ed92"


def test_protected_residual_contract_is_append_only_and_deduplicated() -> None:
    accounting = read_json(AUDIT / "protected-residual-accounting.json")
    assert accounting["bm25_order_preserved"] is True
    assert accounting["dense_append_only"] is True
    assert accounting["candidate_identity_dedup"] is True
    assert accounting["dense_source_top_k"] == 100
    for budget in ("50", "100"):
        stats = accounting["by_bm25_budget"][budget]
        assert stats["bm25_candidate_count"] == int(budget)
        assert stats["maximum_residual_candidates_per_query"] <= 100
        assert stats["union_candidate_count_max"] <= int(budget) + 100


def test_union_identity_and_subset_totals_are_consistent() -> None:
    supply = read_json(AUDIT / "candidate-supply-analysis.json")["by_k"]
    subset = read_json(AUDIT / "subset-residual-analysis.json")["by_subset"]
    for budget in ("50", "100"):
        assert sum(values[budget]["query_count"] for values in subset.values()) == 23088
    assert sum(values["100"]["bm25_hits"] for values in subset.values()) == supply["100"]["bm25_hits"]
    assert sum(values["100"]["union_hits"] for values in subset.values()) == supply["100"]["union_hits"]
    assert sum(values["100"]["dense_unique_rescue"] for values in subset.values()) == supply["100"]["dense_unique_hits"]


def test_rescue_cohort_and_artifact_schema_are_bounded() -> None:
    required = {
        "candidate-supply-analysis.json",
        "protected-residual-accounting.json",
        "subset-residual-analysis.json",
        "dense-rescue-cohort.jsonl",
        "decision.json",
        "README.md",
        "prediction-seal.json",
    }
    assert required <= {path.name for path in AUDIT.iterdir()}
    rows = [json.loads(line) for line in (AUDIT / "dense-rescue-cohort.jsonl").read_text(encoding="utf-8").splitlines()]
    assert 0 < len(rows) <= 50
    assert all(
        {"query_id", "subset", "question", "bm25_gold_rank", "dense_gold_rank", "gold_context_id", "diagnostic_tags"}
        <= row.keys()
        for row in rows
    )


def test_t2_02b_does_not_run_retrieval_or_model_runtime() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "SentenceTransformer" not in source
    assert "from sentence_transformers" not in source
    seal = read_json(AUDIT / "prediction-seal.json")
    assert seal["retrieval_rerun"] is False
    assert seal["model_execution"] is False
    assert seal["parameter_tuning"] is False


def test_decision_threshold_boundaries_are_registered() -> None:
    spec = importlib.util.spec_from_file_location("t2_02b", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.decide_mainline(1.0, 1.0, 1.0) is True
    assert module.decide_mainline(1.000001, 1.0, 10.0) is False
    assert module.decide_mainline(1.0, 0.999999, 10.0) is False


def test_final_decision_is_explicit() -> None:
    decision = read_json(AUDIT / "decision.json")
    assert decision["protected_dense_residual_mainline_rejected"] is True
    assert decision["current_dense_role"] == "diagnostic_baseline_only"
    assert decision["first_stage_retriever"] == "bm25"
    assert decision["next_gate"] == "t2_03_qwen3_cross_encoder"

