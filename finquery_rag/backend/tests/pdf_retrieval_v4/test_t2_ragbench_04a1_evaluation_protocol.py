from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "evaluation"
MODULE_PATH = SCRIPT_DIR / "run_t2_ragbench_04a1_evaluation_protocol.py"
ARTIFACT = ROOT / "artifacts/evaluation/t2-ragbench-04a1-evaluation-protocol"
sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location("t2_04a1_protocol", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_accepts_native_primary_track_without_custom_split() -> None:
    decision = read_json(ARTIFACT / "decision.json")
    protocol = read_json(ARTIFACT / "evaluation-protocol.json")
    assert decision["evaluation_protocol_accepted"] is True
    assert decision["split_contract_accepted"] is False
    assert decision["historical_split_contract_accepted"] is False
    assert decision["custom_split_created"] is False
    assert protocol["primary_track"]["train_queries"] == 15314
    assert protocol["primary_track"]["dev_queries"] == 2025
    assert protocol["primary_track"]["test_queries"] == 2291
    assert protocol["convfinqa_track"]["queries"] == 3458


def test_published_split_inventory_keeps_convfinqa_all() -> None:
    contract = read_json(ARTIFACT / "published-split-contract.json")
    inventory = contract["native_split_inventory"]
    assert inventory["counts"]["FinQA:train"] == 6251
    assert inventory["counts"]["FinQA:dev"] == 883
    assert inventory["counts"]["FinQA:test"] == 1147
    assert inventory["counts"]["TAT-DQA:train"] == 9063
    assert inventory["counts"]["TAT-DQA:dev"] == 1142
    assert inventory["counts"]["TAT-DQA:test"] == 1144
    assert inventory["counts"]["ConvFinQA:all"] == 3458
    assert contract["convfinqa_assigned_to_primary_split"] is False


def test_primary_identity_and_empty_rows_are_preserved() -> None:
    overlap = read_json(ARTIFACT / "identity-overlap-audit.json")
    assert all(value == 0 for value in overlap["primary_query_overlap"].values())
    empty = read_json(ARTIFACT / "empty-question-audit.json")
    assert empty["published_denominator"] == 23088
    assert empty["empty_question_count"] == 11
    assert empty["by_track_group"]["primary_train"]["count"] == 4
    assert empty["by_track_group"]["primary_dev"]["count"] == 0
    assert empty["by_track_group"]["primary_test"]["count"] == 2
    assert empty["by_track_group"]["convfinqa_transfer"]["count"] == 5
    assert empty["silent_exclusion_allowed"] is False


def test_convfinqa_ids_never_enter_primary_ids() -> None:
    train = set(json.loads((ARTIFACT / "primary-train-query-ids.json").read_text()))
    dev = set(json.loads((ARTIFACT / "primary-dev-query-ids.json").read_text()))
    test = set(json.loads((ARTIFACT / "primary-test-query-ids.json").read_text()))
    conv = set(json.loads((ARTIFACT / "convfinqa-transfer-query-ids.json").read_text()))
    assert not (train | dev | test) & conv
    assert len(train) == 15314
    assert len(dev) == 2025
    assert len(test) == 2291
    assert len(conv) == 3458


def test_partition_does_not_invent_convfinqa_split() -> None:
    rows = [
        {"query_id": "f-train", "subset": "FinQA", "split": "train"},
        {"query_id": "t-dev", "subset": "TAT-DQA", "split": "dev"},
        {"query_id": "c-all", "subset": "ConvFinQA", "split": "all"},
    ]
    groups = module.partition(rows)
    assert groups["primary_train"] == ["f-train"]
    assert groups["primary_dev"] == ["t-dev"]
    assert groups["primary_test"] == []
    assert groups["convfinqa_transfer"] == ["c-all"]


def test_score_uses_context_id_and_frozen_rank_order() -> None:
    rows = {
        "q1": {"context_id": "gold-1"},
        "q2": {"context_id": "gold-2"},
    }
    predictions = {"q1": ["gold-1", "x"], "q2": ["x", "gold-2"]}
    scored = module.score_ids(["q1", "q2"], rows, predictions)
    assert scored["count"] == 2
    assert scored["hits"]["1"] == 1
    assert scored["hits"]["3"] == 2
    assert scored["mrr_at_5"] == 0.75


def test_split_id_hashes_are_deterministic(tmp_path: Path) -> None:
    first = module.write_ids(tmp_path / "a.json", ["b", "a", "b"])
    second = module.write_ids(tmp_path / "b.json", ["a", "b", "b"])
    assert first == second
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
