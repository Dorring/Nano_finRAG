from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parents[2] / "scripts" / "evaluation"
MODULE_PATH = SCRIPT_DIR / "run_t2_ragbench_03_r0_1_acceleration.py"
sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location("t2_r0_1_acceleration", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_full_runtime_estimate_uses_fixed_pair_count():
    expected = 1_154_400 / 6.2248253074107165 / 3600
    assert module.estimate_full_hours(6.2248253074107165) == expected


def test_deterministic_query_shard_is_stable_and_bounded():
    values = [module.deterministic_shard("query-1", 3) for _ in range(5)]
    assert len(set(values)) == 1
    assert 0 <= values[0] < 3


def test_qwen_pooling_contract_is_fixed():
    assert module.PROBE_QUERIES == 256
    assert module.PROBE_PAIRS == 12_800
    assert module.VLLM_HF_OVERRIDES["architectures"] == [
        "Qwen3ForSequenceClassification"
    ]
    assert module.VLLM_HF_OVERRIDES["classifier_from_token"] == ["no", "yes"]
    assert module.VLLM_HF_OVERRIDES["is_original_qwen3_reranker"] is True
