import json
from pathlib import Path

from scripts.evaluation.finalize_nf39_r2_acceptance import normalized_stage_metrics


def _metrics(case_hits, source_hits):
    stages = {}
    for name in ("s0_rrf_top40", "s2_reranker_ranked_top20", "s4_final_context_top5"):
        stages[name] = {}
        for k in (5, 20, 40):
            stages[name][f"case_hit_count_at_{k}"] = case_hits
            stages[name][f"source_hit_count_at_{k}"] = source_hits
    return {"stages": stages}


def test_r2_run_metrics_are_compared_without_r1_hardcoded_baseline(tmp_path: Path):
    (tmp_path / "stage-metrics-same-k.json").write_text(json.dumps(_metrics(16, 20)))
    metrics = normalized_stage_metrics(tmp_path)
    assert metrics["rrf_at_5"] == {"case_hits": 16, "source_hits": 20}
    assert metrics["final_at_5"] == {"case_hits": 16, "source_hits": 20}

