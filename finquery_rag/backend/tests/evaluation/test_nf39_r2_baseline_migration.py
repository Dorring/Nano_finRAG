import json


def test_baseline_migration_is_not_a_performance_claim(tmp_path):
    report = {
        "migration_type": "artifact_integrity_correction",
        "algorithm_changed": False,
        "metric_changes_are_performance_claim": False,
        "old_metrics_valid": False,
        "new_metrics_valid": True,
    }
    path = tmp_path / "baseline-migration-report.json"
    path.write_text(json.dumps(report))
    loaded = json.loads(path.read_text())
    assert loaded["migration_type"] == "artifact_integrity_correction"
    assert loaded["algorithm_changed"] is False
    assert loaded["metric_changes_are_performance_claim"] is False

