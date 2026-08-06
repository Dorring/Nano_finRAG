"""Contracts for V4 Gate 06 R2 typed-evidence index inputs."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_pdf_v4_gate_06_r2", ROOT / "scripts" / "evaluation" / "run_pdf_v4_gate_06_r2.py"
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _gzip_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for record in records:
                compressed.write(json.dumps(record, sort_keys=True).encode("utf-8"))
                compressed.write(b"\n")


def test_r0_uses_manifest_record_count_and_tracks_physical_lines(tmp_path):
    stream = tmp_path / "evidence-units.jsonl.gz"
    _gzip_jsonl(
        stream,
        [
            {"stream": "header", "unit_count": 1},
            {"evidence_unit_id": "unit-1", "unit_type": "fact"},
        ],
    )
    (tmp_path / "evidence-units-manifest.json").write_text(
        json.dumps({"record_count": 1}), encoding="utf-8"
    )
    audit = gate._r0_audit(stream)
    assert audit["gate_passed"] is True
    assert audit["parsed_record_count"] == 1
    assert audit["gzip_physical_line_count"] == 2


def test_r4_prediction_is_indexed_by_fact_and_cell_identity(tmp_path):
    path = tmp_path / "temporal-binding-predictions.jsonl.gz"
    _gzip_jsonl(
        path,
        [
            {"stream": "header", "record_count": 1},
            {"fact_id": "fact-1", "cell_id": "cell-1", "fact_semantic_type": "atomic_fact"},
        ],
    )
    by_fact, by_cell = gate._load_r4(tmp_path)
    assert by_fact["fact-1"]["cell_id"] == "cell-1"
    assert by_cell["cell-1"]["fact_id"] == "fact-1"


def test_r2_view_identity_is_deterministic_and_type_scoped():
    unit = {"evidence_unit_id": "unit-1"}
    first = gate._view_id(unit, "atomic_fact")
    second = gate._view_id(dict(unit), "atomic_fact")
    assert first == second
    assert first != gate._view_id(unit, "comparison_fact")
