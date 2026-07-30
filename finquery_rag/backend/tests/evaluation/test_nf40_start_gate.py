import hashlib
import json

import pytest

from src.evaluation.nf40_start_gate import (
    EvaluationIntegrityError,
    require_verified_nf39_r2_inputs,
)


def _write_inputs(tmp_path, *, accepted=True, payload=b"frozen"):
    payload_path = tmp_path / "payload.jsonl"
    payload_path.write_bytes(payload)
    (tmp_path / "acceptance.json").write_text(json.dumps({"artifact_schema": "nf39-r2/v1", "nf40_start_allowed": accepted}))
    (tmp_path / "snapshot.json").write_text(json.dumps({"payload_sha256": hashlib.sha256(payload).hexdigest()}))
    return payload_path


def test_nf40_accepts_verified_r2_snapshot(tmp_path):
    payload = _write_inputs(tmp_path)
    require_verified_nf39_r2_inputs(acceptance_path=tmp_path / "acceptance.json", snapshot_manifest_path=tmp_path / "snapshot.json", frozen_payload_path=payload)


def test_nf40_rejects_payload_hash_mismatch(tmp_path):
    payload = _write_inputs(tmp_path)
    payload.write_bytes(b"changed")
    with pytest.raises(EvaluationIntegrityError):
        require_verified_nf39_r2_inputs(acceptance_path=tmp_path / "acceptance.json", snapshot_manifest_path=tmp_path / "snapshot.json", frozen_payload_path=payload)

