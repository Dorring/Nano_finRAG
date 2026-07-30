"""Fail-closed input gate for the future NF40 frozen-context evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class EvaluationIntegrityError(ValueError):
    """Raised when NF39 R2 artifacts cannot safely start NF40."""


def require_verified_nf39_r2_inputs(
    *,
    acceptance_path: Path,
    snapshot_manifest_path: Path,
    frozen_payload_path: Path,
    expected_payload_sha256: str,
) -> None:
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("artifact_schema") != "nf39-r2/v1":
        raise EvaluationIntegrityError("NF40 requires NF39 R2 artifacts")
    if not acceptance.get("nf40_start_allowed", False):
        raise EvaluationIntegrityError("NF39 R2 gate does not allow NF40")
    manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    actual = hashlib.sha256(frozen_payload_path.read_bytes()).hexdigest()
    if actual != expected_payload_sha256:
        raise EvaluationIntegrityError("Frozen payload differs from explicit expected SHA256")
    if actual != manifest.get("payload_sha256"):
        raise EvaluationIntegrityError("Frozen payload SHA256 mismatch")

