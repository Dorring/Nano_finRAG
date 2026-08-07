"""Full-corpus MinerU runner for Gate 02 R2.

Runs MinerU Hybrid High on each of the 8 frozen benchmark PDFs,
producing deterministic per-document output with checkpoint support.

Key constraints:
  - One document at a time (GPU concurrency = 1)
  - Fixed backend: hybrid-engine / auto / high
  - No per-page backend selection
  - No per-document backend selection
  - Document-level execution with page-level checkpoint
  - Failure resume only if config hash matches
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.pdf_retrieval_v4.frozen_corpus_manifest import sha256_file

MINERU_BACKEND = "hybrid-engine"
MINERU_METHOD = "auto"
MINERU_EFFORT = "high"


@dataclass(frozen=True)
class MinerUConfig:
    """Fixed MinerU configuration for full-corpus ingestion."""

    backend: str = MINERU_BACKEND
    method: str = MINERU_METHOD
    effort: str = MINERU_EFFORT

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            {"backend": self.backend, "method": self.method, "effort": self.effort},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "method": self.method,
            "effort": self.effort,
            "config_hash": self.config_hash,
        }


@dataclass
class DocumentRunResult:
    """Result of running MinerU on one document."""

    document_id: str
    status: str  # "completed" | "failed" | "skipped"
    input_pdf_sha256: str
    config_hash: str
    output_manifest_hash: str | None
    processed_page_count: int
    started_at: str
    completed_at: str
    elapsed_seconds: float
    return_code: int | None
    error: str | None
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "status": self.status,
            "input_pdf_sha256": self.input_pdf_sha256,
            "config_hash": self.config_hash,
            "output_manifest_hash": self.output_manifest_hash,
            "processed_page_count": self.processed_page_count,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "return_code": self.return_code,
            "error": self.error,
            "output_dir": self.output_dir,
        }


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_mineru(
    mineru_bin: Path,
    input_pdf: Path,
    output_dir: Path,
    config: MinerUConfig,
    cuda_visible_devices: str,
    tmpdir: Path,
    timeout_seconds: int,
    log_path: Path,
) -> tuple[int | None, str | None]:
    """Execute MinerU on a single PDF.

    Returns (return_code, error_message).
    """
    command = [
        str(mineru_bin),
        "-p", str(input_pdf),
        "-o", str(output_dir),
        "-b", config.backend,
        "-m", config.method,
    ]
    if config.effort:
        command.extend(["--effort", config.effort])

    env = os.environ.copy()
    env["TMPDIR"] = str(tmpdir)
    env["TEMP"] = env["TMPDIR"]
    env["TMP"] = env["TMPDIR"]
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    tmpdir.mkdir(parents=True, exist_ok=True)

    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                env=env,
            )
        if completed.returncode != 0:
            return completed.returncode, f"mineru_exit:{completed.returncode}"
        return completed.returncode, None
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"mineru_execution:{type(exc).__name__}:{exc}"


def _checkpoint_path(output_root: Path, document_id: str) -> Path:
    return output_root / document_id / "document-run-checkpoint.json"


def _load_checkpoint(
    output_root: Path,
    document_id: str,
    input_pdf_sha256: str,
    config_hash: str,
) -> dict[str, Any] | None:
    """Load a document checkpoint if it matches the current config."""
    ckpt_path = _checkpoint_path(output_root, document_id)
    if not ckpt_path.is_file():
        return None
    try:
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # Only skip if config matches
    if (
        ckpt.get("status") == "completed"
        and ckpt.get("input_pdf_sha256") == input_pdf_sha256
        and ckpt.get("config_hash") == config_hash
    ):
        return ckpt
    return None


def _save_checkpoint(
    output_root: Path,
    document_id: str,
    result: DocumentRunResult,
) -> None:
    ckpt_path = _checkpoint_path(output_root, document_id)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _count_output_files(output_dir: Path) -> int:
    """Count all files in the output directory."""
    if not output_dir.is_dir():
        return 0
    return sum(1 for _ in output_dir.rglob("*") if _.is_file())


def run_document(
    *,
    mineru_bin: Path,
    document_id: str,
    input_pdf: Path,
    output_root: Path,
    config: MinerUConfig,
    cuda_visible_devices: str,
    tmpdir: Path,
    timeout_seconds: int = 7200,
) -> DocumentRunResult:
    """Run MinerU on one document with checkpoint support.

    If a valid checkpoint exists (same PDF hash + config hash), the
    document is skipped.
    """
    input_sha = sha256_file(input_pdf)

    # Check for valid checkpoint
    existing = _load_checkpoint(output_root, document_id, input_sha, config.config_hash)
    if existing is not None:
        return DocumentRunResult(
            document_id=document_id,
            status="skipped",
            input_pdf_sha256=input_sha,
            config_hash=config.config_hash,
            output_manifest_hash=existing.get("output_manifest_hash"),
            processed_page_count=int(existing.get("processed_page_count", 0)),
            started_at=existing.get("started_at", ""),
            completed_at=existing.get("completed_at", ""),
            elapsed_seconds=float(existing.get("elapsed_seconds", 0)),
            return_code=existing.get("return_code"),
            error=None,
            output_dir=existing.get("output_dir", ""),
        )

    output_dir = output_root / document_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run.log"
    started_at = _now_iso()
    started = time.time()

    return_code, error = _run_mineru(
        mineru_bin=mineru_bin,
        input_pdf=input_pdf,
        output_dir=output_dir,
        config=config,
        cuda_visible_devices=cuda_visible_devices,
        tmpdir=tmpdir,
        timeout_seconds=timeout_seconds,
        log_path=log_path,
    )

    elapsed = time.time() - started
    completed_at = _now_iso()

    file_count = _count_output_files(output_dir)
    status = "completed" if return_code == 0 and error is None else "failed"

    result = DocumentRunResult(
        document_id=document_id,
        status=status,
        input_pdf_sha256=input_sha,
        config_hash=config.config_hash,
        output_manifest_hash=None,  # Filled by deterministic manifest builder
        processed_page_count=file_count,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed,
        return_code=return_code,
        error=error,
        output_dir=str(output_dir),
    )

    _save_checkpoint(output_root, document_id, result)
    return result


def run_full_corpus(
    *,
    mineru_bin: Path,
    documents: list[dict[str, Any]],
    output_root: Path,
    config: MinerUConfig,
    cuda_visible_devices: str,
    tmpdir: Path,
    timeout_seconds: int = 7200,
) -> list[DocumentRunResult]:
    """Run MinerU on all documents in deterministic order.

    Documents are sorted by document_id.  One document at a time.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    sorted_docs = sorted(documents, key=lambda d: str(d.get("document_id") or ""))

    results: list[DocumentRunResult] = []
    for doc in sorted_docs:
        doc_id = str(doc["document_id"])
        pdf_path = Path(doc["pdf_path"])
        result = run_document(
            mineru_bin=mineru_bin,
            document_id=doc_id,
            input_pdf=pdf_path,
            output_root=output_root,
            config=config,
            cuda_visible_devices=cuda_visible_devices,
            tmpdir=tmpdir,
            timeout_seconds=timeout_seconds,
        )
        results.append(result)
        print(
            f"  [{result.status}] {doc_id} "
            f"({result.elapsed_seconds:.1f}s, rc={result.return_code})"
        )

    return results


def capture_runtime_environment(
    mineru_bin: Path,
    config: MinerUConfig,
    cuda_visible_devices: str,
) -> dict[str, Any]:
    """Capture the MinerU runtime environment for the manifest."""
    info: dict[str, Any] = {
        "mineru_path": str(mineru_bin),
        "mineru_version": None,
        "python_path": str(mineru_bin.parent / "python"),
        "python_version": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "torch_cuda_available": None,
        "accelerate_version": None,
        "backend": config.backend,
        "effort": config.effort,
        "method": config.method,
        "cuda_visible_devices": cuda_visible_devices,
        "config_hash": config.config_hash,
    }

    # MinerU version
    try:
        version = subprocess.run(
            [str(mineru_bin), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        info["mineru_version"] = (version.stdout or version.stderr).strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass

    python_bin = mineru_bin.parent / "python"
    if python_bin.is_file():
        # Python version
        try:
            pv = subprocess.run(
                [str(python_bin), "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            info["python_version"] = (pv.stdout or pv.stderr).strip() or None
        except (OSError, subprocess.TimeoutExpired):
            pass

        # Torch + accelerate
        probe = (
            "import torch; "
            "print('torch=' + torch.__version__); "
            "print('cuda=' + str(torch.version.cuda)); "
            "print('cuda_avail=' + str(torch.cuda.is_available())); "
            "import importlib; "
            "acc = importlib.import_module('accelerate'); "
            "print('accelerate=' + getattr(acc, '__version__', 'UNKNOWN'))"
        )
        try:
            tp = subprocess.run(
                [str(python_bin), "-c", probe],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            for line in tp.stdout.splitlines():
                line = line.strip()
                if line.startswith("torch="):
                    info["torch_version"] = line.split("=", 1)[1]
                elif line.startswith("cuda="):
                    info["torch_cuda_version"] = line.split("=", 1)[1]
                elif line.startswith("cuda_avail="):
                    info["torch_cuda_available"] = line.split("=", 1)[1].lower() == "true"
                elif line.startswith("accelerate="):
                    val = line.split("=", 1)[1]
                    info["accelerate_version"] = val if val != "NOT_AVAILABLE" else None
        except (OSError, subprocess.TimeoutExpired):
            pass

    return info


def nvidia_snapshot() -> dict[str, Any]:
    """Capture nvidia-smi output for the manifest."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        return {
            "command": command,
            "return_code": completed.returncode,
            "output": completed.stdout.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "return_code": None, "output": "", "error": str(exc)}
