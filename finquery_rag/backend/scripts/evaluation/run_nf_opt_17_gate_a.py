"""NF-OPT-17 Gate A: freeze a disjoint SEC development-corpus manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from src.evaluation.nf_opt_17 import (
    PINNED_DEVELOPMENT_SOURCES,
    build_annotation_contract,
    source_manifest_hash,
    source_record,
    validate_development_sources,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "benchmarks/financial_rag_v1/corpus.json"
DEFAULT_OUT = ROOT / "artifacts/evaluation/nf-opt-17"
DEFAULT_RUNTIME_DIR = Path(os.environ.get("NF_OPT_17_RUNTIME_DIR", ROOT / ".runtime" / "nf-opt-17-dev-corpus"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _download(source_url: str, *, user_agent: str) -> tuple[bytes, str]:
    request = Request(source_url, headers={"User-Agent": user_agent, "Accept": "text/html"})
    with urlopen(request, timeout=60) as response:  # nosec B310: pinned SEC URLs only
        body = response.read()
        content_type = str(response.headers.get("Content-Type") or "")
    if not body:
        raise ValueError(f"empty SEC response: {source_url}")
    if "html" not in content_type.casefold():
        raise ValueError(f"unexpected SEC content type: {content_type}")
    return body, content_type


def run(args: argparse.Namespace) -> int:
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    frozen_filenames = [str(item["filename"]) for item in corpus["documents"]]
    validate_development_sources(PINNED_DEVELOPMENT_SOURCES, frozen_filenames=frozen_filenames)

    records = []
    for source in PINNED_DEVELOPMENT_SOURCES:
        record = source_record(source)
        record["downloaded"] = False
        record["runtime_filename"] = None
        record["content_sha256"] = None
        record["content_bytes"] = None
        record["content_type"] = None
        if args.download:
            content, content_type = _download(source.archive_url, user_agent=args.user_agent)
            runtime_path = args.runtime_dir / source.cik / source.primary_document
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_path.write_bytes(content)
            record.update(
                {
                    "downloaded": True,
                    "runtime_filename": str(runtime_path.relative_to(args.runtime_dir)),
                    "content_sha256": _sha(content),
                    "content_bytes": len(content),
                    "content_type": content_type,
                }
            )
        records.append(record)

    downloaded_count = sum(record["downloaded"] for record in records)
    if args.download and downloaded_count != len(records):
        raise ValueError("all pinned development filings must download before the corpus is frozen")
    manifest = {
        "schema": "nf-opt-17/development-corpus-manifest/v1",
        "source": "SEC EDGAR primary documents",
        "source_manifest_sha256": source_manifest_hash(PINNED_DEVELOPMENT_SOURCES),
        "download_mode": "runtime_shadow_only" if args.download else "manifest_only",
        "runtime_dir_committed": False,
        "documents": records,
    }
    disjointness = {
        "frozen_corpus_filename_count": len(frozen_filenames),
        "development_document_count": len(records),
        "development_cik_count": len({record["cik"] for record in records}),
        "filename_overlap_count": 0,
        "frozen_benchmark_fields_used_for_construction": False,
        "development_documents_written_to_production_index": False,
    }
    acceptance = {
        "schema": "nf-opt-17/gate-a/acceptance/v1",
        "baseline_master_merge_commit": "60c63b9f53ecd2e0ba81f100e3dce6a6f4aa8085",
        "frozen_corpus_sha256": _sha(args.corpus.read_bytes()),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "development_document_count": len(records),
        "downloaded_document_count": downloaded_count,
        "all_downloaded_content_hashed": downloaded_count == len(records),
        "frozen_benchmark_overlap_count": 0,
        "training_examples_created": 0,
        "model_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "decision": "financial_hard_negative_dev_corpus_ready_for_independent_annotation",
        "next_gate": "nf-opt-17-gate-b-independent-hard-negative-annotation",
    }
    _write(args.out_dir / "development-corpus-manifest.json", manifest)
    _write(args.out_dir / "frozen-benchmark-disjointness-report.json", disjointness)
    _write(args.out_dir / "hard-negative-annotation-contract.json", build_annotation_contract())
    _write(args.out_dir / "next-gate.json", {
        "decision": acceptance["decision"],
        "next_gate": acceptance["next_gate"],
        "production_switch_allowed": False,
    })
    _write(args.out_dir / "nf-opt-17-gate-a-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--user-agent", default=os.environ.get("NF_OPT_17_SEC_USER_AGENT", "nano-finance-research contact@example.com"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
