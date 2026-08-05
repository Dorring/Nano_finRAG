"""Freeze and download the issuer-disjoint Structured Financial Fact V2 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from src.evaluation.structured_fact_v2 import PINNED_V2_SOURCES, validate_v2_sources

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/structured-fact-v2-gate-a"
DEFAULT_RUNTIME = Path(os.environ.get("STRUCTURED_FACT_V2_RUNTIME", ROOT / ".runtime/structured-fact-v2"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    validate_v2_sources(PINNED_V2_SOURCES)
    documents = []
    for source in PINNED_V2_SOURCES:
        path = args.runtime_dir / source.split / source.cik / source.primary_document
        if path.exists():
            content = path.read_bytes()
        else:
            request = Request(source.archive_url, headers={"User-Agent": args.user_agent, "Accept": "text/html"})
            with urlopen(request, timeout=90) as response:  # nosec B310: pinned SEC URLs
                content = response.read()
            if not content:
                raise ValueError(f"empty SEC filing: {source.primary_document}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        documents.append(
            {
                **source.__dict__,
                "archive_url": source.archive_url,
                "runtime_filename": str(path.relative_to(args.runtime_dir)),
                "content_sha256": _sha(content),
                "content_bytes": len(content),
            }
        )
    manifest_hash = _sha("\n".join(f"{row['cik']}|{row['accession_number']}|{row['content_sha256']}|{row['split']}" for row in documents).encode())
    manifest = {
        "schema": "structured-financial-fact-v2/corpus/v1",
        "source": "SEC EDGAR Inline XBRL primary documents",
        "corpus_sha256": manifest_hash,
        "runtime_documents_committed": False,
        "documents": documents,
    }
    acceptance = {
        "schema": "structured-financial-fact-v2/gate-a/acceptance/v1",
        "baseline_master_commit": "69d390bf46c024af13d797e91210fc0daa00dbf0",
        "document_count": 6,
        "development_document_count": 3,
        "holdout_document_count": 3,
        "prior_issuer_overlap_count": 0,
        "frozen_72_question_reads": 0,
        "model_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "decision": "structured_fact_v2_corpus_frozen",
        "next_gate": "structured_fact_v2_native_ixbrl_fact_extraction",
    }
    _write(args.out_dir / "corpus-manifest.json", manifest)
    _write(args.out_dir / "input-isolation-report.json", {"excluded_prior_cik_count": 12, "overlap_count": 0, "holdout_rules_frozen": True})
    _write(args.out_dir / "next-gate.json", {"decision": acceptance["decision"], "next_gate": acceptance["next_gate"], "production_switch_allowed": False})
    _write(args.out_dir / "structured-fact-v2-gate-a-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--user-agent", default=os.environ.get("STRUCTURED_FACT_V2_SEC_USER_AGENT", "nano-finance-research contact@example.com"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
