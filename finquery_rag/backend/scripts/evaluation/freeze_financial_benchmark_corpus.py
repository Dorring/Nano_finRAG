"""Convert the ignored runtime corpus manifest into a stable benchmark manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.evaluation.benchmark_foundation import corpus_hash, document_identity_hash
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.evaluation.benchmark_foundation import corpus_hash, document_identity_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-manifest", type=Path, default=Path("runtime/benchmark/financial_rag_v1/corpus-manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/financial_rag_v1/corpus.json"))
    args = parser.parse_args()
    runtime = json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
    documents = []
    for item in runtime["documents"]:
        document = {
            "company": item["company"],
            "document_id": item["document_id"],
            "filename": item["local_filename"],
            "fiscal_year": int(item["fiscal_year"]),
            "source_type": "official_annual_report",
            "source_format": "official_pdf" if item["source_format"] == "official_pdf" else item["source_format"],
            "page_count": int(item["page_count"]),
            "chunk_count": int(item["chunk_count"]),
            "file_sha256": item["sha256"],
        }
        if item["source_format"] == "official_docx_rendered_pdf":
            document["source_format"] = "docx_converted_pdf"
        elif item["source_format"] == "official_sec_html_rendered_pdf":
            document["source_format"] = "sec_html_converted_pdf"
        document["document_identity_hash"] = document_identity_hash(document)
        documents.append(document)
    documents.sort(key=lambda item: item["document_id"])
    payload = {
        "benchmark_id": "financial-rag-v1",
        "schema_version": "1.0",
        "tenant_id": int(runtime["tenant_id"]),
        "document_count": len(documents),
        "total_pages": sum(item["page_count"] for item in documents),
        "total_chunks": sum(item["chunk_count"] for item in documents),
        "documents": documents,
    }
    payload["corpus_hash"] = corpus_hash(documents)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
