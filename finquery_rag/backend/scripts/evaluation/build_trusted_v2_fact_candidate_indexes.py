"""Build a production-compatible four-lane R4 index from a V2 fact store.

The resulting index deliberately uses the exact ``candidate_key`` already
present in the supplied fact store.  It is therefore safe to configure both
artifacts together for ``TrustedFinancialRuntimeV2``: every retrieval result
has a deterministic structured-fact materialization path.

This is an offline asset build.  It reads only source-derived fact records and
the configured embedding encoder; it never reads questions, Gold labels,
answers, conversation history, or model-generated summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexBuilder  # noqa: E402
from src.runtime.trusted_v2_fact_enrichment import (  # noqa: E402
    read_jsonl,
    sha256_file,
)
from src.runtime.trusted_v2_fact_index import (  # noqa: E402
    FACT_CANDIDATE_INDEX_SCHEMA_VERSION,
    build_fact_store_candidate_views,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _output_is_nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--facts",
        type=Path,
        required=True,
        help="sealed source-derived V2 fact-store JSONL",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="new R4 candidate-index directory",
    )
    parser.add_argument(
        "--encoder-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformers model/path used for both dense lanes",
    )
    parser.add_argument(
        "--encoder-device",
        default="cpu",
        help="SentenceTransformers device for this offline build (default: cpu)",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="manifest path (default: <out-dir>/trusted-v2-fact-index-manifest.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing index files in a non-empty --out-dir",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate source fact/index alignment without building dense/BM25 files",
    )
    args = parser.parse_args()

    if not args.facts.is_file():
        parser.error(f"--facts does not exist: {args.facts}")
    if _output_is_nonempty(args.out_dir) and not args.overwrite and not args.dry_run:
        parser.error(f"--out-dir is non-empty (pass --overwrite to replace): {args.out_dir}")

    facts = read_jsonl(args.facts)
    pairs, summary = build_fact_store_candidate_views(facts)
    manifest: dict[str, Any] = {
        "schema_version": FACT_CANDIDATE_INDEX_SCHEMA_VERSION,
        "source_facts": str(args.facts.resolve()),
        "source_facts_sha256": sha256_file(args.facts),
        "out_dir": str(args.out_dir.resolve()),
        "encoder_model": str(args.encoder_model),
        "encoder_device": str(args.encoder_device),
        "candidate_contract": summary.to_dict(),
        "execution_inputs": {
            "questions_read": False,
            "gold_read": False,
            "answer_text_read": False,
            "conversation_history_read": False,
            "generator_called": False,
            "embedding_encoder_called": not args.dry_run,
        },
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    builder = CandidateViewIndexBuilder(
        args.out_dir,
        str(args.encoder_model),
        encoder_device=str(args.encoder_device),
    )
    manifest["r4_index"] = builder.build(pairs)
    manifest["r4_index_compatible_fact_candidate_count"] = summary.candidate_pair_count
    manifest_path = args.manifest_out or args.out_dir / "trusted-v2-fact-index-manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
