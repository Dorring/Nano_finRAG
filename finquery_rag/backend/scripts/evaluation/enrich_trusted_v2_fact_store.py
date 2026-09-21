"""Build an optional source-derived structural context sidecar for V2 facts.

This is an artifact-construction utility, not a retrieval, Binder, or model
runner. It does not read evaluation Gold, questions, answers, or conversation
history. The original fact store remains immutable; write output to a new path
and explicitly configure it only after reviewing the generated manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.trusted_v2_fact_enrichment import (  # noqa: E402
    ENRICHMENT_SCHEMA_VERSION,
    enrich_fact_records,
    read_jsonl,
    sha256_file,
    write_jsonl,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True, help="sealed input native-facts JSONL")
    parser.add_argument(
        "--structured-views",
        type=Path,
        required=True,
        help="existing source-derived structured-views JSONL",
    )
    parser.add_argument("--out", type=Path, required=True, help="new enriched fact-store JSONL")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="output manifest path (default: <out>.manifest.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing output artifact",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the summary without writing output",
    )
    args = parser.parse_args()

    if not args.facts.is_file():
        parser.error(f"--facts does not exist: {args.facts}")
    if not args.structured_views.is_file():
        parser.error(f"--structured-views does not exist: {args.structured_views}")
    if args.out.exists() and not args.overwrite and not args.dry_run:
        parser.error(f"--out already exists (pass --overwrite to replace): {args.out}")

    facts = read_jsonl(args.facts)
    views = read_jsonl(args.structured_views)
    enriched, summary = enrich_fact_records(facts, views)
    manifest: dict[str, object] = {
        "schema_version": ENRICHMENT_SCHEMA_VERSION,
        "source_facts": str(args.facts.resolve()),
        "source_facts_sha256": sha256_file(args.facts),
        "structured_views": str(args.structured_views.resolve()),
        "structured_views_sha256": sha256_file(args.structured_views),
        "output_facts": str(args.out.resolve()),
        "summary": summary,
        "execution_inputs": {
            "questions_read": False,
            "gold_read": False,
            "answer_text_read": False,
            "model_called": False,
        },
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    write_jsonl(args.out, enriched)
    manifest["output_facts_sha256"] = sha256_file(args.out)
    manifest_path = args.manifest_out or args.out.with_suffix(args.out.suffix + ".manifest.json")
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
