"""Build a strict, source-derived V2 Fact Store from parsed canonical corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.runtime.trusted_v2_canonical_fact_store import (
    CANONICAL_FACT_STORE_SCHEMA_VERSION,
    build_canonical_fact_store,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.parsed_root.glob("*/*/document.json"))
    if not paths:
        parser.error(f"no parsed documents at {args.parsed_root}")
    if args.out.exists() and not args.overwrite and not args.dry_run:
        parser.error(f"--out already exists: {args.out}")
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    records, summary = build_canonical_fact_store(documents)
    manifest = {
        "schema_version": CANONICAL_FACT_STORE_SCHEMA_VERSION,
        "parsed_root": str(args.parsed_root.resolve()),
        "parsed_document_count": len(documents),
        "summary": summary.to_dict(),
        "execution_inputs": {"questions_read": False, "gold_read": False, "answer_text_read": False, "conversation_history_read": False, "model_called": False},
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    manifest["output_facts"] = str(args.out.resolve())
    manifest["output_facts_sha256"] = _sha256(args.out)
    target = args.manifest_out or args.out.with_suffix(args.out.suffix + ".manifest.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
