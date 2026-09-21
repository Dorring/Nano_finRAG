#!/usr/bin/env python3
"""P1.8-D1: promote the regenerated cross-entity fixture into the benchmark.

`plan-fixtures-v8` pins two cross-entity rows to the operand order of a question
the benchmark no longer asks.  D1-C3 regenerated them through the authoring
contract into `plan-fixtures-v9`; C3 confirmed that v8 releases those two cases
with the wrong sign and v9 releases them correctly, and C5 added the guard that
would have caught it.  This writes v9 into the benchmark tree and records the
migration, so the sealed benchmark is the one the sealed numbers describe.

    python scripts/evaluation/promote_p1_8_d1_fixture_v9.py \
        --source /path/to/plan-fixtures-v9.jsonl \
        --migration /path/to/p1-8-d1-c3-migration.json

v8 is **not** overwritten.  Its hashes move into `benchmark-version.json`'s
`migrations[]` entry, which is how `p1.6-0h` already records its own
predecessor.  Nothing here reads the fact store or the network: it is a file
operation over bytes that were produced and verified elsewhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[2]
BENCH = BACKEND / "benchmarks/tv2_canonical_v1"

FIXTURE_NAME = "plan-fixtures-v9"
MIGRATION_ID = "p1.8-d1-c3"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def render(rows: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"


def dumps(record: Any) -> str:
    return json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="the regenerated fixture")
    parser.add_argument("--migration", type=Path, required=True, help="the C3 migration record")
    parser.add_argument("--bench", type=Path, default=BENCH)
    args = parser.parse_args(argv)

    record = json.loads(args.migration.read_text(encoding="utf-8"))
    source_bytes = args.source.read_bytes()
    source_digest = sha256_bytes(source_bytes)

    # The bytes promoted must be the bytes that were measured.  The C3 record
    # states the digest it produced; a mismatch means the file moved underneath
    # the run and the sealed numbers describe something else.
    expected = record["new_fixture"]["sha256"]
    if source_digest != expected:
        raise SystemExit(
            f"source digest {source_digest} != the C3 record's {expected}; refusing to promote"
        )

    rows = load_rows(args.source)
    # Re-render and compare so the digest in the manifest is the digest of the
    # file, not of a round trip that might differ by a newline.
    if sha256_bytes(render(rows).encode("utf-8")) != source_digest:
        raise SystemExit("the fixture does not round-trip to its own digest")

    version_path = args.bench / "benchmark-version.json"
    version_text = version_path.read_text(encoding="utf-8")
    version = json.loads(version_text)
    # Formatting is part of the artefact: rewrite only if the file is exactly
    # this serialisation, so a hand-edited file is not silently reformatted.
    if dumps(version) != version_text:
        raise SystemExit(f"{version_path} is not in canonical form; refusing to rewrite it")

    previous = version["files"].get("plan-fixtures-v8.jsonl")
    if previous is None:
        raise SystemExit("benchmark-version.json does not name plan-fixtures-v8.jsonl")

    # --- the fixture and its sidecar ------------------------------------------------------------
    write(args.bench / f"{FIXTURE_NAME}.jsonl", render(rows))
    write(args.bench / f"{FIXTURE_NAME}.jsonl.sha256", source_digest + "\n")

    # --- the manifest ----------------------------------------------------------------------------
    # Only what the rows themselves state is computed here.  The generator's
    # ambiguity audit is not re-derivable from a fixture, so it is deliberately
    # absent rather than carried over from v8, where it described different rows.
    manifest = {
        "stage": "P1.8-D1-C3-FIXTURE-REGENERATION",
        "fixture": FIXTURE_NAME,
        "total": len(rows),
        "fixture_sha256": source_digest,
        "eval_set_sha256": record["provenance"]["eval_set_sha256"],
        "gold_sha256": record["provenance"]["gold_sha256"],
        "intent_distribution": dict(
            sorted(Counter(row["plan"]["intent"] for row in rows).items())
        ),
        "provenance": dict(
            sorted(
                Counter(
                    f"{field}:{tier}"
                    for row in rows
                    for field, tier in row["sourced_from"].items()
                ).items()
            )
        ),
        "coordinate_candidate_histogram": dict(
            sorted(
                Counter(
                    count
                    for row in rows
                    for count in row["coordinate_candidates"].values()
                ).items()
            )
        ),
        "abstention_rows_expecting_no_candidate": sum(
            1 for row in rows if row["expects_no_candidate"]
        ),
        "predecessor": {"plan-fixtures-v8.jsonl": previous["sha256"]},
        "rows_changed": record["rows_changed"],
        "record": f"migrations/{MIGRATION_ID}.json",
        "spec": "docs/evaluation/p1-8-d1-c3-fixture-regeneration.md",
        "not_derivable_here": [
            "ambiguous_slots",
            "ambiguous_sha256",
            "operand_facts_read",
            "fact_store_sha256",
        ],
    }
    write(args.bench / f"{FIXTURE_NAME}.manifest.json", dumps(manifest))

    # --- the migration record, in the tree it is about -------------------------------------------
    write(args.bench / "migrations" / f"{MIGRATION_ID}.json", dumps(record))

    # --- benchmark-version.json ------------------------------------------------------------------
    version["files"] = {
        key: value
        for key, value in version["files"].items()
        if key != "plan-fixtures-v8.jsonl"
    }
    version["files"][f"{FIXTURE_NAME}.jsonl"] = {
        "role": "plan fixtures",
        "rows": len(rows),
        "sha256": source_digest,
    }
    version["migrations"].append(
        {
            "id": MIGRATION_ID,
            "applied_in": "docs/evaluation/p1-8-d1-c3-fixture-regeneration.md",
            "stratum": "cross_entity_comparison",
            "pre": {"plan-fixtures-v8.jsonl": previous["sha256"]},
            "post": {f"{FIXTURE_NAME}.jsonl": source_digest},
            "record": f"migrations/{MIGRATION_ID}.json",
            "summary": (
                "Two cross-entity rows were pinned with the operand order of the "
                "pre-P1.6-0H question. The runtime executed them faithfully and "
                "released both with the wrong sign. Regenerated through the "
                "authoring contract; no question, gold or runtime source changed."
            ),
        }
    )
    version["version"] = "tv2-canonical-v1-post-p1.8-d1-c3"
    version["note"] = (
        "Content changed under unchanged file names at P1.6-0H. The pre-migration "
        "revision is identified by the hashes under migrations[].pre; do not treat "
        "these file names as implying the pre-migration content."
    )
    write(version_path, dumps(version))

    print(f"fixture   {args.bench / (FIXTURE_NAME + '.jsonl')}  {source_digest}")
    print(f"manifest  {args.bench / (FIXTURE_NAME + '.manifest.json')}")
    print(f"migration {args.bench / 'migrations' / (MIGRATION_ID + '.json')}")
    print(f"version   {version_path}  -> {version['version']}")
    print(f"v8 kept   {args.bench / 'plan-fixtures-v8.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
