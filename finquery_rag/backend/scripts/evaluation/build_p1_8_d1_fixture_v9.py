#!/usr/bin/env python3
"""P1.8-D1-C3: regenerate the stale cross-entity fixtures through the contract.

`crossdiff-001` and `crossdiff-002` are pinned with the operand order of a
question the benchmark no longer asks: their slots name Tesla before NVIDIA and
Microsoft before Apple, while the canonical question, the gold's `fact_ids` and
the gold's `expected_value` all name the entity the other way round.

This script does not swap those slots by hand.  It re-runs the authoring
contract that is supposed to produce them -- `build_p1_2_plan_fixtures`, whose
cross-entity branch takes one slot per `gold.fact_ids` entry *in the gold's
order*, each slot naming the entity its fact belongs to -- and writes what that
contract emits.  The correction is therefore derived and re-derivable; a reader
can reproduce v9 from v8 by running the command in `generation.command`.

    .venv/bin/python scripts/evaluation/build_p1_8_d1_fixture_v9.py --dry-run
    .venv/bin/python scripts/evaluation/build_p1_8_d1_fixture_v9.py --apply

Run on 4090-qh: the authoring contract reads the deployment's fact store, and a
gold fact's entity is only knowable from the store that holds it.

What this script deliberately does not do:

* it does not change any question or any gold.  Both are the authority the
  fixture is being reconciled *to*, not the thing under repair;
* it does not regenerate the other three cross-entity cases.  Their operand
  order already agrees with the contract, so they are carried as a **control**
  -- if the contract stopped reproducing them the build fails, because that
  would mean the contract, not the fixture, is what moved;
* it does not run the full `enrich_and_validate` gate.  That gate currently
  aborts on `compare-001` for a reason unrelated to this stratum (see
  `validator_note` in the migration record).  The C5 invariant it needs here is
  re-implemented below and checked directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

BACKEND = Path("/disk/qh/nano-finrag/finquery_rag/backend")
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(BACKEND / "scripts/evaluation") not in sys.path:
    sys.path.insert(0, str(BACKEND / "scripts/evaluation"))

import build_p1_2_plan_fixtures as contract  # noqa: E402
import fixture_integrity  # noqa: E402

V2 = Path("/disk/qh/nano-finrag/artifacts/evaluation/p1-8-c-v2")
OUT_DIR = Path("/disk/qh/nano-finrag/artifacts/evaluation/p1-8-d1-c3")

EVAL_SET = V2 / "canonical-eval-v1.jsonl"
GOLD = V2 / "gold-evidence-v1.jsonl"
V8 = V2 / "plan-fixtures-v8.jsonl"

#: The pinned fixture the C3 brief names.  Their slots disagree with the
#: contract, so they are the ones this migration replaces.
REGENERATE = ("tv2f01-s3-crossdiff-001", "tv2f01-s3-crossdiff-002")

#: Same stratum, same authoring branch, operand order already in agreement.
#: Regenerating them must be a no-op or the contract has drifted underneath the
#: benchmark and nothing below is trustworthy.
CONTROL = (
    "tv2f01-s3-crossdiff-003",
    "tv2f01-s3-crossdiff-004",
    "tv2f01-s3-crossdiff-005",
)

# The mention->entity resolution and the invariant itself live in
# `fixture_integrity`, so the generator and the standalone verifier check the
# same rule with the same code.  A guard that re-implements the rule it guards
# is a second opinion nobody asked for.



def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render(rows: list[dict[str, Any]]) -> str:
    """The fixture file's own rendering, reused so v9's digest is comparable."""

    return contract.render_fixtures(rows)


def row_digest(row: dict[str, Any]) -> str:
    return sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True))


def generate(case: str, record: dict, gold: dict, facts: dict, index: dict) -> dict:
    """One cross-entity fixture row, exactly as the contract emits it."""

    row = contract.author_plan(record["question"], gold, record, facts)
    contract._enrich_slots(row, gold, facts, index, case=case, entity_hint="")
    row["coordinate_candidates"] = {
        slot["slot_id"]: len(contract._slot_candidates(slot, index))
        for slot in row["plan"]["required_slots"]
    }
    row["expects_no_candidate"] = record["stratum"] == "adversarial_abstention"
    return row


def field_diff(old: dict, new: dict) -> dict[str, Any]:
    """Old -> new for every leaf that moved, keyed by dotted path."""

    changes: dict[str, Any] = {}

    def walk(path: str, left: Any, right: Any) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                walk(f"{path}.{key}" if path else key, left.get(key), right.get(key))
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                changes[path] = {"old": left, "new": right}
                return
            for position, (l_item, r_item) in enumerate(zip(left, right)):
                walk(f"{path}[{position}]", l_item, r_item)
        elif left != right:
            changes[path] = {"old": left, "new": right}

    walk("", old, new)
    return changes


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BACKEND,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is recorded, not required
        return "unknown"


def store_digest(path: Path) -> str | None:
    return sha256_text(path.read_text(encoding="utf-8")) if path.is_file() else None


def _sealed_v8_digest() -> str:
    """The frozen v8 digest, from whichever authority this checkout carries.

    The Windows checkout writes a `.sha256` sidecar beside the fixture; the run
    host's is present too, and a checkout that carries `benchmark-version.json`
    states the same digest there.  Both name the same bytes -- verified equal
    before this migration was written -- so either answers the question.
    """

    directory = BACKEND / "benchmarks/tv2_canonical_v1"
    sidecar = directory / "plan-fixtures-v8.jsonl.sha256"
    if sidecar.is_file():
        return sidecar.read_text(encoding="utf-8").strip()
    version = json.loads((directory / "benchmark-version.json").read_text(encoding="utf-8"))
    return version["files"]["plan-fixtures-v8.jsonl"]["sha256"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the files")
    parser.add_argument("--dry-run", action="store_true", help="report only (default)")
    args = parser.parse_args(argv)
    apply_changes = args.apply

    v8_rows = contract._load_records(V8)
    v8_list = [json.loads(line) for line in V8.read_text(encoding="utf-8").splitlines() if line.strip()]
    v8_text = render(v8_list)
    v8_digest = sha256_text(v8_text)
    # The file names alone do not identify a revision -- the content changed
    # under unchanged names at P1.6-0H -- so the frozen bytes are identified by
    # digest, from whichever authority this checkout carries.
    sealed_v8 = _sealed_v8_digest()
    if v8_digest != sealed_v8:
        raise SystemExit(
            f"v8 is not the sealed fixture: recomputed {v8_digest}, sealed {sealed_v8}"
        )

    eval_by_id = contract._load_records(EVAL_SET)
    gold_by_id = contract._load_gold(GOLD)
    stores = [contract.DEFAULT_FACT_STORE, contract.DEFAULT_IXBRL_FACT_STORE]
    facts = contract.load_fact_coordinates(stores, contract.operand_fact_ids(GOLD))
    index = contract.load_coordinate_index(stores)

    cases = list(REGENERATE) + list(CONTROL)
    generated: dict[str, dict] = {}
    invariant: dict[str, dict] = {}
    for case in cases:
        generated[case] = generate(
            case, eval_by_id[case], gold_by_id[case], facts, index
        )
        invariant[case] = fixture_integrity.check_row(
            generated[case], eval_by_id[case]["question"]
        )

    # The control has to hold before anything else is worth reading: the contract
    # must still reproduce the cases the benchmark already agrees with.
    control_failures: dict[str, Any] = {}
    for case in CONTROL:
        old_entities = [s.get("entity") for s in v8_rows[case]["plan"]["required_slots"]]
        new_entities = [s.get("entity") for s in generated[case]["plan"]["required_slots"]]
        if old_entities != new_entities:
            control_failures[case] = {"v8": old_entities, "contract": new_entities}
    if control_failures:
        raise SystemExit(f"control failed -- contract drift: {json.dumps(control_failures, indent=2)}")

    for case in REGENERATE:
        verdict = invariant[case]
        if verdict.get("binds") and not verdict.get("satisfied"):
            raise SystemExit(f"{case}: C5 invariant violated -- {json.dumps(verdict)}")

    v9_rows = {row["id"]: row for row in v8_list}
    changes: dict[str, Any] = {}
    for case in REGENERATE:
        changes[case] = field_diff(v8_rows[case], generated[case])
        v9_rows[case] = generated[case]

    v9_list = [v9_rows[row["id"]] for row in v8_list]
    v9_text = render(v9_list)
    v9_digest = sha256_text(v9_text)
    moved = [row["id"] for row in v8_list if row_digest(v9_rows[row["id"]]) != row_digest(row)]

    record = {
        "migration": "p1.8-d1-c3",
        "stage": "P1.8-D1-C3-CONTRACT-REGENERATION",
        "applied": bool(apply_changes),
        "root_cause_addressed": "STALE_PINNED_PLAN_FIXTURE",
        "why_stale": (
            "The P1.6-0H migration rewrote each cross-entity slot's metric, the "
            "gold's fact_ids and the canonical question, but never re-derived the "
            "slot entities or their order and never refreshed the fixture's own "
            "question string. The slot order is therefore a fossil of the "
            "pre-0H question: `crossdiff-001` was asked as `Additions between "
            "Tesla and NVIDIA` and still stores Tesla as the minuend, while the "
            "canonical question, the gold fact_ids order and the gold "
            "expected_value all put NVIDIA first."
        ),
        "generation": {
            "command": (
                ".venv/bin/python scripts/evaluation/build_p1_8_d1_fixture_v9.py --apply"
            ),
            "code_path": (
                "scripts/evaluation/build_p1_2_plan_fixtures.py::author_plan -> "
                "_multi_evidence_slots (one slot per gold.fact_ids entry, in the "
                "gold's order, entity read from the fact store)"
            ),
            "invoked_as": "module import; the contract is called, not restated",
            "contract_script_sha256": store_digest(
                BACKEND / "scripts/evaluation/build_p1_2_plan_fixtures.py"
            ),
            "driver_script_sha256": store_digest(Path(__file__).resolve()),
            "commit_sha": git_commit(),
        },
        "provenance": {
            "eval_set": str(EVAL_SET),
            "eval_set_sha256": store_digest(EVAL_SET),
            "gold": str(GOLD),
            "gold_sha256": store_digest(GOLD),
            "fact_store": str(contract.DEFAULT_FACT_STORE),
            "fact_store_sha256": store_digest(contract.DEFAULT_FACT_STORE),
            "ixbrl_fact_store": str(contract.DEFAULT_IXBRL_FACT_STORE),
            "ixbrl_fact_store_sha256": store_digest(contract.DEFAULT_IXBRL_FACT_STORE),
        },
        "old_fixture": {"path": str(V8), "sha256": v8_digest},
        "new_fixture": {
            "path": str(OUT_DIR / "plan-fixtures-v9.jsonl"),
            "sha256": v9_digest,
        },
        "rows_changed": moved,
        "row_digests": {
            case: {
                "question": v8_rows[case]["question"],
                "question_sha256": sha256_text(v8_rows[case]["question"]),
                "old_row_sha256": row_digest(v8_rows[case]),
                "new_row_sha256": row_digest(generated[case]),
                "old_slot_entities": [
                    s.get("entity") for s in v8_rows[case]["plan"]["required_slots"]
                ],
                "new_slot_entities": [
                    s.get("entity") for s in generated[case]["plan"]["required_slots"]
                ],
                "new_question": generated[case]["question"],
                "new_question_sha256": sha256_text(generated[case]["question"]),
                "field_diff": changes[case],
            }
            for case in REGENERATE
        },
        "c5_invariant": invariant,
        "control": {
            "cases": list(CONTROL),
            "result": "contract reproduces v8 slot entities for all three",
        },
        "validator_note": (
            "`enrich_and_validate` is not run over the whole set here: it aborts "
            "on `tv2f01-s3-compare-001` with `an answerable slot selects no "
            "fact`. That is pre-existing and unrelated to this stratum -- the "
            "coordinate index folds a record's `metric` field, which is empty on "
            "every rebuilt-iXBRL row, while a slot names the metric in prose. "
            "The C5 invariant is therefore checked directly instead."
        ),
        "out_of_scope_observed": (
            "`crossdiff-003/004/005` carry the same class of stale *question "
            "string* (`difference in Total between ...` where the canonical "
            "question says `Total liabilities`), but their operand order already "
            "agrees with the contract, so they are not operand-order defects and "
            "are left unchanged to keep this migration attributable to D1."
        ),
    }

    print(f"v8 {v8_digest}")
    print(f"v9 {v9_digest}")
    print(f"rows moved: {moved}")
    for case, payload in record["row_digests"].items():
        print(f"\n{case}")
        print(f"  question : {payload['question']}")
        print(f"          -> {payload['new_question']}")
        print(f"  entities : {payload['old_slot_entities']} -> {payload['new_slot_entities']}")
        for path, delta in sorted(payload["field_diff"].items()):
            print(f"    {path}: {delta['old']!r} -> {delta['new']!r}")

    if not apply_changes:
        print("\nDRY RUN -- nothing written.  Re-run with --apply.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plan-fixtures-v9.jsonl").write_text(v9_text, encoding="utf-8", newline="\n")
    (OUT_DIR / "plan-fixtures-v9.jsonl.sha256").write_text(
        v9_digest + "\n", encoding="utf-8", newline="\n"
    )
    (OUT_DIR / "p1-8-d1-c3-migration.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nwrote {OUT_DIR / 'plan-fixtures-v9.jsonl'}")
    print(f"wrote {OUT_DIR / 'p1-8-d1-c3-migration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
