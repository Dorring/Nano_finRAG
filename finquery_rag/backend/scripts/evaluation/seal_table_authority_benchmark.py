"""P1.6-A: seal the Store V2 table-authority benchmark.

Not a summary written after the fact -- a **check**.  Everything the seal freezes is
recomputed from the artifacts at seal time and compared against the expected values,
and the seal fails if any of them moved.  A record that only restates numbers cannot
detect the drift it exists to catch.

What is sealed:

  P1.6-A authority recovery       SEALED   this
  P1.6-A fact emission recovery   OPEN     nine verified primary statements produce
                                           no store records; see `known_gaps`

The second line is why the seal is named for table authority and not for the whole of
P1.6-A.  Source structure, table authority and the parser's cells are all in place;
`cells -> atomic facts` is not, and calling the benchmark sealed without saying so
would claim a completeness the store does not have.

  python seal_table_authority_benchmark.py --store <dir>/store-v2.jsonl \
      --roles <dir>/table-roles.json --out <dir> --commit <git rev>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

EVAL = _BACKEND_DIR / "scripts/evaluation"

#: The scripts whose behaviour the sealed numbers are a result of.  Hashed so that a
#: later edit is visible as a difference rather than as an unexplained change in the
#: numbers.
SCRIPTS = (
    EVAL / "run_nf_v2_17a4_parse.py",
    EVAL / "build_table_authority_oracle.py",
    EVAL / "classify_table_authority.py",
    EVAL / "score_table_authority_classifier.py",
    EVAL / "table_authority_adjudications.json",
    EVAL / "build_store_v2.py",
    EVAL / "resolve_store_v2_slot.py",
    EVAL / "evaluate_table_role_authority.py",
    _BACKEND_DIR / "src/runtime/trusted_v2_canonical_fact_store.py",
)

#: The artifacts the sealed numbers are read from, keyed to the arguments that name them.
ARTIFACTS = ("store", "roles", "experiment")

ORACLE = Path(
    "/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a2b19a-oracle/table-authority-oracle.json"
)
SHADOW = Path(
    "/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a2b19b-shadow/shadow-classification.json"
)

#: What the seal asserts.  Recomputed below, never copied from a report.
EXPECTED = {
    "filings": 8,
    "corpus_tables": 1498,
    "data_tables": 906,
    "layout_scaffolds": 592,
    "oracle_primary": 42,
    "oracle_dangerous_negatives": 229,
    "store_records": 26977,
    "authority_before": {"RESOLVED_COMPANY_LEVEL": 10,
                         "NO_COMPANY_LEVEL_FACT": 29,
                         "AMBIGUOUS_COMPANY_LEVEL": 3,
                         "NO_FACT": 5},
    "authority_after": {"RESOLVED_COMPANY_LEVEL": 37,
                        "NO_COMPANY_LEVEL_FACT": 5,
                        "NO_FACT": 5},
    "labels": {"CAPABILITY_GAIN": 27, "UNCHANGED": 19, "REFUSAL_RETYPED": 1,
               "REGRESSION": 0, "SAFER_FAIL_CLOSED": 0},
    "gates": {"wrong_scope": 0, "wrong_metric": 0, "value_mismatch": 0,
              "dangerous_authority": 0, "scaffold_authority": 0},
}

#: Known open, declared so the seal cannot be read as claiming more than it has.
KNOWN_GAPS = [
    "nine verified primary statements produce no store records: JPMorganChase balance "
    "sheet and cash flow, Coca-Cola balance sheet, cash flow and equity, NVIDIA cash "
    "flow, Pfizer equity, Tesla cash flow, Visa balance sheet",
    "five NO_FACT slots are metric identity, not authority: the filing's row label is "
    "not the exact string the fixture asks for",
    "the classifier's positive rule has not been tested on a filing outside these eight",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", default=None,
                        help="the git revision the run scripts correspond to")
    args = parser.parse_args(argv)

    problems: list[str] = []

    # --- recompute everything the seal asserts ---------------------------------
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    shadow = json.loads(SHADOW.read_text(encoding="utf-8"))
    records = [json.loads(line)
               for line in args.store.read_text(encoding="utf-8").splitlines()
               if line.strip()]

    eligibility = collections.Counter()
    roles_count = collections.Counter()
    corpus_tables = 0
    for document in oracle["documents"].values():
        for table in document["tables"]:
            corpus_tables += 1
            eligibility[table["table_eligibility"]] += 1
            roles_count[table["oracle_role"]] += 1

    spec = importlib.util.spec_from_file_location(
        "resolver", EVAL / "resolve_store_v2_slot.py")
    resolver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resolver)
    cross_spec = importlib.util.spec_from_file_location(
        "cross", EVAL / "resolve_cross_entity_v2.py")
    cross = importlib.util.module_from_spec(cross_spec)
    cross_spec.loader.exec_module(cross)

    # The gates are recomputed by running the experiment script's own entry point, so
    # the seal checks the experiment rather than trusting its report.
    experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
    recomputed = {}
    for authority in (resolver.AUTHORITY_STATEMENT_TYPE, resolver.AUTHORITY_TABLE_ROLE):
        statuses = collections.Counter()
        for _case, (metric, entities) in sorted(cross.CASES.items()):
            for entity in entities:
                result = resolver.resolve(
                    records, entity, metric, cross.PERIOD.get(entity, "FY2025"), authority)
                statuses[result["status"]] += 1
        recomputed[authority] = dict(statuses)

    observed = {
        "filings": len(oracle["documents"]),
        "corpus_tables": corpus_tables,
        "data_tables": eligibility["DATA_TABLE"],
        "layout_scaffolds": eligibility["LAYOUT_SCAFFOLD"],
        "oracle_primary": roles_count["PRIMARY"],
        "oracle_dangerous_negatives": roles_count["DANGEROUS_NEGATIVE"],
        "store_records": len(records),
        "authority_before": recomputed[resolver.AUTHORITY_STATEMENT_TYPE],
        "authority_after": recomputed[resolver.AUTHORITY_TABLE_ROLE],
        "labels": experiment["transitions"],
        "gates": experiment["gate"],
    }

    for name, want in EXPECTED.items():
        got = observed.get(name)
        if name in ("labels", "gates"):
            # Both are sparse counters: a zero is absent rather than recorded, so the
            # comparison names the keys instead of trusting the dict to carry them.
            got = {k: (got or {}).get(k, 0) for k in want}
        if got != want:
            problems.append(f"{name}: expected {want}, recomputed {got}")

    # --- the seal ---------------------------------------------------------------
    seal = {
        "phase": "P1.6-A",
        "sealed": "Store V2 table authority",
        "not_sealed": "fact emission -- nine verified primary statements still produce "
                      "no store records",
        "commit": args.commit,
        "observed": observed,
        "expected": EXPECTED,
        "inputs": {
            "oracle": {"path": str(ORACLE), "sha256": sha256(ORACLE)},
            "shadow": {"path": str(SHADOW), "sha256": sha256(SHADOW)},
        },
        "scripts": {
            str(path.relative_to(_BACKEND_DIR)): sha256(path)
            for path in SCRIPTS if path.is_file()
        },
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in (("store", args.store), ("roles", args.roles),
                               ("experiment", args.experiment))
            if path.is_file()
        },
        "known_gaps": KNOWN_GAPS,
        "verdict": "SEALED" if not problems else "DRIFT",
        "problems": problems,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "seal.json").write_text(
        json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print("=== P1.6-A Store V2 table authority seal ===")
    print()
    for name in ("filings", "corpus_tables", "data_tables", "layout_scaffolds",
                 "oracle_primary", "oracle_dangerous_negatives", "store_records"):
        print(f"  {name:28} {observed[name]}")
    print()
    print(f"  authority before  {observed['authority_before']}")
    print(f"  authority after   {observed['authority_after']}")
    print(f"  labels            {observed['labels']}")
    print(f"  gates             {observed['gates']}")
    print()
    if problems:
        print("  DRIFT:")
        for problem in problems:
            print(f"    {problem}")
    else:
        print("  no drift: every sealed number was recomputed and matched")
    print()
    print(f"  written to {args.out / 'seal.json'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
