"""P1.6-A3-W6-C: seal the fact-emission benchmark.

A seal that only restates numbers cannot detect the drift it exists to catch, so this
recomputes.  Every value below is derived at seal time:

  * the scalars            recomputed here from the primary artifacts -- the two stores,
                           the audit, the funnel -- and **not** from
                           `store-migration-accounting.json`, which is itself a summary
                           and would make the seal a second copy of one derivation
  * the nine-table funnel  recomputed by running `verify_final_state.py` in a **fresh
                           process** into a temporary directory, so the seal checks the
                           verification rather than a stale report of it
  * the reachability       likewise, by re-running `verify_authority_reachability.py`

and every one is compared against a frozen constant.  A constant that moves is a
regression, not a new baseline.

What is frozen:

  implementation commit, the hashes of the modules and tests the numbers are a result of,
  the Store V2 hash and record count, the addition and removal classes, the 47-slot
  distribution and all five safety gates, the provenance counts, the nine-table outcomes,
  and the authority-reachability result.

  python seal_fact_emission_benchmark.py --store <store-v2.jsonl> \
      --previous <store-v2.jsonl> --audit <json> --accounting <json> --funnel <json> \
      --baseline-slots <json> --out <dir> --commit <git rev>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

EVAL = _BACKEND_DIR / "scripts/evaluation"

#: Frozen.  A seal exists to fail when one of these moves.
EXPECTED = {
    "records": 26311,
    "added_by_identity": {"PARTIAL / YEAR": 3368, "RESOLVED / DAY": 3018},
    "removed_by_class": {"LEGACY_FALSE_POSITIVE": 691,
                         "SOURCE_AMBIGUOUS": 4188,
                         "VALID_BUT_OUT_OF_SCOPE_GEOMETRY": 2173},
    "unclassified_added": 0,
    "unclassified_removed": 0,
    "slots": {"RESOLVED_COMPANY_LEVEL": 37, "NO_COMPANY_LEVEL_FACT": 5, "NO_FACT": 5},
    "gates": {"wrong_scope": 0, "wrong_metric": 0, "value_mismatch": 0,
              "dangerous_authority": 0, "scaffold_authority": 0},
    "provenance_complete": 26311,
    "temporal_provenance": 21303,
    "source_cells": 77883,
    "unresolvable_source_cells": 0,
    "reachability": {"legacy_readers": 0, "landmine_tripped": False,
                     "a3_decision_load_bearing": True, "rule_moved": ["unknown"]},
    # The nine zero-record primary statements.  Recovery counts are frozen; a table that
    # is still empty must carry the outcome that explains it.  `WITHHELD_ON_LEGACY_KIND`
    # is not a scope decision: four cash flow statements are withheld because the legacy
    # classifier reads `Operating activities` as a maturity bucket (its `rating`
    # alternative has no word boundary and matches `ope·rating·`).  It is frozen as the
    # outcome because that is what the system does, and declared as a defect below.
    "nine_tables": {
        "jpm_fy2025#63193": "RECOVERED",
        "jpm_fy2025#64546": "WITHHELD_ON_LEGACY_KIND",
        "ko_fy2025#11388": "RECOVERED",
        "ko_fy2025#11899": "WITHHELD_ON_LEGACY_KIND",
        "ko_fy2025#12533": "RECOVERED",
        "nvda_fy2025#8766": "WITHHELD_ON_LEGACY_KIND",
        "pfe_fy2024#24395": "RECOVERED",
        "tsla_fy2025#10407": "WITHHELD_ON_LEGACY_KIND",
        "v_fy2025#9951": "RECOVERED",
    },
}

#: The modules and tests the sealed numbers are a result of.  Hashed so that an edit which
#: is not reflected in the numbers is still visible as a difference.
HASHED = {
    "period_binding": "src/pdf_retrieval_v4/period_binding.py",
    "typed_evidence_emitters": "src/pdf_retrieval_v4/typed_evidence_emitters.py",
    "semantic_graph_models": "src/pdf_retrieval_v4/semantic_graph_models.py",
    "html_semantic_adapter": "src/pdf_retrieval_v4/html_semantic_adapter.py",
    "canonical_fact_store": "src/runtime/trusted_v2_canonical_fact_store.py",
    "parse": "scripts/evaluation/run_nf_v2_17a4_parse.py",
    "period_producer": "scripts/evaluation/period_binding_shadow.py",
    "build_store": "scripts/evaluation/build_store_v2.py",
    "resolver": "scripts/evaluation/resolve_store_v2_slot.py",
    "benchmark": "scripts/evaluation/resolve_cross_entity_v2.py",
    "verify_final_state": "scripts/evaluation/verify_final_state.py",
    "verify_reachability": "scripts/evaluation/verify_authority_reachability.py",
    "test_period_binding": "tests/test_period_binding_v2.py",
    "test_emission_admission": "tests/test_emission_admission.py",
}

VERDICT_NAMES = {"B": "LEGACY_FALSE_POSITIVE", "C": "VALID_BUT_OUT_OF_SCOPE_GEOMETRY",
                 "D": "SOURCE_AMBIGUOUS", "A": "V2_PRODUCER_GAP"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_store(path: Path) -> list[dict]:
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    # Forwarded to `verify_final_state.py` rather than read here: the seal recomputes its
    # own scalars from the stores, and this is only an input to the verifier it re-runs.
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--funnel", type=Path, required=True)
    parser.add_argument("--baseline-slots", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", default=None)
    args = parser.parse_args(argv)

    problems: list[str] = []
    observed: dict = {}

    # --- recomputed here, from the primary artifacts -------------------------------------
    records = read_store(args.store)
    previous = {str(r["cell_id"]) for r in read_store(args.previous)}
    by_cell = {str(r["cell_id"]): r for r in records}
    added = set(by_cell) - previous
    removed = previous - set(by_cell)

    observed["records"] = len(records)

    # Classified from the store's own fields rather than from the accounting summary: the
    # identities are what was persisted, so they are readable from the artifact itself.
    identity = collections.Counter(
        f"{by_cell[c].get('period_binding_status')} / {by_cell[c].get('period_granularity')}"
        for c in added)
    observed["added_by_identity"] = dict(identity)

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    cell_class: dict[str, str] = {}
    for entries in audit["columns_detail"].values():
        for entry in entries:
            for cell_id in entry.get("cell_ids") or ():
                cell_class[str(cell_id)] = str(entry["verdict"])
    by_class = collections.Counter()
    unclassified_removed = 0
    for cell_id in removed:
        verdict = cell_class.get(cell_id)
        if verdict is None:
            unclassified_removed += 1
            continue
        by_class[VERDICT_NAMES.get(verdict, verdict)] += 1
    observed["removed_by_class"] = dict(by_class)
    observed["unclassified_removed"] = unclassified_removed
    # An addition is unclassified when it cannot be attributed to the shadow decision that
    # admitted it; the store carries no such link, so this is the count of additions whose
    # identity is not one the migration ever produced.
    unclassified_added = sum(
        v for k, v in identity.items() if k not in EXPECTED["added_by_identity"])
    observed["unclassified_added"] = unclassified_added

    complete = sum(1 for r in records
                   if str(r.get("period_binding_status")) in ("RESOLVED", "PARTIAL")
                   and r.get("period_binding_method") and r.get("period_target_scope")
                   and r.get("period_granularity") not in (None, "UNKNOWN")
                   and r.get("normalized_period") and r.get("period_source_cells"))
    observed["provenance_complete"] = complete
    observed["temporal_provenance"] = sum(
        1 for r in records if str(r.get("temporal_kind") or "UNKNOWN") != "UNKNOWN"
        and r.get("temporal_kind_method") and r.get("temporal_kind_source_cells"))

    # --- the 47 slots, recomputed ---------------------------------------------------------
    def load(path: Path, name: str):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    resolver = load(EVAL / "resolve_store_v2_slot.py", "resolver")
    cross = load(EVAL / "resolve_cross_entity_v2.py", "cross")
    roles = json.loads((args.store.parent / "table-roles.json").read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_slots.read_text(encoding="utf-8"))


    caption = re.compile(r"\s*\((?:in|except|dollars in|amounts in)[^)]*\)\s*$", re.I)

    statuses = collections.Counter()
    gate = collections.Counter()
    for row in baseline["slots"]:
        entity, metric = row["entity"], row["metric"]
        period = cross.PERIOD.get(entity, "FY2025")
        out = resolver.resolve(records, entity, metric, period,
                               resolver.AUTHORITY_TABLE_ROLE)
        statuses[out["status"]] += 1
        if out["status"] != "RESOLVED_COMPANY_LEVEL":
            continue
        record = by_cell.get(str(out.get("cell_id")), {})
        role = roles.get(str(out.get("table_fragment_id")), {})
        if str(out.get("table_role")) != "PRIMARY_FINANCIAL_STATEMENT":
            gate["wrong_scope"] += 1
        folded = caption.sub("", " ".join(str(record.get("row_label") or "").split()).casefold())
        if folded.strip() != " ".join(metric.split()).casefold():
            gate["wrong_metric"] += 1
        expected_value = cross.EXPECTED.get((row["case_id"], entity))
        if expected_value is not None and str(out.get("value_raw")) != expected_value:
            gate["value_mismatch"] += 1
        if role.get("oracle_role") == "DANGEROUS_NEGATIVE":
            gate["dangerous_authority"] += 1
        if role.get("table_eligibility") == "LAYOUT_SCAFFOLD":
            gate["scaffold_authority"] += 1
    observed["slots"] = dict(statuses)
    observed["gates"] = {name: gate.get(name, 0) for name in EXPECTED["gates"]}

    # --- the verifiers, re-run in fresh processes -----------------------------------------
    print("=== re-running the verifiers in fresh processes ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        final_dir, reach_dir = tmpdir / "final", tmpdir / "reach"
        runs = {
            "final": [sys.executable, str(EVAL / "verify_final_state.py"),
                      "--store", str(args.store),
                      "--accounting", str(args.accounting),
                      "--funnel", str(args.funnel),
                      "--baseline-slots", str(args.baseline_slots),
                      "--out", str(final_dir)],
            "reachability": [sys.executable, str(EVAL / "verify_authority_reachability.py"),
                             "--document", "ko_fy2025", "--out", str(reach_dir)],
        }
        print("    (a full re-parse of the corpus; this takes a few minutes)")
        for name, command in runs.items():
            proc = subprocess.run(command, capture_output=True, text=True, cwd=str(_BACKEND_DIR))
            print(f"    {name}: exit {proc.returncode}")
            if proc.returncode != 0:
                problems.append(f"{name} FAILED: {(proc.stderr or proc.stdout)[-400:]}")

        final = json.loads((final_dir / "final-state-verification.json").read_text("utf-8"))
        reach = json.loads((reach_dir / "authority-reachability.json").read_text("utf-8"))

    observed["nine_tables"] = {k: v["outcome"] for k, v in final["nine_tables"].items()}
    observed["unresolvable_source_cells"] = sum(
        final["provenance"]["unresolvable"].values())
    observed["source_cells"] = final["provenance"]["source_cells"]
    observed["reachability"] = {
        "legacy_readers": len(reach["legacy_readers"]),
        "landmine_tripped": reach["legacy_landmine_tripped"],
        "a3_decision_load_bearing": reach["a3_decision_load_bearing"],
        "rule_moved": reach["rule_delta"]["moved_over_legacy_domain"],
        "path_modules": len(reach["path_modules"]),
    }

    # --- compare ---------------------------------------------------------------------------
    print()
    print("=== against the frozen constants ===")
    for name, want in EXPECTED.items():
        got = observed.get(name)
        mark = "ok " if got == want else "FAIL"
        print(f"    {mark} {name}")
        if got != want:
            problems.append(f"{name}: expected {want}, recomputed {got}")

    # --- the seal ---------------------------------------------------------------------------
    seal = {
        "phase": "P1.6-A3-W6-C",
        "sealed": "A3 fact emission -- admission authority, period identity, provenance",
        "commit": args.commit,
        "hashes": {name: sha256(_BACKEND_DIR / rel) for name, rel in sorted(HASHED.items())},
        "store_sha256": sha256(args.store),
        "table_roles_sha256": sha256(args.store.parent / "table-roles.json"),
        "benchmark_sha256": sha256(EVAL / "resolve_cross_entity_v2.py"),
        "observed": observed,
        "expected": EXPECTED,
        "problems": problems,
        "verdict": "SEALED" if not problems else "NOT SEALED",
    }
    # The legacy table-authority seal predates this one and is not superseded by it: it
    # seals which tables may speak for the company, this seals whether a fact exists.
    seal["relationship_to_previous_seal"] = (
        "p1-6-a2b20-store-v2-role/seal.json seals table authority (which table may speak "
        "for the company); this seals fact emission (whether a valued cell becomes a "
        "fact). They are independent and neither supersedes the other."
    )
    # Still open, declared so the seal cannot be read as claiming more than it has.
    seal["known_gaps"] = [
        "** a legacy classifier defect, not a scope decision: `_BUCKET_RE` in "
        "temporal_axis_graph.py lists `rating` without word boundaries, so it matches "
        "inside `operating` (`ope·rating· activities`). Every cash flow statement section "
        "labelled `Operating activities` is classified `bucket`, which routes to "
        "DISAGGREGATION_AXIS and withholds the whole table. Four of the nine verified "
        "primary statements -- JPMorganChase, Coca-Cola, NVIDIA and Tesla cash flow -- "
        "still produce no records for this reason alone. The A3 routing rule is still a "
        "kind question by design, so a mislabelled kind still decides. Fixing it is a "
        "behaviour change and was deliberately kept out of W6",
        "SECTION scope / SECTION_BRACKET for row-major equity statements is unimplemented; "
        "2173 cells are withheld as VALID_BUT_OUT_OF_SCOPE_GEOMETRY",
        "the resolver does not consume YEAR(2025) or granularity; 3368 year-only facts "
        "have no `period_end` for it to match on",
        "row-label numeric extraction (`Balance at January 1` -> 1) is unfixed",
        "trusted_v2_canonical_fact_store.py has two `normalized_period` keys in one dict "
        "literal; the first is dead code, and removing either is a behaviour change",
        "SOURCE_AMBIGUOUS holds 4188 facts withheld because the source does not settle "
        "them; recovering any is a later phase",
        "REAL_BUCKET_POSITIVE_UNOBSERVED: no corpus case exercises a real bucket-positive",
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "seal.json").write_text(
        json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print()
    print(f"  verdict: {seal['verdict']}")
    for problem in problems:
        print(f"      {problem}")
    print(f"  written to {args.out / 'seal.json'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
