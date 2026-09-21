"""P1.6-A3-W4-A4: do the 474 wrong-period primary facts reach a query?

Diagnosis only.  W4-A3 established that 474 oracle-PRIMARY cells are stored under a period
that is not theirs -- 281 under the opening balance of their own section, 191 under the
table's earliest date.  The open question was whether that is a latent defect in the store
or an active one at the resolver, and this settles it by asking the resolver rather than by
reasoning about it.

It matters more than it looks, because of one line in `resolve_store_v2_slot`:

    str(r.get("period_end") or "")[:4] == want_year

Period matching is **by year**.  So a fact carrying the opening instant of its section --
the year-ago date -- is filed under the *previous* year, which is a slot some other table
answers correctly.

The hypothesis going in was that this would produce false `AMBIGUOUS_COMPANY_LEVEL`
readings: two primary facts in one pool, the resolver refusing to choose, a refusal where
the store holds a correct answer.  **The measurement says otherwise, and the hypothesis was
wrong.**  Nothing landed in AMBIGUOUS.  What happens instead is that most of the wrong-
period facts are simply unreachable at the year they claim, and where something else does
answer at that year the two agree on a value -- so the wrong fact is inert there.

Three questions per fact, asked in order:

    1. is it in Store V2 at all?
    2. queried under the period it claims, does it win, collide, or agree?
    3. queried under the period it *should* carry, does it appear?  (it should not)

  python diagnose_wrong_period_reachability.py --store <store-v2.jsonl> \
      --equity <equity-geometry.json> --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

RESOLVER = _BACKEND_DIR / "scripts/evaluation/resolve_store_v2_slot.py"
CROSS = _BACKEND_DIR / "scripts/evaluation/resolve_cross_entity_v2.py"

RESOLVED = "RESOLVED_COMPANY_LEVEL"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def year_of(text) -> str | None:
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(text or ""))
    return match.group(1) if match else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--equity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args(argv)

    resolver = load("resolver", RESOLVER)
    cross = load("cross", CROSS)

    records = [json.loads(line) for line in args.store.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    by_cell = {str(r.get("cell_id")): r for r in records if r.get("cell_id")}
    equity = json.loads(args.equity.read_text(encoding="utf-8"))

    # The cells W4-A3 diagnosed, with the year their section actually closes on.
    wrong: list[dict] = []
    for key, table in equity["tables"].items():
        for cell in table["cells"]:
            wrong.append({
                "key": key,
                "cell_id": cell["cell_id"],
                "row_label": cell["row_label"],
                "legacy_period": cell["legacy_normalized_period"],
                "anchor_above": cell["anchor_above"],
                "anchor_below": cell["anchor_below"],
                "verdict": cell["verdict"],
                "claimed_year": year_of(cell["legacy_normalized_period"]),
                "real_year": year_of(cell["anchor_below"]),
            })

    report = {"phase": "P1.6-A3-W4-A4", "mutation": "none", "found_in_store": 0,
              "total": len(wrong), "verdicts": {}, "collisions": [], "reachable": [],
              "slots": {}}
    tally = collections.Counter()

    print(f"=== {len(wrong)} cells diagnosed in W4-A3 ===")
    for entry in wrong:
        record = by_cell.get(str(entry["cell_id"]))
        entry["in_store"] = record is not None
        if record is None:
            tally["not in Store V2"] += 1
            continue
        tally["in Store V2"] += 1
        entry["entity"] = record.get("entity")
        entry["stored_period_end"] = record.get("period_end")
        entry["stored_year"] = str(record.get("period_end") or "")[:4]
        entry["table_fragment_id"] = record.get("table_fragment_id")
        entry["table_role"] = record.get("table_role")
        entry["value_raw"] = record.get("value_raw")
        entry["value"] = record.get("value")
        entry["row_label"] = record.get("row_label")
        entry["column_header"] = record.get("column_header")

    report["found_in_store"] = tally["in Store V2"]
    print(f"    in Store V2: {tally['in Store V2']}   not found: {tally['not in Store V2']}")
    print()

    # The resolver matches periods with `str(r.get("period_end") or "")[:4]`, and Store V2
    # carries no `normalized_period` field at all -- so a fact whose legacy axis set the
    # normalised string while leaving `period_end` unset is stored with a period the
    # resolver cannot read.  That is the difference between "the store holds a wrong-period
    # fact" and "a query can reach one", and it is the first thing to count.
    invisible = sum(1 for e in wrong if e.get("in_store")
                    and e.get("stored_period_end") in (None, ""))
    visible = sum(1 for e in wrong if e.get("in_store")
                  and e.get("stored_period_end") not in (None, ""))
    print("=== can the resolver see the period at all? ===")
    print(f"    {invisible:>5}  stored with period_end NULL -- unreachable by construction")
    print(f"    {visible:>5}  stored with a period_end -- a query can reach these")
    report["period_end_null"] = invisible
    report["period_end_present"] = visible
    print()

    # --- 2/3. ask the resolver, under the period it claims and under the one it should
    for entry in wrong:
        if not entry.get("in_store"):
            continue
        # The metric a user would actually name.  `_matches_metric` compares the *record's*
        # label after stripping a trailing unit caption (`Net income (in millions)`) against
        # the query metric, so passing the raw label asks for something the resolver can
        # never match: the first version of this did exactly that and reported 331 of the
        # 474 as unreachable when they were reachable.  Building the query with the
        # resolver's own normaliser is the only way the count means anything.
        metric = resolver._ROW_CAPTION.sub("", resolver._norm(entry["row_label"])).strip()
        entity = str(entry["entity"] or "")
        if not metric or not entity:
            tally["no entity or metric to query with"] += 1
            continue

        claimed = resolver.resolve(records, entity, metric, entry["claimed_year"])
        entry["claimed_query"] = claimed["status"]
        if claimed["status"] == RESOLVED and claimed.get("cell_id") == entry["cell_id"]:
            # The outcome is set on `entry` *before* it is copied into the report, so the
            # recorded fact carries its own verdict rather than needing the reader to
            # reconstruct it from which list it landed in.
            entry["claimed_outcome"] = (
                "WINS the query for the year it claims -- an outright wrong answer")
            report["reachable"].append({**entry, "how": "wins"})
        elif claimed["status"] == RESOLVED:
            # Not "outvoted": this branch means every primary record in the pool agreed on
            # a value, so the wrong-period fact is inert *here* only because something at
            # that period has the same number.  Recorded separately for that reason.
            entry["claimed_outcome"] = (
                f"agrees with {claimed.get('table_fragment_id')} at the same period")
            report.setdefault("agreed", []).append(
                {**entry, "how": "agreed", "winner_value": claimed.get("value_raw")})
        elif claimed["status"].startswith("AMBIGUOUS"):
            entry["claimed_outcome"] = (
                "collides -- AMBIGUOUS where a primary fact exists")
            report["collisions"].append({**entry, "how": "ambiguous",
                                         "values": claimed.get("values")})
        else:
            entry["claimed_outcome"] = f"unreachable ({claimed['status']})"
        tally[entry["claimed_outcome"]] += 1

        if entry.get("real_year") and entry["real_year"] != entry["claimed_year"]:
            real = resolver.resolve(records, entity, metric, entry["real_year"])
            entry["real_query"] = real["status"]
            if real["status"] == RESOLVED and real.get("cell_id") == entry["cell_id"]:
                tally["... and it also answers the year it should not"] += 1

    print("=== what the resolver does with each, queried under the year it claims ===")
    for name, count in tally.most_common():
        print(f"    {count:>5}  {name}")
    print()

    # --- the 47 slots, in case one of them is the one that lands on these
    print("=== the 47 slots under table_role authority ===")
    slot_hits = 0
    wrong_cells = {str(e["cell_id"]) for e in wrong}
    for case_id, (metric, entities) in sorted(cross.CASES.items()):
        for entity in entities:
            period = cross.PERIOD.get(entity, "FY2025")
            out = resolver.resolve(records, entity, metric, period,
                                   resolver.AUTHORITY_TABLE_ROLE)
            hit = str(out.get("cell_id") or "") in wrong_cells
            if hit:
                slot_hits += 1
                report["slots"][f"{case_id}|{entity}"] = {
                    "metric": metric, "period": period, "cell_id": out.get("cell_id"),
                    "value_raw": out.get("value_raw")}
            print(f"    {case_id:<14} {entity:<12} {metric:<28} {period:<8} "
                  f"{out['status']:<26} {'<-- WRONG-PERIOD FACT' if hit else ''}")
    print()
    print(f"    slots whose answer is one of the {len(wrong)} wrong-period facts: {slot_hits}")

    report["verdicts"] = {k: v for k, v in tally.items()}
    report["slot_hits"] = slot_hits

    print()
    print("=== examples of the ones that win or collide ===")
    for entry in (report["reachable"] + report["collisions"])[:args.samples]:
        print(f"    {entry['entity']} {str(entry['row_label'])[:26]!r} "
              f"stored {entry['stored_period_end']} (claims {entry['claimed_year']}) "
              f"real section closes {entry['real_year']} -> {entry['claimed_outcome']}")

    print()
    print("=== every fact that WINS a query for a year it does not belong to ===")
    for entry in report["reachable"]:
        print(f"    {entry['entity']:<12} {str(entry['row_label'])[:34]!r:<36} "
              f"stored {entry['stored_period_end']}  section closes {entry['real_year']}"
              f"  value={entry['value']!r}")
        print(f"        raw: {str(entry['value_raw'])[:96]!r}   [{entry['key']}]")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "wrong-period-reachability.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"\n  written to {args.out / 'wrong-period-reachability.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
