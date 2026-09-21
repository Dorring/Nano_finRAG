#!/usr/bin/env python3
"""P1.8-C migration: build Benchmark V2 from V1, emitting a full audit record.

Writes NEW files; never mutates the inputs. Every changed case records its
old value/status, new value/status, reason, authority rule and source evidence.

  python build_p1_8c_benchmark_v2.py --base <dir> --out <dir>

STATUS VOCABULARY.  The benchmark recognises exactly two values for
``expected_outcome``: ``ANSWER`` and ``ABSTENTION``.  The scorer tests
``!= "ABSTENTION"``, so an unknown word such as ``ABSTAIN`` silently leaves a
case counted as answerable.  The first draft of this migration made that
mistake; V2 now asserts the vocabulary before writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# ---------------------------------------------------------------- repoints
# case -> (new expected_value, new fact_ids, reason, authority rule, evidence)
REPOINTS = {
    "tv2f01-s2-sum-007": {
        "expected_value": "23754.0000",
        "fact_ids": [
            "v2fact:de7e0d2a617d2527c6518057fad69f44",   # FY2024 9,992
            "v2fact:b42b7ddfc0c8775d0fa70a3febc68ebe",   # FY2025 13,762
        ],
        "operands": {"a": "9,992", "b": "13,762"},
        "reason": (
            "The gold drew from the equity-method investee summary "
            "(13,426 + 12,536 = 25,962). The question names The Coca-Cola "
            "Company's Operating income and no scope, so the company-level "
            "row governs."
        ),
        "authority_rule": (
            "Benchmark Authority Contract s2: a question's coordinate denotes "
            "the company-level fact unless the question text names the scope. "
            "Company-level == XBRL dimension_count == 0. The investee row "
            "carries EquityMethodInvestmentNonconsolidatedInvesteeAxis."
        ),
        "source_evidence": [
            "ko_fy2025 primary.html, 'THE COCA-COLA COMPANY AND SUBSIDIARIES / "
            "CONSOLIDATED STATEMENTS OF INCOME': Operating Income | 13,762 | "
            "9,992 | 11,311",
            "segment table 'Information about our Company's operations by "
            "operating segment and Corporate' reconciles to 13,762 (FY2025) "
            "and 9,992 (FY2024)",
            "the replaced row is preceded by 'A summary of financial "
            "information for our equity method investees in the aggregate'",
        ],
    },
    "tv2f01-s1-tsla-032": {
        "expected_value": "57165",
        "fact_ids": ["v2fact:fa675bff805644c425252216c77c87a2"],
        "reason": (
            "The gold (5,708) is the row's year-over-year CHANGE column, not "
            "its value. The row's FY2025 value is 57,165."
        ),
        "authority_rule": (
            "The row 'Total automotive cost of revenues' states one value per "
            "period; the change columns are comparisons, not values."
        ),
        "source_evidence": [
            "tsla_fy2025 primary.html p.56 'Cost of Revenues and Gross "
            "Margin' (Dollars in millions): Total automotive cost of revenues "
            "| 57,165 | 62,873 | 66,389 | (5,708) | (9) %",
            "the same row in the Consolidated Statements of Operations "
            "(p.49): Total automotive cost of revenues | 57,165 | 62,873 | "
            "66,389",
        ],
    },
    "tv2f01-s1-tsla-033": {
        "expected_value": "94827",
        "fact_ids": ["v2fact:c806e9ac1be6e89eee99529835e19c1c"],
        "reason": (
            "The gold 82,056 is the automotive SEGMENT revenue from the "
            "reportable-segment table. The question names no scope, so the "
            "company-level total revenue governs: 94,827."
        ),
        "authority_rule": (
            "Benchmark Authority Contract s2, as for sum-007. 94,827 appears "
            "at dimension_count == 0 under us-gaap:Revenues and "
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax; "
            "82,056 appears at no company-level concept."
        ),
        "source_evidence": [
            "tsla_fy2025 primary.html p.49 Consolidated Statements of "
            "Operations: Total revenues | 94,827 | 97,690 | 96,773",
            "p.55 Results of Operations: Total revenues | $ 94,827 | ...",
            "p.69 Revenue by source: Total revenues | $ 94,827 | ...",
            "p.59 reportable-segment table: 'Automotive segment | Revenues | "
            "$ 82,056'",
        ],
    },
}

# ------------------------------------------------------------ abstentions
# case -> (reason, authority rule, source evidence)
ABSTENTIONS = {
    "tv2f01-s1-aapl-003": (
        "The question names a row label ('the figure for Services') that four "
        "real quantities in four different tables satisfy, and names no column.",
        "Contract s3 class A: where the question names neither a column nor a "
        "scope and several real quantities satisfy it, no source-level "
        "evidence selects between them.",
        ["aapl_fy2025 p.22 Consolidated Statements of Operations: Services "
         "net sales 109,158",
         "aapl_fy2025 p.23 Gross Margin: Services gross margin 82,314",
         "aapl_fy2025 p.24 Gross margin percentage: Services 75.4% (the old gold)",
         "aapl_fy2025 p.22 cost of sales: Services 26,844"],
    ),
    "tv2f01-s1-aapl-001": (
        "The question asks for 'the reported Deferred' with no qualifier; three "
        "different deferred balances share the label.",
        "Contract s3 class A.",
        ["aapl_fy2025 p.43: Deferred | (1,804) | (3,080) | (3,644)",
         "aapl_fy2025 p.43: Deferred | (139) | (298) | (49)",
         "aapl_fy2025 p.43: Deferred | 604 | 347 | 669"],
    ),
    "tv2f01-s1-aapl-005": (
        "The competing rows' text did not survive extraction; the question "
        "names no column and the source cannot be shown to select one value.",
        "Contract s3 class A.",
        ["store records at (Apple, Hedge accounting fair value adjustments, "
         "FY2025): (294) and (358), both with blank row text"],
    ),
    "tv2f01-s1-jpm-008": (
        "The question names a securities category that the source reports as "
        "four quantities (amortized cost, unrealized gains, unrealized losses, "
        "fair value) and names none of them.",
        "Contract s3 class A.",
        ["jpm_fy2025 p.228: 'U.S. GSEs and government agencies | $ 92,112 | "
         "$ 1,075 | $ 2,215 | $ 90,972' under headers Amortized cost(c)(d) | "
         "Gross unrealized gains | Gross unrealized losses | Fair value",
         "jpm_fy2025 p.261: a second table states 89,073 / 57 / 9,200 / 79,930"],
    ),
    "tv2f01-s1-ko-011": (
        "The question asks for 'the reported Net foreign currency translation "
        "adjustments'; three tables state one, each a different quantity, and "
        "the question names no table, measure or scope.",
        "Contract s3 class A.",
        ["ko_fy2025 p.57 CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME: "
         "Net foreign currency translation adjustments | 2,868 | (2,893) | 736",
         "ko_fy2025 p.102: Net foreign currency translation adjustments | "
         "(32) | 25 | 1 | (2)",
         "ko_fy2025 p.108: Net foreign currency translation adjustments | $ | "
         "(12,673) $ | (15,610)"],
    ),
    "tv2f01-s1-ko-015": (
        "The question asks for 'Intersegment'. All 16 records at this "
        "coordinate share one row whose text is 'Total net operating revenues' "
        "and whose columns are the segment totals; the metric and the row "
        "disagree.",
        "Contract s3 class A, and additionally STORE_SEMANTIC_DEFECT: the row "
        "is mislabelled, so there is no correct row to re-point to.",
        ["ko_fy2025 p.119: all 16 records share row_id row:190f8d640521d1..., "
         "row text 'Total net operating revenues | 11,513 | 6,334 | 19,586 | "
         "5,638 | 5,735 | 48,806 | 144 | (1,009) | 47,941'",
         "the metric stored is 'Intersegment'"],
    ),
    "tv2f01-s1-nvda-022": (
        "The question asks how much NVIDIA 'reported for' a named individual. "
        "The compensation table states salary, stock awards and total "
        "compensation for that person, and the question names no column.",
        "Contract s3 class A.",
        ["nvda_fy2025 p.74: Colette M. Kress row carries 47,890 / 6,830,072 / "
         "95,780 / 13,660,144 across its columns",
         "nvda_fy2025 p.70: the same name heads a summary row including the "
         "year 2025 as a cell"],
    ),
    "tv2f01-s1-nvda-023": (
        "Same shape as nvda-022 for a different named individual.",
        "Contract s3 class A.",
        ["nvda_fy2025 p.74: Ajay K. Puri row carries 46,510 / 6,633,256 / "
         "93,020 / 13,266,512"],
    ),
    "tv2f01-s1-tsla-034": (
        "The question asks for 'the State value'. 'State' heads a "
        "tax-jurisdiction row, not a quantity the question can name.",
        "Contract s3 class A.",
        ["tsla_fy2025 p.117: 'State | 5 | 45 | 57' and "
         "'State | 21 | (49) | (653)'"],
    ),
    "tv2f01-s1-v-036": (
        "The question asks for 'the reported U.S. Treasury securities'; the "
        "source reports amortized cost, unrealized gains and fair value under "
        "that one row and the question names none.",
        "Contract s3 class A.",
        ["v_fy2025 p.78: 'U.S. Treasury securities | 2,101 | 15 | - | 2,116' "
         "under headers Amortized Cost | Gross Unrealized Gains | Losses | "
         "Fair Value"],
    ),
    "tv2f01-s1-v-037": (
        "The question asks for 'Total nominal payments volume(4)'; the row "
        "states several columns and the question names no period or region.",
        "Contract s3 class A.",
        ["v_fy2025 p.59: 'Total nominal payments volume(4) ... $ 6,788 | "
         "$ 6,388 | $ 6,045 | $ 7,106 | $ 6,600 | $ 6,044 | $ 13,894 | "
         "$ 12,988 | $ 12,088'"],
    ),
    "tv2f01-s1-v-039": (
        "The question asks for 'the Income tax effect value'; the coordinate "
        "holds five unlabelled numbered cells.",
        "Contract s3 class A.",
        ["v_fy2025 p.76: five records with values 5 / 4 / 36 / (13) / 101, "
         "none carrying a distinguishing column label"],
    ),
    "tv2f01-s2-diff-002": (
        "The question asks for the change in 'Beginning balance at January 1'; "
        "the rollforward states one beginning balance per portfolio segment and "
        "the question names no segment.",
        "Contract s3 class A.",
        ["jpm_fy2025: seven values at FY2024 (1,856 / 12,450 / 8,114 / 22,420 "
         "/ 75 / 1,899 / 1,974) and seven at FY2025, one column per portfolio "
         "segment of the allowance-for-credit-losses rollforward"],
    ),
    "tv2f01-s2-diff-004": (
        "The question asks for the change in 'Commercial(3)'; the rows carry "
        "no label agreeing with the metric and state several portfolio values.",
        "Contract s3 class A, and STORE_SEMANTIC_DEFECT: the emitted metric "
        "disagrees with the row text ('Consumer debit(2)').",
        ["v_fy2025: records with metric 'Commercial(3)' whose row text reads "
         "'Consumer debit(2) ...', values 1,042 / 613 / 1,655 (FY2024) and "
         "1,084 / 1,739 (FY2025)"],
    ),
    "tv2f01-s2-growth-003": (
        "The question asks for the growth rate in 'Total "
        "noninvestment-grade'; the coordinate holds several portfolio values "
        "per year.",
        "Contract s3 class A.",
        ["jpm_fy2025: 218,726 / 48,152 / 74,646 / 95,928 (FY2024) and "
         "107,461 / 251,003 / 46,670 (FY2025), one column per category"],
    ),
    "tv2f01-s2-growth-008": (
        "The question asks for the change in 'Other letters of credit(d)'; two "
        "values share the coordinate and the question names no column.",
        "Contract s3 class A.",
        ["jpm_fy2025: 37 / 4,354 (FY2024) and 13 / 214 / 4,529 / 6 (FY2025)"],
    ),
    "tv2f01-s2-pctshare-005": (
        "The operands are correct but the question does not say whether the "
        "share carries the sign of the expense: -0.5309 and 0.5309 are both "
        "defensible readings of 'what percentage of Total net sales was Cost "
        "of sales'.",
        "Contract s3 class A: the question lacks the measure that would select "
        "one reading.",
        ["aapl_fy2025 p.28 Consolidated Statements of Operations: Cost of "
         "sales 220,960 (stated as a positive expense); Total net sales "
         "416,161",
         "aapl_fy2025 p.32 segment table states the same cost of sales as "
         "(220,960), parenthesised"],
    ),
    "tv2f01-s2-pctshare-006": (
        "The question asks for 'Total non-current portion of term debt as a "
        "share of iPhone'; the numerator's reading is not determined and the "
        "coordinate also holds $ 85,750.",
        "Contract s3 class A.",
        ["aapl_fy2025: $ 78,328 and $ 85,750 both sit at (Apple, Total "
         "non-current portion of term debt, FY2025)",
         "iPhone net sales 209,586 is unambiguous (p.22 and p.24)"],
    ),
}

#: Cases whose root cause is a Store attribution error rather than the gold
#: pointing at a wrong cell. Recorded, never repaired by a re-point.
STORE_SEMANTIC_DEFECTS = {
    "tv2f01-s1-ko-015": "row mislabelled 'Intersegment'; its text is "
                        "'Total net operating revenues'",
    "tv2f01-s1-msft-020": "a derivative-table cell attributed the metric "
                          "'Other contracts'; the gold value 21 is correct",
    "tv2f01-s2-diff-004": "records whose metric 'Commercial(3)' disagree with "
                          "their own row text 'Consumer debit(2)'",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base_gold = args.base / "gold-evidence-v1.jsonl"
    base_eval = args.base / "canonical-eval-v1.jsonl"
    old = {"gold": sha256(base_gold), "eval": sha256(base_eval)}

    args.out.mkdir(parents=True, exist_ok=True)
    changes: list[dict] = []

    # ---- gold + eval rows, rewritten in place, order preserved ----
    gold_rows = []
    for line in base_gold.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row["id"]
        before = {"expected_outcome": row.get("expected_outcome"),
                  "expected_value": row.get("expected_value"),
                  "fact_ids": row.get("fact_ids"),
                  "operands": row.get("operands")}
        if cid in REPOINTS:
            spec = REPOINTS[cid]
            row["expected_value"] = spec["expected_value"]
            row["fact_ids"] = spec["fact_ids"]
            if "operands" in spec:
                row["operands"] = spec["operands"]
            changes.append({
                "case_id": cid, "change": "REPOINT_GOLD",
                "old_gold": before, "new_gold": {
                    "expected_outcome": row.get("expected_outcome"),
                    "expected_value": row["expected_value"],
                    "fact_ids": row["fact_ids"],
                    "operands": row.get("operands")},
                "reason": spec["reason"],
                "authority_rule": spec["authority_rule"],
                "source_evidence": spec["source_evidence"],
                "store_semantic_defect": cid in STORE_SEMANTIC_DEFECTS,
            })
        elif cid in ABSTENTIONS:
            reason, rule, evidence = ABSTENTIONS[cid]
            # The benchmark's own vocabulary, read from the V1 file: the only
            # two values are ANSWER and ABSTENTION.  A first draft wrote
            # "ABSTAIN" here, which the scorer does not recognise -- it tests
            # `!= "ABSTENTION"`, so those cases still counted as answerable and
            # the run measured the V1 denominator.
            row["expected_outcome"] = "ABSTENTION"
            row["expected_value"] = None
            changes.append({
                "case_id": cid, "change": "STATUS_ABSTENTION",
                "old_gold": before,
                "new_gold": {"expected_outcome": "ABSTENTION",
                             "expected_value": None,
                             "fact_ids": row.get("fact_ids"),
                             "operands": None},
                "reason": reason, "authority_rule": rule,
                "source_evidence": evidence,
                "store_semantic_defect": cid in STORE_SEMANTIC_DEFECTS,
            })
        gold_rows.append(row)

    eval_rows = []
    for line in base_eval.read_text(encoding="utf-8").splitlines():
        if line.strip():
            eval_rows.append(json.loads(line))

    def dump(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False,
                                        sort_keys=True) + "\n")

    gold_v2 = args.out / "gold-evidence-v1.jsonl"
    eval_v2 = args.out / "canonical-eval-v1.jsonl"
    fixtures_src = base_eval.parent / "plan-fixtures-v8.jsonl"
    fixtures_v2 = args.out / "plan-fixtures-v8.jsonl"

    # Fail loudly rather than write a status the scorer does not know.  The
    # scorer's only test is `!= "ABSTENTION"`, so an unrecognised word does not
    # raise -- it silently counts the case as answerable and the run measures
    # the wrong denominator.
    allowed = {"ANSWER", "ABSTENTION"}
    stray = sorted({str(r.get("expected_outcome")) for r in gold_rows} - allowed)
    if stray:
        raise SystemExit(
            "expected_outcome vocabulary violation: %s (allowed: %s)"
            % (stray, sorted(allowed))
        )

    dump(gold_v2, gold_rows)
    dump(eval_v2, eval_rows)
    if fixtures_src.is_file():
        fixtures_v2.write_bytes(fixtures_src.read_bytes())

    new = {"gold": sha256(gold_v2), "eval": sha256(eval_v2)}
    record = {
        "migration_id": "p1.8-c",
        "status": "PROPOSED",
        "benchmark": "tv2-canonical-v1",
        "spec": "docs/evaluation/p1-8-c3-adjudication-closure.md",
        "adjudication": {
            "gold_attribution": "docs/evaluation/p1-8-c1-gold-adjudication.md",
            "column_scope": "docs/evaluation/p1-8-c2-column-adjudication.md",
        },
        "reason": (
            "Cases whose questions do not determine the quantity they ask for "
            "are moved to abstention; cases whose gold denoted the wrong table "
            "or a change column are re-pointed. No case is deleted and no "
            "stratum is changed."
        ),
        "pre": old,
        "post": new,
        "unresolved": 0,
        "counts": {
            "repoint_gold": len(REPOINTS),
            "status_abstention": len(ABSTENTIONS),
            "store_semantic_defect": len(STORE_SEMANTIC_DEFECTS),
            "cases": len(gold_rows),
        },
        "store_semantic_defects": STORE_SEMANTIC_DEFECTS,
        "changes": changes,
    }
    (args.out / "p1-8-c-migration.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("old gold %s" % old["gold"])
    print("old eval %s" % old["eval"])
    print("new gold %s" % new["gold"])
    print("new eval %s" % new["eval"])
    print("changes: %d repoint, %d abstain" % (len(REPOINTS), len(ABSTENTIONS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
