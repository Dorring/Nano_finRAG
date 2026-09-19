# VERIFY-02 — Adjudicate 11 contested cross-entity gold operands

**Audience:** an independent agent/engineer, with access to the 4090 benchmark host.
**Deliverable:** a filled `verdicts.json` recording, per slot, whether the fixture's
gold operand is the company-level figure the question asks for.
**Estimated effort:** 11 slots. Each is one page of one PDF.

---

## 1. Background — what is being checked and why

The system under test answers questions about SEC filings by retrieving facts,
binding them to a plan, and executing a deterministic operation (comparison,
ranking, difference). A fact store keyed by `(entity, metric, period)` supplies the
numeric operands.

**The core problem.** That coordinate has no dimension for *where inside the
statement the number came from*. Financial filings break a metric down along many
axes — business segment, jurisdiction, revenue line, derivative type, instrument —
and each breakdown cell becomes its own stored fact with the **same**
`(entity, metric, period)`. So one coordinate can hold many different, all-legitimate
values.

Example (already settled, quoted so the failure mode is concrete — `compare-009`):

```
JPMorganChase / Net income / FY2025, from one row of one table
  x=188   4,520   -> column 165-312   headed "Corporate"
  x=444  57,048   -> column 430-562   headed "Total"
```

`$4,520` is the *Corporate segment*; `$57,048` is the *Firm*. The fixture's gold for
that slot is `$4,520`, i.e. the segment. The question asks about the company.

**Your task is to determine, per slot, which of these three holds:**

| Verdict | Meaning |
|---|---|
| `VALID_GOLD` | The gold value **is** the company-level figure the question asks for. |
| `WRONG_SCOPE_GOLD` | The gold value is a breakdown component (segment / jurisdiction / line item) while the company-level figure is a different number. |
| `SOURCE_STILL_AMBIGUOUS` | The filing genuinely does not let a reader decide which value is company-level. |

**A wrong `VALID_GOLD` is the worst possible output.** Certifying a segment sub-total
as the company figure silently marks a defective fixture as sound, and everything
downstream inherits it. If you cannot establish the company figure from the filing,
say `SOURCE_STILL_AMBIGUOUS` and give the reason. Do not guess, and do not default
to `VALID_GOLD`.

---

## 2. Inputs

The audit artifact:

```
/disk/qh/nano-finrag/artifacts/evaluation/p1-6-0d-source-truth/cross-entity-source-truth.json
```

The filings (read-only — **see §5**):

```
/disk/qh/nano-finrag/data/raw_pdfs/
/disk/qh/nano-finrag/data/raw_pdfs/corpus-manifest.json
```

Python with PyMuPDF is available:

```bash
cd /disk/qh/nano-finrag/finquery_rag/backend && .venv/bin/python ...
```

---

## 3. Scope — the 11 slots

Select every slot in the artifact with `decision == "SOURCE_STILL_AMBIGUOUS"`.
There are 11, across 9 cases. They are, for orientation only — **derive your list
from the artifact, not from this table:**

| case | entity | metric (as stored) | gold | file | page |
|---|---|---|---|---|---|
| `tv2f01-s3-compare-001` | The Coca-Cola Company | `United States` | `$ 5,678` | ko_fy2025_10k.pdf | 103 |
| `tv2f01-s3-compare-003` | Tesla | `Gross profit` | `$ 13,292` | tsla_fy2025_10k.pdf | 129 |
| `tv2f01-s3-compare-004` | Apple | `Foreign exchange contracts` | `$ 62,647` | aapl_fy2025_10k.pdf | 42 |
| `tv2f01-s3-compare-004` | Microsoft | `Foreign exchange contracts` | `(809)` | msft_fy2025_annual_report.pdf | 58 |
| `tv2f01-s3-compare-005` | Apple | `Current` | `$ 11,487` | aapl_fy2025_10k.pdf | 43 |
| `tv2f01-s3-compare-010` | Microsoft | `Other current liabilities` | `$ 5,424` | msft_fy2025_annual_report.pdf | 71 |
| `tv2f01-s3-crossdiff-003` | Apple | `Total` | `9,683` | aapl_fy2025_10k.pdf | 43 |
| `tv2f01-s3-crossdiff-003` | Tesla | `Total` | `$ 13,279` | tsla_fy2025_10k.pdf | 100 |
| `tv2f01-s3-crossdiff-004` | The Coca-Cola Company | `Long-term debt` | `850` | ko_fy2025_10k.pdf | 76 |
| `tv2f01-s3-rank-002` | Tesla | `Research and development` | `$ 6,411` | tsla_fy2025_10k.pdf | 58 |
| `tv2f01-s3-rank-005` | The Coca-Cola Company | `Interest rate contracts` | `16` | ko_fy2025_10k.pdf | 84 |

Note the stored `metric` is sometimes itself a breakdown label (`United States`,
`Current`, `Total`) — the extractor filed a *cell caption* as the metric. Treat the
stored metric as a hint about which table the fact came from, **not** as a reliable
description of the quantity.

Each slot also carries `comparable_distinct_values` — the other values at the same
coordinate. Those are the competing candidates. Read them; the company figure is
often among them.

---

## 4. Method

For each slot:

1. Open the PDF at the `page` recorded in the slot.
2. Locate the value. The row it sits in, and the table that row belongs to, are what
   you are reading — not the value alone.
3. Establish the **scope** of that value. Two shapes occur in this corpus:
   - **Column-scoped.** The row's header names the columns
     (`Corporate … Total`). A value under a column headed `Total`/`Consolidated`/`Firm`
     is company-level; one under a segment name is not.
   - **Row-scoped.** The segments are rows, and the company figure is a separate row
     labelled `Total <metric>`. Beware: `Gross profit total automotive & services and
     other segment` *contains* the word "total" and is still a **segment** row — the
     word modifies the grouping. Match position, not the word.
4. Decide the verdict from what the filing says.

**Recurrence is a useful but insufficient signal.** A consolidated figure is normally
restated across the filing (statement, highlights, MD&A, notes) while a breakdown cell
appears once. This holds often but not always, and it is a heuristic — use it to
corroborate a reading, never as the sole basis for `VALID_GOLD`. (An earlier
rule-based pass in this project over-trusted it and produced a false `VALID_GOLD`.)

**State your evidence as page + the exact row/column/block text.** "It looked like the
total" is not evidence.

---

## 5. Hard constraints

- **Read-only.** The filings are source-of-truth material for auditing. Do **not**
  re-run the parser, rebuild the fact store, or regenerate gold from them. Three
  separate things — "the benchmark is wrong", "the extractor is wrong", "the system
  behaved wrongly" — must stay separate, and rebuilding collapses them.
- **Do not modify** any fixture, any file under `benchmarks/`, or the fact store.
- Put your output only where §6 says.

---

## 6. Output

Write to:

```
/disk/qh/nano-finrag/artifacts/evaluation/p1-6-0d-source-truth/verdicts.json
```

```json
{
  "verifier": "<name/handle>",
  "generated_at": "<ISO-8601>",
  "verdicts": [
    {
      "case_id": "tv2f01-s3-crossdiff-003",
      "entity": "Apple",
      "gold_value": "9,683",
      "verdict": "WRONG_SCOPE_GOLD",
      "company_level_value": "$ 20,719",
      "evidence": {
        "pdf_page": 43,
        "row_or_column_text": "Provision for income taxes $ 20,719  $ 29,749  $ 16,741",
        "why": "9,683 is the 'Total' of the Federal: grouping (9,683 + 1,541 + 9,495 = 20,719); the company figure is the Provision for income taxes row."
      }
    }
  ]
}
```

Per-verdict requirements:

- `verdict` ∈ `{"VALID_GOLD", "WRONG_SCOPE_GOLD", "SOURCE_STILL_AMBIGUOUS"}`
- `company_level_value` — the value you believe is company-level. Required when you
  claim `VALID_GOLD` or `WRONG_SCOPE_GOLD`; may be `null` for `SOURCE_STILL_AMBIGUOUS`.
- `evidence.pdf_page` and `evidence.row_or_column_text` — verbatim text from the
  filing that carries your decision.
- `evidence.why` — one or two sentences a reader can check.

---

## 7. Independence — read this before starting

Record your own verdicts **before** reading §8. The artifact's
`decision`/`reason`/`page_context_blocks` fields are a machine's first pass that
already produced one confirmed false positive; anchoring on them defeats the purpose
of an independent check.

You may use `page_context_blocks` as a *pointer to which page to open*. You may not
treat it as the answer.

## 8. Reconciliation (only after your verdicts are written)

A first-pass rule set settled 1 of the 10 cases (`compare-009`, proven at
column level). The author's tentative, **unverified** leads on the remaining slots
were:

- `compare-001` — gold `$ 5,678` appears to be the `United States` jurisdiction row of
  a `United States / International / Total` table; company total possibly `$ 15,998`.
- `compare-005` — gold `$ 11,487` is `Federal: Current`; the company figure for a
  "Current" tax line may be the sum across Federal/State/Foreign (= `22,058`).
- `crossdiff-003` (Apple) — gold `9,683` is the Federal sub-total, corroborated above.
- `rank-002` — gold `$ 6,411` may in fact be **VALID**: its block is titled
  "Research and Development Expense" and the row reads
  `Research and development $ 6,411  $ 4,540  $ 3,969`, which looks company-level.

Report agreements and disagreements. **Where you disagree, the filing wins** —
cite the page text and the case will be re-read against it.
