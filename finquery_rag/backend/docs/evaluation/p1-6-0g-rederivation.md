# P1.6-0G — re-derivation of the cross-entity stratum

Chosen over patching (option A). `p1-6-0e` is superseded.

## The design rule

A case survives only if **one concept is disclosed as a single company-level line by
every entity in it**. `rank-004` is the model: all four companies put `Interest expense`
on their own income statement, so the question has one subject.

This replaces the old construction, which matched a metric *string* against the store.
That string matched several concepts per filing — `Total` lives in every note, `United
States` is both a revenue geography and a tax jurisdiction — and it is why only 2 of the
20 cases held one quantity for all entities.

## How the candidates were found, and what they are worth

`derive_cross_entity_metrics.py` filters the store to metric strings present for every
entity in a case **at that case's period** with no ambiguous coordinate, and ranks the
survivors. `read_company_line.py` then reads the chosen line out of the filing.

**Values from the line reader are proposals, not ground truth.** It restricts the read to
primary-statement pages and requires the label to lead the row, and it still gets roughly
half wrong: it reports Coca-Cola's net income as `$3.05` (the diluted EPS two rows down)
and Apple's total liabilities as `$359,241` (liabilities *and equity*). Automated reading
of these filings is the P1.6-A problem, not a shortcut past it. Every value below is
marked for whether it was read and confirmed in this work, or is machine-proposed.

## The stratum

`C` = confirmed against the filing in this session · `P` = machine-proposed, **needs a
source read before it becomes gold** · `—` = not yet attempted

### Comparison (10)

| case | entities | new concept | values | status |
|---|---|---|---|---|
| compare-001 | Coca-Cola, Tesla | Net income | KO ?, Tesla `$3,855` | KO —, TSLA P |
| compare-002 | Apple, Visa | Net income | Apple `$112,010`, Visa `$20,058` | AAPL C, V P |
| compare-003 | Coca-Cola, Tesla | Net income | KO ?, Tesla `$3,855` | KO —, TSLA P |
| compare-004 | Apple, Microsoft | Net income | Apple `$112,010`, Microsoft `$101,832` | AAPL C, MSFT P |
| compare-005 | JPMorganChase, Apple | **Diluted earnings per share** | JPM `20.02`, Apple `$7.46` | both C |
| compare-006 | JPMorganChase, Tesla | **Comprehensive income** | JPM `$65,214`, Tesla `4,886` | both C |
| compare-007 | JPMorganChase, Pfizer | Net income | JPM `$57,048`, Pfizer `8,062` | JPM C, PFE P |
| compare-008 | JPMorganChase, Apple | **Diluted earnings per share** | JPM `20.02`, Apple `$7.46` | both C |
| compare-009 | Apple, JPMorganChase | **Net income** | Apple `$112,010`, JPM `$57,048` | both C |
| compare-010 | Apple, Microsoft | Net income | Apple `$112,010`, Microsoft `$101,832` | AAPL C, MSFT P |

### Difference (5)

| case | entities | new concept | values | status |
|---|---|---|---|---|
| crossdiff-001 | NVIDIA, Tesla | Net income | NVIDIA `$72,880`, Tesla `$3,855` | NVDA P, TSLA P |
| crossdiff-002 | Microsoft, Apple | Net income | Microsoft `$101,832`, Apple `$112,010` | MSFT P, AAPL C |
| crossdiff-003 | Apple, Tesla | **Total liabilities** | Apple `285,508`, Tesla `54,941` | both C |
| crossdiff-004 | Coca-Cola, Visa | **Long-term debt** | KO `42,119`, Visa `19,602` | both C |
| crossdiff-005 | NVIDIA, Coca-Cola | Total liabilities | NVIDIA `32,274`, KO ? | NVDA P, KO — |

### Ranking (5)

| case | entities | new concept | values | status |
|---|---|---|---|---|
| rank-001 | Tesla, Coca-Cola, Visa | Net income | TSLA `$3,855`, KO ?, Visa `$20,058` | TSLA P, KO —, V P |
| rank-002 | Apple, Microsoft, Tesla | **Research and development** | Apple `34,550`, Microsoft `32,488`, Tesla `$6,411` | all C |
| rank-003 | JPMorganChase, Microsoft, Tesla, Visa | **Comprehensive income** | `$65,214` / `$104,075` / `4,886` / `$20,614` | JPM,TSLA C; MSFT,V P |
| rank-004 | JPMorganChase, Visa, Coca-Cola, Microsoft | **Interest expense** | `97,898` / `(589)` / `1,654` / `(2,385)` | all C |
| rank-005 | Apple, Coca-Cola, Microsoft | — | — | **RETIRE** |

## Retirements

- **rank-005** — no concept is disclosed company-level by all three. Apple's `12,875` is
  a derivative *notional*, Coca-Cola's `16` is an OCI *gain*. `derive_cross_entity_metrics`
  finds no shared concept at all, which agrees.

The cases the coherence pass called GARBAGE (`compare-006`, `compare-007`,
`crossdiff-002`) are **not** retired here: each turns out to have a usable shared concept
(`Comprehensive income`, `Net income`, `Net income`), so their defect was the old metric's
choice rather than the entity set.

## What this changes for the stratum

- **19 of 20 cases** are re-derived; 1 retires.
- **`rank-002`'s answer changes.** With Apple's R&D read as positive `34,550` the ranking
  is `Apple > Microsoft > Tesla`, where the old gold — built from the mis-signed value —
  said `Microsoft > Tesla > Apple`.
- **`compare-009` and `compare-003`** now rest on `Net income`, so the JPMorganChase
  `Corporate`-column defect and the Coca-Cola equity-method-investees defect both go away
  by construction rather than by value patch.
- Several cases now share a concept (nine use `Net income`). That is a coverage cost —
  the stratum tests fewer distinct quantities — accepted because coherence is the
  property it lacked.

## Files

```
docs/evaluation/
  p1-6-0g-rederivation.md              this document
  p1-6-0f-coherence-pass.md            why the old stratum is not patchable
  p1-6-0f-partner-coherence-audit.md   the partner check that started this
scripts/evaluation/
  derive_cross_entity_metrics.py       candidate finder (store-side)
  read_company_line.py                 source reader (proposal only)
  audit_cross_entity_coherence.py      per-entity evidence workbook

artifacts/evaluation/p1-6-0g-rederivation/
  cross-entity-metric-candidates.json  candidates per case, with store statuses
```

## What is still open

1. **Every `P` and `—` value needs a source read** before it can be gold. That is the
   remaining work, and it is per-value: the reader above is a pointer, not an answer.
2. **The `Net income` label differs by filer** — Microsoft's statements were not detected
   by the reader at all, so its values come from the store's candidate list and need
   confirming.
3. **No fixture has been changed.**
