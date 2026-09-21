# P1.6-A3-1a — period header reconstruction, in shadow

**Nothing applied.** A3-0 proved the chain; this reconstructs the split header and reports
what it *would* bind, so the controls can be shown unmoved before anything downstream moves.

## The rule

Geometric, not a string search: **a month-day fragment applies to the run of bare-year
columns that follows it within the same header row, and every column keeps its own year.**
A rule that looked left for the nearest month would cross header groups and collapse a
multi-year row onto one date. The contract is a *period expression* — `Years ended` is
carried with the date so the semantics stay `duration` rather than degrading to an instant.

## Result

```
                          columns   newly dated
jpm_fy2025#63193  BS            12             6      col 3  -> 'December 31, 2025'
jpm_fy2025#64546  CF            18             9      col 3  -> 'December 31, 2025'
ko_fy2025#11388   BS             9             6      col 3  -> 'December 31, 2025'
ko_fy2025#11899   CF            12             9      col 3  -> 'December 31, 2025'
----------------------------------------------------------------------------------
ko_fy2025#12533   EQ            12             0
nvda_fy2025#8766  CF            18             0
pfe_fy2024#24395  EQ            75             0
tsla_fy2025#10407 CF            18             0
v_fy2025#9951     BS            12             0
----------------------------------------------------------------------------------
aapl_fy2025#4731  IS   (ctrl)   18             0      unchanged
aapl_fy2025#5212  CI   (ctrl)   18             0      unchanged
```

**Four of nine reconstruct and five do not.** Both controls are unmoved, which is the
property the shadow pass exists to check.

The join behaves as specified on the four: `December 31,` and the bare year are joined, the
2025 columns and the 2024 columns take **their own** years, and no column consumes another's.
Coca-Cola's balance sheet is the clean case — the same table whose `December 31,` and `2025`
were in separate columns at A3-0.

## The five that did not reconstruct

They are not failures of the rule; they are a **different header shape**, and the shadow
output names it: their header rows expose no bare-year column to the scan at all.

```
ko_fy2025#12533   first column reads 'Equity Attributable to Shareowners of The Coca-Cola Comp…'
nvda_fy2025#8766  first column reads 'Cash flows from operating activities:'
tsla_fy2025#10407 first column reads 'Cash Flows from Operating Activities'
pfe_fy2024#24395  75 columns; first reads '(MILLIONS, EXCEPT PER SHARE DATA) / Balance, January 1,'
v_fy2025#9951     first column is empty
```

For these the year is not a bare-year cell in the header rows the scan reads. Whether it is
in a header row `header_idx` did not select, or fused into a longer cell, or the grid's
column geometry differs, is **not established here** — and asserting one of those would be
the guess A3-0 was written to avoid.

## What this does and does not establish

**Established.** The split-date hypothesis is confirmed where it applies: the join is
geometric, each column keeps its own year, and the two controls are unaffected.

**Not established.** That reconstructing four headers is enough — five of the nine are
untouched, so **A3-1b must not be wired on the strength of this**. Applying it now would
recover JPMorganChase's and Coca-Cola's balance sheets and cash flow statements and leave
five tables exactly where they are, while making the fix look complete.

The next step is a second diagnosis on the five, on the same terms: read their header
geometry, do not change the rule to reach them.

## Safety, restated

The requirement is **not** that dangerous `NON_PRIMARY` tables emit no facts. Store V2's job
is to hold facts faithfully; `table_role` and the resolver's authority test decide what may
speak for the company. A period fix that lets a note's cells reach the store is a data
quality *improvement*, and the gates that must hold are:

```
dangerous_authority = 0     wrong_scope = 0
wrong_metric = 0            value_mismatch = 0
```

**Extraction failure must not be the safety mechanism.**

## Status

Nothing changed. Evidence at
`artifacts/evaluation/p1-6-a3-1a-shadow/period-reconstruction-shadow.json`.
