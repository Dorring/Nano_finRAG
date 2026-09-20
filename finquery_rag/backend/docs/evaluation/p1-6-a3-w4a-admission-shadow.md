# P1.6-A3-W4-A — the admission decision, measured beside the authoritative one

Shadow only. The legacy rule in `typed_evidence_emitters` still decides; nothing was
switched. What follows is the delta W4-B would cause, split so that each part is
attributable.

```
python emission_admission_shadow.py --out artifacts/evaluation/p1-6-a3-w4a-shadow
python diagnose_admission_losses.py --delta .../emission-admission-shadow.json
```

## Why the delta is split in two

Switching admission authority moves two things at once, and one old-vs-new count would
blend them:

```
V1   the decision rule      the kind whitelist becomes a negative form;
                            the period is still the legacy axis's
V2   V1 + the new period source   what W4-B as originally planned would do
```

```
legacy -> V1   the rule change        +14,802 / -0
V1     -> V2   the producer change    +0      / -18,104
```

The first smoke run is why they are apart. On Apple alone it showed 386 cells the legacy
admitted and V2 withheld, and reading them named tables of contents and exhibit indexes:

```
ord=4622   'Page / years ended September 27 / 29 / 30'   role=OPEN
ord=13280  'Exhibit Number / as of August 20'            role=OPEN
```

The legacy axis classifier had read a date-shaped token out of their prose and minted
period facts. The binder refuses to bind a period to a table of contents because no period
header is there. That is the *producer* being right, and filing it under "the admission
change" would have been a false account of both.

## The rule change is exactly one kind wide

```
gained 14802   lost 0
    + kind:unknown -> ADMIT_GRAIN_LIMITED   3399
    + kind:unknown -> ADMIT                 2987
    + kind:unknown -> WITHHOLD              8416   (gained by the rule, withheld by V2)
```

Zero loss is not luck: the two rule expressions differ on `unknown` and on nothing else,
which `test_the_new_rule_differs_from_the_old_one_on_exactly_unknown` asserts by reading
the sets against each other. Under V1 alone all 14,802 are admitted; the 8,416 are cells
the rule would deliver and the producer then refuses, so they are counted separately as
`rule_gain_then_producer_loss` rather than being credited to the rule.

## The producer change is strictly narrower

```
gained 0   lost 18104        all of it `NO_BINDING`
```

`+0` is the finding. The new binder never binds a cell the legacy axis missed, so V2 is not
a differently-shaped path — it is a smaller one. Of the 18,104, **9,688 were admitted by
the legacy rule** and are what the store holds today; the other 8,416 were never stored, so
withholding them loses nothing.

The 9,688, by what the authority oracle says the table is:

```
OPEN                 6411
DANGEROUS_NEGATIVE   1329
WEAK_NON_PRIMARY     1240
PRIMARY               708
```

## The 708 primary cells

```
432   a month-day with no year      nvda ord=7899  'Retained / Earnings / ... / as of Jan 30'
      in the column's header
274   month-day and year both       nvda ord=6754  'Year Ended / Jan 26, 2025'
      present, still unbound
  2   empty header -- correctly unbound
```

The first version of this section split the same 708 into "352 abbreviated month name /
332 `as of` with no year / 22 / 2", and that split was wrong — see W4-A2 below, where the
prediction built on it missed by a third. A cell whose problem is a missing year was being
counted under "abbreviated month name" because an abbreviation happened to appear in its
header text. The buckets overlapped and the first match won.

The structural reading is the one that survives: **the dominant primary gap is a month-day
whose year is not in the column's header.** The adjacent-year join fires only on a bare
year cell (`^2025$`), and in a stockholders' equity statement the year travels with other
text or sits in a sibling column.

The non-primary losses are mostly the producer being right — `June 29, 2025 to August 2,
2025:` is a range and declares no single period; a table of contents is not a period
header. Reporting one number would have read as catastrophe where the detail reads as a
short list.

## Why there is no rule-only escape

The obvious way out would be to switch the rule now and leave the period source alone —
`V1` is finished, one kind wide, zero loss. That option does not exist, and checking rather
than assuming is what showed it:

```
unknown-kind cells the legacy rule drops            14802
  ... and the legacy axis carries NO period         14802
  ... and it has a normalized_period                    0
```

Every one of them. Not "most" — the tally reconciles cell for cell with the shadow's
`rule_gain` of 14,802 by an independent route, and a legacy `unknown` column is unknown
precisely *because* `_classify_column_temporal` falls through to
`return ("unknown", None, None, None)`. Admitting them under the legacy period source would
store 14,802 facts with no period, which is worse than withholding them.

So the rule and the period source cannot be switched apart: the new rule's whole content is
"a usable binding is what makes a fact storable", and there is no binding to be had from the
path being replaced.

## Verdict

**W4-B is blocked, and it is blocked by the producer, not by the rule.** Switching today
would remove facts the store holds — 8,369 after the abbreviated-month widening, 474 of
them from oracle-PRIMARY statements, 472 of those one single gap — and the 14,802 the rule
would add cannot be stored without a period at all.

The rule half is finished and measured. The producer half needs the second gap closed and a
re-measurement before anything switches, and its entry condition is now sayable:

```
producer coverage >= legacy axis coverage on oracle-PRIMARY tables
```

An earlier plan would have switched both at once and discovered this in the Store V2
rebuild, where the only visible symptom would have been a count going down.

## Coverage gaps this leaves

Registered rather than papered over, in the same way as `REAL_BUCKET_POSITIVE_UNOBSERVED`:

- **`CONFLICTED_PERIOD` is contracted and tested but not exercised.** `period_for` composes
  at most one column binding with one row binding, so `resolve_period_evidence` cannot
  return a `Conflict` on this path. A zero here means "not reachable from this producer",
  not "verified safe".
- The non-primary loss buckets are classified by header text, not adjudicated cell by cell.
  The primary buckets are the ones that decide the phase and those are.

## W4-A2 — the first slice, and a prediction that was wrong

The prediction was written down **before** the abbreviated-month widening ran:

```
oracle-PRIMARY regression       708  ->  ~356
total regression              9,688  ->  ~7,467
```

What happened:

```
oracle-PRIMARY regression       708  ->   474     recovered 234, predicted 352
total regression              9,688  -> 8,369     recovered 1,319, predicted 2,221
```

**The model over-predicted, and the reason it did is worth more than the fix.** The
classifier bucketed a cell by whether an abbreviated month name appeared anywhere in its
column header. That is a *lexical* test; the boundary that matters is *structural*. Of the
352 cells it blamed on abbreviations, about 100 had no year in the column at all —
`as of Jan 30` — so no widening of the month pattern could ever have reached them. A
lexical bucket straddled a structural boundary and reported one cause as two.

`abbreviated month` is no longer a bucket. After the widening both spellings take the same
path, so the distinction explains nothing, and keeping it would have preserved the same
error in a new shape. The structural question gives a clean answer:

```
PRIMARY                      before   after
  no month-day expression         2       2
  month-day, no year in column  432     432
  both present, still unbound   274      40
```

**472 of the 474 remaining primary losses are one gap: the year is not in the cell that
carries the month-day.** `as of Jan 30`, `as of December 31`. The abbreviated-month work is
finished; what is left is the second gap and nothing else.

### What the widening did and did not move

```
                          before    after
unchanged_admitted        17,604   18,923    +1,319
producer_loss             18,104   16,785    -1,319
regression (stored)        9,688    8,369    -1,319
rule_gain                 14,802   14,802         0
producer_gain                  0        0         0
unchanged_withheld        13,740   13,740         0
```

One number up by exactly what another is down by, and nothing else moved: **the widening is
monotone.** It added bindings and removed none — a widening month pattern cannot unbind a
column it used to bind, and this is the measurement that says so rather than the argument.

`producer_gain` is still **0**. The producer still never binds a cell the legacy axis
missed, so the entry condition for W4-B has not moved.

## W4-A2 implementation notes

A pure widening of the month pattern — full names and abbreviations through one path. Each
alternative is the full name plus a trailing optional run, so `Mar` cannot match inside
`Marketing`: the alternation is followed by `\s+\d` and `keting` is not that, pinned against
`Marketing 5`, `Junction 7`, `Decembering 3`, `Mayo 2`, `Augment 9`.

One structural consequence: `_MONTH_DAY_YEAR`'s month is no longer a capture group, so both
call sites read the year from group 1. Indexing the wrong group would not crash — it would
produce a period whose year is the day — so the arity is pinned by a test.

Both producers also gained a guard the widening makes reachable in principle: a `RESOLVED`
binding whose `normalized_period` is None is the incoherent state the contract forbids and
would read downstream as "resolved, but to nothing". They fall through instead.

The first assertion written for the NVDA case was wrong in the useful direction — it
expected two fiscal years and the filing has three (`2025-01-26`, `2024-01-28`,
`2023-01-29`). A hand-written expectation met a computed value and lost, rather than
agreeing with it and pinning nothing.

## Behaviour-neutral, provably

Since the W1 baseline `8ef4c45` the **only** production file changed is
`period_binding.py`, and `test_the_production_path_does_not_import_this_module` shows
nothing in the parse → build → resolve path reads it. No rebuild was needed to establish
that W4-A changed no emitted fact.
