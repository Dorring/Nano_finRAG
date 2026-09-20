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

## The 708 primary cells are two mechanical gaps

```
352   abbreviated month name            nvda ord=6754  'Year Ended / Jan 26, 2025'
332   `as of <month day>`, year in a     tsla ord=9172  'Common Stock / Shares / as of December 31'
      sibling cell
 22   `Year ended December 31` in every  jpm ord=63944  'Year ended December 31 / ...'
      column, year one step further out
  2   empty header -- correctly unbound
```

684 of 708 have two named causes, and both are narrow:

- **the month pattern lists full names only**, so NVIDIA's `Jan 26, 2025` fiscal calendar
  matches nothing at all;
- **the adjacent-year join only fires on a bare year cell** (`^2025$`), and in a
  stockholders' equity statement the year travels with other text.

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
would remove 3,302 facts net (6,386 in, 9,688 out), 708 of them from oracle-PRIMARY
statements, and the 14,802 the rule would add cannot be stored without a period.

The rule half is finished and measured. The producer half needs its two gaps closed and a
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

## Behaviour-neutral, provably

Since the W1 baseline `8ef4c45` the **only** production file changed is
`period_binding.py`, and `test_the_production_path_does_not_import_this_module` shows
nothing in the parse → build → resolve path reads it. No rebuild was needed to establish
that W4-A changed no emitted fact.
