# P1.6-A10 — `rank-004`: an expense is a positive tagged value

Found while checking whether signs survive into the rebuilt store, and it changes one
case's expected answer. No fixture has been migrated yet, which is why it was worth
looking before writing.

## The two readings

`rank-004` asks four filers to be ranked by interest expense, highest to lowest.

```
                    tagged value        as presented
JPMorganChase       97,898              97,898
Microsoft            2,385              (2,385)
The Coca-Cola Co.    1,654              1,654
Visa                   589              (589)
```

Ranked by tagged value — which is expense **magnitude**:

```
JPMorganChase > Microsoft > The Coca-Cola Company > Visa
```

Ranked by presented signed value:

```
JPMorganChase > The Coca-Cola Company > Visa > Microsoft
```

The re-derived stratum recorded the second. The rebuilt store produces the first.

## Why the tagged value is the right one here

US-GAAP expense concepts carry a **debit** balance type: a positive tagged amount *is* an
expense. The filing's own inline XBRL confirms it — Microsoft's fact reads

```xml
<ix:nonFraction name="us-gaap:InterestExpenseNonoperating" ...>2,385</ix:nonFraction>
```

with no `sign` attribute, so `+2,385` is what the filer asserted. The parentheses appear on
the income statement because the line is a deduction from income, which is presentation
rather than value. Visa's is tagged the same way.

So "rank by interest expense, highest to lowest" means largest expense first, and
`JPMorganChase > Microsoft > The Coca-Cola Company > Visa` is the answer to that question.

The signed reading is not incoherent — it answers "which filer's income statement line
sits highest" — but it puts the *smallest* expense second-from-last and the largest
third, which is a strange thing for the question to be asking and is an artefact of two
filers presenting the line as a deduction and two not.

## What this is really an instance of

The same split that ran through `net_income` and `comprehensive_income`, in a different
guise: **the value a filer tags and the value a reader sees are not always the same
number.** There it was consolidated against parent-only; here it is the concept's balance
type against the statement's presentation. In both cases the tagged value is the one that
is comparable across filers, because it follows the taxonomy rather than the typography.

That is an argument for the iXBRL path rather than against it — this is exactly the
information the table-parsing path could not carry. It is also a reason the stratum's
gold must be read from the store once the store is right, rather than hand-assembled from
what a page appeared to show.

## Consequence for the migration

`rank-004`'s `expected_ranking` becomes:

```
["JPMorganChase", "Microsoft", "The Coca-Cola Company", "Visa"]
```

and the question should say what it means — *rank by interest expense, largest first* —
so that a reader and the runtime are answering the same thing.

Worth checking the same way, before or after the migration: every other case whose values
could have a balance-type/presentation divergence. `rank-002` (R&D) and the arithmetic
cases are the candidates; the rest are income or balance-sheet amounts whose tagged and
presented signs agree in this corpus.
