# Annotation guide

## Page convention

`page` is the physical PDF page number, 1-based, and is the page used for
evaluation matching.  `printed_page` is optional and may preserve the number
printed inside the report.  Do not substitute printed page numbers for
physical PDF pages.

## Numeric normalization

- Store the canonical numeric value without thousands separators.
- Convert `million`, `billion`, `thousand`, `K`, `M`, and `B` into the chosen
  `scale`; keep the original unit in the source note.
- Record currency separately (`USD`, `EUR`, etc.).
- Percentages are proportions in normalized form only when the label explicitly
  says so; otherwise preserve the report's percentage representation and set
  `unit="percentage"`.
- Parenthesized values are negative values.
- Do not silently convert a percentage into a plain number.

## Period and table binding

Bind a value to the exact table column header and period.  A row with 2023,
2024, and 2025 values needs one source binding per selected column.  Never
bind all row values to the first year.  Record table title, row label, column
header, unit, and scale when available.

## Calculations

Record the operation, every operand, its period, and the source index.  For
growth rate use `(new-old)/old`; retain the unrounded operands and specify the
display precision separately if needed.

## Multi-source questions

List every required source separately.  A question is not complete when only
one of its documents or metrics is cited.

## No-answer questions

Search the likely report sections and relevant synonyms before labeling a
question unavailable.  “Not retrieved” is not evidence of “not disclosed”.
Record the search terms and reviewer notes, but do not fabricate a source.

## Promotion gate

Draft records have `source_verified=false` and `review_status=unreviewed`.
They are never Golden or Sealed.  Promotion requires question, answer,
source, and (when applicable) calculation review, plus
`ready_for_golden=true`.
