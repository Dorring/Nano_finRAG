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
- The current Draft answer keys store percentage points as strings (for
  example `46.9` means 46.9%), with a separate `display_value` such as
  `46.90%`.  Reviewers must confirm the report representation before Golden
  promotion; do not silently convert percentage points to proportions.
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
`answer_key_status=entered_unverified` only records that an answer key has
been entered for annotation; it does not verify the PDF source.  No-answer
records remain `pending_negative_evidence` until a full-text negative search
is documented.  Drafts are never Golden or Sealed.  Promotion requires question, answer,
source, and (when applicable) calculation review, plus
`ready_for_golden=true`.

The current review actions are workflow states, not historical edit commands:
answerable records use `manual_answer_source_review`, and no-answer records use
`manual_negative_evidence_review`.  The prior `replace` or `rewrite` action is
retained only as `superseded_review_action`.  Source review is counted per
expected source record, so a two-source question contributes two evidence
items to the Golden gate.

## PDF verification and indexed identity

`pdf_page_verified=true` and `pdf_content_verified=true` mean that a reviewer
or a documented verification pass checked the disclosed value against the
actual ingested PDF, using the physical 1-based PDF page. These fields do not
make a source Golden. `source_verified` remains false until the source has a
stable indexed candidate identity and the human source-review workflow is
complete. If row-level identity is unavailable, record the limitation rather
than inventing a `row_id`.

An automated full-document term scan for a no-answer case is provisional
evidence only. It does not satisfy the manual negative-evidence gate and must
not set `negative_evidence_reviewed=true`.
