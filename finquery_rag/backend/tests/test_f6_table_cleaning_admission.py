"""H2A-3D: the live table-cleaning path, before and after.

F6's defect was the authority transition at what is now
``process_tables.enhance_table_with_context``.  The model's answer was matched
with one regex and returned under the key ``content`` -- the *same* key and
shape the three failure fallbacks return -- so from that line onward nothing in
the pipeline could tell model-produced text from parser-produced text, and the
answer was embedded, BM25-indexed, expanded into ``parent_excerpt`` and read as
evidence.

The before/after below is deliberate, and the "before" is a record rather than a
re-run.  ``_pre_fix_content`` reproduces the removed decision in five lines --
the CLEANED TABLE regex and the truthiness guard, which is all there was -- so
that the claim "this alteration used to become chunk content" stays checkable
after the code that did it is gone.  That is the same reason
``b3_legacy_context_baseline_v1`` is kept: a defect nobody can reproduce is a
defect nobody can confirm was real.

The live path is driven with a stubbed HTTP call, so this file needs no network
and no API key.  Before this phase nothing tested ``enhance_table_with_context``
at all -- partly because importing its module required Camelot, which the
function itself has nothing to do with.  That import is now where it is used.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from src.services import process_tables

#: The authoritative extraction: two labels, two columns, four figures.
SOURCE_TABLE = """| Metric | FY2024 | FY2023 |
| --- | --- | --- |
| Revenue | 100 | 90 |
| Cost | 80 | 70 |
"""

#: The same table with one digit moved.  Everything else is identical.
ALTERED_TABLE = """| Metric | FY2024 | FY2023 |
| --- | --- | --- |
| Revenue | 101 | 90 |
| Cost | 80 | 70 |
"""

#: The same table with the two values swapped between rows.
REASSIGNED_TABLE = """| Metric | FY2024 | FY2023 |
| --- | --- | --- |
| Revenue | 80 | 90 |
| Cost | 100 | 70 |
"""

#: A faithful cleaning: whitespace tidied, numbers untouched.
CLEANED_TABLE = """| Metric | FY2024 | FY2023 |
| --- | --- | --- |
| Revenue | 100 | 90 |
| Cost | 80 | 70 |
"""

SOURCE_REF = "annual-report.pdf::page_7::table_1"

TABLE = {"md": SOURCE_TABLE, "bbox": None}


# --- the removed path, kept so the defect stays reproducible -------------------------------


def _pre_fix_content(model_answer: str, fallback: str) -> str:
    """What ``enhance_table_with_context`` returned before H2A-3D.

    The whole of the old decision: take the ``CLEANED TABLE:`` section if there
    is one and it is non-empty, otherwise the authoritative markdown.  A record
    of superseded behaviour, not a second implementation -- nothing calls it but
    the reproduction below.
    """

    match = re.search(r"CLEANED TABLE:\s*(.*)", model_answer, re.DOTALL)
    parsed = match.group(1).strip() if match else ""
    return parsed if parsed else fallback


def _model_answer(summary: str, table: str) -> str:
    return f"TABLE SUMMARY:\n{summary}\n\nCLEANED TABLE:\n{table}"


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _StubRequests:
    """A ``requests`` module that answers with one canned completion, or raises."""

    def __init__(self, answer: str | Exception) -> None:
        self.answer = answer
        self.calls = 0

    def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return _FakeResponse({"choices": [{"message": {"content": self.answer}}]})


def _enhance(
    monkeypatch: pytest.MonkeyPatch,
    answer: str | Exception,
    *,
    table: dict | None = None,
) -> tuple[_StubRequests, dict]:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    stub = _StubRequests(answer)
    monkeypatch.setattr(process_tables, "requests", stub)
    result = process_tables.enhance_table_with_context(
        table if table is not None else TABLE,
        page_text="Revenue rose year over year.",
        page_num=7,
        source_reference=SOURCE_REF,
    )
    return stub, result


# --- the before/after ------------------------------------------------------------------------


def test_the_removed_path_would_have_promoted_the_altered_table() -> None:
    """The "before", reproduced.

    The alteration genuinely is in the model's answer, and the old path's only
    transform was the regex above it -- so this is what used to become the
    chunk's ``content``, its ``parent_excerpt``, and the text the answer model
    read.  If this assertion ever becomes false, the defect is being
    misdescribed.
    """

    promoted = _pre_fix_content(
        _model_answer("A revenue table.", ALTERED_TABLE), SOURCE_TABLE
    )

    assert promoted == ALTERED_TABLE.strip()
    assert "101" in promoted
    assert "Revenue | 100" not in promoted


def test_the_live_path_blocks_the_same_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The "after", on the same adversarial answer.

    Same regexable model answer, same source table -- and what leaves the
    function is now the authoritative one, with a record saying why.
    """

    stub, result = _enhance(monkeypatch, _model_answer("A revenue table.", ALTERED_TABLE))

    assert stub.calls == 1
    assert result["content"] == SOURCE_TABLE
    assert result["content"] != ALTERED_TABLE.strip()

    record = result["admission_record"]
    assert record is not None
    assert record["admitted"] is False
    assert record["reason"] == "NUMERIC_VALUE_CHANGED"
    assert record["location"] == "revenue / fy2024"
    assert record["source_reference"] == SOURCE_REF


def test_a_reassignment_is_blocked_on_the_live_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every figure survives and the meaning does not."""

    _, result = _enhance(
        monkeypatch, _model_answer("A revenue table.", REASSIGNED_TABLE)
    )

    assert result["content"] == SOURCE_TABLE
    assert result["admission_record"]["reason"] == "NUMERIC_VALUE_REASSIGNED"


def test_a_faithful_cleaning_is_admitted_and_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verifier admits too -- a contract that only ever refuses is not one.

    The admitted content is the model's table, so this is not passing by
    discarding everything the model produced.
    """

    _, result = _enhance(monkeypatch, _model_answer("A revenue table.", CLEANED_TABLE))

    assert result["content"] == CLEANED_TABLE.strip()
    assert result["admission_record"]["admitted"] is True
    assert result["admission_record"]["reason"] == "FIDELITY_PRESERVED"
    assert result["admission_record"]["model_id"] == "meta/llama-3.1-8b-instruct"


def test_presentation_only_normalisation_is_admitted_on_the_live_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``1,000`` for ``1000`` is the same quantity, and the contract allows it."""

    source = "| Metric | FY2024 |\n| --- | --- |\n| Revenue | 1000 |\n"
    cleaned = "| Metric | FY2024 |\n| --- | --- |\n| Revenue | 1,000 |\n"

    _, result = _enhance(
        monkeypatch,
        _model_answer("Revenue.", cleaned),
        table={"md": source, "bbox": None},
    )

    assert result["admission_record"]["admitted"] is True
    assert result["content"] == cleaned.strip()


# --- the model's prose never becomes content -------------------------------------------------


def test_the_model_summary_never_reaches_the_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same defect one step removed.

    The summary is unverifiable -- nothing in the source says what a description
    of the table ought to say -- and it used to be folded into both ``content``
    and ``parent_excerpt``.  A summary carrying a figure the table never stated
    is exactly what a fidelity check on the table would not catch.
    """

    answer = _model_answer(
        "Revenue was 999 million, up from 111 million.", CLEANED_TABLE
    )

    _, result = _enhance(monkeypatch, answer)

    assert "999" not in result["content"]
    assert "111" not in result["content"]
    assert "Summary" not in result["content"]
    assert "TABLE SUMMARY" not in result["content"]


# --- the fallbacks are unchanged, and now say what they are -------------------------------------


def test_no_api_key_returns_the_authoritative_table_with_no_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    result = process_tables.enhance_table_with_context(
        TABLE, page_text="", page_num=7, source_reference=SOURCE_REF
    )

    assert result == {"content": SOURCE_TABLE, "admission_record": None}


def test_a_failed_call_returns_the_authoritative_table_with_no_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub, result = _enhance(monkeypatch, RuntimeError("connection reset"))

    assert stub.calls == 1
    assert result["content"] == SOURCE_TABLE
    assert result["admission_record"] is None


def test_an_answer_without_a_table_has_nothing_to_admit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a rejection: no candidate was produced, so the source content stands."""

    _, result = _enhance(monkeypatch, "TABLE SUMMARY:\nA table, apparently.")

    assert result["content"] == SOURCE_TABLE
    assert result["admission_record"] is None


# --- the write site -------------------------------------------------------------------------


def _process_pdf() -> ast.FunctionDef:
    tree = ast.parse(
        pathlib.Path("src/services/ingest.py").read_text(encoding="utf-8")
    )
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "process_pdf"
    )


def test_the_table_loop_writes_only_the_enhanced_content_field() -> None:
    """Only admitted-or-authoritative content reaches the trusted fields.

    Checked on the syntax tree rather than by running ``process_pdf``, which
    needs a real PDF and the PyMuPDF/Camelot stack.  What is asserted is exact:
    one assignment to ``table_content``, and its value is the ``content`` field
    the admitted artifact produced -- not a concatenation, not a second field,
    and not a summary.
    """

    function = _process_pdf()
    assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "table_content"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1, "table content must be established once"

    value = assignments[0].value
    assert isinstance(value, ast.Subscript), ast.unparse(value)
    assert ast.unparse(value.value) == "enhanced_table"
    assert isinstance(value.slice, ast.Constant) and value.slice.value == "content"

    # No in-place edit afterwards, so nothing can compose a summary onto it.
    for node in ast.walk(function):
        if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            target = node.target
            assert not (isinstance(target, ast.Name) and target.id == "table_content")


def test_the_parent_excerpt_is_the_same_admitted_text() -> None:
    """``parent_excerpt`` is what the answer model actually reads -- context
    expansion replaces a child's content with it -- so it must be derived from
    the same string, not from a second source."""

    arguments = [
        node.args[0]
        for node in ast.walk(_process_pdf())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_compact_parent_excerpt"
        and node.args
    ]

    assert arguments, "the excerpt must be built somewhere"
    assert any(
        isinstance(argument, ast.Name) and argument.id == "table_content"
        for argument in arguments
    ), ast.unparse(arguments[0])


def test_the_chunk_carries_the_admission_record_it_was_given() -> None:
    """The metadata write is the record the function produced, not a second
    derivation of it -- so there is one place the record is built."""

    function = _process_pdf()
    values = [
        value
        for node in ast.walk(function)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "derived_admission"
    ]
    assert len(values) == 1, "one admission record per table chunk"

    target = values[0]
    assert isinstance(target, ast.Subscript), ast.unparse(target)
    assert ast.unparse(target.value) == "enhanced_table"
    assert (
        isinstance(target.slice, ast.Constant)
        and target.slice.value == "admission_record"
    )


# --- the record itself ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("table", "admitted", "reason"),
    [
        (CLEANED_TABLE, True, "FIDELITY_PRESERVED"),
        (ALTERED_TABLE, False, "NUMERIC_VALUE_CHANGED"),
    ],
)
def test_the_admission_record_carries_names_and_never_content(
    monkeypatch: pytest.MonkeyPatch,
    table: str,
    admitted: bool,
    reason: str,
) -> None:
    _, result = _enhance(monkeypatch, _model_answer("A table.", table))
    record = result["admission_record"]

    assert set(record) == {
        "transformation",
        "admitted",
        "reason",
        "location",
        "source_reference",
        "artifact_id",
        "model_id",
    }
    assert record["transformation"] == "TABLE_CLEANING"
    assert record["admitted"] is admitted
    assert record["reason"] == reason
    assert record["source_reference"] == SOURCE_REF
    assert record["artifact_id"] == f"{SOURCE_REF}::table-cleaning"

    rendered = repr(record)
    for value in ("100", "101", "80", "90", "70"):
        assert value not in rendered, value
