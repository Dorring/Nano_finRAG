from __future__ import annotations

import json
from pathlib import Path

from rag_v2.contracts.errors import ContractError
from rag_v2.contracts.query import QuestionEnvelope


def load_question_envelopes(path: Path) -> tuple[QuestionEnvelope, ...]:
    """Load question-only records without consulting answers or labels."""

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    envelopes = tuple(QuestionEnvelope.from_benchmark_record(row) for row in rows)
    ids = [item.question_id for item in envelopes]
    if len(ids) != len(set(ids)):
        raise ContractError("question IDs must be unique")
    return envelopes
