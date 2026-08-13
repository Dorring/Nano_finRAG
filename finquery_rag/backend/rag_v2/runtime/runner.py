"""Final-evaluation harness seam with prediction-seal ordering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .contracts import TrustedRAGQueryV2, TrustedRAGResponseV2
from .metrics import RuntimeMetricAggregatorV1
from .runtime import TrustedRAGRuntimeV2


@dataclass(frozen=True)
class EvaluationRunResultV1:
    responses: tuple[TrustedRAGResponseV2, ...]
    prediction_seal: str
    metrics: dict[str, Any]
    gold_loaded_after_seal: bool


class V2FinalEvaluationRunner:
    """Skeleton for the later 72-question sealed evaluation."""

    def __init__(self, runtime: TrustedRAGRuntimeV2) -> None:
        self.runtime = runtime

    def run(self, queries: Iterable[TrustedRAGQueryV2], *, post_seal_evaluator: Callable[[tuple[TrustedRAGResponseV2, ...]], Any] | None = None) -> EvaluationRunResultV1:
        responses = tuple(self.runtime.handle(query) for query in queries)
        payload = json.dumps([item.to_dict() for item in responses], sort_keys=True, separators=(",", ":")).encode()
        seal = hashlib.sha256(payload).hexdigest()
        aggregator = RuntimeMetricAggregatorV1()
        for response in responses:
            aggregator.observe(response)
        # The evaluator is explicitly invoked only after the prediction seal exists.
        post_seal_evaluator(responses) if post_seal_evaluator else None
        return EvaluationRunResultV1(responses, seal, aggregator.snapshot(), post_seal_evaluator is not None)
