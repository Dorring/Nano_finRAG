"""Provider abstraction, registry, mock provider, and sealed replay provider."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .contracts import AnswerEnvelopeV1, GenerationInputV1


@dataclass(frozen=True)
class GeneratorProviderMetadataV1:
    provider_id: str
    model_id: str
    revision: str | None = None


class GeneratorProviderV1(Protocol):
    @property
    def metadata(self) -> GeneratorProviderMetadataV1: ...

    def generate(self, generation_input: GenerationInputV1, generation_context: Mapping[str, Any]) -> AnswerEnvelopeV1: ...


ResponseFactory = Callable[[GenerationInputV1, Mapping[str, Any]], Mapping[str, Any] | AnswerEnvelopeV1]


class MockGeneratorProviderV1:
    def __init__(self, provider_id: str = "mock", model_id: str = "mock-model",
                 response: Mapping[str, Any] | AnswerEnvelopeV1 | ResponseFactory | None = None) -> None:
        self._metadata = GeneratorProviderMetadataV1(provider_id, model_id)
        self.response = response
        self.calls = 0

    @property
    def metadata(self) -> GeneratorProviderMetadataV1:
        return self._metadata

    def generate(self, generation_input: GenerationInputV1, generation_context: Mapping[str, Any]) -> AnswerEnvelopeV1:
        self.calls += 1
        value: Any = self.response
        if callable(value):
            value = value(generation_input, generation_context)
        if value is None:
            value = {"query_id": generation_input.query_id, "route": generation_input.route,
                     "answer_text": "Evidence is available [EV-1].", "citation_ids": ["EV-1"],
                     "generation_status": "complete", "generator_model": self.metadata.model_id}
        if isinstance(value, AnswerEnvelopeV1):
            return value
        return AnswerEnvelopeV1.from_dict(value, provider_id=self.metadata.provider_id,
                                          attempt_index=int(generation_context.get("attempt_index", 0)))


class ReplayGeneratorProviderV1:
    """Replays sealed V2-06 envelopes without invoking a model."""

    def __init__(self, predictions: Mapping[str, Mapping[str, Any] | list[Mapping[str, Any]]], model_id: str,
                 provider_id: str = "replay") -> None:
        self._metadata = GeneratorProviderMetadataV1(provider_id, model_id)
        self.predictions = predictions
        self.calls = 0

    @property
    def metadata(self) -> GeneratorProviderMetadataV1:
        return self._metadata

    def generate(self, generation_input: GenerationInputV1, generation_context: Mapping[str, Any]) -> AnswerEnvelopeV1:
        self.calls += 1
        raw_rows = self.predictions.get(generation_input.query_id)
        if raw_rows is None:
            raise KeyError(f"no sealed replay prediction for {generation_input.query_id}")
        rows = raw_rows if isinstance(raw_rows, list) else [raw_rows]
        actual = str(generation_input.packet.get("packet_sha256", ""))
        row = next((candidate for candidate in rows
                    if not candidate.get("packet_sha256") or candidate.get("packet_sha256") == actual), rows[0])
        expected = str(row.get("packet_sha256", ""))
        if expected and actual and expected != actual:
            raise ValueError(f"sealed packet mismatch for {generation_input.query_id}")
        envelope = row.get("answer_envelope")
        if not isinstance(envelope, Mapping):
            raise ValueError("sealed replay row lacks answer_envelope")
        return AnswerEnvelopeV1.from_dict(envelope, provider_id=self.metadata.provider_id,
                                          attempt_index=int(generation_context.get("attempt_index", 0)))


class ProviderRegistryV1:
    def __init__(self, providers: Mapping[str, GeneratorProviderV1] | None = None) -> None:
        self._providers = dict(providers or {})

    def register(self, provider_id: str, provider: GeneratorProviderV1) -> None:
        if not provider_id.strip():
            raise ValueError("provider_id must not be empty")
        self._providers[provider_id] = provider

    def resolve(self, provider_id: str | None) -> GeneratorProviderV1 | None:
        return self._providers.get(provider_id) if provider_id else None

    @classmethod
    def from_config(cls, config: Mapping[str, Any], providers: Mapping[str, GeneratorProviderV1]) -> "ProviderRegistryV1":
        selected = config.get("generation", config)
        ids = {selected.get("primary_provider"), selected.get("fallback_provider")} if isinstance(selected, Mapping) else set()
        return cls({key: value for key, value in providers.items() if key in ids})


def packet_set_sha256(packets: list[Mapping[str, Any]]) -> str:
    payload = json.dumps(packets, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
