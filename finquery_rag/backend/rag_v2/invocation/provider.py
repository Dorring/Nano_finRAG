"""The provider boundary: one interface, and the adapter that reaches the old one.

A provider owns transport, serialization, authentication, and the conversion of
its own failures and responses into the neutral contracts beside this file.  It
does not own evidence selection, disclosure, pack construction, binding,
calculation, verification or prompt semantics -- and the reason that list is
worth writing down is that every one of those is reachable from an
``AgentContextPackV1``, which is exactly what a provider never receives.

This module imports nothing from ``rag_v2.context`` and nothing from ``src``.
That is not a coincidence of what it happens to use: it is the boundary.  A
provider author who wants a semantic field has to get it through the renderer
that put it in the prompt, which is the only route Disclosure Authority governs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .contracts import (
    ModelProviderError,
    ModelRequestV1,
    ModelResponseV1,
    ProviderFailureKind,
)

__all__ = [
    "LegacyPromptProviderAdapterV1",
    "ModelProviderV1",
]

#: Where a backend convention puts the answer inside a returned mapping.  A
#: compatibility detail of the ``generate(prompt)`` backends that predate this
#: contract, listed rather than chained with ``or`` so that a field which is
#: present and empty is not silently replaced by a different one.
_ANSWER_KEYS = ("answer_text", "answer", "raw_output")

_FINISH_KEYS = ("finish_reason", "finish_status")


@runtime_checkable
class ModelProviderV1(Protocol):
    """What the Harness requires of anything that runs a model.

    One method, one request, one response.  There is no ``stream``, no ``health``
    and no ``close`` here because nothing in the runtime uses them yet, and a
    protocol that guessed at a provider's shape would be a worse spec than a
    small one -- a future provider that needs a capability adds it here, next to
    the requirement that justified it.
    """

    def invoke(self, request: ModelRequestV1) -> ModelResponseV1: ...


def _answer_text(raw: Mapping[str, Any]) -> str:
    for key in _ANSWER_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _citations(raw: Mapping[str, Any]) -> tuple[str, ...]:
    value = raw.get("citation_ids", ())
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(str(item) for item in value)


def _finish_status(raw: Mapping[str, Any]) -> str:
    for key in _FINISH_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "unknown"


def _usage(raw: Mapping[str, Any]) -> dict[str, int] | None:
    """Token counts the backend reported about its own run, when it reported any.

    These are *observed* counts from the provider's own tokenizer, and they are
    not a token-efficiency metric of any kind: nothing in the Harness compares
    them to a bound, and H2A-4's rules about what may be claimed from a count
    apply to anything that later reads them.  A backend that does not report
    them gets ``None`` -- absence stated as absence.
    """

    usage: dict[str, int] = {}
    if isinstance(raw.get("rendered_input_length"), int):
        usage["prompt_tokens"] = int(raw["rendered_input_length"])
    if isinstance(raw.get("tokens_generated"), int):
        usage["completion_tokens"] = int(raw["tokens_generated"])
    return usage or None


class LegacyPromptProviderAdapterV1:
    """Wrap an existing ``generate(prompt) -> text`` backend as a provider.

    H2A-3C.  There are eleven of these backends -- two production, nine in the
    suite -- and every one of them already satisfies the only thing the Harness
    actually needs from a model: take the text, give back the text.  Requiring
    each to implement a new interface would have been churn that proved nothing,
    and would have left the boundary described by eleven signatures instead of
    one.

    So this is the **migration adapter**, not a second source of truth.  It does
    not render, does not select, does not project and does not repair.  It calls
    the backend with ``request.prompt`` and reads what comes back into a
    ``ModelResponseV1``, normalizing the two things a provider owns: how a
    failure is classified, and how a response is shaped.

    One thing it deliberately *stops* doing.  The legacy boundary merged a
    provider-declared ``metadata`` mapping into the candidate's
    ``generation_metadata``, so whatever a backend chose to put in that bag
    became part of the runtime's record of the run.  A provider's own bag is not
    runtime truth -- that is the rule this contract exists to state -- so it is
    dropped rather than forwarded.  No production backend ever populated it.
    """

    def __init__(self, backend: Any) -> None:
        if not callable(getattr(backend, "generate", None)):
            raise TypeError("specialist backend must expose generate")
        self.backend = backend
        self.calls = 0

    def invoke(self, request: ModelRequestV1) -> ModelResponseV1:
        self.calls += 1
        try:
            raw = self.backend.generate(request.prompt)
        except ModelProviderError:
            # Already normalized by an adapter further in; do not re-wrap and
            # lose a kind someone deliberately chose.
            raise
        except TimeoutError as exc:
            raise ModelProviderError(
                ProviderFailureKind.TIMEOUT,
                f"{type(self.backend).__name__} timed out",
            ) from exc
        except Exception as exc:
            # Every provider-specific failure lands on one kind.  The cause is
            # chained rather than flattened: what the Harness needs is a
            # classification, and what a reader needs is still the original.
            raise ModelProviderError(
                ProviderFailureKind.UNAVAILABLE,
                f"{type(self.backend).__name__} failed: {exc}",
            ) from exc

        if isinstance(raw, str):
            return ModelResponseV1(text=raw, finish_status="unknown")

        if isinstance(raw, Mapping):
            return ModelResponseV1(
                text=_answer_text(raw),
                finish_status=_finish_status(raw),
                citations=_citations(raw),
                usage=_usage(raw),
            )

        raise ModelProviderError(
            ProviderFailureKind.INVALID_RESPONSE,
            f"{type(self.backend).__name__} returned "
            f"{type(raw).__name__}, which is neither text nor a response mapping",
        )
