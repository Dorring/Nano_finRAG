"""The model invocation contracts: what a provider is asked, and what it answers.

H2A-3C.  The Context Runtime had a working production path by H2A-3B3, but the
model boundary was still shaped by the one boundary that had crossed it: the
specialist backend took a rendered prompt because that is what the specialist
needed.  That is a contract by accident, and it is the wrong place for the
accident to sit -- the whole point of the Harness is that a financial model and
a DeepSeek endpoint are interchangeable behind it.

So this package names the boundary:

    AgentContextPackV1
      -> ContextRendererV1        (a role decides what the model is shown)
      -> ModelRequestV1           (the text, and where it is going)
      -> ModelProviderV1          (transport, serialization, errors)
      -> ModelResponseV1          (the text that came back)

**What a provider may receive is deliberately narrow.**  A `ModelRequestV1` is
text, an identity, and configuration.  It is not a pack, not evidence, not a
RunState, not a calculation.  The reason is not tidiness: a provider that could
read the pack would be able to reach semantic fields the role's renderer chose
not to expose, and the Disclosure Authority's decision about what a model may
see would be one implementation detail away from being advisory.  Keeping the
pack out of the provider's reach is what makes the renderer the *only* thing
that decides what crosses.

That is also why this module imports nothing from ``rag_v2.context``.  Its
sibling ``renderer.py`` does, because a renderer's whole job is a pack -- but the
contracts a provider handles are free of it, and ``provider.py`` is guarded
against acquiring the dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

__all__ = [
    "InvocationIntegrityError",
    "ModelProviderError",
    "ModelRequestV1",
    "ModelResponseV1",
    "ProviderFailureKind",
]


class InvocationIntegrityError(ValueError):
    """Raised when a request or response would be built with a shape it must not have."""


class ProviderFailureKind(str, Enum):
    """The smallest provider-neutral classification the Harness needs.

    Deliberately not a cloud-provider error taxonomy.  Three kinds, because
    three are what the runtime distinguishes today:

    * ``UNAVAILABLE`` -- the provider could not be reached or could not run.  A
      missing checkpoint, an unreachable endpoint and a crashed engine are the
      same thing here, and they fail closed the same way.
    * ``TIMEOUT`` -- a bound elapsed.  Kept apart from ``UNAVAILABLE`` because a
      caller may reasonably retry one and not the other; RunBudget owns whether
      it does.
    * ``INVALID_RESPONSE`` -- the provider answered, and what it answered could
      not be read as a model output at all.  Distinct from an *empty* answer,
      which is a real model output and is judged by the boundary that knows what
      the boundary asked for.

    Nothing above is mapped onto release behaviour.  A normalized failure still
    propagates and still fails closed; what changes is that the Harness can tell
    three situations apart without importing a provider's SDK.
    """

    UNAVAILABLE = "UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    INVALID_RESPONSE = "INVALID_RESPONSE"


class ModelProviderError(RuntimeError):
    """A provider failure, normalized to a kind the Harness can act on.

    Providers wrap their own failures in this and chain the original
    (``raise ... from exc``), so a provider-specific cause is never lost -- it
    stops being the *interface*.
    """

    def __init__(self, kind: ProviderFailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _freeze_scalars(value: Mapping[str, Any], *, where: str) -> Mapping[str, Any]:
    """A read-only copy of a configuration mapping, scalars only.

    The same rule the pack applies to evidence, for the same reason: a named
    field is not a licence for a structure stored under it.  Usage counters are
    numbers -- a container here would be a second object travelling inside a
    response, and the boundary this module exists to keep narrow would have a
    hole in it.
    """

    frozen = dict(value)
    for name, item in frozen.items():
        if not isinstance(name, str) or not name:
            raise InvocationIntegrityError(
                f"{where} keys must be non-empty strings, got {name!r}"
            )
        if isinstance(item, (Mapping, list, tuple, set, frozenset)):
            raise InvocationIntegrityError(
                f"{where}[{name!r}] is a container; generation configuration is "
                f"scalars, and a container here is a second object crossing the "
                f"provider boundary"
            )
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class ModelRequestV1:
    """One provider invocation: the text, the identity, and the configuration.

    The field set is the boundary.  There is no ``pack``, no ``evidence``, no
    ``state``, no ``metadata`` bag and no provider SDK object, and adding one
    would be the change this contract exists to prevent -- a provider that could
    read the pack could reach fields the role's renderer chose not to expose.

    There is deliberately no ``generation`` field.  H2A-3C carried an empty one
    as a marker for where per-invocation sampling configuration would go, and
    H2A-3E removed it: an audit found no producer, no consumer, and no test
    beyond the one asserting the field set.  A field with nothing on either end
    of it is a claim about a future requirement nobody has stated, and the
    current provider owns its sampling configuration in its own constructor --
    which is where a real requirement will say it belongs when it arrives.
    """

    invocation_id: str
    role: str
    prompt: str
    provider_id: str
    model_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("invocation_id", "role", "provider_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise InvocationIntegrityError(f"{name} must be a non-empty string")
        if not isinstance(self.prompt, str):
            raise InvocationIntegrityError("prompt must be a string")
        if not self.prompt.strip():
            raise InvocationIntegrityError(
                "prompt must not be blank; a renderer that produced no input is "
                "a defect, not an invocation"
            )
        if self.model_id is not None and (
            not isinstance(self.model_id, str) or not self.model_id.strip()
        ):
            raise InvocationIntegrityError("model_id must be a non-empty string or None")


@dataclass(frozen=True)
class ModelResponseV1:
    """What a provider returned: text, and the facts about how it returned.

    ``citations`` is the one field beyond the obvious, and it is here because
    the existing Harness needs it rather than because it seemed useful.  A
    grounded generator declares which of the handles it was shown it used, the
    capability filters that against the handles the pack actually issued, and
    the release path carries the survivors.  Those are *handles from the
    request* -- E1, C1 -- so they are provider-neutral in the only sense that
    matters: the same vocabulary the renderer wrote and the validator resolves.

    ``text`` may be empty.  An empty answer is a real model output, and it is
    not this contract's job to judge it: the boundary knows what it asked for
    and judges grounding.  ``INVALID_RESPONSE`` is for a response that could not
    be read at all, which is a different thing.
    """

    text: str
    finish_status: str = "unknown"
    citations: tuple[str, ...] = ()
    usage: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise InvocationIntegrityError("text must be a string")
        if not isinstance(self.finish_status, str) or not self.finish_status.strip():
            raise InvocationIntegrityError("finish_status must be a non-empty string")

        citations: list[str] = []
        seen: set[str] = set()
        for citation in self.citations:
            text = str(citation).strip()
            if not text:
                raise InvocationIntegrityError("a citation handle must not be blank")
            if text not in seen:
                seen.add(text)
                citations.append(text)
        object.__setattr__(self, "citations", tuple(citations))

        if self.usage is not None:
            usage = _freeze_scalars(self.usage, where="usage")
            for name, value in usage.items():
                if isinstance(value, bool) or not isinstance(value, int):
                    raise InvocationIntegrityError(
                        f"usage[{name!r}] must be an integer count"
                    )
            object.__setattr__(self, "usage", usage)
