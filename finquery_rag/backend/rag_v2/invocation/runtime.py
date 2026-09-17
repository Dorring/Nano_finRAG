"""The invocation runtime: one pack in, one response out, one of each step.

    AgentContextPackV1
      -> binding.renderer.render(pack)      -> rendered text
      -> ModelRequestV1                     -> what the provider is asked
      -> binding.provider.invoke(request)   -> ModelResponseV1

That is the whole of it.  This layer owns wiring and nothing else: it does not
select context, does not compile, does not verify, does not retry.  Retrying is
RunBudget's question and this object has no opinion about it -- which is the
point of it being this small, because a runtime that could retry would also be a
runtime that could render twice.

**Exactly one of each step, made observable.**  The counters here are the
evidence for that claim, and they exist because the claim is otherwise
unfalsifiable from outside: a second rendering would produce the same text and a
second provider call would produce the same answer, so nothing downstream could
notice.  They are attempts, not successes -- a render that raises still counts,
because the thing being ruled out is a second call, not a failed one.
"""

from __future__ import annotations

from rag_v2.context import AgentContextPackV1

from .binding import ModelBindingV1
from .contracts import InvocationIntegrityError, ModelRequestV1, ModelResponseV1

__all__ = ["ModelInvocationRuntimeV1"]


class ModelInvocationRuntimeV1:
    """Run one bound model invocation for one compiled pack."""

    def __init__(self, binding: ModelBindingV1) -> None:
        if not isinstance(binding, ModelBindingV1):
            raise InvocationIntegrityError(
                f"an invocation runtime is constructed from a binding, got "
                f"{type(binding).__name__}"
            )
        self.binding = binding
        self.render_count = 0
        self.provider_count = 0
        #: The last request and response, held for the same reason the
        #: capability holds the last pack: they are the artefacts of this
        #: invocation, and a test that rebuilt them would be observing a second
        #: invocation rather than the one the runtime performed.
        self.last_request: ModelRequestV1 | None = None
        self.last_response: ModelResponseV1 | None = None

    def invoke(self, pack: AgentContextPackV1) -> ModelResponseV1:
        # Counted before the call, so a renderer that raises is still recorded
        # as having been called once rather than zero times.
        self.render_count += 1
        prompt = self.binding.renderer.render(pack)

        request = ModelRequestV1(
            invocation_id=pack.invocation_id,
            role=pack.role.value,
            prompt=prompt,
            provider_id=self.binding.provider_id,
            model_id=self.binding.model_id,
        )
        self.last_request = request

        self.provider_count += 1
        response = self.binding.provider.invoke(request)

        # A provider that returns something other than the contract's response
        # is a contract violation, not a provider failure, and it must not be
        # allowed to travel as runtime truth.
        if not isinstance(response, ModelResponseV1):
            raise InvocationIntegrityError(
                f"provider {self.binding.provider_id!r} returned "
                f"{type(response).__name__}, not a ModelResponseV1"
            )
        self.last_response = response
        return response
