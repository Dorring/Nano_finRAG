"""The binding: which provider, which renderer, and which tokenizer.

H2A-3C.  This is the seam the whole phase is for.  Adding a financial model or a
DeepSeek endpoint should be a new binding and nothing else -- no change to
evidence handling, binding, disclosure, compilation or the pack, because none of
those knows what a provider is.

    financial_binding = ModelBindingV1(
        provider=FinancialModelProvider(...),
        renderer=FinancialModelContextRendererV1(),
        provider_id="financial-model",
        model_id="...",
        exact_token_counter=FinancialModelTokenCounter(...),
    )

The three things it binds answer three different questions, and they are bound
together rather than configured separately because they are only meaningful
together: a tokenizer belongs to a model, and a renderer's layout is what that
model was trained to read.  A binding that could pair one model's tokenizer with
another's prompt format would be configurable into nonsense.

**The counter is the tokenizer seam and nothing else.**  It is optional, and the
rule it participates in is unchanged from H2A-3B1: a configured token bound with
an exact counter is enforced; a configured token bound without one is a
configuration failure.  No word count, character count or substitute tokenizer
may be introduced anywhere to make a missing counter go away.  B3 configures no
token bound, so the counter here is ``None`` and nothing is enforced -- which is
the honest state, not an omission.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_v2.context import ExactTokenCounterV1

from .contracts import InvocationIntegrityError
from .provider import ModelProviderV1
from .renderer import ContextRendererV1

__all__ = ["ModelBindingV1"]


@dataclass(frozen=True)
class ModelBindingV1:
    """One model boundary: how it is reached, how it is addressed, what it counts.

    Validated at construction rather than at first use, for the same reason
    ``ContextCompilerV1`` refuses an unmeasurable token bound there: a binding is
    deployment configuration, and a misconfigured one should fail where it is
    written rather than on the first request that happens to reach it.
    """

    provider: ModelProviderV1
    renderer: ContextRendererV1
    provider_id: str
    model_id: str | None = None
    exact_token_counter: ExactTokenCounterV1 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, ModelProviderV1):
            raise InvocationIntegrityError(
                f"a binding's provider must implement ModelProviderV1.invoke, got "
                f"{type(self.provider).__name__}"
            )
        if not isinstance(self.renderer, ContextRendererV1):
            raise InvocationIntegrityError(
                f"a binding's renderer must implement ContextRendererV1.render, got "
                f"{type(self.renderer).__name__}"
            )
        if not isinstance(self.provider_id, str) or not self.provider_id.strip():
            raise InvocationIntegrityError("provider_id must be a non-empty string")
        if self.model_id is not None and (
            not isinstance(self.model_id, str) or not self.model_id.strip()
        ):
            raise InvocationIntegrityError(
                "model_id must be a non-empty string or None"
            )
        if self.exact_token_counter is not None and not isinstance(
            self.exact_token_counter, ExactTokenCounterV1
        ):
            raise InvocationIntegrityError(
                f"exact_token_counter must implement ExactTokenCounterV1, got "
                f"{type(self.exact_token_counter).__name__}"
            )

    @property
    def renderer_id(self) -> str:
        """A name for the renderer, for a trace that must not carry its output."""

        return getattr(
            self.renderer, "renderer_id", type(self.renderer).__name__
        )
