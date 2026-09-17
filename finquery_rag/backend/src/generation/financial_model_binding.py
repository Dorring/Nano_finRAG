"""The Financial Model's production model binding.

P1.1.  H2A-3C made the model boundary a contract, and P1 proved the real
Step-156 checkpoint travels it unchanged: the capability reaches the model
through ``ModelBindingV1``, adapts a ``generate(prompt)`` backend, and renders
the compiled pack -- with no Harness core movement.  What P1 left open is that
the binding built in production carried **no identity and no tokenizer**: it was
an anonymous adapter over a checkpoint, so the F1 debt was closed in a
verification script rather than in the contract.

This module is where that closes.  One named factory, one place that knows what
the financial model *is*:

    provider_id         = what the backend declares, or its type name
    model_id            = what the backend declares, never invented here
    renderer            = the specialist renderer
    exact_token_counter = built from the backend's own tokenizer, when it has one

**The binding does not configure a token bound.**  ``ContextBudgetV1`` and the
counter are two different decisions: a counter is a *capability* the binding can
declare, and a bound is a *policy* someone has to justify with a measured input
distribution.  P1 proved the counter is exact; nobody has yet shown that any
particular truncation threshold is safe, so ``max_input_tokens`` stays unset and
the compiler enforces nothing.  Wiring a cap in at the same moment as the
counter would make an unmeasured number look like a considered one.

**Identity is delegated, not stamped.**  ``model_id`` comes from the backend's
own declaration.  A constant applied to whatever backend is passed would label a
test double as the financial model, which is the same class of mistake as the
fabricated ``Scope:`` default the specialist prompt used to carry -- an unknown
turned into an assertion.  A backend that declares no identity yields ``None``,
and ``None`` is the honest answer for a double.

**The counter counts the text it is given, and nothing else.**
``count(text) == len(tokenizer.encode(text))``.  It deliberately does not add
the four chat-frame tokens the provider wraps around a prompt, because the
compiler calls it once on the payload it assembled and a payload never carries a
frame.  A quantity whose unit changes with its caller is not a quantity.  The
distinction the two numbers draw is worth keeping visible:

    Context Compiler token count  = tokenizer.encode(compiler payload)
    model input tokens            = that, plus the renderer's static text,
                                    plus the provider's framing

Both are real; they answer different questions.  Anything that reports "how many
tokens the model actually received" must use the provider's own observed count
(``ModelResponseV1.usage``), never this counter -- which is exactly the
cross-check P1 ran to prove the two sides share one tokenizer.
"""

from __future__ import annotations

from typing import Any

from rag_v2.invocation import (
    CallableContextRendererV1,
    LegacyPromptProviderAdapterV1,
    ModelBindingV1,
    ModelProviderV1,
)

from src.generation.specialist_prompt import render_specialist_prompt

__all__ = [
    "NanochatRustBPEExactTokenCounter",
    "build_financial_model_binding",
]

#: The tokenizer family, for the counter id.  Deliberately not the model step:
#: the counter is a property of the tokenizer, and two checkpoints sharing one
#: tokenizer share one counter id -- which is what makes the id comparable.
_TOKENIZER_FAMILY = "nanochat-rustbpe"


class NanochatRustBPEExactTokenCounter:
    """``ExactTokenCounterV1`` over a checkpoint's own tokenizer.

    The counter's *identity* is derived from the tokenizer rather than asserted
    beside it: ``counter_id`` reports the family and the vocabulary size the
    tokenizer actually has.  An id written as a literal could survive a
    tokenizer swap and go on describing the old one, and the id exists precisely
    so a reader can tell which counter produced a number.
    """

    def __init__(self, tokenizer: Any) -> None:
        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError(
                "an exact token counter needs a tokenizer with encode(), got "
                f"{type(tokenizer).__name__}"
            )
        self._tokenizer = tokenizer

    @property
    def counter_id(self) -> str:
        size = getattr(self._tokenizer, "get_vocab_size", None)
        try:
            vocab = int(size()) if callable(size) else None
        except Exception:
            vocab = None
        return (
            f"{_TOKENIZER_FAMILY}-vocab{vocab}"
            if vocab is not None
            else f"{_TOKENIZER_FAMILY}-unlabelled"
        )

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError(f"a token counter counts text, got {type(text).__name__}")
        return len(self._tokenizer.encode(text))

    def __repr__(self) -> str:
        return f"NanochatRustBPEExactTokenCounter({self.counter_id})"


def build_financial_model_binding(
    model_backend: Any,
    *,
    renderer: Any | None = None,
) -> ModelBindingV1:
    """Bind one financial-model backend into a ``ModelBindingV1``.

    ``model_backend`` is either an object that already implements
    ``ModelProviderV1`` -- used as itself -- or a legacy ``generate(prompt)``
    backend, adapted.  That ``isinstance`` is the whole extension seam: a
    DeepSeek provider arrives as a provider and is not wrapped.

    Two declarations are read from the backend, and both are optional because a
    double has neither:

    * ``model_id`` -- the model's stable, auditable registry identity.  A path
      is not an identity: an absolute Linux path in a semantic field would
      describe one host's filesystem layout, and the checkpoint's SHA256 is
      deployment metadata that belongs with the rest of the fingerprint, not in
      the name of the model.
    * ``tokenizer`` -- present and non-``None`` only after a real checkpoint has
      loaded.  Its presence *is* the claim "this backend can count its own
      tokens", which is the claim ``exact_token_counter`` exists to carry.

    The renderer defaults to the specialist renderer, because a financial
    model's binding is what decides that this model reads this pack in this
    layout.  It is overridable so a test can bind a double without pretending it
    is the production renderer.
    """

    provider = (
        model_backend
        if isinstance(model_backend, ModelProviderV1)
        else LegacyPromptProviderAdapterV1(model_backend)
    )

    declared_provider = getattr(model_backend, "provider_id", None)
    provider_id = (
        declared_provider
        if isinstance(declared_provider, str) and declared_provider.strip()
        else type(model_backend).__name__
    )

    declared_model = getattr(model_backend, "model_id", None)
    model_id = (
        declared_model
        if isinstance(declared_model, str) and declared_model.strip()
        else None
    )

    # ``None`` before ``load()`` and for every double: no tokenizer is no
    # counter, and the binding says so rather than reaching for a substitute.
    tokenizer = getattr(model_backend, "tokenizer", None)
    counter = (
        None if tokenizer is None else NanochatRustBPEExactTokenCounter(tokenizer)
    )

    return ModelBindingV1(
        provider=provider,
        renderer=(
            renderer
            if renderer is not None
            else CallableContextRendererV1(render_specialist_prompt)
        ),
        provider_id=provider_id,
        model_id=model_id,
        exact_token_counter=counter,
    )
