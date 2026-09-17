"""The rendering boundary: a role decides what the model is shown.

A renderer turns a governed pack into the text a model reads.  It is the last
place a semantic decision is made, and the first place something model-specific
is -- which is why it sits between the pack and the provider rather than inside
either.

**What a renderer may decide:** textual layout, static instructions, how the
disclosed evidence is represented, and whether an explicitly permitted piece of
support topology is rendered.

**What a renderer may not do:** read a RunState, retrieve raw evidence, invoke
Disclosure Authority again, reconstruct a slot binding, canonicalise a financial
value, or repair missing context.  Every one of those is a decision another
component already made and recorded, and a renderer repeating it would be a
second answer to a settled question.  A pack is deliberately the *only* input a
renderer has, so most of that list is unreachable rather than forbidden.

The protocol is a method rather than a callable because a renderer is a named
role artefact -- a binding says which one it uses, and a trace naming
``render_specialist_prompt`` is more useful than a trace naming ``<function>``.
``CallableContextRendererV1`` exists for the renderer the repository already
has, which is a plain function and does not need to become a class to be
adopted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from rag_v2.context import AgentContextPackV1

__all__ = [
    "CallableContextRendererV1",
    "ContextRendererV1",
]


@runtime_checkable
class ContextRendererV1(Protocol):
    """One role's answer to "how is this pack turned into model input?"."""

    def render(self, pack: AgentContextPackV1) -> str: ...


class CallableContextRendererV1:
    """Adapt a plain ``pack -> str`` function to the renderer protocol.

    Not a convenience: the specialist renderer already exists as a function with
    exactly this signature, and giving it a class would be ceremony that changed
    no behaviour while making every existing call site construct something.  The
    adapter says the function *is* a renderer, which it is.

    It also refuses a callable that is not callable, for the same reason
    ``ModelBindingV1`` refuses a provider without ``invoke``: the failure should
    happen where the binding is configured, not where the first request is made.
    """

    def __init__(self, render: Callable[[AgentContextPackV1], str]) -> None:
        if not callable(render):
            raise TypeError("a context renderer must be callable")
        self._render = render

    @property
    def renderer_id(self) -> str:
        return getattr(self._render, "__name__", type(self._render).__name__)

    def render(self, pack: AgentContextPackV1) -> str:
        return self._render(pack)

    def __repr__(self) -> str:
        return f"CallableContextRendererV1({self.renderer_id})"
