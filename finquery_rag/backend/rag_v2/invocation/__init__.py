"""The model invocation framework: the boundary a provider plugs into.

    AgentContextPackV1
      -> ContextRendererV1     renderer.py   (a role decides what is shown)
      -> ModelRequestV1        contracts.py  (text, identity, configuration)
      -> ModelProviderV1       provider.py   (transport, serialization, errors)
      -> ModelResponseV1       contracts.py  (what came back)

assembled by ``ModelBindingV1`` (binding.py) and driven by
``ModelInvocationRuntimeV1`` (runtime.py).

The single rule this package exists to make structural: **a provider never
receives an ``AgentContextPackV1``, and nothing else from the Harness's
authorities either.**  A provider handed the pack could read every disclosed
field and every support group, including ones the role's renderer chose not to
put in the prompt -- and "what may this model see" would then have two answers,
one governed and one depending on what a provider felt like looking at.  The
renderer is the only component that decides what crosses, and the request is the
only thing that does.
"""

from .binding import ModelBindingV1
from .contracts import (
    InvocationIntegrityError,
    ModelProviderError,
    ModelRequestV1,
    ModelResponseV1,
    ProviderFailureKind,
)
from .provider import LegacyPromptProviderAdapterV1, ModelProviderV1
from .renderer import CallableContextRendererV1, ContextRendererV1
from .runtime import ModelInvocationRuntimeV1

__all__ = [
    "CallableContextRendererV1",
    "ContextRendererV1",
    "InvocationIntegrityError",
    "LegacyPromptProviderAdapterV1",
    "ModelBindingV1",
    "ModelInvocationRuntimeV1",
    "ModelProviderError",
    "ModelProviderV1",
    "ModelRequestV1",
    "ModelResponseV1",
    "ProviderFailureKind",
]
