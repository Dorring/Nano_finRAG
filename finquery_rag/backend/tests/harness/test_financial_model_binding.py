"""P1.1: the financial model's binding declares its identity and its tokenizer.

P1 proved the real Step-156 checkpoint travels the migrated path unchanged -- the
capability reaches the model through ``ModelBindingV1``, adapts a
``generate(prompt)`` backend, and renders the compiled pack, with no Harness core
movement.  It proved that in a verification *script*, which is not the same as
proving it in the *contract*: the binding built in production carried no identity
and no tokenizer, so F1 was closed in a script and still open in the wiring.

This file is about the difference.  Two declarations move onto the binding --
``model_id`` and ``exact_token_counter`` -- and one policy deliberately does
**not** move with them.

That asymmetry is the point, not an omission.  A counter is a *capability* the
binding can declare because the backend genuinely has one.  A bound is a *policy*
somebody has to justify with a measured input distribution, and nobody has yet
shown that any particular truncation threshold is safe for this model.  Wiring a
cap in at the moment the counter arrives would make an unmeasured number look
like a considered one.

The expectations here are read off authored fixtures, never off the code under
test.  ``_Tokenizer`` records what it was asked to encode, which is how "the
counter counts the text it is given and nothing else" is checked as a fact
rather than restated as a definition.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest

from rag_v2.context import ExactTokenCounterV1
from rag_v2.invocation import LegacyPromptProviderAdapterV1, ModelProviderV1
from src.generation.financial_model_binding import (
    NanochatRustBPEExactTokenCounter,
    build_financial_model_binding,
)
from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability
from tests.harness.b3_legacy_context_baseline import build_state

BACKEND_SOURCE = pathlib.Path("src/generation/local_specialist_generator.py")


class _Tokenizer:
    """A tokenizer double whose answers are authored, and which remembers.

    One token per character, so an expected count can be read off the input.
    ``encoded`` is the load-bearing part: it is how a test can tell that the
    counter passed the text through unchanged, rather than wrapping it in four
    chat-frame tokens before asking the tokenizer.
    """

    def __init__(self, vocab_size: int = 65000) -> None:
        self._vocab_size = vocab_size
        self.encoded: list[str] = []

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        self.encoded.append(text)
        return list(range(len(text)))


class _FinancialBackend:
    """The shape a *loaded* specialist presents to the factory."""

    def __init__(
        self,
        *,
        tokenizer: Any = None,
        model_id: Any = "nano-finance-2.08b-step156",
    ) -> None:
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> dict[str, Any]:
        self.prompts.append(prompt)
        return {
            "raw_output": "an answer",
            "finish_reason": "stop",
            "tokens_generated": 3,
            "rendered_input_length": 11,
        }


class _BareDouble:
    """A double that declares neither an identity nor a tokenizer."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "an answer"


class _Provider:
    """An object that is already a provider, so it must not be wrapped."""

    provider_id = "a-declared-provider"

    def invoke(self, request: Any) -> Any:  # pragma: no cover - never called here
        raise AssertionError("this test only inspects the binding")


# --- the counter counts the text it is given, and nothing else ---------------------------------


def test_the_counter_counts_exactly_the_text_it_is_given() -> None:
    """No chat frame, no BOS, no decoration -- the string, encoded.

    This is the rule the interface's unit depends on.  The compiler calls
    ``count`` on the payload it assembled, and a payload never carries a chat
    frame; a counter that added one would mean "payload" when the compiler
    called it and "prompt framing" when a provider did, and a quantity whose
    unit changes with its caller is not a quantity.
    """

    tokenizer = _Tokenizer()
    counter = NanochatRustBPEExactTokenCounter(tokenizer)

    assert counter.count("abcd") == 4
    # The load-bearing assertion: the tokenizer saw the string itself.
    assert tokenizer.encoded == ["abcd"]
    # And the frame is absent rather than merely unnoticed -- four extra tokens
    # is what the mistake would look like.
    assert counter.count("abc") == 3


def test_the_counter_reports_the_tokenizer_it_actually_uses() -> None:
    """``counter_id`` is derived, not asserted beside the thing it describes.

    A literal id could survive a tokenizer swap and go on naming the old one,
    and the id exists precisely so a reader can tell which counter produced a
    number.
    """

    small = NanochatRustBPEExactTokenCounter(_Tokenizer(vocab_size=32000))
    large = NanochatRustBPEExactTokenCounter(_Tokenizer(vocab_size=65000))

    assert small.counter_id != large.counter_id
    assert "32000" in small.counter_id
    assert "65000" in large.counter_id
    assert isinstance(large, ExactTokenCounterV1)


def test_the_counter_refuses_something_that_is_not_a_tokenizer() -> None:
    with pytest.raises(TypeError):
        NanochatRustBPEExactTokenCounter(object())


def test_the_counter_refuses_something_that_is_not_text() -> None:
    counter = NanochatRustBPEExactTokenCounter(_Tokenizer())

    with pytest.raises(TypeError):
        counter.count(b"bytes")  # type: ignore[arg-type]


# --- what the binding declares ------------------------------------------------------------------


def test_a_loaded_financial_backend_binds_its_identity_and_its_tokenizer() -> None:
    binding = build_financial_model_binding(
        _FinancialBackend(tokenizer=_Tokenizer())
    )

    assert binding.model_id == "nano-finance-2.08b-step156"
    assert binding.exact_token_counter is not None
    assert isinstance(binding.exact_token_counter, ExactTokenCounterV1)
    assert binding.exact_token_counter.counter_id.endswith("vocab65000")
    assert binding.provider_id == "_FinancialBackend"


def test_a_double_that_declares_nothing_is_labelled_as_nothing() -> None:
    """The anti-fabrication rule, applied to identity.

    A constant stamped on whatever backend was passed would label a test double
    as the financial model -- the same class of mistake as the ``Scope:``
    default the specialist prompt used to invent.  ``None`` is the honest answer
    for a double, and it is also the answer that makes the production binding's
    ``model_id`` worth having.
    """

    binding = build_financial_model_binding(_BareDouble())

    assert binding.model_id is None
    assert binding.exact_token_counter is None
    assert binding.provider_id == "_BareDouble"


def test_an_unloaded_backend_has_an_identity_but_no_tokenizer() -> None:
    """Identity describes which model this is; the tokenizer describes whether
    one is resident.  A specialist that has not loaded declares the first and
    not the second."""

    binding = build_financial_model_binding(_FinancialBackend(tokenizer=None))

    assert binding.model_id == "nano-finance-2.08b-step156"
    assert binding.exact_token_counter is None


def test_a_blank_identity_is_absent_rather_than_used() -> None:
    for blank in ("", "   ", None):
        binding = build_financial_model_binding(
            _FinancialBackend(tokenizer=_Tokenizer(), model_id=blank)
        )
        assert binding.model_id is None, blank


def test_a_provider_arrives_as_itself_and_a_legacy_backend_arrives_wrapped() -> None:
    provider = _Provider()

    as_provider = build_financial_model_binding(provider)
    as_legacy = build_financial_model_binding(_BareDouble())

    assert as_provider.provider is provider
    assert isinstance(provider, ModelProviderV1)
    assert as_provider.provider_id == "a-declared-provider"
    assert isinstance(as_legacy.provider, LegacyPromptProviderAdapterV1)


def test_the_default_renderer_is_the_specialist_renderer() -> None:
    assert build_financial_model_binding(_BareDouble()).renderer_id == (
        "render_specialist_prompt"
    )


def test_the_model_id_is_a_registry_name_and_not_a_path() -> None:
    """A path is not an identity.

    ``local_specialist_generator`` imports torch, so the constant is read from
    the syntax tree rather than by importing the module -- and the rule being
    checked is about the value's *shape*, which an AST read answers exactly.
    """

    tree = ast.parse(BACKEND_SOURCE.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODEL_ID":
                    assert isinstance(node.value, ast.Constant)
                    values.append(str(node.value.value))

    assert len(values) == 1, "MODEL_ID should be declared exactly once"
    model_id = values[0]
    assert "/" not in model_id and "\\" not in model_id, model_id
    assert not model_id.endswith(".pt"), model_id
    assert model_id.strip() == model_id and model_id


# --- the counter is wired; the bound is deliberately not ----------------------------------------


def test_the_binding_carries_a_counter_and_configures_no_token_bound() -> None:
    """The P1.1 decision, pinned as a behaviour rather than as a comment.

    A counter without a bound is a real state, not a half-finished one: the
    binding can say exactly how many tokens a payload is, and the compiler
    enforces nothing until somebody justifies a number.
    """

    capability = TrustedV2GenerationCapability(
        model_backend=_FinancialBackend(tokenizer=_Tokenizer())
    )

    assert capability.binding is not None
    assert capability.binding.exact_token_counter is not None
    # The counter reaches the compiler...
    assert capability.context_compiler.token_counter is (
        capability.binding.exact_token_counter
    )
    # ...and no bound was configured alongside it.
    assert capability.context_compiler.budget.max_input_tokens is None


def test_the_pack_records_the_counter_and_no_bound() -> None:
    """Measured, not enforced.

    A counter with no bound still measures: the pack carries which counter
    produced its number, so the number is attributable even while nothing acts
    on it.  Both halves are asserted together, because "the counter is wired"
    and "a cap is enforced" are the two claims most easily confused.
    """

    backend = _FinancialBackend(tokenizer=_Tokenizer())
    capability = TrustedV2GenerationCapability(model_backend=backend)

    capability.generate(build_state("multi_fact"))

    pack = capability.last_context_pack
    assert pack is not None
    assert pack.budget.token_counter_id == "nanochat-rustbpe-vocab65000"
    assert pack.budget.max_input_tokens is None
    assert isinstance(pack.budget.selected_context_tokens, int)
    assert pack.budget.selected_context_tokens > 0
