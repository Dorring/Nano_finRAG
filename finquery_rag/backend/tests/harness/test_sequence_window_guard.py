"""The provider's own window, and what it does when a request exceeds it.

Nothing upstream bounds the prompt: no evidence-count limit is configured, no
token bound is configured, and the prompt grows with whatever the binder
admitted.  Handing an over-long sequence to the engine does not return an error
-- it trips a CUDA assertion and takes the process, and with it the service.

So the provider refuses the request instead.  This is a *window* check and not a
context policy: it says this engine cannot serve this request, which is the same
class of statement as the checkpoint being missing.  It does not say how much
context anyone should send -- that is a bound somebody still has to justify with
a measured input distribution, and it is deliberately still unset.

Skipped where torch is absent rather than deleted: the module it tests imports
torch, and the provisioned host is where it runs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from src.generation.local_specialist_generator import exceeds_sequence_window  # noqa: E402

WINDOW = 2048


def test_a_prompt_that_leaves_room_for_generation_is_allowed() -> None:
    assert not exceeds_sequence_window(182, 128, WINDOW)
    assert not exceeds_sequence_window(WINDOW - 128, 128, WINDOW)


def test_the_generated_tokens_count_towards_the_window() -> None:
    """The engine holds the prompt *and* what it generates.

    A prompt that fits on its own can still overflow once generation starts, so
    a check that counted only the prompt would pass exactly the requests that
    crash.  One token over either way is the boundary.
    """

    assert not exceeds_sequence_window(WINDOW - 128, 128, WINDOW)
    assert exceeds_sequence_window(WINDOW - 128 + 1, 128, WINDOW)


def test_an_undeclared_window_is_not_guessed_at() -> None:
    """``None`` means the checkpoint declared no window.

    Checking against a made-up number would be the same class of mistake as the
    fabricated renderer defaults this project already removed once.
    """

    assert not exceeds_sequence_window(10_000, 128, None)


def test_the_numbers_this_corpus_actually_produces_are_nowhere_near_the_window() -> None:
    """Measured, not assumed: the B3 scenarios run 182 to 298 prompt tokens."""

    for measured in (182, 231, 239, 263, 298):
        assert not exceeds_sequence_window(measured, 128, WINDOW)
