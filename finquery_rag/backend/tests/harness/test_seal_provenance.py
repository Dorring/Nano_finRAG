"""The seal has to name the revision its numbers belong to.

For the whole P1.3-0D run the seal recorded `commit_sha: null`, because the
field read an environment variable that nobody had exported.  Nothing failed,
nothing warned, and the run produced a seal that could not be checked against
any revision -- which is the one thing a seal exists to do.

So the value is read from the repository, and a revision that cannot be read
raises.  An empty provenance field and a missing one look identical to a
reader, and only one of them is a bug.
"""

from __future__ import annotations

import subprocess

import pytest

from scripts.evaluation import run_p1_2_dual_track_benchmark as runner


@pytest.fixture(autouse=True)
def _no_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("P1_2_COMMIT_SHA", raising=False)


def test_the_override_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run from an exported tree is not a checkout and must still seal."""

    monkeypatch.setenv("P1_2_COMMIT_SHA", "deadbeef")

    assert runner._git_commit_sha() == "deadbeef"


def test_an_unreadable_repository_raises_rather_than_sealing_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("git is not on this machine")

    monkeypatch.setattr(runner.subprocess, "run", boom)

    with pytest.raises(RuntimeError, match="cannot determine the commit"):
        runner._git_commit_sha()


def test_a_git_that_fails_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fails(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(runner.subprocess, "run", fails)

    with pytest.raises(RuntimeError, match="cannot determine the commit"):
        runner._git_commit_sha()


def test_empty_output_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """`check=True` says it ran; it does not say it said anything."""

    class Result:
        stdout = "   \n"

    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: Result())

    with pytest.raises(RuntimeError, match="returned nothing"):
        runner._git_commit_sha()


def test_a_real_checkout_reports_a_real_revision() -> None:
    commit = runner._git_commit_sha()

    assert len(commit) == 40
    assert all(character in "0123456789abcdef" for character in commit)
