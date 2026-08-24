"""Conversation runtime configuration helpers.

I5 exposes only off and shadow. Active conversation rewriting is reserved
for the later rewrite-bypass integration gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ALLOWED_MODES = frozenset({"off", "shadow"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


def resolve_multiturn_context_mode(
    mode: Any = None,
    legacy_enabled: Any = None,
    environ: Mapping[str, Any] | None = None,
) -> str:
    """Return the validated I5 mode, with the new variable taking precedence.

    MULTITURN_CONTEXT_ENABLED is accepted only as a deprecated compatibility
    fallback when MULTITURN_CONTEXT_MODE is absent. The value on is rejected
    until the active rewrite/bypass gate is implemented.
    """
    if mode is None and environ is not None:
        if "MULTITURN_CONTEXT_MODE" in environ:
            mode = environ.get("MULTITURN_CONTEXT_MODE")
        elif legacy_enabled is None:
            legacy_enabled = environ.get("MULTITURN_CONTEXT_ENABLED")
    if mode is None:
        if legacy_enabled is None:
            return "off"
        legacy = str(legacy_enabled).strip().lower()
        if legacy in _TRUE_VALUES:
            return "shadow"
        if legacy in _FALSE_VALUES:
            return "off"
        raise ValueError(
            "MULTITURN_CONTEXT_ENABLED must be a boolean value when "
            "MULTITURN_CONTEXT_MODE is not set",
        )

    normalized = str(mode).strip().lower()
    if normalized == "on":
        raise ValueError(
            "MULTITURN_CONTEXT_MODE=on is not available in I5; "
            "use off or shadow until active rewrite bypass is integrated",
        )
    if normalized not in _ALLOWED_MODES:
        raise ValueError("MULTITURN_CONTEXT_MODE must be one of: off, shadow")
    return normalized
