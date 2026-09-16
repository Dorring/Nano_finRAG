"""Runtime mode selection for the trusted V2 agent harness.

NF-V3 H1 changes where answer production happens: ``legacy`` keeps generation
outside the harness loop (the coordinator's candidate stage), while
``harness_v3`` closes the loop so calculation, generation, verification and
release are harness phases.

The default is ``legacy``.  That makes the change an ablation rather than a
replacement: same supervisor, retrieval, binder, calculator, generator and
validator, with only the execution model differing.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import Enum

ENV_VAR = "NF_AGENT_RUNTIME_MODE"


class AgentRuntimeMode(str, Enum):
    """Which execution model the bounded coordinator runs."""

    LEGACY = "legacy"
    HARNESS_V3 = "harness_v3"


class AgentRuntimeModeError(ValueError):
    """Raised when NF_AGENT_RUNTIME_MODE names an unknown mode."""


def resolve_agent_runtime_mode(
    environ: Mapping[str, str] | None = None,
) -> AgentRuntimeMode:
    """Read the runtime mode from the environment, defaulting to legacy."""

    source = os.environ if environ is None else environ
    raw = source.get(ENV_VAR)
    if raw is None or not str(raw).strip():
        return AgentRuntimeMode.LEGACY
    try:
        return AgentRuntimeMode(str(raw).strip())
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in AgentRuntimeMode)
        raise AgentRuntimeModeError(
            f"{ENV_VAR}={raw!r} is not a known mode; expected one of: {allowed}"
        ) from exc


def coerce_agent_runtime_mode(
    value: AgentRuntimeMode | str | None,
) -> AgentRuntimeMode:
    """Normalize an explicit mode, falling back to the environment."""

    if value is None:
        return resolve_agent_runtime_mode()
    if isinstance(value, AgentRuntimeMode):
        return value
    try:
        return AgentRuntimeMode(str(value).strip())
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in AgentRuntimeMode)
        raise AgentRuntimeModeError(
            f"{value!r} is not a known runtime mode; expected one of: {allowed}"
        ) from exc
