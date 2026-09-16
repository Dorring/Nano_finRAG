"""NF-V3 H1-D5: the runtime mode flag that makes H1 an ablation.

Default must stay ``legacy`` so that nothing changes unless it is asked for.
"""

from __future__ import annotations

import pytest

from src.runtime.harness_runtime_mode import (
    ENV_VAR,
    AgentRuntimeMode,
    AgentRuntimeModeError,
    coerce_agent_runtime_mode,
    resolve_agent_runtime_mode,
)


def test_defaults_to_legacy_when_unset() -> None:
    assert resolve_agent_runtime_mode({}) is AgentRuntimeMode.LEGACY


def test_blank_value_is_treated_as_unset() -> None:
    assert resolve_agent_runtime_mode({ENV_VAR: "   "}) is AgentRuntimeMode.LEGACY


@pytest.mark.parametrize("mode", list(AgentRuntimeMode))
def test_reads_each_known_mode(mode: AgentRuntimeMode) -> None:
    assert resolve_agent_runtime_mode({ENV_VAR: mode.value}) is mode


def test_unknown_mode_is_rejected_loudly() -> None:
    with pytest.raises(AgentRuntimeModeError) as excinfo:
        resolve_agent_runtime_mode({ENV_VAR: "multi_agent"})
    assert ENV_VAR in str(excinfo.value)
    assert "legacy" in str(excinfo.value)


def test_coerce_prefers_an_explicit_value_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_VAR, "harness_v3")
    assert coerce_agent_runtime_mode(AgentRuntimeMode.LEGACY) is AgentRuntimeMode.LEGACY
    assert coerce_agent_runtime_mode("harness_v3") is AgentRuntimeMode.HARNESS_V3
    # None means "ask the environment", which the monkeypatch has set.
    assert coerce_agent_runtime_mode(None) is AgentRuntimeMode.HARNESS_V3


def test_coordinator_defaults_to_legacy_mode() -> None:
    from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
    from src.runtime.trusted_v2_capabilities import TrustedV2CapabilityPorts
    from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator

    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(DeterministicFallbackProvider({})),
        capabilities=TrustedV2CapabilityPorts(),
    )
    assert coordinator.runtime_mode is AgentRuntimeMode.LEGACY
