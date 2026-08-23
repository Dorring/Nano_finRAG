"""Structured Conversation & Resolution Contracts.

Defines immutable/dataclass contracts for:
- DialogueTurn: Single conversation turn metadata
- DialogueState: Semantic dialogue state (not authoritative financial facts)
- ConversationResolution: Structured query reconstruction and ambiguity output
- ReasonCode: Standardized decision codes
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ReasonCode:
    """Standardized decision reason codes for conversation resolution."""
    NO_CONTEXT_REQUIRED = "NO_CONTEXT_REQUIRED"
    EXPLICIT_QUERY_OVERRIDE = "EXPLICIT_QUERY_OVERRIDE"
    
    ENTITY_INHERITED = "ENTITY_INHERITED"
    METRIC_INHERITED = "METRIC_INHERITED"
    PERIOD_INHERITED = "PERIOD_INHERITED"
    SCOPE_INHERITED = "SCOPE_INHERITED"
    
    REFERENCE_RESOLVED = "REFERENCE_RESOLVED"
    ELLIPSIS_RESOLVED = "ELLIPSIS_RESOLVED"
    RELATIVE_PERIOD_RESOLVED = "RELATIVE_PERIOD_RESOLVED"
    CROSS_TURN_CALCULATION_RESOLVED = "CROSS_TURN_CALCULATION_RESOLVED"
    
    TOPIC_SWITCH = "TOPIC_SWITCH"
    
    AMBIGUOUS_REFERENCE = "AMBIGUOUS_REFERENCE"
    AMBIGUOUS_METRIC = "AMBIGUOUS_METRIC"
    AMBIGUOUS_ENTITY = "AMBIGUOUS_ENTITY"
    AMBIGUOUS_PERIOD = "AMBIGUOUS_PERIOD"
    
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    
    RESOLVER_BYPASS = "RESOLVER_BYPASS"
    RESOLVER_RETRY_EXHAUSTED = "RESOLVER_RETRY_EXHAUSTED"
    INVALID_RESOLUTION = "INVALID_RESOLUTION"


@dataclass
class DialogueTurn:
    """Represents a single conversational turn.
    
    Note on Trust Boundary:
    referenced_evidence_ids preserves provenance metadata only.
    It must NEVER be treated as automatically bound evidence for subsequent turns.
    """
    turn_id: str
    user_query: str
    standalone_query: str
    timestamp: float = field(default_factory=time.time)
    assistant_response: str | None = None
    referenced_evidence_ids: list[str] = field(default_factory=list)
    topic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DialogueState:
    """Maintains compressed semantic state across dialogue turns.
    
    Note on Trust Boundary:
    DialogueState stores semantic intent coordinates (entity, metric, period),
    NEVER unverified numeric claims or authoritative financial facts.
    """
    conversation_id: str
    
    # Active focus
    active_entity: str | None = None
    active_metric: str | None = None
    active_period: str | None = None
    active_scope: str | None = None
    
    # Comparison targets (for relative/growth queries)
    comparison_entity: str | None = None
    comparison_metric: str | None = None
    comparison_period: str | None = None
    
    active_topic: str | None = None
    last_resolved_query: str | None = None
    referenced_turn_ids: list[str] = field(default_factory=list)
    
    # Hierarchical history components
    recent_turns: list[DialogueTurn] = field(default_factory=list)
    compressed_history: str | None = None
    turn_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "active_entity": self.active_entity,
            "active_metric": self.active_metric,
            "active_period": self.active_period,
            "active_scope": self.active_scope,
            "comparison_entity": self.comparison_entity,
            "comparison_metric": self.comparison_metric,
            "comparison_period": self.comparison_period,
            "active_topic": self.active_topic,
            "last_resolved_query": self.last_resolved_query,
            "referenced_turn_ids": list(self.referenced_turn_ids),
            "recent_turns": [t.to_dict() for t in self.recent_turns],
            "compressed_history": self.compressed_history,
            "turn_count": self.turn_count,
        }


@dataclass
class ConversationResolution:
    """Structured output from ContextualQueryResolver.
    
    The Conversation Layer outputs ONLY this structure and never directly generates
    financial answers or modifies runtime release authority.
    """
    supported: bool = True
    requires_context: bool = False
    standalone_query: str = ""
    
    # Resolved semantic coordinates
    resolved_entity: str | None = None
    resolved_metric: str | None = None
    resolved_period: str | None = None
    resolved_scope: str | None = None
    
    # Precedence tracking
    inherited_fields: list[str] = field(default_factory=list)
    explicit_fields: list[str] = field(default_factory=list)
    
    # State flags
    topic_switch: bool = False
    ambiguity_detected: bool = False
    clarification_required: bool = False
    clarification_question: str | None = None
    clarification_options: list[str] = field(default_factory=list)
    
    # Provenance and telemetry
    relevant_turn_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
