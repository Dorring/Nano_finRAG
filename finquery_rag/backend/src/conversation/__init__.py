"""NanoFinance Conversation Context Layer."""

from .config import resolve_multiturn_context_mode
from .contracts import (
    ConversationResolution,
    DialogueState,
    DialogueTurn,
    ReasonCode,
)


from .sqlite_store import (
    ConversationStateConflictError,
    ConversationStateIdentityError,
    ConversationStateSerializationError,
    ConversationStateStoreError,
    ConversationStateVersionError,
    SQLiteConversationStateStore,
)
from .store import ConversationStateStore, InMemoryConversationStore
from .shadow_service import ConversationShadowObservation, ConversationShadowService


__all__ = [
    "DialogueTurn",
    "DialogueState",
    "ConversationResolution",
    "ReasonCode",
    "ConversationStateStore",
    "InMemoryConversationStore",
    "SQLiteConversationStateStore",
    "ConversationStateStoreError",
    "ConversationStateSerializationError",
    "ConversationStateVersionError",
    "ConversationStateConflictError",
    "ConversationStateIdentityError",
    "ConversationShadowObservation",
    "ConversationShadowService",
    "resolve_multiturn_context_mode",
]
