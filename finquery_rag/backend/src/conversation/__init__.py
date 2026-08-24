"""NanoFinance Conversation Context Layer."""

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
]
