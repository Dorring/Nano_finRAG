"""Conversation State Store Interface and In-Memory Implementation.

Provides thread-safe session isolation for multi-turn conversation states.
"""

from __future__ import annotations

import collections
import threading
from abc import ABC, abstractmethod

from .contracts import DialogueState


class ConversationStateStore(ABC):
    """Abstract interface for session-isolated conversation state storage."""

    @abstractmethod
    def get_state(self, conversation_id: str) -> DialogueState | None:
        """Retrieves dialogue state for a conversation."""
        pass

    @abstractmethod
    def save_state(self, state: DialogueState) -> None:
        """Saves dialogue state for a conversation."""
        pass

    @abstractmethod
    def clear_state(self, conversation_id: str) -> None:
        """Clears state for a conversation."""
        pass


class InMemoryConversationStore(ConversationStateStore):
    """Thread-safe in-memory LRU store for dialogue states."""

    def __init__(self, max_conversations: int = 1000) -> None:
        self.max_conversations = max_conversations
        self._store: collections.OrderedDict[str, DialogueState] = collections.OrderedDict()
        self._lock = threading.Lock()

    def get_state(self, conversation_id: str) -> DialogueState | None:
        with self._lock:
            state = self._store.get(conversation_id)
            if state:
                self._store.move_to_end(conversation_id)
            return state

    def save_state(self, state: DialogueState) -> None:
        with self._lock:
            self._store[state.conversation_id] = state
            self._store.move_to_end(state.conversation_id)
            if len(self._store) > self.max_conversations:
                self._store.popitem(last=False)

    def clear_state(self, conversation_id: str) -> None:
        with self._lock:
            self._store.pop(conversation_id, None)
