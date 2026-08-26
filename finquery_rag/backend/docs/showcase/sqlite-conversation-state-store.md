# SQLite Conversation State Store (I4)

I4 adds a persistent implementation of the existing
ConversationStateStore capability. It is a persistence milestone only:
 /query, /query/stream, ConversationContextManager, legacy rewrite, and
V2 remain unchanged.

## Contract audit

The existing store contract in src/conversation/store.py is synchronous and
has three methods:

- get_state(conversation_id)
- save_state(state)
- clear_state(conversation_id)

InMemoryConversationStore remains the default implementation used by
ConversationContextManager and remains available for component tests.

DialogueState is a dataclass of semantic coordinates and provenance
references. I4 adds referenced_evidence_ids and a formal from_dict
round-trip constructor. It still does not contain authoritative answer values
or calculator operands.

SessionManager in src/services/session_manager.py is the raw dialogue source
of truth. It uses a thread-local sqlite3 connection, WAL mode,
busy_timeout=5000, the SESSIONS_DB_PATH environment variable, and the shared
schema_version component-migration table. The new store follows that same
connection and migration pattern and defaults to the same database path.

## Persistence boundary

~~~text
SQLite database
├── conversations
│   └── SessionManager: raw user/assistant messages
└── conversation_states
    └── SQLiteConversationStateStore: structured semantic projection
~~~

The conversation_states primary key is (user_id, session_id). Explicit
production-shaped methods are get, save/put, delete, exists, and
get_state_version. The historical one-key get_state/save_state/
clear_state methods remain available for existing component callers; their
unscoped save path uses a reserved legacy user id of 0 and must not be used
for future production session wiring.

Each row stores:

- schema_version: JSON/table contract version, currently 1
- state_version: monotonically increasing update version
- structured_state_json: validated semantic DialogueState
- compressed_history: non-authoritative context summary
- turn_count
- UTC epoch created_at/updated_at

Writes use one BEGIN IMMEDIATE transaction. Optional
expected_state_version provides deterministic compare-and-set conflict
detection. Connections are thread-local and never shared across threads.

## Serialization and trust boundary

The persisted JSON is produced from DialogueState.to_dict() and validated
with DialogueState.from_dict(). recent_turns is deliberately replaced by an
empty list before persistence: raw turns and assistant text remain owned by
SessionManager. referenced_turn_ids and referenced_evidence_ids are
provenance/reference metadata only; they are not verified evidence or future
calculator operands.

Malformed JSON, mismatched session IDs, forbidden authoritative fields, and
unknown schema versions raise explicit store errors. They never silently
become an empty state. Pickle, repr, answer-value fields, and independent TTL
are not used.

## Production status

I4 status: PERSISTENCE_CAPABILITY_IMPLEMENTED.

- /query: unchanged I3 Contract -> V1 Adapter -> V1 path
- /query/stream: unchanged legacy direct V1 path
- ConversationContextManager: not called by production endpoints
- default ConversationContextManager store: still InMemoryConversationStore
- SessionManager clear/delete lifecycle: not yet wired to this store
- independent conversation TTL: disabled/not introduced

The next integration milestone can explicitly inject
SQLiteConversationStateStore and bind its lifecycle to
(user_id, session_id) without changing the raw-message authority.
